"""
GERÇEK Pipeline Backtest Motoru — toy 5-kural değil, asıl karar zinciri.

Mevcut backtest/engine.py 5 sabit kuralı test eder; bu modül ise GERÇEK
DecisionEngine + RiskEngine'i (konsensüs kapısı + scalp 3-hedef + kademeli
çıkış) bar-bar koşturur. Look-ahead yok: her bar yalnızca o ana kadarki veriyle
oy üretir; giriş bir SONRAKİ barın açılışından yapılır; çıkışlar barların
high/low'una göre tetiklenir.

Çevrimdışı dürüstlük: ağ-bağımlı katmanlar (sentiment, news, onchain, macro,
fear_greed) geçmiş veride yeniden üretilemez → bu testte DEVRE DIŞI. Test
edilen, fiyat-türevli konsensüs: technical · pattern · trendline · smc ·
multi_timeframe. (Haber vetosu canlıda ayrıca devrededir.)

3-hedef kademeli çıkış Portfolio'daki mantığı birebir yansıtır:
  TP1/TP2'de kalanın 1/3'ü realize edilir, TP1 sonrası stop girişe çekilir,
  TP3 (veya stop/iptal/max-hold) kalanı kapatır.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import Config
from ..core.models import AnalysisResult, Direction
from ..risk.risk_engine import RiskEngine
from ..decision.decision_engine import DecisionEngine
from ..analysis.indicators import (
    technical_vote, atr as atr_fn, ema, rsi, macd, adx, supertrend, psar,
    bollinger, cmf, stoch, mfi, vwap, sma, ichimoku,
)
from ..analysis.patterns import detect_patterns, extreme_analysis
from ..analysis.trendlines import trendline_vote
from ..analysis.smc import smc_vote

FEE = 0.0004        # (eski sabit — geriye dönük uyumluluk; artık config kullanılır)
SLIPPAGE = 0.0002


def load_csv(path: str) -> pd.DataFrame:
    """data_6m CSV'lerini datetime-index'li OHLCV'ye yükle."""
    df = pd.read_csv(path)
    if "dt" in df.columns:
        df.index = pd.to_datetime(df["dt"])
    elif "ts" in df.columns:
        df.index = pd.to_datetime(df["ts"], unit="ms")
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def precompute_tech(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """technical_vote'un ihtiyaç duyduğu ~20 göstergeyi TÜM df üzerinde BİR KEZ
    (kayan/özyinelemeli, look-ahead'siz) hesapla → bar-bar O(1) indeksleme.
    Bu, pencere başına compute_all_indicators (123 ms) maliyetini ortadan kaldırır."""
    c = df["close"]
    pre: Dict[str, np.ndarray] = {}
    pre["close"] = c.values
    for n in (21, 50, 55, 200):
        pre[f"ema_{n}"] = ema(c, n).values
    pre["rsi_14"] = rsi(c, 14).values
    pre["macd_hist_12_26"] = macd(c)[2].values
    adx_v, pdi, mdi = adx(df)
    pre["adx_14"] = adx_v.values; pre["plus_di"] = pdi.values; pre["minus_di"] = mdi.values
    pre["supertrend_dir"] = supertrend(df)[1].values
    pre["psar"] = psar(df).values
    bb_u, bb_m, bb_l = bollinger(c)
    pre["bb_pct_b"] = ((c - bb_l) / (bb_u - bb_l + 1e-12)).values
    ten, kij, span_a, span_b = ichimoku(df)
    sa = span_a.values.copy(); sb = span_b.values.copy()
    tv = ten.values; kv = kij.values
    sa[np.isnan(sa)] = tv[np.isnan(sa)]; sb[np.isnan(sb)] = kv[np.isnan(sb)]
    pre["ichimoku_span_a"] = sa; pre["ichimoku_span_b"] = sb
    pre["cmf_20"] = cmf(df).values
    pre["rel_volume"] = (df["volume"] / (sma(df["volume"], 20) + 1e-12)).values
    pre["stoch_k"] = stoch(df)[0].values
    pre["mfi_14"] = mfi(df, 14).values
    pre["vwap"] = vwap(df).values
    pre["heikin_ashi"] = _heikin_ashi_series(df)
    pre["atr_14"] = atr_fn(df, 14).values
    return pre


def _heikin_ashi_series(df: pd.DataFrame) -> np.ndarray:
    """heikin_ashi_signal'in vektörize hâli (+1/-1 güçlü, ±0.5 trend, 0 kararsız)."""
    o = df["open"].values; h = df["high"].values
    l = df["low"].values; c = df["close"].values
    n = len(df)
    ha_close = (o + h + l + c) / 4
    ha_open = np.empty(n)
    if n:
        ha_open[0] = o[0]
        for i in range(1, n):
            ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2
    ha_high = np.maximum.reduce([h, ha_open, ha_close])
    ha_low = np.minimum.reduce([l, ha_open, ha_close])
    out = np.where(ha_close > ha_open, 0.5, np.where(ha_close < ha_open, -0.5, 0.0))
    out = np.where((ha_close > ha_open) & (ha_low == ha_open), 1.0, out)
    out = np.where((ha_close < ha_open) & (ha_high == ha_open), -1.0, out)
    return out


def _tech_scalar(pre: Dict[str, np.ndarray], i: int) -> Dict[str, float]:
    return {k: float(v[i]) for k, v in pre.items()}


def _offline_votes(w: pd.DataFrame, ind_i: Dict[str, float]) -> list:
    """Fiyat-türevli katman oyları (look-ahead'siz, ağ katmanları hariç).
    technical önceden hesaplanmış skalar göstergeyi kullanır; pattern/trendline/smc
    pencereyi kullanır. Konsensüs için 4 aktif katman → min 3 kuralı sağlanır."""
    votes: list = []
    for fn in (lambda: technical_vote(w, ind_i),
               lambda: detect_patterns(w)[1],
               lambda: trendline_vote(w),
               lambda: smc_vote(w)):
        try:
            votes.append(fn())
        except Exception:
            pass
    return votes


def run_real_backtest(df: pd.DataFrame, config: Config, symbol: str = "?", tf: str = "1h",
                      lookback: int = 400, warmup: int = 220, position_fraction: float = 1.0,
                      max_hold: int = 48) -> Dict:
    """Tek parite üzerinde gerçek karar zincirini koştur, işlem + equity üret."""
    risk = RiskEngine(config)
    decision = DecisionEngine(config, risk)

    o = df["open"].values; h = df["high"].values
    l = df["low"].values; c = df["close"].values
    idx = df.index
    n = len(df)
    if n < warmup + 50:
        return {"error": "yetersiz veri", "symbol": symbol}

    pre = precompute_tech(df)          # tüm göstergeleri bir kez hesapla
    atr_arr = pre["atr_14"]
    # FAZ4 — gerçekçi yürütme maliyeti
    rc = config.get("risk", {})
    fee_taker = float(rc.get("fee_taker", 0.0005))
    slip_base = float(rc.get("slippage_base", 0.0003))
    slip_vol_k = float(rc.get("slippage_vol_k", 0.15))
    # FAZ3 — risk-temelli boyutlama
    sizing_mode = str(rc.get("sizing_mode", "full")).lower()
    risk_per_trade = float(rc.get("risk_per_trade", 0.01))
    max_lev = float(rc.get("max_leverage", 1.0))
    dd_scaling = bool(rc.get("dd_risk_scaling", True))
    cooldown_n = int(rc.get("loss_cooldown_n", 3))
    consec_loss = 0

    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    eq_curve = [equity]
    trades: List[Dict] = []
    pos: Optional[Dict] = None
    eval_count = 0
    actionable_count = 0

    i = warmup
    while i < n - 1:
        # ---------- açık pozisyon yönetimi (giriş barından SONRAKİ barlar) ----------
        if pos is not None:
            side = pos["side"]
            hi, lo = h[i], l[i]
            closed = False

            # 1) STOP (konservatif: TP'den önce kontrol)
            stop_hit = (lo <= pos["stop"]) if side > 0 else (hi >= pos["stop"])
            if stop_hit:
                _book(pos, pos["stop"], pos["remaining"], side)
                _finalize(pos, trades, idx[i], "STOP")
                equity *= (1 + pos["size"] * pos["ret"])
                consec_loss = consec_loss + 1 if pos["ret"] <= 0 else 0
                pos = None; closed = True

            # 2) TP merdiveni (kademeli)
            if not closed:
                tps = pos["tps"]
                for k in range(pos["tp_hits"], len(tps)):
                    tp = tps[k]
                    tp_hit = (hi >= tp) if side > 0 else (lo <= tp)
                    if not tp_hit:
                        break
                    pos["tp_hits"] = k + 1
                    if k + 1 >= len(tps):                 # son hedef → tümü kapat
                        _book(pos, tp, pos["remaining"], side)
                        _finalize(pos, trades, idx[i], f"TP{k+1}")
                        equity *= (1 + pos["size"] * pos["ret"])
                        consec_loss = consec_loss + 1 if pos["ret"] <= 0 else 0
                        pos = None; closed = True
                        break
                    portion = 1.0 / 3.0                   # ara hedef → kalanın 1/3'ü
                    _book(pos, tp, portion, side)
                    pos["remaining"] = max(0.0, pos["remaining"] - portion)
                    pos["stop"] = pos["entry"]            # breakeven

            # 3) max-hold süre dolumu
            if not closed and (i - pos["entry_bar"]) >= max_hold:
                _book(pos, c[i], pos["remaining"], side)
                _finalize(pos, trades, idx[i], "SÜRE")
                equity *= (1 + pos["size"] * pos["ret"])
                consec_loss = consec_loss + 1 if pos["ret"] <= 0 else 0
                pos = None; closed = True

            peak = max(peak, equity)
            max_dd = max(max_dd, (peak - equity) / peak)
            eq_curve.append(equity)
            i += 1
            continue

        # ---------- pozisyon yok: karar üret ----------
        w = df.iloc[max(0, i - lookback):i + 1]
        ind_i = _tech_scalar(pre, i)
        votes = _offline_votes(w, ind_i)
        analysis = AnalysisResult(symbol=symbol, timeframe=tf, last_price=float(c[i]),
                                  votes=votes, indicators=ind_i, extremes=extreme_analysis(w))
        atr_val = float(atr_arr[i]) if not np.isnan(atr_arr[i]) else float(c[i]) * 0.01
        sig = decision.decide(symbol, tf, float(c[i]), atr_val, votes, analysis, df=w)
        eval_count += 1

        if sig.actionable and sig.direction != Direction.FLAT and i + 1 < n:
            actionable_count += 1
            side = 1 if sig.direction == Direction.LONG else -1
            entry = float(o[i + 1])                       # sonraki barın açılışı (fiyatta kayma yok)
            atr_pct = float(atr_arr[i]) / (entry + 1e-12) if not np.isnan(atr_arr[i]) else 0.01
            slip_pct = slip_base + slip_vol_k * atr_pct   # volatilite-ölçekli kayma (her yön)
            cost_rt = 2 * fee_taker + 2 * slip_pct        # round-trip toplam maliyet (kesir)
            # FAZ3 — pozisyon boyutu: sabit-fraksiyonel risk (vol hedefleme)
            if sizing_mode == "fixed_fraction":
                stop_dist = abs(entry - float(sig.stop_loss)) / (entry + 1e-12)
                rb = risk_per_trade
                if dd_scaling:
                    cur_dd = (peak - equity) / (peak + 1e-12)
                    rb *= max(0.3, 1 - 2 * cur_dd)        # drawdown'da riski azalt
                if consec_loss >= cooldown_n:
                    rb *= 0.5                              # ardışık-kayıp soğuması
                size = min(rb / (stop_dist + 1e-9), max_lev)
            else:
                size = position_fraction
            pos = {"side": side, "entry": entry, "stop": float(sig.stop_loss),
                   "tps": [float(t) for t in sig.take_profits], "remaining": 1.0,
                   "tp_hits": 0, "ret": 0.0, "entry_bar": i + 1, "cost_rt": cost_rt,
                   "size": float(size),
                   "entry_ts": idx[i + 1], "confidence": sig.confidence,
                   "direction": sig.direction.value}
            i += 1   # girişi yaptık; bir sonraki bardan yönetime geç
            eq_curve.append(equity)
            continue

        eq_curve.append(equity)
        i += 1

    return _summarize(trades, eq_curve, equity, max_dd, symbol, tf,
                      eval_count, actionable_count, df)


def walk_forward(df: pd.DataFrame, config: Config, symbol: str = "?", tf: str = "1h",
                 folds: int = 3) -> Dict:
    """Veriyi `folds` ardışık pencereye böl, her birinde ayrı backtest koştur →
    sağlamlık: konfig tek döneme mi uydu (overfit) yoksa dönemler arası tutarlı mı?
    Her pencere kendi warmup'ını içerir (look-ahead yok)."""
    n = len(df)
    seg = n // folds
    out = []
    for k in range(folds):
        s0 = k * seg
        s1 = (k + 1) * seg if k < folds - 1 else n
        sub = df.iloc[s0:s1]
        r = run_real_backtest(sub, config, symbol=symbol, tf=tf)
        out.append({"fold": k + 1, "bars": len(sub),
                    "from": str(sub.index[0])[:10], "to": str(sub.index[-1])[:10],
                    "trades": r.get("trades", 0), "win_rate": r.get("win_rate", 0),
                    "return_pct": r.get("total_return_pct", 0),
                    "max_dd": r.get("max_drawdown_pct", 0)})
    pos = sum(1 for f in out if f["return_pct"] > 0)
    rets = [f["return_pct"] for f in out]
    return {"symbol": symbol, "folds": out, "positive_folds": pos, "n_folds": folds,
            "mean_fold_return": round(float(np.mean(rets)), 2) if rets else 0.0,
            "robust": pos >= folds - 0 if folds <= 2 else pos >= folds - 1}


def _book(pos: Dict, exit_price: float, portion: float, side: int) -> None:
    """Bir porsiyonu realize et (FAZ4: volatilite-ölçekli round-trip maliyet düşülür)."""
    gross = side * (exit_price / pos["entry"] - 1)
    net = gross - pos.get("cost_rt", 2 * FEE + 2 * SLIPPAGE)   # round-trip fee + kayma
    pos["ret"] += portion * net


def _finalize(pos: Dict, trades: List[Dict], exit_ts, reason: str) -> None:
    trades.append({
        "symbol_ts": str(pos["entry_ts"]), "exit_ts": str(exit_ts),
        "direction": pos["direction"], "entry": round(float(pos["entry"]), 6),
        "reason": reason, "ret_pct": round(float(pos["ret"]) * 100, 4),
        "win": bool(pos["ret"] > 0), "confidence": round(float(pos["confidence"]), 3),
        "tp_hits": int(pos["tp_hits"]),
    })


def _psr(rets, sr_benchmark: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio — TEK KAYNAK: `research.validation.psr`.

    Buradaki kopya, doğrulama katmanı kurulmadan önce yazılmıştı. İki ayrı
    uygulamanın zamanla ayrışmaması için delege ediliyor (davranış aynı)."""
    from ..research.validation import psr as _canonical_psr
    return _canonical_psr(rets, sr_benchmark)


def _summarize(trades, eq_curve, equity, max_dd, symbol, tf, evals, actionable, df) -> Dict:
    rets = [t["ret_pct"] / 100 for t in trades]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    win_rate = 100 * len(wins) / len(rets) if rets else 0.0
    gross_win = sum(wins); gross_loss = abs(sum(losses))
    pf = gross_win / (gross_loss + 1e-9) if gross_loss else (gross_win and 99.0 or 0.0)
    arr = np.array(rets) if rets else np.array([0.0])
    expectancy = float(arr.mean() * 100) if rets else 0.0
    sharpe = float(arr.mean() / (arr.std() + 1e-9) * np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    psr = round(_psr(rets), 3) if len(rets) >= 5 else 0.0   # FAZ5: istatistiksel güven

    # aylık dağılım (çıkış zaman damgasına göre, ay içinde bileşik)
    monthly: Dict[str, float] = {}
    for t in trades:
        ym = str(t["exit_ts"])[:7]
        monthly[ym] = monthly.get(ym, 1.0) * (1 + t["ret_pct"] / 100)
    monthly_pct = {k: round((v - 1) * 100, 2) for k, v in sorted(monthly.items())}
    n_months = max(1, len(monthly_pct))
    total_ret = (equity - 1) * 100
    avg_monthly = float(np.mean(list(monthly_pct.values()))) if monthly_pct else 0.0

    # zaman aralığı (gün)
    try:
        days = (df.index[-1] - df.index[0]).days or 1
    except Exception:
        days = 1
    daily_avg = total_ret / days if days else 0.0

    return {
        "symbol": symbol, "timeframe": tf,
        "bars": len(df), "evaluations": evals, "actionable_signals": actionable,
        "trades": len(trades),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(float(pf), 2),
        "expectancy_pct_per_trade": round(expectancy, 4),
        "total_return_pct": round(total_ret, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "sharpe": round(sharpe, 2),
        "months": n_months,
        "avg_monthly_pct": round(avg_monthly, 2),
        "est_daily_pct": round(daily_avg, 3),
        "monthly_returns": monthly_pct,
        "sample_trades": trades[:8],
    }
