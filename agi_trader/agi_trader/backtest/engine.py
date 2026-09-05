"""
Backtest + Walk-Forward Motoru (#3) — Backtest/Strateji Doğrulama rolü.

Event-driven: her bar için teknik karar skoru hesaplanır (look-ahead bias yok —
yalnızca o ana kadarki veri), skor eşiği aşılırsa pozisyon açılır, sonraki
barların high/low'una göre SL/TP veya ters sinyalle kapanır. Gerçekçi fee +
slippage uygulanır.

Çıktı: trade sayısı, kazanç oranı, profit factor, net getiri, max drawdown,
Sharpe, equity curve + walk-forward kat bazlı dağılım.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ..analysis.indicators import ema, rsi, macd, atr, supertrend

FEE = 0.0004       # taker ~%0.04
SLIPPAGE = 0.0002


def _signal_series(df: pd.DataFrame) -> pd.Series:
    """Her bar için -1..+1 teknik karar skoru (vektörize, look-ahead'siz)."""
    c = df["close"]
    e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
    r = rsi(c, 14)
    _, _, hist = macd(c)
    _, st_dir = supertrend(df)

    s = pd.Series(0.0, index=df.index)
    s += np.where((c > e50) & (e50 > e200), 0.30, np.where((c < e50) & (e50 < e200), -0.30, 0.0))
    s += np.where(e20 > e50, 0.15, -0.15)
    s += np.where(r < 35, 0.20, np.where(r > 65, -0.20, 0.0))
    s += np.where(hist > 0, 0.15, -0.15)
    s += np.where(st_dir > 0, 0.20, -0.20)
    return s.clip(-1, 1)


def _simulate(df: pd.DataFrame, sig: pd.Series, thr: float = 0.35,
              atr_mult: float = 2.0, rr: float = 2.0,
              start: int = 200) -> Dict:
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    a = atr(df, 14).values

    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    curve = [equity]
    trades: List[float] = []

    pos = None  # dict: side, entry, stop, tp
    for i in range(start, len(df) - 1):
        if pos is None:
            score = sig.iloc[i]
            if abs(score) >= thr and a[i] > 0:
                side = 1 if score > 0 else -1
                entry = close[i] * (1 + side * SLIPPAGE)
                stop = entry - side * atr_mult * a[i]
                tp = entry + side * atr_mult * a[i] * rr
                pos = {"side": side, "entry": entry, "stop": stop, "tp": tp}
        else:
            side = pos["side"]
            hit_tp = high[i] >= pos["tp"] if side == 1 else low[i] <= pos["tp"]
            hit_sl = low[i] <= pos["stop"] if side == 1 else high[i] >= pos["stop"]
            exit_price = None
            if hit_sl:                       # konservatif: SL önce
                exit_price = pos["stop"]
            elif hit_tp:
                exit_price = pos["tp"]
            elif np.sign(sig.iloc[i]) == -side and abs(sig.iloc[i]) >= thr:
                exit_price = close[i]        # ters sinyal
            if exit_price is not None:
                ret = side * (exit_price - pos["entry"]) / pos["entry"] - 2 * FEE
                equity *= (1 + ret)
                trades.append(ret)
                peak = max(peak, equity)
                max_dd = max(max_dd, (peak - equity) / peak)
                pos = None
        curve.append(equity)

    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    win_rate = 100 * len(wins) / len(trades) if trades else 0.0
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    pf = gross_win / (gross_loss + 1e-9) if gross_loss > 0 else (gross_win and 99.0 or 0.0)
    rets = np.array(trades) if trades else np.array([0.0])
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252)) if len(rets) > 1 else 0.0

    return {
        "trades": len(trades),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(float(pf), 2),
        "total_return": round((equity - 1) * 100, 2),
        "max_drawdown": round(max_dd * 100, 2),
        "sharpe": round(sharpe, 2),
        "equity_curve": [round(x, 4) for x in curve],
        "trade_returns": [round(float(t), 5) for t in trades],
    }


def monte_carlo_backtest(trade_returns: List[float], n_sims: int = 1000, seed: int = 42) -> Dict:
    """İşlem getirilerini 1000× bootstrap (yerine koyarak) yeniden örnekle →
    sıralamadan bağımsız sağlam istatistik: getiri dağılımı, kâr olasılığı,
    en kötü drawdown. Tek bir backtest'in 'şansa' bağlı olup olmadığını gösterir."""
    if not trade_returns or len(trade_returns) < 5:
        return {"available": False, "reason": "Monte Carlo için ≥5 işlem gerekli"}
    rng = np.random.default_rng(seed)
    arr = np.array(trade_returns, dtype=float)
    finals = np.empty(n_sims)
    dds = np.empty(n_sims)
    for i in range(n_sims):
        sample = rng.choice(arr, size=len(arr), replace=True)
        eq = np.cumprod(1 + sample)
        finals[i] = eq[-1] - 1
        peak = np.maximum.accumulate(eq)
        dds[i] = ((peak - eq) / peak).max()
    finals *= 100
    dds *= 100
    return {
        "available": True, "sims": n_sims, "n_trades": len(arr),
        "return_mean": round(float(finals.mean()), 2),
        "return_median": round(float(np.median(finals)), 2),
        "return_p5": round(float(np.percentile(finals, 5)), 2),
        "return_p95": round(float(np.percentile(finals, 95)), 2),
        "prob_profit": round(float((finals > 0).mean() * 100), 1),
        "maxdd_median": round(float(np.median(dds)), 2),
        "maxdd_p95": round(float(np.percentile(dds, 95)), 2),
        "verdict": ("sağlam" if (finals > 0).mean() > 0.7 else
                    "kırılgan" if (finals > 0).mean() < 0.5 else "orta"),
    }


def _score(res: Dict) -> float:
    """Risk-ayarlı sağlamlık skoru: getiri/drawdown (Calmar-benzeri), işlem sayısı
    kapısı + profit factor bonusu. Yetersiz örnek/zarar eden config elenir."""
    if res["trades"] < 6 or res["profit_factor"] < 1.0:
        return -999.0
    calmar = res["total_return"] / (res["max_drawdown"] + 2.0)
    return round(calmar * (1 + min(res["profit_factor"], 4) / 5), 3)


# Aranan parametre ızgarası (yürütme: stop/TP/eşik) — sinyal mantığı sabit
_GRID_ATR = [1.5, 2.0, 2.5, 3.0]      # ATR stop çarpanı
_GRID_RR = [1.5, 2.0, 2.5, 3.0]       # TP / stop oranı
_GRID_THR = [0.30, 0.40, 0.50]        # sinyal eşiği


def optimize_parameters(orch, symbol: str, timeframe: str = "4h", bars: int = 800) -> Dict:
    """Stop/TP/eşik ızgarasını tarar, risk-ayarlı skora göre en iyi kombinasyonu
    bulur ve kazananı Monte Carlo ile doğrular. 'Doğru miktarlarda al-sat' için
    motor parametrelerini optimize eder."""
    try:
        df = orch.data.fetch_ohlcv(symbol, timeframe, limit=bars)
    except Exception as e:
        return {"error": f"veri alınamadı: {e}"}
    if df is None or len(df) < 260:
        return {"error": "yetersiz veri (en az ~260 bar gerekli)"}

    sig = _signal_series(df)               # bir kez hesapla (paramsız)
    results: List[Dict] = []
    for am in _GRID_ATR:
        for rr in _GRID_RR:
            for thr in _GRID_THR:
                res = _simulate(df, sig, thr=thr, atr_mult=am, rr=rr)
                results.append({
                    "atr_mult": am, "rr": rr, "thr": thr,
                    "trades": res["trades"], "win_rate": res["win_rate"],
                    "profit_factor": res["profit_factor"], "total_return": res["total_return"],
                    "max_drawdown": res["max_drawdown"], "sharpe": res["sharpe"],
                    "score": _score(res),
                    "_returns": res["trade_returns"],
                })
    results.sort(key=lambda r: r["score"], reverse=True)
    best = results[0]
    best_mc = monte_carlo_backtest(best.get("_returns", []))

    # mevcut (varsayılan) config ile kıyas
    cur_am = float(orch.config.get("risk.atr_stop_mult", 2.0))
    baseline = _simulate(df, sig, thr=0.35, atr_mult=cur_am, rr=2.0)

    for r in results:
        r.pop("_returns", None)
    return {
        "symbol": symbol, "timeframe": timeframe, "tested": len(results),
        "best": {k: best[k] for k in ("atr_mult", "rr", "thr", "trades", "win_rate",
                                      "profit_factor", "total_return", "max_drawdown", "sharpe", "score")},
        "best_monte_carlo": best_mc,
        "baseline": {"atr_mult": cur_am, "rr": 2.0, "thr": 0.35,
                     "total_return": baseline["total_return"], "max_drawdown": baseline["max_drawdown"],
                     "profit_factor": baseline["profit_factor"], "trades": baseline["trades"]},
        "top": results[:10],
    }


def run_backtest(orch, symbol: str, timeframe: str = "4h", bars: int = 600) -> Dict:
    try:
        df = orch.data.fetch_ohlcv(symbol, timeframe, limit=bars)
    except Exception as e:
        return {"error": f"veri alınamadı: {e}"}
    if df is None or len(df) < 260:
        return {"error": "yetersiz veri (en az ~260 bar gerekli)"}

    sig = _signal_series(df)
    full = _simulate(df, sig)

    # Walk-forward: 4 kat, her katta ayrı performans
    folds = 4
    n = len(df)
    seg = (n - 200) // folds
    wf = []
    for k in range(folds):
        s0 = 200 + k * seg
        s1 = 200 + (k + 1) * seg if k < folds - 1 else n
        sub = df.iloc[max(0, s0 - 200):s1]
        if len(sub) < 230:
            continue
        res = _simulate(sub, _signal_series(sub))
        wf.append({"fold": k + 1, "trades": res["trades"], "return": res["total_return"],
                   "win_rate": res["win_rate"]})

    full["walkforward_folds"] = len(wf)
    full["walkforward"] = wf
    full["monte_carlo"] = monte_carlo_backtest(full.get("trade_returns", []))
    full.pop("trade_returns", None)   # ham listeyi yanıttan çıkar (MC özeti yeterli)
    full["symbol"] = symbol
    full["timeframe"] = timeframe
    return full
