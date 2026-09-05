"""
REPLAY / BİLİMSEL KABUL BORU HATTI — komiteyi GERÇEK geçmiş 1 dk veride bar bar oynatır.

  HistoryFetcher   ccxt ile sayfalı 1m/1h/4h OHLCV (runs/history/*.csv önbelleği)
  ReplayExchange   ccxt-benzeri istemci: imleç zamanına kadar olan barları verir (LOOKAHEAD YOK),
                   defter = kapanış ± spread (sentetik), işlem akışı yok
  run_replay       RunnerRegistry + LiveRunner'ı simülasyon saatiyle döngü döngü koşturur
  analyze          sleeve başına beklenti, bootstrap %95 CI, maliyet×2 dayanıklılığı, alt-dönem
                   tutarlılığı, Sharpe/PSR/DSR (n_trials ile deflasyon), CSCV-PBO (sleeve'ler = strateji kümesi)
  write_evidence   lifecycle kayıt defterine kanıt (kapılar: OOS>0 · CI>0 · maliyet×2 · tutarlılık · DSR>0 · PBO<0,5 · n≥30)

Sınırlar (dürüst): ağır bağlam (formasyon/300 gösterge/mover) replay'de YOK — yalnız hafif bağlam;
dolum modeli bar low/high (kuyruk önceliği yok); haber yok. Sonuç "canlıdan iyimser" olabilir, bu
yüzden kapılar maliyet×2 ve DSR ile sıkılır.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TF_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


# ---------------------------------------------------------------- geçmiş veri
class HistoryFetcher:
    def __init__(self, exchange_id: str = "mexc", cache_dir: Optional[Path] = None, client_factory: Optional[Callable] = None,
                 rate_sleep: float = 0.12):
        self.exchange_id = exchange_id
        self.cache = Path(cache_dir) if cache_dir else Path(__file__).resolve().parents[2] / "runs" / "history"
        self.cache.mkdir(parents=True, exist_ok=True)
        self._factory = client_factory
        self._client = None
        self.rate_sleep = rate_sleep

    def client(self):
        if self._client is None:
            if self._factory is not None:
                self._client = self._factory(self.exchange_id, {"enableRateLimit": True})
            else:
                import ccxt
                self._client = getattr(ccxt, self.exchange_id)({"enableRateLimit": True})
        return self._client

    def _path(self, symbol: str, tf: str, since_ms: int, until_ms: int) -> Path:
        return self.cache / f"{self.exchange_id}_{symbol.replace('/', '-')}_{tf}_{since_ms // 1000}_{until_ms // 1000}.csv"

    def fetch(self, symbol: str, tf: str, since_ms: int, until_ms: int, page: int = 1000) -> pd.DataFrame:
        p = self._path(symbol, tf, since_ms, until_ms)
        if p.exists():
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, utc=True)
            return df
        ex = self.client()
        rows: List[List[float]] = []
        cur = since_ms
        step = TF_MS[tf]
        while cur < until_ms:
            batch = ex.fetch_ohlcv(symbol, tf, since=cur, limit=page)
            if not batch:
                break
            rows.extend(b for b in batch if b[0] < until_ms)
            nxt = batch[-1][0] + step
            if nxt <= cur:
                break
            cur = nxt
            time.sleep(self.rate_sleep)
        if not rows:
            raise RuntimeError(f"{symbol} {tf}: geçmiş veri boş")
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).drop_duplicates("ts")
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.set_index("ts").sort_index()
        df.to_csv(p)
        return df

    def bundle(self, symbols: List[str], days: float, end_ms: Optional[int] = None,
               progress: Optional[Callable[[str], None]] = None) -> Dict[Tuple[str, str], pd.DataFrame]:
        end_ms = int(end_ms or time.time() * 1000)
        out: Dict[Tuple[str, str], pd.DataFrame] = {}
        for s in symbols:
            spans = {"1m": int(days * 86_400_000) + 200 * 60_000, "1h": 130 * 3_600_000, "4h": 320 * 14_400_000}
            for tf, span in spans.items():
                try:
                    out[(s, tf)] = self.fetch(s, tf, end_ms - span, end_ms)
                    if progress:
                        progress(f"{s} {tf}: {len(out[(s, tf)])} bar")
                except Exception as e:
                    if progress:
                        progress(f"{s} {tf}: HATA {type(e).__name__}: {str(e)[:60]}")
        return out


# ---------------------------------------------------------------- replay istemcisi
class Cursor:
    def __init__(self, ms: int = 0):
        self.ms = int(ms)


class ReplayExchange:
    """ccxt-benzeri; imleçten sonraki barı ASLA vermez (lookahead yok). Son bar = 'devam eden' bar sayılır."""
    spread_bps = 3.0
    depth_usd = 400_000.0

    def __init__(self, hist: Dict[Tuple[str, str], pd.DataFrame], cursor: Cursor, params=None):
        self.hist = hist
        self.cursor = cursor
        self._arr: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
        for k, df in hist.items():
            ts = np.array([int(t.timestamp() * 1000) for t in df.index], dtype=np.int64)   # pandas 3: çözünürlük değişebilir → ms garantisi
            self._arr[k] = (ts, df[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float))
        self.markets = {s: {"symbol": s} for (s, _) in hist}

    def load_markets(self):
        return self.markets

    def market(self, s):
        return {"limits": {"cost": {"min": 5.0}, "amount": {"min": 0.0}}, "precision": {"amount": 6}}

    def amount_to_precision(self, s, a):
        return f"{float(a):.6f}"

    def fetch_ohlcv(self, symbol, timeframe="1m", limit=150, since=None):
        k = (symbol, timeframe)
        if k not in self._arr:
            return []
        ts, ohlcv = self._arr[k]
        i = int(np.searchsorted(ts, self.cursor.ms, side="right"))
        lo = max(0, i - limit)
        rows = ohlcv[lo:i]
        return [[int(ts[lo + j]), *map(float, rows[j])] for j in range(len(rows))]

    def _last(self, symbol) -> float:
        rows = self.fetch_ohlcv(symbol, "1m", 1)
        return float(rows[-1][4]) if rows else float("nan")

    def fetch_ticker(self, symbol):
        return {"last": self._last(symbol)}

    def fetch_tickers(self):
        return {}

    def fetch_order_book(self, symbol, limit=20):
        px = self._last(symbol)
        sp = self.spread_bps / 2e4
        return {"bids": [[px * (1 - sp), self.depth_usd / px]], "asks": [[px * (1 + sp), self.depth_usd / px]]}

    def fetch_trades(self, symbol, limit=200):
        return []

    def fetch_balance(self):
        return {"free": {"USDT": 0.0}}


def make_factory(hist: Dict[Tuple[str, str], pd.DataFrame], cursor: Cursor) -> Callable:
    def factory(exchange_id, params=None):
        return ReplayExchange(hist, cursor, params)
    return factory


# ---------------------------------------------------------------- replay koşumu
def run_replay(hist: Dict[Tuple[str, str], pd.DataFrame], symbols: List[str], out_dir: Path,
               cfg_overrides: Optional[Dict] = None, step_sec: int = 60, warmup_bars: int = 200,
               progress: Optional[Callable[[str], None]] = None, max_cycles: Optional[int] = None) -> Dict:
    from . import live_runner as LR
    from . import simulator as SIM
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    syms = [s for s in symbols if (s, "1m") in hist]
    if not syms:
        raise RuntimeError("replay için 1m verisi olan parite yok")
    t0 = max(int(hist[(s, "1m")].index[warmup_bars].timestamp() * 1000) for s in syms if len(hist[(s, "1m")]) > warmup_bars)
    t1 = min(int(hist[(s, "1m")].index[-1].timestamp() * 1000) for s in syms)
    cursor = Cursor(t0)
    ctx = LR.Context(cm_signal=lambda s: None, qual_cell=lambda *a: None, system_health=lambda: {"overall": "GREEN"},
                     regime=lambda df: None, slow_ctx=lambda s: None, candidate_symbols=lambda: None,
                     news_for=lambda s: None, market_news=lambda: None)
    reg = LR.RunnerRegistry(output_dir=str(out_dir), ctx=ctx, client_factory=make_factory(hist, cursor))
    cfg = LR.RunnerConfig.from_dict({**SIM.default_config("mexc").to_dict(), "symbols": syms, "symbols_mode": "fixed",
                                     **(cfg_overrides or {})})
    r = reg.create(0, cfg)
    n = 0
    t = t0
    while t <= t1:
        cursor.ms = t
        r.run_cycle(now=t / 1000.0)
        n += 1
        t += step_sec * 1000
        if progress and n % 120 == 0:
            progress(f"döngü {n} · {time.strftime('%Y-%m-%d %H:%M', time.gmtime(t / 1000))} · açık {len(r.positions)} · kapanan {len(r.trades)} · özsermaye {r.equity():.2f}")
        if max_cycles and n >= max_cycles:
            break
    cursor.ms = t1
    r.close_all("REPLAY_END")
    st = r.stats()
    result = {"symbols": syms, "start": t0, "end": t1, "n_cycles": n, "trades": r.trades, "stats": st,
              "equity_curve": r.equity_curve, "missed": r.missed.report(), "capital": cfg.capital_usdt,
              "config": cfg.to_dict(), "limits": ["ağır bağlam yok (yalnız hafif)", "dolum = bar low/high", "haber yok", "sentetik defter"]}
    return result


# ---------------------------------------------------------------- analiz
def _bootstrap_ci(x: np.ndarray, n: int = 2000, seed: int = 7) -> Tuple[float, float]:
    if len(x) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(n)])
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def _pbo_cscv(trades: List[Dict], n_groups: int = 4) -> Optional[float]:
    """CSCV: sleeve'ler strateji kümesi; her bölmede IS'de en iyi sleeve OOS'ta medyanın altına düşüyor mu?"""
    sleeves = sorted({t.get("sleeve") or t.get("trigger") for t in trades})
    if len(sleeves) < 2 or len(trades) < 16:
        return None
    import itertools
    order = sorted(trades, key=lambda t: t["closed_ts"])
    groups = np.array_split(np.arange(len(order)), n_groups)          # zaman sıralı bitişik gruplar (CSCV)
    splits = []
    for test_g in itertools.combinations(range(n_groups), n_groups // 2):
        te = np.concatenate([groups[g] for g in test_g])
        tr = np.concatenate([groups[g] for g in range(n_groups) if g not in test_g])
        splits.append((tr, te))
    below = 0; total = 0
    for tr_idx, te_idx in splits:
        tr_set, te_set = set(tr_idx.tolist()), set(te_idx.tolist())
        perf_is = {}; perf_oos = {}
        for s in sleeves:
            a = [float(order[i]["net_pct_realized"]) for i in tr_set if (order[i].get("sleeve") or order[i].get("trigger")) == s]
            b = [float(order[i]["net_pct_realized"]) for i in te_set if (order[i].get("sleeve") or order[i].get("trigger")) == s]
            if a and b:
                perf_is[s] = float(np.mean(a)); perf_oos[s] = float(np.mean(b))
        if len(perf_is) < 2:
            continue
        best = max(perf_is, key=perf_is.get)
        rank = sorted(perf_oos.values()).index(perf_oos[best]) / max(1, len(perf_oos) - 1)
        below += int(rank < 0.5); total += 1
    return (below / total) if total else None


def analyze(result: Dict, n_trials: int = 20) -> Dict:
    from ..research.validation import deflated_sharpe, psr, sharpe
    trades = result.get("trades") or []
    cap = float(result.get("capital") or 1000.0)
    out: Dict = {"n_trades": len(trades), "capital": cap}
    if not trades:
        out["note"] = "işlem yok — kanıt üretilemez"
        return out
    rets = np.array([float(t["net_pct_realized"]) for t in trades])
    fees2 = np.array([(float(t["gross_pnl"]) - 2.0 * float(t["fees"])) / max(1e-9, float(t["notional"])) * 100.0 for t in trades])
    eq = np.array([float(x["equity"]) for x in (result.get("equity_curve") or [])], dtype=float)
    if eq.size == 0:
        eq = np.array([cap])
    peak = np.maximum.accumulate(eq)
    dd = float(((peak - eq) / peak).max() * 100.0) if len(eq) else 0.0
    half = len(rets) // 2
    ds = deflated_sharpe(rets.tolist(), n_trials=n_trials, periods_per_year=365.0 * 24) if len(rets) >= 5 else {"dsr": None}
    out.update({
        "net_pnl": round(float(sum(t["net_pnl"] for t in trades)), 4), "return_pct": round(float(sum(t["net_pnl"] for t in trades)) / cap * 100.0, 3),
        "win_rate": round(float((rets > 0).mean()), 3), "expectancy_pct": round(float(rets.mean()), 4),
        "ci95": [round(v, 4) for v in _bootstrap_ci(rets)], "expectancy_cost_x2_pct": round(float(fees2.mean()), 4),
        "sharpe_per_trade": round(float(sharpe(rets.tolist())), 3) if len(rets) >= 3 else None,
        "psr": round(float(psr(rets.tolist())), 3) if len(rets) >= 5 else None,
        "dsr": (None if ds.get("dsr") is None else round(float(ds["dsr"]), 3)), "n_trials": n_trials,
        "subperiod": {"first_half": round(float(rets[:half].mean()), 4) if half else None, "second_half": round(float(rets[half:].mean()), 4) if half else None,
                      "consistent": bool(half and rets[:half].mean() > 0 and rets[half:].mean() > 0)},
        "max_drawdown_pct": round(dd, 3), "pbo": (None if _pbo_cscv(trades) is None else round(_pbo_cscv(trades), 3)),
        "fee_share_of_gross_pct": round(100.0 * sum(float(t["fees"]) for t in trades) / max(1e-9, abs(sum(float(t["gross_pnl"]) for t in trades))), 1),
        "exit_reasons": dict(pd.Series([t["reason"] for t in trades]).value_counts()),
        "avg_peak_capture": (round(float(np.mean([t["peak_capture"] for t in trades if t.get("peak_capture") is not None])), 3)
                             if any(t.get("peak_capture") is not None for t in trades) else None),
    })
    per: Dict[str, Dict] = {}
    for s in sorted({t.get("sleeve") or t.get("trigger") for t in trades}):
        ts = [t for t in trades if (t.get("sleeve") or t.get("trigger")) == s]
        r_ = np.array([float(t["net_pct_realized"]) for t in ts])
        f2 = np.array([(float(t["gross_pnl"]) - 2.0 * float(t["fees"])) / max(1e-9, float(t["notional"])) * 100.0 for t in ts])
        h = len(r_) // 2
        d_ = deflated_sharpe(r_.tolist(), n_trials=n_trials, periods_per_year=365.0 * 24) if len(r_) >= 5 else {"dsr": None}
        lo, hi = _bootstrap_ci(r_)
        ev = {"n_trades": len(ts), "win_rate": round(float((r_ > 0).mean()), 3), "oos_expectancy": round(float(r_.mean()), 4),
              "ci_lower": round(lo, 4), "ci_upper": round(hi, 4), "expectancy_cost_x2": round(float(f2.mean()), 4),
              "subperiod_consistent": bool(h and r_[:h].mean() > 0 and r_[h:].mean() > 0),
              "dsr": (None if d_.get("dsr") is None else round(float(d_["dsr"]), 3)), "pbo": out["pbo"],
              "net_pnl": round(float(sum(t["net_pnl"] for t in ts)), 4)}
        per[s] = ev
    out["per_sleeve"] = per
    return out


def write_evidence(lifecycle, analysis: Dict, source: str = "replay") -> List[Dict]:
    rows = []
    for s, ev in (analysis.get("per_sleeve") or {}).items():
        lifecycle.record_evidence(s, {**{k: v for k, v in ev.items() if k in ("oos_expectancy", "ci_lower", "expectancy_cost_x2",
                                                                              "subperiod_consistent", "dsr", "pbo", "n_trades")},
                                      "source": source})
        g = lifecycle.gates(s)
        rows.append({"sleeve": s, "passed": g["passed"], "checks": g["checks"], **ev})
    return rows


def trials_registry_path() -> Path:
    return Path(__file__).resolve().parents[2] / "runs" / "research" / "trials.jsonl"


def trials_count() -> int:
    try:
        return sum(1 for _ in open(trials_registry_path(), encoding="utf-8"))
    except Exception:
        return 0


def save_result(result: Dict, analysis: Dict, out_dir: Path) -> Path:
    """Sonuç dosyası + DENEME KAYDI (silinmez; DSR bu sayıyla deflate edilir) + VALIDATION_REPORT.json."""
    import hashlib
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    p = out_dir / f"replay_{run_id}.json"
    slim = {k: v for k, v in result.items() if k not in ("equity_curve",)}
    slim["equity_curve"] = (result.get("equity_curve") or [])[-500:]
    p.write_text(json.dumps({"result": slim, "analysis": analysis}, ensure_ascii=False, default=str), encoding="utf-8")
    try:
        cfg = result.get("config") or {}
        params_hash = hashlib.sha256(json.dumps({"params": cfg.get("params"), "exit": cfg.get("exit"), "symbols": result.get("symbols")},
                                                sort_keys=True, default=str).encode()).hexdigest()[:12]
        rp = trials_registry_path(); rp.parent.mkdir(parents=True, exist_ok=True)
        with open(rp, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": time.time(), "run_id": run_id, "symbols": result.get("symbols"), "n_cycles": result.get("n_cycles"),
                                 "params_hash": params_hash, "n_trades": analysis.get("n_trades"), "expectancy_pct": analysis.get("expectancy_pct"),
                                 "dsr": analysis.get("dsr"), "pbo": analysis.get("pbo"), "return_pct": analysis.get("return_pct")},
                                ensure_ascii=False, default=str) + "\n")
        vr = rp.parent / "VALIDATION_REPORT.json"
        vr.write_text(json.dumps({"run_id": run_id, "generated": time.time(), "mode": "REPLAY (hafif bağlam, sentetik defter, haber yok)",
                                  "status": ("UNVERIFIED" if (analysis.get("n_trades") or 0) < 30 else "PAPER_EVIDENCE"),
                                  "n_trials_registry": trials_count(), "analysis": analysis,
                                  "limits": result.get("limits"), "symbols": result.get("symbols"), "start": result.get("start"), "end": result.get("end")},
                                 ensure_ascii=False, default=str, indent=1), encoding="utf-8")
    except Exception:
        pass
    return p
