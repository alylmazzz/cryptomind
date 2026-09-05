"""
OLASILIK KALİBRASYONU + MONTE CARLO — fişteki p_win gerçekleşmeyle uyuşuyor mu; sermaye eğrisi ne kadar kırılgan?

  reliability_table  p_win kovaları (0,4–0,5 · 0,5–0,6 · …) × gerçekleşen kazanma oranı; Brier skoru;
                     sleeve bazında ayrı. Kaynak: kapanan işlemler (fiş p_win) + çözülen gölgeler (fiş yoksa 0,5).
  monte_carlo        işlem net% dizisinden bootstrap (2000 yol): P5/P50/P95 son özsermaye, maksimum
                     drawdown dağılımı, günlük zarar limitini aşma olasılığı, iflas (drawdown ≥ %30) olasılığı.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

BINS = [0.0, 0.4, 0.5, 0.6, 0.7, 1.01]


def reliability_table(trades: List[Dict]) -> Dict:
    rows = []
    for t in trades:
        p = t.get("p_win")
        if p is None:
            p = ((t.get("decision") or {}).get("ticket") or {}).get("p_win")
        if p is None:
            p = (t.get("ticket") or {}).get("p_win")
        if p is None:
            continue
        rows.append((float(p), 1.0 if t.get("win") else 0.0, t.get("sleeve") or t.get("trigger") or "?"))
    if not rows:
        return {"n": 0, "brier": None, "bins": [], "per_sleeve": {}, "note": "fişte p_win yok"}
    P = np.array([r[0] for r in rows]); Y = np.array([r[1] for r in rows])
    brier = float(np.mean((P - Y) ** 2))
    base = float(Y.mean())
    brier_ref = float(np.mean((base - Y) ** 2))
    bins = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (P >= lo) & (P < hi)
        if m.sum():
            bins.append({"bin": f"{lo:.1f}–{min(hi, 1.0):.1f}", "n": int(m.sum()), "p_mean": round(float(P[m].mean()), 3),
                         "realized": round(float(Y[m].mean()), 3), "gap": round(float(Y[m].mean() - P[m].mean()), 3)})
    per = {}
    for s in sorted({r[2] for r in rows}):
        m = np.array([r[2] == s for r in rows])
        per[s] = {"n": int(m.sum()), "p_mean": round(float(P[m].mean()), 3), "realized": round(float(Y[m].mean()), 3),
                  "brier": round(float(np.mean((P[m] - Y[m]) ** 2)), 4)}
    verdict = ("iyi kalibre" if brier <= brier_ref * 0.95 else "aşırı iyimser" if float((P - Y).mean()) > 0.08
               else "aşırı kötümser" if float((P - Y).mean()) < -0.08 else "bilgisiz (taban oranı kadar)")
    return {"n": len(rows), "brier": round(brier, 4), "brier_reference": round(brier_ref, 4), "skill": round(1 - brier / max(1e-9, brier_ref), 3),
            "bins": bins, "per_sleeve": per, "verdict": verdict,
            "note": "Brier < referans → tahminler taban oranından bilgili; gap > 0 = gerçekleşen > tahmin (kötümser)"}


def monte_carlo(trades: List[Dict], capital: float = 1000.0, n_paths: int = 2000, horizon: Optional[int] = None,
                daily_loss_limit_pct: float = 5.0, ruin_dd_pct: float = 30.0, seed: int = 11) -> Dict:
    rets = np.array([float(t["net_pnl"]) / max(1e-9, float(t.get("notional") or 1.0)) for t in trades])
    notionals = np.array([float(t.get("notional") or 0.0) for t in trades])
    if len(rets) < 5:
        return {"n_trades": len(rets), "note": "en az 5 işlem gerekir"}
    rng = np.random.default_rng(seed)
    H = int(horizon or max(20, len(rets)))
    avg_notional = float(np.median(notionals)) if notionals.size else 100.0
    finals = np.empty(n_paths); mdds = np.empty(n_paths); breach = 0
    for i in range(n_paths):
        idx = rng.integers(0, len(rets), size=H)
        pnl = rets[idx] * avg_notional
        eq = capital + np.cumsum(pnl)
        peak = np.maximum.accumulate(np.concatenate([[capital], eq]))[1:]
        mdd = float(((peak - eq) / peak).max() * 100.0)
        finals[i] = eq[-1]; mdds[i] = mdd
        if mdd >= daily_loss_limit_pct:
            breach += 1
    return {"n_trades": len(rets), "paths": n_paths, "horizon_trades": H, "avg_notional": round(avg_notional, 2),
            "final_p5": round(float(np.quantile(finals, 0.05)), 2), "final_p50": round(float(np.quantile(finals, 0.5)), 2),
            "final_p95": round(float(np.quantile(finals, 0.95)), 2),
            "max_dd_p50_pct": round(float(np.quantile(mdds, 0.5)), 2), "max_dd_p95_pct": round(float(np.quantile(mdds, 0.95)), 2),
            "p_breach_daily_limit": round(breach / n_paths, 3), "p_ruin": round(float((mdds >= ruin_dd_pct).mean()), 4),
            "expectancy_per_trade_usdt": round(float(rets.mean() * avg_notional), 4),
            "note": "bootstrap: işlemler bağımsız ve aynı dağılımdan varsayılır (rejim değişimi yansımaz)"}
