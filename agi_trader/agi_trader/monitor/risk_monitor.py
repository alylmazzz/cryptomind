"""
Ops/İzleme Katmanı — canlı işletme güvenliği (FAZ6+).

Trend paper portföyünün durumundan (runs/trend_state.json) risk & sağlık
metrikleri üretir: gerçekleşen Sharpe, model-drift tespiti (beklenen Sharpe'a
göre bozulma), VaR/CVaR, maruziyet, drawdown ve uyarılar. /api/risk ile sunulur.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

EXPECTED_SHARPE = 1.36      # nihai stratejinin OOS beklentisi (diversifiye+kaldıraç)
EXPECTED_DD = 13.3          # OOS max drawdown (%)


def risk_report(state: Dict, expected_sharpe: float = EXPECTED_SHARPE) -> Dict:
    hist: List[Dict] = state.get("history", []) or []
    eq = [float(h.get("equity", 0)) for h in hist if h.get("equity")]
    weights = state.get("weights", {}) or {}
    init = float(state.get("initial", 10000))
    out: Dict = {"available": len(eq) >= 2, "samples": len(eq),
                 "expected_sharpe": expected_sharpe, "expected_dd_pct": EXPECTED_DD}
    alerts: List[str] = []

    # maruziyet
    gross = sum(abs(v) for v in weights.values())
    out["gross_exposure_pct"] = round(gross * 100, 1)
    out["in_market"] = [s for s, v in weights.items() if abs(v) > 1e-4]
    if gross > 3.0:
        alerts.append(f"⚠️ Yüksek kaldıraç: gross %{gross*100:.0f} (>%300)")

    if len(eq) < 20:
        out["alerts"] = alerts + ["ℹ️ Yeterli geçmiş yok (≥20 gün gerekli metrikler için)"]
        return out

    arr = np.array(eq, dtype=float)
    rets = np.diff(arr) / arr[:-1]
    mu, sd = float(rets.mean()), float(rets.std())
    realized_sharpe = mu / (sd + 1e-12) * np.sqrt(365)
    peak = np.maximum.accumulate(arr)
    max_dd = float(((peak - arr) / peak).max() * 100)
    cur_dd = float((peak[-1] - arr[-1]) / peak[-1] * 100)

    var95 = float(-np.percentile(rets, 5) * 100)
    tail = rets[rets <= np.percentile(rets, 5)]
    cvar95 = float(-tail.mean() * 100) if len(tail) else var95

    out.update({
        "return_pct": round((arr[-1] / init - 1) * 100, 2),
        "realized_sharpe": round(realized_sharpe, 2),
        "max_drawdown_pct": round(max_dd, 1), "current_drawdown_pct": round(cur_dd, 1),
        "daily_vol_pct": round(sd * 100, 2), "ann_vol_pct": round(sd * np.sqrt(365) * 100, 1),
        "var95_pct": round(var95, 2), "cvar95_pct": round(cvar95, 2),
    })

    # --- DRIFT tespiti (model çürümesi) ---
    if len(eq) >= 40:
        if realized_sharpe < expected_sharpe * 0.3:
            alerts.append(f"🔴 DRIFT: gerçekleşen Sharpe {realized_sharpe:.2f} << beklenen {expected_sharpe:.2f} "
                          f"— model çürümesi olabilir, gözden geçir.")
        elif realized_sharpe < expected_sharpe * 0.6:
            alerts.append(f"🟡 Zayıf: Sharpe {realized_sharpe:.2f} beklenenin altında ({expected_sharpe:.2f}).")
    # --- DRAWDOWN uyarısı ---
    if max_dd > EXPECTED_DD * 1.5:
        alerts.append(f"🔴 DD {max_dd:.1f}% beklenen tavanı (%{EXPECTED_DD})×1.5 aştı — riski azalt/dur.")
    elif cur_dd > EXPECTED_DD:
        alerts.append(f"🟡 Güncel DD %{cur_dd:.1f} — beklenen tavana yakın.")
    if not alerts:
        alerts.append("🟢 Sağlıklı — metrikler beklenen aralıkta.")
    out["alerts"] = alerts
    out["health"] = "red" if any("🔴" in a for a in alerts) else ("yellow" if any("🟡" in a for a in alerts) else "green")
    return out
