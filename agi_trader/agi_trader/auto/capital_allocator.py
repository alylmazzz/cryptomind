"""
SERMAYE TAHSİSİ — hangi katmana ne kadar risk verilir, ÖLÇÜLMÜŞ sonuca göre.

Neden gerekli (2026-09-04 ölçümü): sistemde iki katman var ve ikisi zıt yönde çalışıyordu —
  · trend takip (günlük yeniden dengeleme, 17 varlık): 48 günde +%4,29 · Sharpe 2,68 · DD %2,0
  · scalping (1 dk komite): 105 işlemde −%0,62 · komisyon brütün 3 katı
Sermaye buna rağmen eşit muamele görüyordu. Bu modül tahsisi ÖLÇÜME bağlar: kazanan katman
ağırlığı alır, kaybeden katman yalnız ÖLÇÜM BÜTÇESİ (küçük, sabit) ile yaşamaya devam eder —
kapatılmaz, çünkü kapatılan katman bir daha ölçülemez ve rejim değişince fark edilemez.

Kural (basit ve savunulabilir; Kelly değil):
  w_i ∝ max(0, Sharpe_i − SHARPE_FLOOR)        Sharpe eşiğin altındaysa pay YOK
  Sharpe ölçülmemişse (n < MIN_N) → yalnız ölçüm bütçesi (kanıt birikene dek)
  negatif/eşik altı katman → ölçüm bütçesi (MEASURE_BUDGET_PCT)
  tek katman tavanı MAX_WEIGHT (yoğunlaşma sınırı)

Kelly KULLANILMADI: f* = μ/σ² trend katmanı için ~21 çıkıyor (yani 21× kaldıraç). Kelly tek
varlık ve bilinen dağılım varsayar; burada ne dağılım bilinir ne de kaldıraç serbesttir.
"""
from __future__ import annotations

import json
import math
import statistics as st
import time
from pathlib import Path
from typing import Dict, List, Optional

MIN_N = 20                    # trend katmanı için asgari GÜN sayısı
MIN_DAYS_SCALP = 5            # scalping günlük Sharpe'ı için asgari gün (işlem sayısı değil — gün)
SHARPE_FLOOR = 0.0            # bu eşiğin altındaki katman risk sermayesi almaz
MEASURE_BUDGET_PCT = 2.0      # kanıtsız/eşik altı katmanın ölçüm için aldığı pay (%)
MAX_WEIGHT = 95.0             # tek katman tavanı (%)
ANNUALIZE_DAYS = 365.0


def _sharpe_daily(returns: List[float]) -> Optional[float]:
    """Günlük getiri serisinden yıllıklandırılmış Sharpe. n < 5 ya da σ = 0 → None."""
    r = [float(x) for x in returns if x is not None and math.isfinite(float(x))]
    if len(r) < 5:
        return None
    mu = st.mean(r)
    sd = st.stdev(r)
    if sd <= 1e-12:
        return None
    return float(mu / sd * math.sqrt(ANNUALIZE_DAYS))


def trend_metrics(state: Dict) -> Dict:
    """`runs/trend_state.json` → gerçekleşen metrikler (NaN günler atlanır)."""
    h = [x for x in (state.get("history") or [])
         if isinstance(x.get("equity"), (int, float)) and math.isfinite(x["equity"])]
    if len(h) < 2:
        return {"layer": "trend", "n": len(h), "sharpe": None, "return_pct": None,
                "max_dd_pct": None, "measured": False, "note": "yetersiz geçmiş"}
    eq = [float(x["equity"]) for x in h]
    init = float(state.get("initial") or eq[0])
    rets = [eq[i] / eq[i - 1] - 1 for i in range(1, len(eq)) if eq[i - 1] > 0]
    peak, dd = eq[0], 0.0
    for e in eq:
        peak = max(peak, e)
        dd = min(dd, e / peak - 1)
    return {"layer": "trend", "n": len(eq), "sharpe": _sharpe_daily(rets),
            "return_pct": round((eq[-1] / init - 1) * 100, 3),
            "mean_daily_pct": round(st.mean(rets) * 100, 4) if rets else None,
            "max_dd_pct": round(dd * 100, 2), "equity": round(eq[-1], 2), "initial": init,
            "measured": len(eq) >= MIN_N, "days": len(eq),
            "not_measured_reason": (None if len(eq) >= MIN_N else f"{len(eq)} gün < {MIN_N} gün")}


def scalp_metrics(trades: List[Dict], capital: float) -> Dict:
    """Kapanan işlemlerden GÜNLÜK getiri serisi kurar (işlem başına değil — katmanlar arası
    karşılaştırma günlük ölçekte yapılmalı, yoksa 65 işlem/gün yapan katman yanıltıcı görünür)."""
    if not trades:
        return {"layer": "scalp", "n": 0, "sharpe": None, "return_pct": None,
                "max_dd_pct": None, "measured": False, "note": "işlem yok"}
    by_day: Dict[str, float] = {}
    for t in trades:
        d = time.strftime("%Y-%m-%d", time.gmtime(float(t.get("closed_ts") or 0)))
        by_day[d] = by_day.get(d, 0.0) + float(t.get("net_pnl") or 0.0)
    days = sorted(by_day)
    eq, cur = [], float(capital)
    for d in days:
        cur += by_day[d]
        eq.append(cur)
    rets = [by_day[d] / capital for d in days]
    peak, dd = float(capital), 0.0
    for e in eq:
        peak = max(peak, e)
        dd = min(dd, e / peak - 1)
    net = sum(float(t.get("net_pnl") or 0.0) for t in trades)
    fees = sum(float(t.get("fees") or 0.0) for t in trades)
    gross = sum(float(t.get("gross_pnl") or 0.0) for t in trades)
    return {"layer": "scalp", "n": len(trades), "days": len(days),
            "sharpe": _sharpe_daily(rets),
            "return_pct": round(net / capital * 100, 3),
            "mean_daily_pct": round(st.mean(rets) * 100, 4) if rets else None,
            "max_dd_pct": round(dd * 100, 2), "equity": round(capital + net, 2), "initial": float(capital),
            "net": round(net, 2), "fees": round(fees, 2), "gross": round(gross, 2),
            "fee_share_of_gross_pct": (round(abs(fees / gross) * 100, 1) if abs(gross) > 1e-9 else None),
            "measured": len(days) >= MIN_DAYS_SCALP,
            "not_measured_reason": (None if len(days) >= MIN_DAYS_SCALP
                                    else f"{len(days)} gün < {MIN_DAYS_SCALP} gün (günlük Sharpe için)"),
            "trades_per_day": (round(len(trades) / len(days), 1) if days else None)}


def allocate(layers: List[Dict], measure_budget_pct: float = MEASURE_BUDGET_PCT,
             sharpe_floor: float = SHARPE_FLOOR, max_weight: float = MAX_WEIGHT) -> Dict:
    """Ölçülmüş Sharpe'a göre risk payı. Ölçülmemiş/eşik altı katman yalnız ölçüm bütçesi alır."""
    rows = []
    for m in layers:
        sh = m.get("sharpe")
        eligible = bool(m.get("measured")) and sh is not None and sh > sharpe_floor
        rows.append({**m, "eligible": eligible, "score": (max(0.0, sh - sharpe_floor) if eligible else 0.0)})
    total = sum(r["score"] for r in rows)
    n_measure = sum(1 for r in rows if not r["eligible"])
    budget = measure_budget_pct * n_measure
    out = []
    for r in rows:
        if not r["eligible"]:
            w = measure_budget_pct
            why = (f"ölçülmedi ({r.get('not_measured_reason') or 'yetersiz gözlem'}) → ölçüm bütçesi"
                   if not r.get("measured")
                   else "Sharpe %s ≤ eşik %s → risk sermayesi YOK, yalnız ölçüm bütçesi"
                        % (("—" if r.get("sharpe") is None else f"{r['sharpe']:.2f}"), sharpe_floor))
        else:
            w = (100.0 - budget) * (r["score"] / total) if total > 0 else 0.0
            why = f"ölçülmüş Sharpe {r['sharpe']:.2f} (n {r.get('n')})"
        out.append({**r, "weight_pct": round(min(max_weight, w), 2), "reason": why})
    # NAKİT açıkça bir kalemdir. NORMALİZE ETMEK YANLIŞTI: iki katman da eşik altındayken
    # %2 + %2 = %4 toplamı normalize edilince %50/%50 oluyordu — yani "ikisi de kaybediyor"
    # durumu "sermayenin tamamını ikiye böl"e dönüşüyordu. Kalan pay nakitte durur.
    used = sum(r["weight_pct"] for r in out)
    if used > 100.0:                                # yalnız taşma hâlinde oransal kırp
        for r in out:
            r["weight_pct"] = round(r["weight_pct"] * 100.0 / used, 2)
        used = sum(r["weight_pct"] for r in out)
    cash = round(max(0.0, 100.0 - used), 2)
    return {"layers": out, "cash_pct": cash, "invested_pct": round(used, 2),
            "measure_budget_pct": measure_budget_pct, "sharpe_floor": sharpe_floor,
            "max_weight_pct": max_weight, "min_n": MIN_N,
            "note": ("Pay ÖLÇÜLMÜŞ Sharpe'la verilir; kalan NAKİTTE durur (normalize edilmez — "
                     "iki katman da kaybederken normalize etmek 'sermayeyi ikiye böl' demek olurdu). "
                     "Eşik altı katman kapatılmaz: kapatılan katman bir daha ölçülemez ve rejim "
                     "değişince fark edilmez, bu yüzden küçük bir ölçüm bütçesiyle yaşar. "
                     "Kelly kullanılmaz (trend için 21× kaldıraç önerirdi).")}


def scalp_risk_budget(weight_pct: float, capital: float, baseline: Optional[Dict] = None) -> Dict:
    """Scalping katmanının payından koşucu limitlerini türetir (kod-tanımlı tavanlar).

    Pay %2 iken 200 $'lık emirle işlem açmak payı anlamsız kılar: limitler payla ORANTILI olmalı."""
    b = baseline or {"max_order_usdt": 200.0, "risk_per_trade_pct": 1.0, "max_open": 5,
                     "max_exposure_pct": 75.0, "max_trades_per_day": 200}
    f = max(0.0, min(1.0, weight_pct / 100.0))
    return {
        "max_order_usdt": round(max(10.0, b["max_order_usdt"] * f), 2),
        "risk_per_trade_pct": round(max(0.1, b["risk_per_trade_pct"] * f), 3),
        "max_open": max(2, int(round(b["max_open"] * f))),      # taban 2: tek pozisyon çeşitlendirmeyi bitirir
        "max_exposure_pct": round(max(5.0, b["max_exposure_pct"] * f), 1),
        "max_trades_per_day": max(10, int(round(b["max_trades_per_day"] * f))),
        "weight_pct": weight_pct, "capital_at_risk_usdt": round(capital * f, 2),
    }


def report(trend_state: Optional[Dict], scalp_trades: Optional[List[Dict]],
           scalp_capital: float = 1000.0) -> Dict:
    """Panel/API için tam tahsis raporu."""
    layers = []
    tm = trend_metrics(trend_state or {})
    tm["title"] = "Trend takip (günlük yeniden dengeleme, 17 varlık)"
    layers.append(tm)
    sm = scalp_metrics(scalp_trades or [], scalp_capital)
    sm["title"] = "Scalping (1 dk komite, çok-sleeve)"
    layers.append(sm)
    alloc = allocate(layers)
    scalp_w = next((r["weight_pct"] for r in alloc["layers"] if r["layer"] == "scalp"), MEASURE_BUDGET_PCT)
    alloc["scalp_budget"] = scalp_risk_budget(scalp_w, scalp_capital)
    tot_eq = sum(float(r.get("equity") or 0.0) for r in layers)
    tot_init = sum(float(r.get("initial") or 0.0) for r in layers)
    alloc["combined"] = {"equity": round(tot_eq, 2), "initial": round(tot_init, 2),
                         "return_pct": (round((tot_eq / tot_init - 1) * 100, 3) if tot_init > 0 else None)}
    return alloc


def load_trend_state(path: str | Path) -> Optional[Dict]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
