"""ANA ÜRÜN METRİĞİ — Realized Net EV / Predicted Net EV (§CVI).

NEDEN TEK METRİK BU

P&L tek başına yanıltıcıdır: iyi bir tahmin kötü bir yürütmeyle kaybettirebilir,
kötü bir tahmin şansla kazandırabilir. Buna karşılık

    gerçekleşen net getiri  ÷  tahmin edilen net getiri

**dört katmanın hepsini birden** ölçer: tahmin doğru muydu, maliyet doğru
tahmin edildi mi, emir beklenen fiyattan doldu mu, risk kapıları doğru olanı mı
geçirdi. Oran 1'e yakınsa sistem kendini tanıyordur.

SAPMA TEK BAŞINA YETMEZ — HANGİ KATMANIN BOZULDUĞU AYRIŞTIRILIR

    toplam sapma = olasılık sapması + maliyet sapması + yürütme sapması

  olasılık : tahmin edilen P(hedef) ile gerçekleşen hedef-önce oranı farkı
  maliyet  : tahmin edilen maliyet ile gerçekleşen maliyet farkı
  yürütme  : beklenen giriş fiyatı ile gerçek dolum farkı (kayma)

Bu ayrıştırma olmadan "model bozuldu" ile "borsa pahalılaştı" ayırt edilemez.

⚠️ ÖLÇÜM ANCAK ÇÖZÜLMÜŞ TAHMİNLERDEN YAPILIR. Açık pozisyonlar dahil edilmez;
"henüz kaybetmedi" kazanç sayılmaz.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Sequence

import numpy as np

PRICE_OUTCOMES = ("TP_FIRST", "SL_FIRST", "TIMEOUT")
MIN_SAMPLE = 20


@dataclass
class Attribution:
    n: int
    predicted_ev_mean: Optional[float]
    realized_net_mean: Optional[float]
    ratio: Optional[float]
    gap: Optional[float]
    probability_gap: Optional[float]
    cost_gap: Optional[float]
    execution_gap: Optional[float]
    unexplained_gap: Optional[float]
    predicted_p_mean: Optional[float]
    realized_tp_rate: Optional[float]
    verdict: str
    reasons: List[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


def _ort(x: Sequence[float]) -> Optional[float]:
    v = [float(a) for a in x if a is not None and math.isfinite(float(a))]
    return (sum(v) / len(v)) if v else None


def attribute(predictions: Sequence[Dict], outcomes: Dict[str, Dict],
              min_sample: int = MIN_SAMPLE) -> Attribution:
    """Tahmin ile gerçekleşen arasındaki farkı katmanlara ayır."""
    tahmin_ev, gercek_net = [], []
    tahmin_p, gercek_tp = [], []
    tahmin_maliyet, gercek_maliyet = [], []
    giris_kayma = []

    for p in predictions:
        o = outcomes.get(p.get("prediction_id"))
        if not o or o.get("outcome") not in PRICE_OUTCOMES:
            continue                      # açık ya da işlem olmamış: sayılmaz
        if p.get("robust_ev") is not None:
            tahmin_ev.append(float(p["robust_ev"]))
        if o.get("realized_net_pct") is not None:
            gercek_net.append(float(o["realized_net_pct"]))
        if p.get("p_target_first") is not None:
            tahmin_p.append(float(p["p_target_first"]))
        gercek_tp.append(1.0 if o["outcome"] == "TP_FIRST" else 0.0)
        if p.get("cost_pct") is not None:
            tahmin_maliyet.append(float(p["cost_pct"]))
        if o.get("realized_cost_pct") is not None:
            gercek_maliyet.append(float(o["realized_cost_pct"]))
        # yürütme: planlanan giriş ile gerçek dolum farkı
        if p.get("entry") and o.get("entry_vwap"):
            e, f = float(p["entry"]), float(o["entry_vwap"])
            if e > 0:
                yon = 1.0 if p.get("direction") == "LONG" else -1.0
                giris_kayma.append((f - e) / e * 100.0 * yon)

    n = len(gercek_tp)
    if n < min_sample:
        return Attribution(
            n=n, predicted_ev_mean=_ort(tahmin_ev),
            realized_net_mean=_ort(gercek_net), ratio=None, gap=None,
            probability_gap=None, cost_gap=None, execution_gap=None,
            unexplained_gap=None, predicted_p_mean=_ort(tahmin_p),
            realized_tp_rate=_ort(gercek_tp),
            verdict="INSUFFICIENT_SAMPLE",
            reasons=[f"çözülmüş tahmin {n} < {min_sample} — oran anlamsız"],
            note="Açık pozisyonlar ölçüme GİRMEZ; 'henüz kaybetmedi' kazanç değildir.")

    pev, rnet = _ort(tahmin_ev), _ort(gercek_net)
    pp, rtp = _ort(tahmin_p), _ort(gercek_tp)
    pc, rc = _ort(tahmin_maliyet), _ort(gercek_maliyet)
    kayma = _ort(giris_kayma)

    oran = (rnet / pev) if (pev not in (None, 0) and rnet is not None) else None
    fark = (rnet - pev) if (pev is not None and rnet is not None) else None

    # Olasılık sapmasının PARA karşılığı: (gerçek − tahmin) × ortalama kazanç.
    # Kazanç ölçeği net %1 hedefidir; tahminin kendi kazanç varsayımı budur.
    olasilik_fark = ((rtp - pp) * 1.0) if (pp is not None and rtp is not None) else None
    maliyet_fark = (-(rc - pc)) if (pc is not None and rc is not None) else None
    yurutme_fark = (-kayma) if kayma is not None else None

    aciklanan = sum(x for x in (olasilik_fark, maliyet_fark, yurutme_fark)
                    if x is not None)
    aciklanmayan = (fark - aciklanan) if fark is not None else None

    neden: List[str] = []
    if oran is not None:
        if oran < 0.5:
            neden.append(f"gerçekleşen, tahminin yalnız {oran:.2f} katı — "
                         f"sistem kendini iyimser tanıyor")
        elif oran > 1.5:
            neden.append(f"gerçekleşen tahminin {oran:.2f} katı — tahmin "
                         f"fazla temkinli ya da örneklem şanslı")
    if olasilik_fark is not None and abs(olasilik_fark) > 0.05:
        neden.append(f"olasılık sapması {olasilik_fark:+.3f} puan "
                     f"(tahmin %{(pp or 0)*100:.1f} → gerçek %{(rtp or 0)*100:.1f})")
    if maliyet_fark is not None and abs(maliyet_fark) > 0.02:
        neden.append(f"maliyet sapması {maliyet_fark:+.3f}% "
                     f"(tahmin %{pc:.3f} → gerçek %{rc:.3f})")
    if yurutme_fark is not None and abs(yurutme_fark) > 0.02:
        neden.append(f"yürütme kayması {kayma:+.3f}% — emirler planlanan "
                     f"fiyattan dolmuyor")

    if oran is None:
        verdikt = "UNMEASURED"
    elif 0.7 <= oran <= 1.3:
        verdikt = "ALIGNED"
    elif oran > 1.3:
        verdikt = "CONSERVATIVE"
    else:
        verdikt = "OPTIMISTIC"

    return Attribution(
        n=n, predicted_ev_mean=pev, realized_net_mean=rnet,
        ratio=(None if oran is None else round(oran, 4)),
        gap=(None if fark is None else round(fark, 5)),
        probability_gap=(None if olasilik_fark is None else round(olasilik_fark, 5)),
        cost_gap=(None if maliyet_fark is None else round(maliyet_fark, 5)),
        execution_gap=(None if yurutme_fark is None else round(yurutme_fark, 5)),
        unexplained_gap=(None if aciklanmayan is None else round(aciklanmayan, 5)),
        predicted_p_mean=pp, realized_tp_rate=rtp,
        verdict=verdikt, reasons=neden,
        note=("Oran 1'e yakınsa sistem kendini tanıyor demektir. Sapma "
              "olasılık / maliyet / yürütme katmanlarına ayrıştırılır; "
              "aksi hâlde 'model bozuldu' ile 'borsa pahalılaştı' ayırt edilemez."))


VERDICT_TR = {
    "ALIGNED": "tahmin gerçekleşenle uyumlu",
    "OPTIMISTIC": "tahmin fazla iyimser",
    "CONSERVATIVE": "tahmin fazla temkinli",
    "INSUFFICIENT_SAMPLE": "örneklem yetersiz",
    "UNMEASURED": "ölçülemedi",
}


def by_cell(led, min_sample: int = MIN_SAMPLE) -> List[Dict]:
    """Parite × ufuk × yön kırılımıyla mutabakat."""
    son = led.outcomes()
    gruplar: Dict[str, List[Dict]] = {}
    for p in led.predictions():
        k = f"{p['symbol']}|{p['horizon']}|{p['direction']}"
        gruplar.setdefault(k, []).append(p)
    out = []
    for k, ps in sorted(gruplar.items()):
        a = attribute(ps, son, min_sample)
        sym, hz, d = k.split("|")
        out.append({"symbol": sym, "horizon": hz, "direction": d,
                    **a.to_dict()})
    return out


def overall(led, days: Optional[int] = None,
            min_sample: int = MIN_SAMPLE,
            source: Optional[str] = None) -> Dict:
    """Bütün sistemin tek mutabakat sayısı (§CVI).

    `source` verilirse yalnız o kaynağın kayıtları sayılır. Gölge ve yayımlanan
    kayıtlar ASLA aynı oranda toplanmaz: ikisi farklı soruları cevaplar."""
    ps = led.predictions(source=source) if source else led.predictions()
    if days:
        kesim = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
        sec = []
        for p in ps:
            try:
                if dt.datetime.fromisoformat(
                        p["timestamp"].replace("Z", "+00:00")) >= kesim:
                    sec.append(p)
            except Exception:
                continue
        ps = sec
    a = attribute(ps, led.outcomes(), min_sample)
    d = a.to_dict()
    d["window_days"] = days
    d["source"] = source or "hepsi"
    d["verdict_tr"] = VERDICT_TR.get(a.verdict, a.verdict)
    return d
