"""Robust EV, ufuk optimizeri, giriş planı — şartname 12, 13, 23, 24, 25,
27, 63, 64, 65, 66, 68, 69, 94, 95, 96.

NEDEN HAM EV YETMEZ
EV = p·kazanç − q·kayıp formülü p'yi KESİN bilindiği varsayar. Örneklem
küçükse p'nin kendisi geniş bir aralıktır ve EV o aralığın üst ucundan
hesaplanınca sistem kendi belirsizliğini kâr sanır. Bu yüzden:

    RobustEV = EV(p_alt_sınır) − KuyrukCezası − SürüklenmeCezası

Üç ceza da AYRI raporlanır; tek bir "düzeltilmiş sayı" gösterip nereden
geldiğini saklamak, ölçümü gizlemektir.

EN İYİ UFUK = argmax P(hedef) DEĞİLDİR (şartname 11, 12)
24 saat beklemek olasılığı yükseltir ama sermayeyi bağlar, funding ödetir ve
rejim değişimine maruz bırakır. Karşılaştırma BİRİM ZAMAN BAŞINA yapılır:

    RobustUtility(H) = RobustEV(H) / beklenen_tutma_saati(H)

ve yalnız kapıları geçen ufuklar karşılaştırmaya girer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

from .lift import Payoff, expected_value

EPS_HOURS = 1e-6
# Sürüklenme ölçülemediğinde uygulanan sabit tıraş. Bu bir tahmin değil,
# BİLİNMEYENİN maliyetidir ve `drift_measured=False` ile beyan edilir.
UNMEASURED_DRIFT_HAIRCUT_PCT = 0.05


@dataclass
class RobustResult:
    ev: Optional[float]
    ev_lower: Optional[float]
    uncertainty_penalty: Optional[float]
    tail_penalty: Optional[float]
    drift_penalty: Optional[float]
    robust_ev: Optional[float]
    expected_holding_hours: Optional[float]
    robust_utility: Optional[float]
    drift_measured: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


def robust_expected_value(p_tp: float, p_sl: float, p_to: float,
                          pay: Payoff,
                          p_tp_lower: Optional[float] = None,
                          tail_excess_pct: float = 0.0,
                          psi: Optional[float] = None,
                          expected_holding_hours: Optional[float] = None
                          ) -> RobustResult:
    """Şartname 13 — belirsizlik cezalı beklenen değer.

    p_tp_lower     : hedef olasılığının %95 ALT sınırı. Eksilen olasılık
                     STOP'a yazılır (iyimser tarafa değil).
    tail_excess_pct: stop vurulduğunda stop seviyesinin ÖTESİNE geçen
                     ortalama fazla kayıp (gerçek MAE dağılımından ölçülür).
    psi            : özellik/olasılık sürüklenme göstergesi. None ise
                     ölçülmemiştir ve sabit tıraş uygulanır.
    """
    ev = expected_value(p_tp, p_sl, p_to, pay)
    notlar: List[str] = []

    if p_tp_lower is None:
        ev_alt = ev
        belirsizlik = 0.0
        notlar.append("olasılık alt sınırı yok — belirsizlik cezası "
                      "UYGULANMADI, bu iyimserdir")
    else:
        kayip = max(0.0, p_tp - p_tp_lower)
        ev_alt = expected_value(p_tp_lower, p_sl + kayip, p_to, pay)
        belirsizlik = ev - ev_alt

    kuyruk = float(p_sl * max(0.0, tail_excess_pct))

    if psi is None:
        surukleme = UNMEASURED_DRIFT_HAIRCUT_PCT
        olculdu = False
        notlar.append("sürüklenme ÖLÇÜLMEDİ — sabit tıraş uygulandı")
    else:
        olculdu = True
        surukleme = float(max(0.0, min(1.0, psi / 0.25)) * UNMEASURED_DRIFT_HAIRCUT_PCT)

    rev = ev_alt - kuyruk - surukleme
    saat = expected_holding_hours
    fayda = None
    if saat is not None and saat > EPS_HOURS:
        fayda = rev / saat
    return RobustResult(ev, ev_alt, belirsizlik, kuyruk, surukleme, rev,
                        saat, fayda, olculdu, notlar)


def expected_holding_hours(p_tp: float, p_sl: float, p_to: float,
                           hours_to_tp: Optional[float],
                           hours_to_sl: Optional[float],
                           horizon_hours: float) -> float:
    """Şartname 95 — beklenen tutma süresi.

    Nominal ufuk YANILTICIDIR: 24h ufkunda hedef ortalama 3 saatte geliyorsa
    sermaye 24 saat bağlanmaz. Zaman aşımı olasılığı tam ufku öder."""
    s = p_tp + p_sl + p_to
    if s <= 0:
        return horizon_hours
    t = (p_tp * (hours_to_tp if hours_to_tp is not None else horizon_hours)
         + p_sl * (hours_to_sl if hours_to_sl is not None else horizon_hours)
         + p_to * horizon_hours) / s
    return float(max(EPS_HOURS, t))


# ── ufuk seçimi ────────────────────────────────────────────────────────────

def best_horizon(cells: List[Dict]) -> Optional[Dict]:
    """Şartname 12/48 — yalnız QUALIFIED/HIGH_CONFIDENCE hücreler yarışır.

    Kapıyı geçen yoksa **None** döner; "en iyisi bu olsun" diye düşük kaliteli
    bir hücre seçmek YASAKTIR (şartname 59, 62)."""
    aday = [c for c in cells
            if c.get("tradable") and c.get("robust_utility") is not None]
    if not aday:
        return None
    return max(aday, key=lambda c: c["robust_utility"])


def earliest_qualified_horizon(cells: List[Dict]) -> Optional[Dict]:
    """Şartname 94 — EN İYİ ile EN ERKEN farklı sorulardır.

    Kullanıcı 'en yüksek olasılık' değil 'hedefe en çabuk ulaşabileceğim
    doğrulanmış ufuk' da isteyebilir."""
    aday = [c for c in cells if c.get("tradable")]
    if not aday:
        return None
    return min(aday, key=lambda c: c.get("horizon_minutes", 1 << 30))


def horizon_narrative(cells: List[Dict], secilen: Optional[Dict]) -> str:
    """Şartname 55/91 — 'neden bu ufuk?' açıklaması.

    ⚠️ Bu metin DETERMİNİSTİK metriklerden üretilir. LLM karar vermez, LLM
    yazmaz; aşağıdaki karşılaştırmalar sayıların kendisidir."""
    if secilen is None:
        olculen = [c for c in cells if c.get("p_target_first") is not None]
        if not olculen:
            return ("Hiçbir ufukta ölçülebilir taban oranı yok — veri yetersiz.")
        en_iyi = max(olculen, key=lambda c: c["p_target_first"])
        return (f"Doğrulanmış ufuk yok. Ham taban oranı en yüksek ufuk "
                f"{en_iyi['horizon']} (%{en_iyi['p_target_first']*100:.1f}) fakat "
                f"durumu {en_iyi['status']}: "
                f"{', '.join(en_iyi.get('rejection_reasons_tr', [])) or 'kanıt eksik'}. "
                f"Taban oranı bir tahmin değildir; modelin bu tabanı aşması "
                f"gerekir ve bu ölçülmedi.")

    parcalar = [f"{secilen['horizon']} seçildi."]
    digerleri = [c for c in cells if c is not secilen
                 and c.get("robust_utility") is not None]
    daha_yuksek_p = [c for c in digerleri
                     if (c.get("p_target_first") or 0) > (secilen.get("p_target_first") or 0)]
    if daha_yuksek_p:
        d = max(daha_yuksek_p, key=lambda c: c["p_target_first"])
        parcalar.append(
            f"{d['horizon']} ufkunda hedef olasılığı daha yüksek "
            f"(%{d['p_target_first']*100:.1f} vs %{secilen['p_target_first']*100:.1f}) "
            f"fakat beklenen tutma süresi {d.get('expected_holding_hours', 0):.1f} sa "
            f"olduğu için birim zaman başına robust EV daha düşük "
            f"({d['robust_utility']:+.4f} vs {secilen['robust_utility']:+.4f}).")
    daha_kisa = [c for c in cells
                 if c.get("horizon_minutes", 1 << 30) < secilen.get("horizon_minutes", 0)
                 and not c.get("tradable")]
    if daha_kisa:
        adlar = ", ".join(c["horizon"] for c in daha_kisa[-3:])
        parcalar.append(f"Daha kısa ufuklar ({adlar}) kapıları geçemedi.")
    return " ".join(parcalar)


# ── giriş planı (şartname 23, 24, 25, 65, 66) ──────────────────────────────

@dataclass
class EntryPlan:
    order_type: str                      # LIMIT | MARKETABLE_LIMIT | MARKET
    entry_low: Optional[float]
    optimal_entry: Optional[float]
    entry_high: Optional[float]
    max_chase_price: Optional[float]
    fill_probability: Optional[float]
    fill_probability_measured: bool
    adverse_selection_bps: Optional[float]
    reason: str

    def to_dict(self) -> Dict:
        return asdict(self)


def build_entry_plan(direction: str, mid: float, best_bid: Optional[float],
                     best_ask: Optional[float], atr_pct_now: Optional[float],
                     microprice: Optional[float] = None,
                     queue_data: bool = False) -> EntryPlan:
    """Giriş bölgesi ve emir türü.

    DOLUM OLASILIĞI — DÜRÜST BOŞLUK (şartname 24)
    Limit emrin dolma ihtimali kuyruk derinliği, agresif akış hızı ve venue
    gecikmesi gerektirir. Kaydedici bu alanları henüz biriktirmedi; bu yüzden
    limit girişte `fill_probability=None` döner ve emir türü olarak
    MARKETABLE_LIMIT önerilir. "Daha ucuz" diye dolmayacak bir limit seçmek
    (şartname 24'ün açık uyarısı) böylece engellenir.
    """
    if best_bid is None or best_ask is None or mid <= 0:
        return EntryPlan("MARKETABLE_LIMIT", None, None, None, None,
                         None, False, None,
                         "L2 en iyi kotasyon yok — giriş bölgesi kurulamadı")

    # Giriş bandı = yarım ufuk-sigması. ¼ ile ölçüldüğünde bant σ_bar'ın
    # altına düşüyor ve sinyalin yarı ömrü taban değere (30 sn) çakılıyordu;
    # yani sistem pratikte dolmayacak bir giriş öneriyordu.
    bant = (atr_pct_now or 0.2) / 100.0 * mid * 0.5
    if direction == "LONG":
        optimal = best_ask
        low, high = mid - bant, best_ask
        chase = best_ask * (1.0 + (atr_pct_now or 0.2) / 100.0 * 0.5)
    else:
        optimal = best_bid
        low, high = best_bid, mid + bant
        chase = best_bid * (1.0 - (atr_pct_now or 0.2) / 100.0 * 0.5)

    if queue_data:
        return EntryPlan("LIMIT", low, optimal, high, chase, None, False, None,
                         "kuyruk verisi var ama dolum modeli KALİBRE EDİLMEDİ")
    return EntryPlan(
        "MARKETABLE_LIMIT", float(low), float(optimal), float(high), float(chase),
        None, False, None,
        "dolum olasılığı ÖLÇÜLMEDİ (kuyruk/akış geçmişi yok) → pasif limit "
        "önerilmez; maliyet taker varsayımıyla hesaplandı")


def order_type_recommendation(maker_ev: Optional[float],
                              taker_ev: Optional[float],
                              fill_p: Optional[float]) -> Tuple[str, str]:
    """Şartname 65 — maker EV × dolum olasılığı vs taker EV.

    Dolum olasılığı bilinmiyorsa maker EV KIYASLANAMAZ; taker seçilir."""
    if fill_p is None or maker_ev is None or taker_ev is None:
        return ("MARKETABLE_LIMIT",
                "maker/taker karşılaştırması için dolum olasılığı gerekli — "
                "ölçülmedi, taker ekonomisi varsayıldı")
    if maker_ev * fill_p > taker_ev:
        return ("LIMIT", f"maker EV × dolum ({maker_ev * fill_p:+.4f}) > "
                         f"taker EV ({taker_ev:+.4f})")
    return ("MARKETABLE_LIMIT", f"taker EV ({taker_ev:+.4f}) ≥ "
                               f"maker EV × dolum ({maker_ev * fill_p:+.4f})")


# ── stres senaryoları (şartname 69) ────────────────────────────────────────

STRESS_SCENARIOS = [
    ("oynaklık ×2", {"vol_mult": 2.0}),
    ("spread ×3", {"spread_mult": 3.0}),
    ("derinlik −%50", {"depth_mult": 0.5}),
    ("gecikme ×3", {"latency_mult": 3.0}),
    ("ani BTC hareketi", {"vol_mult": 3.0, "spread_mult": 2.0}),
    ("borsa bozulması", {"spread_mult": 5.0, "depth_mult": 0.25,
                         "latency_mult": 5.0}),
]


def stress_test(base_ev: float, base_cost_bps: float, p_sl: float,
                net_loss_pct: float) -> List[Dict]:
    """Her senaryoda EV'yi yeniden hesapla. Ağır negatife düşen senaryo
    risk skorunu yükseltir."""
    out = []
    for ad, k in STRESS_SCENARIOS:
        ek_maliyet = base_cost_bps * (k.get("spread_mult", 1.0) - 1.0) * 0.5 \
            + base_cost_bps * (1.0 / max(0.1, k.get("depth_mult", 1.0)) - 1.0) * 0.3 \
            + 2.0 * (k.get("latency_mult", 1.0) - 1.0)
        ek_kayip = p_sl * abs(net_loss_pct) * (k.get("vol_mult", 1.0) - 1.0) * 0.5
        ev = base_ev - ek_maliyet / 100.0 - ek_kayip
        out.append({"scenario": ad, "ev_pct": round(float(ev), 4),
                    "extra_cost_bps": round(float(ek_maliyet), 2),
                    "severe": bool(ev < -0.5)})
    return out
