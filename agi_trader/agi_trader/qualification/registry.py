"""Kanıt sicili — şartname 31, 32, 33, 38, 39 (2. mesaj 38, 39).

TEK KURAL: bir sinyal ailesinin YÖN AĞIRLIĞI alabilmesi için OOS yön kenarının
ÖLÇÜLMÜŞ ve KANITLANMIŞ olması gerekir. Kanıtlanmamış aile grafikte
gösterilebilir ama karar motoruna ağırlık VEREMEZ.

Bu, bu programda yapılan ölçümlerin sonucudur; bir tercih değil:

  • 384 göstergelik panelin konsensüsü ölçüldü → TERS yönde çalışıyor
  • klasik formasyonların geometrisi düzeltildi, çift tepe/dip'in ayırt
    ediciliği hiçbir kapı ayarında rastgeleyi geçemedi
  • harmonik formasyonların ŞEKLİ gerçek (+27 puan), YÖNÜ yok
  • mum formasyonlarında anlamlı çıkan iki sonuç TERS işaretli
  • mikroyapıda yön var (spread t=+4,66) ama maliyeti 2,6–5,2× karşılamıyor

Bu yüzden `directional_weight` hepsinde 0'dır. İleride ayrı ve kilitli bir OOS
çalışması gerçek lift gösterirse ağırlık >0 yapılabilir — ölçüm olmadan asla.
"""
from __future__ import annotations

from typing import Dict, List, Optional

VERIFIED = "VERIFIED"
UNVERIFIED = "UNVERIFIED"
REFUTED = "REFUTED"
UNMEASURED = "UNMEASURED"

# Yön ağırlığı — şartname 31. Bu sabit 0'dır ve testle kilitlenmiştir.
INDICATOR_DIRECTIONAL_WEIGHT = 0.0


def _row(name: str, claim: str, status: str, *, sample: Optional[int] = None,
         oos_rate: Optional[float] = None, random_rate: Optional[float] = None,
         lift: Optional[float] = None, p_value: Optional[float] = None,
         dsr: Optional[float] = None, pbo: Optional[float] = None,
         note: str = "") -> Dict:
    return {"pattern_name": name, "direction_claim": claim,
            "sample_size": sample, "oos_target_first_rate": oos_rate,
            "matched_random_rate": random_rate, "lift": lift,
            "p_value": p_value, "dsr": dsr, "pbo": pbo,
            "status": status,
            "directional_weight": 0.0,
            "display": "grafikte gösterilebilir",
            "note": note}


def signal_registry() -> List[Dict]:
    """Şartname 32 — her ailenin kaydı. Hiçbiri yön ağırlığı taşımaz."""
    return [
        _row("gosterge_konsensusu", "AL/SAT sayımı yön verir", REFUTED,
             note="384 gösterge ölçüldü; konsensüsü takip etmek para "
                  "kaybettiriyor. Panel AÇIKLAYICI kalır, sinyal değildir."),
        _row("ucgen_kama_kanal_dikdortgen", "kırılım yönü", UNVERIFIED,
             note="geometri düzeltildi (zarf uydurma, pivot seçimi); yön "
                  "doğruluğu 152/152 ama ileri getiri kenarı ölçülmedi"),
        _row("cift_tepe_dip", "dönüş yönü", REFUTED,
             note="hiçbir kapı ayarında gerçek oran eşleşen rastgele "
                  "yürüyüşü geçemedi; ileri çalışmada işaret TERS çıktı"),
        _row("harmonik", "PRZ'den dönüş", REFUTED,
             note="ŞEKİL gerçek (rastgeleye göre +27 puan) fakat YÖN yok"),
        _row("mum_formasyonlari", "tek/iki barlık dönüş", REFUTED,
             note="24 formasyon ölçüldü; anlamlı çıkan iki sonuç TERS işaretli"),
        _row("mikroyapi_L2", "defter dengesizliği / spread", UNVERIFIED,
             note="spread t=+4,66 ve 9/10 paritede doğru işaret — programda "
                  "ilk doğru işaretli sinyal; fakat maliyeti 2,6–5,2× "
                  "karşılamıyor. 34,5 saatlik tek rejim: KANIT DEĞİL."),
        _row("sosyal_haber", "olay sonrası yön", UNMEASURED,
             note="olay örneklemi yetersiz — 'veri yok' ile 'etki yok' aynı "
                  "şey değildir (şartname 33)"),
        _row("net1_softmax_modeli", "P(net +%1 hedef, stop'tan önce)", UNVERIFIED,
             note="ölçülüyor; durum makinesi her hücre için ayrı karar verir"),
    ]


def registry_note() -> Dict:
    return {
        "indicator_directional_weight": INDICATOR_DIRECTIONAL_WEIGHT,
        "rule": ("Kanıtlanmamış aile karar motoruna ağırlık veremez; yalnız "
                 "görsel bağlam olarak gösterilir."),
        "how_to_promote": ("Ayrı ve kilitli bir OOS çalışmasında gerçek lift "
                           "kanıtlanırsa ağırlık >0 yapılabilir."),
        "statuses": [VERIFIED, UNVERIFIED, REFUTED, UNMEASURED],
    }
