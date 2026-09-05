"""PARİTE × UFUK NET +%1 NİTELENDİRME MOTORU.

Bu paket tek bir soruya cevap arar:

    "Bu parite için, maliyet sonrası NET +%1 hedefin stop bariyerinden ÖNCE
     görülme olasılığı hangi zaman ufkunda en yüksek ve bu ölçüm GÜVENİLİR mi?"

MUTLAK KURAL — KODLA ZORLANIR
`GUARANTEED` diye bir durum bu pakette YOKTUR ve `state.py` bunu bir testle
kilitler. Sistem hiçbir koşulda "garanti" demez. Çıktı iki ayrı satırdır:

    GARANTİ: YOK
    NET +%1 İÇİN EN YÜKSEK DOĞRULANMIŞ OLASILIK: X%   (ya da: YOK)

İkincisi ancak kanıt kapılarının hepsi geçilirse bir sayı taşır; aksi hâlde
`BEST QUALIFIED HORIZON: NONE` döner. Kanıt yokken tahmin ÜRETİLMEZ.
"""
from .horizons import HORIZONS, HORIZON_MIN, horizon_bars, primary_horizons
from .state import QualificationState, RejectionCode, decide_state

__all__ = ["HORIZONS", "HORIZON_MIN", "horizon_bars", "primary_horizons",
           "QualificationState", "RejectionCode", "decide_state"]
