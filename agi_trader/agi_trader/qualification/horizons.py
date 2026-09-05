"""Ufuk ızgarası — şartname 1 ve 2.

İKİ KAVRAM AYRI TUTULUR (şartname 2):

  feature_timeframe   : modelin GİRDİLERİNİN çözünürlüğü (5m, 1h, 4h, 1g …)
  prediction_horizon  : hedefin içinde aranacağı SÜRE (bu dosya)

"4 saatlik grafik" ile "4 saatlik tahmin ufku" aynı şey DEĞİLDİR. Panelde
yalnız `Prediction Horizon` yazılır.

TABAN ÇÖZÜNÜRLÜK 5 DAKİKA
Bütün ufuklar 5 dakikalık barın katıdır. Bunun sebebi keyfi değil: hedef ile
stop aynı barın içinde vurulduğunda hangisinin önce geldiği bilinemez. Bar ne
kadar küçükse bu belirsizlik o kadar seyrekleşir. 5m altındaki ufuklar (1m,
tick) veri maliyeti nedeniyle kapsam dışıdır ve bu yüzden ızgarada YOKTUR —
"ölçemediğimiz ufku listelemeyiz".
"""
from __future__ import annotations

from typing import Dict, List, Tuple

BASE_TF_MIN = 5

# (etiket, dakika). 48h yalnız REFERANS: kullanıcı günlük %1 hedeflediği için
# ana karşılaştırma 5m–24h arasında yapılır (şartname 1).
HORIZONS: List[Tuple[str, int]] = [
    ("5m", 5),
    ("15m", 15),
    ("30m", 30),
    ("1h", 60),
    ("2h", 120),
    ("4h", 240),
    ("6h", 360),
    ("8h", 480),
    ("12h", 720),
    ("24h", 1440),
    ("48h", 2880),
]

HORIZON_MIN: Dict[str, int] = {k: v for k, v in HORIZONS}
REFERENCE_ONLY = {"48h"}


def horizon_bars(label: str, tf_min: int = BASE_TF_MIN) -> int:
    """Ufkun kaç TABAN bara denk geldiği."""
    dk = HORIZON_MIN[label]
    if dk % tf_min:
        raise ValueError(f"{label} ({dk} dk) {tf_min} dakikanın katı değil")
    return dk // tf_min


def horizon_hours(label: str) -> float:
    return HORIZON_MIN[label] / 60.0


def primary_horizons() -> List[str]:
    """Karar için karşılaştırılan ufuklar — 48h dışarıda (yalnız referans)."""
    return [k for k, _ in HORIZONS if k not in REFERENCE_ONLY]


def all_horizons() -> List[str]:
    return [k for k, _ in HORIZONS]


def max_bars(tf_min: int = BASE_TF_MIN) -> int:
    return horizon_bars(HORIZONS[-1][0], tf_min)
