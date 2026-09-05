"""
Hafif mod bayrağı.

`AGI_LIGHT_MODE=1` iken ağır ML bağımlılıkları (torch, transformers, XGBoost,
LightGBM, RL) KURULU OLSA BİLE hiç import edilmez. Sadece `_HAS_*` bayrağını
sonradan False yapmak yetmez: import zaten gerçekleştiği için torch tek başına
~400 MB RSS tutar. Paylaşımlı ARM VPS'te (CryptoMind paneli) bu bellek diğer
uygulamaları düşürebilir.

Kullanım:
    from ..core.light import LIGHT_MODE
    if not LIGHT_MODE:
        try:
            import torch
            ...
"""
from __future__ import annotations

import os

LIGHT_MODE: bool = os.environ.get("AGI_LIGHT_MODE", "").strip().lower() in ("1", "true", "yes")
