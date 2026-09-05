"""
Kripto Korku & Açgözlülük Endeksi (alternative.me) — ÜCRETSİZ, ANAHTARSIZ.

0 = Aşırı Korku (genelde dip/alım fırsatı), 100 = Aşırı Açgözlülük (tepe/risk).
Kontraryen yorum: aşırı korku hafif boğa, aşırı açgözlülük hafif ayı sinyalidir.
Piyasa geneli bir göstergedir (parite-bağımsız); 1 saat önbelleklenir.
"""
from __future__ import annotations

import time
from typing import Dict, Optional

from ..core.models import LayerVote

try:
    import requests
    _HAS_REQUESTS = True
except Exception:  # pragma: no cover
    requests = None  # type: ignore
    _HAS_REQUESTS = False

_CACHE: Dict[str, object] = {"ts": 0.0, "data": None}
_TTL = 3600  # 1 saat


def get_fear_greed() -> Optional[Dict]:
    """{value:0-100, label, score:-1..1, ts} veya None."""
    now = time.time()
    if _CACHE["data"] is not None and now - float(_CACHE["ts"]) < _TTL:
        return _CACHE["data"]  # type: ignore
    if not _HAS_REQUESTS:
        return None
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        if r.status_code != 200:
            return None
        d = (r.json() or {}).get("data", [])
        if not d:
            return None
        value = int(d[0]["value"])
        label = d[0].get("value_classification", "")
        # kontraryen skor: 50 nötr; <25 korku -> +; >75 açgözlülük -> -
        score = (50 - value) / 50.0  # value 0 -> +1 (korku=boğa), 100 -> -1 (açgözlü=ayı)
        out = {"value": value, "label": label, "score": round(score, 3), "ts": now}
        _CACHE["data"], _CACHE["ts"] = out, now
        return out
    except Exception:
        return None


def fear_greed_vote() -> LayerVote:
    fg = get_fear_greed()
    if not fg:
        return LayerVote(name="fear_greed", score=0.0, confidence=0.0,
                         reasons=["Korku/Açgözlülük endeksi alınamadı (ağ?) — nötr"],
                         detail={"connected": False})
    v = fg["value"]
    # güven uçlarda yüksek (aşırı korku/açgözlülük daha bilgi verici)
    conf = 0.35 + 0.45 * (abs(50 - v) / 50.0)
    dir_txt = ("aşırı korku → kontraryen ALIM" if v <= 25 else
               "aşırı açgözlülük → kontraryen SATIŞ" if v >= 75 else
               "nötr bölge")
    return LayerVote(
        name="fear_greed", score=float(fg["score"]), confidence=float(min(0.8, conf)),
        reasons=[f"Korku/Açgözlülük: {v}/100 ({fg['label']}) — {dir_txt}"],
        detail={"value": v, "label": fg["label"], "connected": True},
    )
