"""
NaN-GÜVENLİ JSON YANITI — FastAPI varsayılanı `allow_nan=False` ile NaN/inf görünce 500 verir
(canlıda /api/risk ve /api/trend bu yüzden düşüyordu; panel her 5 sn'de bir hata yazıyordu).
Bütün uygulamalar bu yanıt sınıfını varsayılan yapar: NaN/inf → null, numpy skalerleri → Python.
"""
from __future__ import annotations

import json
import math
from typing import Any

from fastapi.responses import JSONResponse


def clean(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [clean(v) for v in x]
    if isinstance(x, bool) or x is None or isinstance(x, (str, int)):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    try:                                             # numpy skalerleri / pandas Timestamp vb.
        import numpy as np
        if isinstance(x, np.generic):
            v = x.item()
            return clean(v)
    except Exception:
        pass
    if hasattr(x, "isoformat"):
        try:
            return x.isoformat()
        except Exception:
            return str(x)
    return x


class SafeJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return json.dumps(clean(content), ensure_ascii=False, allow_nan=False, separators=(",", ":"), default=str).encode("utf-8")
