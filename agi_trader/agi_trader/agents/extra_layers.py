"""
Hafif ek analiz katmanları: On-chain proxy ve Makro.

Spec on-chain (whale, exchange akışı, funding, OI, CVD, order flow ...) ve makro
(FED, ETF, CPI ...) için tam veri harici API gerektirir. API anahtarları
verildiğinde gerçek sağlayıcılara (Glassnode/CryptoQuant/Nansen) bağlanacak
şekilde genişletilebilir. Anahtar yokken bu katmanlar fiyat/hacim türevlerinden
PROXY üretir ve DÜŞÜK güvenle oy verir — böylece karar motorunda yerleri korunur
ama aşırı ağırlık almazlar.
"""
from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..core.models import LayerVote
from ..analysis.indicators import obv, ema

try:
    import requests
    _HAS_REQUESTS = True
except Exception:  # pragma: no cover
    requests = None  # type: ignore
    _HAS_REQUESTS = False

_FRED_CACHE: Dict[str, object] = {"ts": 0.0, "data": None}
_FRED_TTL = 6 * 3600  # makro veri yavaş değişir → 6 saat önbellek


def _fred_series(series_id: str, key: str, limit: int = 13):
    """FRED'den bir serinin son gözlemleri (yeni→eski)."""
    try:
        r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                         params={"series_id": series_id, "api_key": key, "file_type": "json",
                                 "sort_order": "desc", "limit": limit}, timeout=12)
        if r.status_code != 200:
            return None
        obs = [float(o["value"]) for o in (r.json() or {}).get("observations", [])
               if o.get("value") not in (".", "", None)]
        return obs or None
    except Exception:
        return None


def _fetch_macro(key: str) -> Optional[Dict]:
    """FED faizi + CPI yıllık enflasyon → risk-on/off eğilimi (önbellekli)."""
    now = time.time()
    if _FRED_CACHE["data"] is not None and now - float(_FRED_CACHE["ts"]) < _FRED_TTL:
        return _FRED_CACHE["data"]  # type: ignore
    if not _HAS_REQUESTS:
        return None
    fed = _fred_series("FEDFUNDS", key, 4)
    cpi = _fred_series("CPIAUCSL", key, 13)
    if not fed and not cpi:
        return None
    data: Dict[str, float] = {}
    if fed and len(fed) >= 2:
        data["fed_rate"] = fed[0]
        data["fed_delta"] = fed[0] - fed[1]          # faiz artıyor mu (risk-off)
    if cpi and len(cpi) >= 13:
        data["cpi_yoy"] = (cpi[0] / cpi[12] - 1) * 100  # yıllık enflasyon %
    _FRED_CACHE["data"], _FRED_CACHE["ts"] = data, now
    return data


def onchain_proxy_vote(df: pd.DataFrame) -> LayerVote:
    """OBV eğimi + hacim baskısından net akış proxy'si."""
    o = obv(df)
    slope = (o.iloc[-1] - o.iloc[-10]) / (abs(o.iloc[-10]) + 1e-9) if len(o) > 10 else 0.0
    vol_trend = (df["volume"].tail(5).mean() - df["volume"].tail(20).mean()) / (df["volume"].tail(20).mean() + 1e-9)

    score = float(np.tanh(slope * 2 + vol_trend * 0.5))
    reasons = [
        f"OBV eğimi {slope:+.2%} (akım yönü proxy)",
        f"Hacim trendi {vol_trend:+.2%}",
        "Not: gerçek on-chain (whale/exchange flow/funding/OI) için API anahtarı gerekli",
    ]
    return LayerVote(name="onchain", score=score, confidence=0.25, reasons=reasons,
                     detail={"obv_slope": slope, "proxy": True})


def macro_vote(df: pd.DataFrame, config=None) -> LayerVote:
    """FRED API anahtarı varsa gerçek makro (FED faizi + CPI enflasyon) → risk-on/off.
    Anahtar yoksa nötr, çok düşük güven (yerini korur, aşırı ağırlık almaz)."""
    key = config.secret("FRED_API_KEY") if config else None
    if key:
        m = _fetch_macro(key)
        if m:
            score = 0.0
            reasons = []
            # Faiz artışı/yüksek faiz → kripto için risk-off (ayı); düşüş → risk-on (boğa)
            if "fed_delta" in m:
                if m["fed_delta"] > 0.01:
                    score -= 0.4; reasons.append(f"FED faizi artıyor (%{m['fed_rate']:.2f}) → risk-off")
                elif m["fed_delta"] < -0.01:
                    score += 0.4; reasons.append(f"FED faizi düşüyor (%{m['fed_rate']:.2f}) → risk-on")
                else:
                    reasons.append(f"FED faizi sabit (%{m['fed_rate']:.2f})")
            if "cpi_yoy" in m:
                if m["cpi_yoy"] > 4:
                    score -= 0.3; reasons.append(f"Yüksek enflasyon (CPI %{m['cpi_yoy']:.1f} yıllık) → sıkı para")
                elif m["cpi_yoy"] < 2.5:
                    score += 0.3; reasons.append(f"Düşük enflasyon (CPI %{m['cpi_yoy']:.1f}) → gevşek para")
                else:
                    reasons.append(f"Enflasyon ılımlı (CPI %{m['cpi_yoy']:.1f})")
            return LayerVote(
                name="macro", score=float(np.clip(score, -1, 1)), confidence=0.5,
                reasons=reasons or ["FRED makro verisi alındı"],
                detail={"connected": True, "source": "FRED", **m},
            )
    return LayerVote(
        name="macro", score=0.0, confidence=0.1,
        reasons=["Makro veri kaynağı bağlı değil (FRED_API_KEY ekleyin: FED/CPI canlı) — nötr"],
        detail={"connected": False},
    )
