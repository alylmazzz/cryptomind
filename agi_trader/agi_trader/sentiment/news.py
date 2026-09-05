"""
Haber sentiment katmanı — ÜCRETSİZ API'ler (NewsAPI / CryptoPanic).

Başlık metinlerinden hafif sözlük-tabanlı (lexicon) boğa/ayı skoru üretir.
Anahtar yoksa nötr/sıfır güven döner. Parite bazında 15 dk önbelleklenir.

Sağlayıcı önceliği:
  1. CryptoPanic (CRYPTOPANIC_API_KEY)  — kripto-özel, ücretsiz 5000/ay
  2. NewsAPI     (NEWSAPI_API_KEY)      — genel haber, ücretsiz 100/gün
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from ..core.models import LayerVote

try:
    import requests
    _HAS_REQUESTS = True
except Exception:  # pragma: no cover
    requests = None  # type: ignore
    _HAS_REQUESTS = False

# Basit finansal duygu sözlüğü (TR + EN)
_BULL = ["surge", "rally", "soar", "bullish", "breakout", "adoption", "partnership",
         "approval", "etf", "inflow", "upgrade", "gain", "jump", "record", "boom",
         "yükseliş", "ralli", "rekor", "onay", "ortaklık", "kazanç", "patlama"]
_BEAR = ["crash", "plunge", "dump", "bearish", "selloff", "hack", "exploit", "ban",
         "lawsuit", "outflow", "downgrade", "fear", "liquidation", "collapse", "fraud",
         "düşüş", "çöküş", "hack", "dava", "yasak", "likidasyon", "dolandırıcılık"]

_CACHE: Dict[str, Dict] = {}
_TTL = 900  # 15 dk


def _score_headlines(titles: List[str]) -> Tuple[float, int, int]:
    bull = bear = 0
    for t in titles:
        tl = (t or "").lower()
        if any(w in tl for w in _BULL):
            bull += 1
        if any(w in tl for w in _BEAR):
            bear += 1
    total = bull + bear
    score = (bull - bear) / (total + 1e-9) if total else 0.0
    return score, bull, bear


def _base(symbol: str) -> str:
    return symbol.split("/")[0].upper() if "/" in symbol else symbol.upper()


def _fetch_cryptopanic(base: str, key: str) -> Optional[List[str]]:
    try:
        r = requests.get("https://cryptopanic.com/api/v1/posts/",
                         params={"auth_token": key, "currencies": base, "public": "true"},
                         timeout=10)
        if r.status_code != 200:
            return None
        return [p.get("title", "") for p in (r.json() or {}).get("results", [])][:30]
    except Exception:
        return None


def _fetch_newsapi(base: str, key: str) -> Optional[List[str]]:
    try:
        q = {"BTC": "bitcoin", "ETH": "ethereum"}.get(base, base)
        r = requests.get("https://newsapi.org/v2/everything",
                         params={"q": q + " crypto", "sortBy": "publishedAt",
                                 "pageSize": 30, "language": "en", "apiKey": key},
                         timeout=10)
        if r.status_code != 200:
            return None
        return [a.get("title", "") for a in (r.json() or {}).get("articles", [])][:30]
    except Exception:
        return None


def get_news_sentiment(symbol: str, config) -> Optional[Dict]:
    base = _base(symbol)
    now = time.time()
    c = _CACHE.get(base)
    if c and now - c["ts"] < _TTL:
        return c
    if not _HAS_REQUESTS:
        return None
    titles: Optional[List[str]] = None
    provider = None
    cpk = config.secret("CRYPTOPANIC_API_KEY") if config else None
    nak = config.secret("NEWSAPI_API_KEY") if config else None
    if cpk:
        titles = _fetch_cryptopanic(base, cpk)
        provider = "cryptopanic"
    if titles is None and nak:
        titles = _fetch_newsapi(base, nak)
        provider = "newsapi"
    if not titles:
        return None
    score, bull, bear = _score_headlines(titles)
    out = {"score": round(score, 3), "count": len(titles), "bull": bull, "bear": bear,
           "provider": provider, "headlines": titles[:5], "ts": now}
    _CACHE[base] = out
    return out


def news_vote(symbol: str, config) -> LayerVote:
    n = get_news_sentiment(symbol, config)
    if not n:
        return LayerVote(name="news", score=0.0, confidence=0.0,
                         reasons=["Haber API anahtarı yok (CryptoPanic/NewsAPI) — nötr"],
                         detail={"connected": False})
    total = n["bull"] + n["bear"]
    conf = min(0.7, 0.3 + 0.04 * total)
    return LayerVote(
        name="news", score=float(n["score"]), confidence=float(conf),
        reasons=[f"Haber sentiment {n['score']:+.2f} ({n['bull']}↑/{n['bear']}↓, {n['count']} başlık · {n['provider']})"],
        detail={"connected": True, "provider": n["provider"], "headlines": n["headlines"]},
    )
