"""
Gerçek komisyon adaptörü — statik tablo yalnız yedek.

Anahtar varsa (testnet/canlı) ccxt `fetch_trading_fee(symbol)` / `fetch_trading_fees()` ile
hesaba özgü maker/taker oranı çekilir; TTL 6 saat; hata → statik tabloya düşülür ve
`verified=False` işaretlenir. Doğrulanmamış ücretle HIGH_CONFIDENCE üretilmez
(komite: verified False → güven ×0,85, "ücret doğrulanmadı" notu).
"""
from __future__ import annotations

import time
from typing import Dict, Optional

from ..strategies import fees as FE

TTL_SEC = 6 * 3600
_CACHE: Dict[tuple, Dict] = {}


def fetch_account_fee(exchange_client, exchange_id: str, symbol: str,
                      now: Optional[float] = None, ttl: float = TTL_SEC) -> Dict:
    """{maker_bps, taker_bps, verified, source, ts, stale, fee_currency, discounts}."""
    now = time.time() if now is None else now
    key = (exchange_id, symbol)
    hit = _CACHE.get(key)
    if hit and now - hit["ts"] < ttl:
        return {**hit, "stale": False}
    static = FE.venue_fee(exchange_id)
    fallback = {"maker_bps": static.maker_bps, "taker_bps": static.taker_bps, "verified": False,
                "source": "static_table", "ts": now, "stale": False, "fee_currency": None,
                "discounts": None, "note": static.note}
    if exchange_client is None:
        _CACHE[key] = fallback
        return fallback
    try:
        d = None
        if hasattr(exchange_client, "fetch_trading_fee"):
            d = exchange_client.fetch_trading_fee(symbol)
        elif hasattr(exchange_client, "fetch_trading_fees"):
            d = (exchange_client.fetch_trading_fees() or {}).get(symbol)
        if not d or d.get("maker") is None or d.get("taker") is None:
            raise ValueError("ücret alanı yok")
        out = {"maker_bps": round(float(d["maker"]) * 1e4, 3), "taker_bps": round(float(d["taker"]) * 1e4, 3),
               "verified": True, "source": f"{exchange_id}:fetch_trading_fee", "ts": now, "stale": False,
               "fee_currency": (d.get("info") or {}).get("feeCurrency") if isinstance(d.get("info"), dict) else None,
               "discounts": None, "note": "hesaba özgü oran"}
        _CACHE[key] = out
        return out
    except Exception as e:
        if hit:                                             # bayat ama doğrulanmış: kullan, işaretle
            return {**hit, "stale": True, "note": f"yenileme başarısız: {type(e).__name__}"}
        fb = {**fallback, "note": f"API hatası → statik tablo ({type(e).__name__})"}
        _CACHE[key] = fb
        return fb


def clear_cache() -> None:
    _CACHE.clear()
