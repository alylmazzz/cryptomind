"""
Venue router — "en düşük maker ücreti" ≠ "en ucuz". Her venue için tümü-dahil maliyet:

  ALL_IN(v) = giriş ücreti + çıkış ücreti + spread + beklenen kayma (derinliğe göre etki)
            + gecikme rezervi + dolum-hatası rezervi (+ funding/borrow varsa)

VENUE* = argmin ALL_IN, yalnız aynı fırsat o venue'da gerçekten varsa (parite listeli, veri taze).
Sağlık: devre kesici açık / veri STALE olan venue elenir. Kıyas yalnız Top-K adaylar için ve
60 sn önbellekle yapılır (paylaşımlı sunucuda ağ yükü).
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

from ..opportunity.costs import LATENCY_RESERVE_BPS, FAILURE_RESERVE_BPS, linear_impact_bps
from ..strategies import fees as FE

_CACHE: Dict[str, Dict] = {}
TTL = 60.0


def all_in_cost_bps(maker_bps: float, taker_bps: float, spread_bps: float, bid_depth: float,
                    ask_depth: float, notional: float, p_maker_fill: float = 0.5,
                    funding_bps: float = 0.0) -> Dict:
    fee = p_maker_fill * 2 * maker_bps + (1 - p_maker_fill) * 2 * taker_bps
    gi, _ = linear_impact_bps(notional, ask_depth if ask_depth else 1e9)
    ci, _ = linear_impact_bps(notional, bid_depth if bid_depth else 1e9)
    total = fee + spread_bps + gi + ci + LATENCY_RESERVE_BPS + FAILURE_RESERVE_BPS + funding_bps
    return {"fee_bps": round(fee, 3), "spread_bps": round(spread_bps, 3),
            "impact_bps": round(gi + ci, 3), "latency_bps": LATENCY_RESERVE_BPS,
            "failure_bps": FAILURE_RESERVE_BPS, "funding_bps": funding_bps, "total_bps": round(total, 3)}


def compare(symbol: str, venues: List[str], book_fn: Callable[[str, str], Optional[Dict]],
            fee_fn: Callable[[str, str], Dict], notional: float, p_maker_fill: float = 0.5,
            health_fn: Optional[Callable[[str], str]] = None, now: Optional[float] = None) -> Dict:
    """book_fn(venue, symbol) → {spread_bps, bid_depth_usd, ask_depth_usd} | None (listeli değil)."""
    now = time.time() if now is None else now
    key = f"{symbol}|{','.join(venues)}|{int(notional)}"
    hit = _CACHE.get(key)
    if hit and now - hit["ts"] < TTL:
        return hit["out"]
    rows = []
    for v in venues:
        if health_fn and health_fn(v) != "CLOSED":
            rows.append({"venue": v, "available": False, "why": "devre kesici açık"})
            continue
        try:
            b = book_fn(v, symbol)
        except Exception as e:
            rows.append({"venue": v, "available": False, "why": f"{type(e).__name__}"})
            continue
        if not b:
            rows.append({"venue": v, "available": False, "why": "parite listeli değil / veri yok"})
            continue
        f = fee_fn(v, symbol)
        c = all_in_cost_bps(f["maker_bps"], f["taker_bps"], float(b.get("spread_bps") or 0.0),
                            float(b.get("bid_depth_usd") or 0.0), float(b.get("ask_depth_usd") or 0.0),
                            notional, p_maker_fill)
        rows.append({"venue": v, "available": True, "fee_verified": bool(f.get("verified")),
                     "maker_bps": f["maker_bps"], "taker_bps": f["taker_bps"], **c,
                     "bid_depth_usd": b.get("bid_depth_usd"), "ask_depth_usd": b.get("ask_depth_usd")})
    ok = [r for r in rows if r.get("available")]
    ok.sort(key=lambda r: r["total_bps"])
    out = {"symbol": symbol, "notional": notional, "rows": rows,
           "best": (ok[0]["venue"] if ok else None),
           "best_total_bps": (ok[0]["total_bps"] if ok else None),
           "note": ("en düşük TÜM-DAHİL maliyet; yalnız listeli+sağlıklı venue'lar" if ok else "uygun venue yok")}
    _CACHE[key] = {"ts": now, "out": out}
    return out


def static_fee(venue: str, symbol: str) -> Dict:
    f = FE.venue_fee(venue)
    return {"maker_bps": f.maker_bps, "taker_bps": f.taker_bps, "verified": False}
