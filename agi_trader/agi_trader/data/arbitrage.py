"""
Çoklu-Borsa Füzyon + Arbitraj Tarayıcı (#8).

scan_arbitrage(): her parite için tüm bağlı borsalardan anlık fiyatı çeker,
en ucuz/en pahalı borsayı ve yüzde spread'i bulur. Spread (tipik işlem
maliyeti ~0.2%'in) üzerindeyse arbitraj fırsatı olarak işaretler.
fused_price(): borsalar arası likidite/öncelik ağırlıklı birleşik fiyat.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .exchange_manager import EXCHANGE_PRIORITY

# Tek yön (al+sat) için kaba toplam maliyet eşiği
FEE_THRESHOLD_PCT = 0.20


def _tickers(em, symbol: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for ex_id, ex in getattr(em, "exchanges", {}).items():
        try:
            t = ex.fetch_ticker(symbol)
            if t and t.get("last"):
                out[ex_id] = float(t["last"])
        except Exception:
            continue
    return out


def fused_price(em, symbol: str) -> Optional[float]:
    px = _tickers(em, symbol)
    if not px:
        return None
    num = sum(p * EXCHANGE_PRIORITY.get(ex, 0.5) for ex, p in px.items())
    den = sum(EXCHANGE_PRIORITY.get(ex, 0.5) for ex in px)
    return num / (den + 1e-9)


def scan_arbitrage(em, symbols: List[str]) -> List[Dict]:
    out: List[Dict] = []
    for sym in symbols:
        px = _tickers(em, sym)
        if len(px) < 2:
            continue
        cheap = min(px, key=px.get)
        dear = max(px, key=px.get)
        lo, hi = px[cheap], px[dear]
        spread = (hi - lo) / lo * 100
        actionable = spread > FEE_THRESHOLD_PCT
        out.append({
            "symbol": sym,
            "cheapest": cheap, "min_price": lo,
            "dearest": dear, "max_price": hi,
            "spread_pct": round(spread, 3),
            "fused_price": round(fused_price(em, sym) or 0, 4),
            "actionable": actionable,
            "note": f"AL {cheap} → SAT {dear} (net ~%{spread - FEE_THRESHOLD_PCT:.2f})" if actionable else "",
            "exchanges": px,
        })
    out.sort(key=lambda x: -x["spread_pct"])
    return out
