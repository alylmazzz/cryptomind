"""
Borsa komisyon tablosu + en ucuz venue seçimi.

Videonun ana dersi "komisyon kârı yedi" idi. Bu modül iki şey yapar:
  1. Her borsanın SPOT maker/taker ücretini (bps) beyan eder — kamuya açık
     ücret çizelgeleri (VIP0, indirimsiz). BEYANDIR; borsa değiştirebilir.
     Canlı hesapta gerçek ücret `fetch_trading_fees` ile okunur (anahtar ister).
  2. Verilen adaylar arasından gidiş-dönüş maliyeti en düşük venue'yu seçer;
     maker-öncelikli politikada maker ücreti ağırlıklıdır.

Sıralama ölçütü: beklenen gidiş-dönüş bps = p_maker·(2·maker) + (1−p_maker)·(2·taker)
p_maker = maker emirlerin dolma oranı (ölçülmediyse muhafazakâr 0,5).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class VenueFee:
    exchange_id: str
    maker_bps: float
    taker_bps: float
    note: str = ""

    def roundtrip_bps(self, p_maker: float = 0.5) -> float:
        p = min(1.0, max(0.0, p_maker))
        return 2.0 * (p * self.maker_bps + (1.0 - p) * self.taker_bps)


# Kamuya açık VIP0 spot çizelgeleri (2026 başı). İndirim jetonu (BNB/OKB/KCS/GT/MX)
# uygulanmamış hâli; kullanıcı panelden indirimi ayrıca açabilir.
FEE_TABLE: Dict[str, VenueFee] = {
    "mexc":     VenueFee("mexc",     0.0,  5.0,  "spot maker %0 · taker %0,05 — en düşük"),
    "okx":      VenueFee("okx",      8.0,  10.0, "spot %0,08 / %0,10"),
    "binance":  VenueFee("binance",  10.0, 10.0, "spot %0,10 / %0,10 (BNB ile %25 indirim)"),
    "bybit":    VenueFee("bybit",    10.0, 10.0, "spot %0,10 / %0,10"),
    "kucoin":   VenueFee("kucoin",   10.0, 10.0, "spot %0,10 / %0,10 (KCS ile indirim)"),
    "bitget":   VenueFee("bitget",   10.0, 10.0, "spot %0,10 / %0,10"),
    "gateio":   VenueFee("gateio",   20.0, 20.0, "spot %0,20 / %0,20 (GT ile indirim)"),
    "htx":      VenueFee("htx",      20.0, 20.0, "spot %0,20 / %0,20"),
    "kraken":   VenueFee("kraken",   25.0, 40.0, "spot %0,25 / %0,40"),
    "coinbase": VenueFee("coinbase", 60.0, 120.0, "Advanced %0,60 / %1,20 — en pahalı"),
}


def venue_fee(exchange_id: str) -> VenueFee:
    return FEE_TABLE.get(exchange_id, VenueFee(exchange_id, 10.0, 10.0, "tabloda yok — %0,10 varsayıldı"))


def ranked_venues(candidates: Optional[Sequence[str]] = None,
                  p_maker: float = 0.5) -> List[Dict]:
    ids = list(candidates) if candidates else list(FEE_TABLE)
    rows = []
    for ex in ids:
        f = venue_fee(ex)
        rows.append({"exchange_id": ex, "maker_bps": f.maker_bps, "taker_bps": f.taker_bps,
                     "roundtrip_bps": round(f.roundtrip_bps(p_maker), 2), "note": f.note})
    rows.sort(key=lambda r: (r["roundtrip_bps"], r["taker_bps"]))
    return rows


def cheapest_venue(candidates: Optional[Sequence[str]] = None, p_maker: float = 0.5,
                   probe: Optional[Callable[[str], bool]] = None) -> Dict:
    """En ucuz venue; `probe(exchange_id)` verilirse (canlı veri çekilebiliyor mu?)
    ilk geçen seçilir. Hiçbiri geçmezse en ucuz yine döner ama `probed=False`."""
    rows = ranked_venues(candidates, p_maker)
    if probe is None:
        return {**rows[0], "probed": None, "skipped": []}
    skipped = []
    for r in rows:
        try:
            ok = bool(probe(r["exchange_id"]))
        except Exception:
            ok = False
        if ok:
            return {**r, "probed": True, "skipped": skipped}
        skipped.append(r["exchange_id"])
    return {**rows[0], "probed": False, "skipped": skipped}
