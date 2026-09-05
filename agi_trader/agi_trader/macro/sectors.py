"""
Sektör Rotasyonu — kripto sektörleri arası para akışı / göreli güç.

Temsili bir sepetteki coinleri sektörlere ayırır (L1/L2/DeFi/Meme/AI/Borsa/Ödeme/
Oracle/Gaming), her sektörün son dönem ortalama getirisini (7 ve 30 bar) hesaplar
ve sıralar. Hızlanan (mom7 > mom30) sektörlere para GİRİYOR, yavaşlayanlardan
ÇIKIYOR yorumu üretir → rotasyon sinyali. 30 dk önbelleklenir.
"""
from __future__ import annotations

import time
from typing import Dict, List

# Temsili sepet (sektör başına 2-3 likit coin)
_BASKET: Dict[str, List[str]] = {
    "L1 (Layer-1)": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT"],
    "L2 (Layer-2)": ["MATIC/USDT", "ARB/USDT", "OP/USDT"],
    "DeFi": ["UNI/USDT", "AAVE/USDT", "MKR/USDT"],
    "Meme": ["DOGE/USDT", "SHIB/USDT", "PEPE/USDT"],
    "AI": ["FET/USDT", "RNDR/USDT", "TAO/USDT"],
    "Borsa Token": ["BNB/USDT", "OKB/USDT"],
    "Ödeme": ["XRP/USDT", "LTC/USDT", "BCH/USDT"],
    "Oracle": ["LINK/USDT", "BAND/USDT"],
    "Gaming/Metaverse": ["SAND/USDT", "MANA/USDT", "AXS/USDT"],
}

_CACHE: Dict[str, object] = {"ts": 0.0, "data": None, "tf": None}
_TTL = 1800  # 30 dk


def compute_sector_rotation(orch, tf: str = "1d") -> Dict:
    now = time.time()
    if (_CACHE["data"] is not None and _CACHE["tf"] == tf and now - float(_CACHE["ts"]) < _TTL):
        return _CACHE["data"]  # type: ignore

    sectors: List[Dict] = []
    for sector, syms in _BASKET.items():
        m7, m30, members = [], [], []
        for s in syms:
            try:
                df = orch.data.fetch_ohlcv(s, tf, limit=35)
            except Exception:
                df = None
            if df is None or len(df) < 31:
                continue
            c = df["close"]
            m7.append((float(c.iloc[-1]) / float(c.iloc[-7]) - 1) * 100)
            m30.append((float(c.iloc[-1]) / float(c.iloc[-30]) - 1) * 100)
            members.append(s.split("/")[0])
        if not m7:
            continue
        avg7 = sum(m7) / len(m7)
        avg30 = sum(m30) / len(m30)
        sectors.append({
            "sector": sector, "mom7": round(avg7, 2), "mom30": round(avg30, 2),
            "accelerating": avg7 * 7 > avg30,   # haftalık hız aylık ortalamadan yüksek
            "members": members, "n": len(members),
        })

    sectors.sort(key=lambda x: x["mom7"], reverse=True)
    inflow = [s["sector"] for s in sectors if s["mom7"] > 0 and s["accelerating"]][:3]
    outflow = [s["sector"] for s in sectors if s["mom7"] < 0][-3:]
    note = ""
    if inflow:
        note += "Para GİRİYOR: " + ", ".join(inflow) + ". "
    if outflow:
        note += "Para ÇIKIYOR: " + ", ".join(outflow) + "."
    out = {"tf": tf, "sectors": sectors, "inflow": inflow, "outflow": outflow,
           "note": note or "Belirgin rotasyon yok.", "updated": now}
    _CACHE["data"], _CACHE["ts"], _CACHE["tf"] = out, now, tf
    return out
