"""
Likidasyon Haritası + CVD / Order Flow (#2).

- cvd_analysis(): borsa işlem akışından Kümülatif Hacim Deltası (taker alım -
  taker satım) ve büyük-taker net yönü.
- liquidation_map(): mevcut fiyat + yaygın kaldıraç seviyeleri (5x/10x/25x/50x/
  100x) kullanarak long ve short pozisyonların tahmini likidasyon fiyatlarını
  (likidasyon bölgeleri / heatmap) üretir. Open Interest varsa bölge yoğunluğu
  ölçeklenir.

Not: Borsaların gerçek toplam likidasyon emir defteri public değildir; bu modül
standart kaldıraç matematiğiyle bölgeleri TAHMİN eder (Coinglass benzeri yaklaşım).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def cvd_analysis(engine, symbol: str) -> Dict:
    ex = engine._spot_client()
    if ex is None:
        return {"cvd": 0.0, "trend": "veri yok", "net_taker_label": "—"}
    try:
        trades = ex.fetch_trades(symbol, limit=1000)
    except Exception:
        return {"cvd": 0.0, "trend": "veri yok", "net_taker_label": "—"}

    delta = 0.0
    cum = []
    big_buy = big_sell = 0.0
    for t in trades:
        amt = float(t.get("amount") or 0)
        cost = float(t.get("cost") or 0)
        d = amt if t.get("side") == "buy" else -amt
        delta += d
        cum.append(delta)
        if cost > 50_000:
            if t.get("side") == "buy":
                big_buy += cost
            else:
                big_sell += cost

    trend = "yükselen (alıcı baskısı)" if len(cum) > 10 and cum[-1] > cum[-min(50, len(cum))] \
        else "düşen (satıcı baskısı)" if len(cum) > 10 else "yatay"
    net = big_buy - big_sell
    label = (f"ALIM ${net/1e6:.2f}M" if net > 0 else f"SATIM ${abs(net)/1e6:.2f}M")
    return {
        "cvd": round(delta, 4),
        "trend": trend,
        "big_taker_buy_usd": round(big_buy, 0),
        "big_taker_sell_usd": round(big_sell, 0),
        "net_taker_label": label,
    }


def liquidation_map(engine, symbol: str) -> Dict:
    ex = engine._spot_client()
    price = None
    if ex is not None:
        try:
            price = float(ex.fetch_ticker(symbol)["last"])
        except Exception:
            price = None
    if price is None:
        return {"zones": [], "note": "fiyat alınamadı"}

    oi = engine.open_interest(symbol)
    leverages = [5, 10, 25, 50, 100]
    zones: List[Dict] = []
    # bakım marjini yaklaşımı: likidasyon ~ giriş * (1 ∓ 1/kaldıraç)
    for lev in leverages:
        long_liq = price * (1 - 1 / lev)
        short_liq = price * (1 + 1 / lev)
        # yoğunluk: düşük kaldıraç daha az ama daha büyük pozisyon; basit ağırlık
        density = (oi or 1.0) / lev
        zones.append({"side": "long", "leverage": lev, "price": round(long_liq, 4),
                      "distance_pct": round((long_liq - price) / price * 100, 2),
                      "density": round(density, 2)})
        zones.append({"side": "short", "leverage": lev, "price": round(short_liq, 4),
                      "distance_pct": round((short_liq - price) / price * 100, 2),
                      "density": round(density, 2)})
    # fiyata en yakın bölgeler önce
    zones.sort(key=lambda z: abs(z["distance_pct"]))
    return {"ref_price": round(price, 4), "open_interest": oi, "zones": zones,
            "note": "Standart kaldıraç matematiğiyle tahmini likidasyon bölgeleri"}
