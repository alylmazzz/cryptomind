"""
Sentetik OHLCV üretici.

Amaç: ccxt / internet / API anahtarı olmadan sistemin uçtan uca çalışabilmesi.
Gerçekçi bir geometrik Brownian hareket + rejim değişimleri + hacim üretir,
böylece indikatör/formasyon motorları anlamlı veri üzerinde test edilebilir.
Deterministiktir (symbol+timeframe seed) — tekrarlanabilir sonuç verir.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

# Zaman dilimi -> dakika
TF_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480,
    "12h": 720, "1d": 1440, "3d": 4320, "1w": 10080, "1M": 43200,
    "1y": 525600,
}

# Parite için kabaca gerçekçi başlangıç fiyatları
_BASE_PRICE = {
    "BTC": 65000.0, "ETH": 3400.0, "SOL": 150.0, "BNB": 580.0,
    "XRP": 0.62, "ADA": 0.45, "DOGE": 0.16, "AVAX": 35.0,
}


def _seed_for(symbol: str, timeframe: str) -> int:
    return abs(hash(f"{symbol}|{timeframe}")) % (2**32)


def generate_ohlcv(symbol: str, timeframe: str, limit: int = 400) -> pd.DataFrame:
    """Gerçekçi sentetik OHLCV DataFrame döndürür."""
    rng = np.random.default_rng(_seed_for(symbol, timeframe))
    base_asset = symbol.split("/")[0].upper()
    price = _BASE_PRICE.get(base_asset, 100.0)

    minutes = TF_MINUTES.get(timeframe, 240)
    # zaman dilimine göre ölçeklenen volatilite
    vol = 0.004 * np.sqrt(minutes / 60.0)

    # rejim dizisi: trend yukarı / aşağı / yatay
    n = limit
    regimes = rng.choice([1, -1, 0], size=(n // 40) + 1, p=[0.4, 0.3, 0.3])
    drift = np.repeat(regimes, 40)[:n] * vol * 0.35

    shocks = rng.normal(0, vol, n)
    log_returns = drift + shocks
    closes = price * np.exp(np.cumsum(log_returns))

    opens = np.empty(n)
    opens[0] = price
    opens[1:] = closes[:-1]

    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, vol * 0.6, n)))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, vol * 0.6, n)))

    base_vol = 1_000_000 / max(price, 1e-9)
    volume = base_vol * (1 + np.abs(rng.normal(0, 0.8, n))) * (1 + np.abs(log_returns) * 50)

    end = pd.Timestamp.now("UTC").floor("min")
    idx = pd.date_range(end=end, periods=n, freq=f"{minutes}min")

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volume,
        },
        index=idx,
    )
    df.index.name = "timestamp"
    return df
