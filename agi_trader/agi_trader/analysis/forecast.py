"""
Sonraki Periyot Tahmin Motoru + Alış/Satış Baskısı.

İki çıktı üretir:

1) forecast_next(): Bir SONRAKİ mumun (örn. bir sonraki 4s) beklenen
   maksimum/minimum/kapanış fiyatını + güven aralıklarını (±1σ ≈ %68,
   ±2σ ≈ %95) tahmin eder. Yön eğilimi karar motorunun birleşik skorundan,
   büyüklük ATR + getiri volatilitesinden gelir. Bu, "geleceği kesin bilme"
   değil; volatilite-temelli, açıklanabilir bir BEKLENTİ aralığıdır.

2) buy_sell_pressure(): Tüm katman oyları + hacim akışı + mum gövdesi baskısını
   tek bir "alıcı %X / satıcı %Y" oranına indirir.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .indicators import atr as atr_fn


def forecast_next(df: pd.DataFrame, agg_score: float, p_up: float,
                  confidence: float) -> Dict:
    """Bir sonraki mumun maks/min/kapanış beklentisi ve güven bantları."""
    close = float(df["close"].iloc[-1])
    atr = float(atr_fn(df, 14).iloc[-1])

    rets = df["close"].pct_change().dropna()
    sigma_ret = float(rets.tail(30).std()) if len(rets) >= 5 else 0.01
    price_sigma = close * sigma_ret

    # Yön eğilimi: birleşik skor kadar, volatiliteyle ölçekli kayma
    tilt = float(np.clip(agg_score, -1, 1))
    drift = tilt * price_sigma * 1.2
    expected_close = close + drift

    # Beklenen bar aralığı ≈ ATR (boğa eğiliminde tepeye, ayıda dibe asimetri)
    up_skew = 0.6 + 0.4 * max(0.0, tilt)
    dn_skew = 0.6 + 0.4 * max(0.0, -tilt)
    expected_high = expected_close + up_skew * atr
    expected_low = expected_close - dn_skew * atr

    # Olasılıksal bantlar
    band68_high = expected_close + price_sigma
    band68_low = expected_close - price_sigma
    band95_high = expected_close + 2 * price_sigma
    band95_low = expected_close - 2 * price_sigma

    # Beklenen hareket yüzdeleri
    up_move_pct = (expected_high - close) / close * 100
    down_move_pct = (expected_low - close) / close * 100

    return {
        "ref_price": round(close, 6),
        "atr": round(atr, 6),
        "prob_up": round(float(p_up), 3),
        "expected_close": round(expected_close, 6),
        "expected_high": round(expected_high, 6),     # sonraki periyot beklenen MAKS
        "expected_low": round(expected_low, 6),       # sonraki periyot beklenen MİN
        "up_move_pct": round(up_move_pct, 2),
        "down_move_pct": round(down_move_pct, 2),
        "band68": [round(band68_low, 6), round(band68_high, 6)],
        "band95": [round(band95_low, 6), round(band95_high, 6)],
        "note": "Volatilite-temelli beklenti aralığı (kesin tahmin değildir)",
    }


def buy_sell_pressure(df: pd.DataFrame, agg_score: float, n: int = 20) -> Tuple[float, float, str]:
    """Katman skoru + hacim akışı + mum gövdesi baskısından alıcı/satıcı %."""
    recent = df.tail(n)
    up_vol = float(recent.loc[recent["close"] > recent["open"], "volume"].sum())
    dn_vol = float(recent.loc[recent["close"] < recent["open"], "volume"].sum())
    vol_buy = up_vol / (up_vol + dn_vol + 1e-9)        # 0..1

    body = recent["close"] - recent["open"]
    body_buy = float(body.clip(lower=0).sum()) / (float(body.abs().sum()) + 1e-9)

    score_buy = (np.clip(agg_score, -1, 1) + 1) / 2     # 0..1

    buy = 0.45 * score_buy + 0.35 * vol_buy + 0.20 * body_buy
    buy_pct = round(float(np.clip(buy, 0, 1)) * 100, 1)
    sell_pct = round(100 - buy_pct, 1)

    if buy_pct >= 55:
        label = "ALICILAR HÂKİM"
    elif buy_pct <= 45:
        label = "SATICILAR HÂKİM"
    else:
        label = "DENGELİ"
    return buy_pct, sell_pct, label
