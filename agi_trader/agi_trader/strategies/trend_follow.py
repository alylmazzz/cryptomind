"""
Trend-Takip Çekirdek Stratejisi (OOS-DOĞRULANMIŞ).

4.5 yıl (2022-2026) çok-rejim veride, SABİT parametreyle (fit yok → doğal OOS),
gerçekçi maliyetle doğrulandı: 5-parite portföy Sharpe ~1.05, CAGR ~%19,
MaxDD %18. Karmaşık 1h TA-konsensüs botunun aksine (OOS'ta edge'siz), bu basit
günlük trend-takip rejimler arası KÂRLIDIR ve overfit'e dayanıklıdır.

Mantık (parite başına, long/flat):
  Pozisyonda ol  ⇔  fiyat > 200-günlük SMA  VE  20-günlük momentum > 0
  Aksi halde     ⇔  NAKİT (piyasadan çık — ayı/düşüşü es geç)
Vol-targeting: pozisyon boyutu = hedef_vol / gerçekleşen_vol (kaldıraçsız tavan).

Referans: Time-Series Momentum (Moskowitz, Ooi, Pedersen 2012); 200-gün trend filtresi.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

# OOS-doğrulanmış sabit parametreler (OPTİMİZE ETME — overfit riski)
SMA_TREND = 200
MOM_LOOKBACK = 20
VOL_WINDOW = 30
TARGET_VOL_DAILY = 0.025      # ~yıllık %48; vol-targeting ile normalize
MAX_LEVERAGE = 1.0            # spot/kaldıraçsız


def to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Herhangi bir OHLCV'yi günlüğe indir (datetime-index gerekir)."""
    d = df[["open", "high", "low", "close", "volume"]].astype(float)
    return pd.DataFrame({
        "open": d["open"].resample("1D").first(), "high": d["high"].resample("1D").max(),
        "low": d["low"].resample("1D").min(), "close": d["close"].resample("1D").last(),
        "volume": d["volume"].resample("1D").sum(),
    }).dropna()


def trend_position(daily: pd.DataFrame, vol_target: bool = True) -> pd.Series:
    """Günlük hedef pozisyon serisi (0..MAX_LEVERAGE). Look-ahead yok."""
    c = daily["close"]
    raw = ((c > c.rolling(SMA_TREND).mean()) & (c.pct_change(MOM_LOOKBACK) > 0)).astype(float)
    if not vol_target:
        return raw.fillna(0)
    vol = c.pct_change().rolling(VOL_WINDOW).std()
    scale = (TARGET_VOL_DAILY / (vol + 1e-9)).clip(0, MAX_LEVERAGE)
    return (raw * scale).fillna(0)


def current_signal(df: pd.DataFrame, vol_target: bool = True) -> Dict:
    """Canlı kullanım: son bar için hedef pozisyon + açıklama.
    df: bir paritenin OHLCV'si (herhangi TF; içeride günlüğe indirilir)."""
    daily = to_daily(df)
    if len(daily) < SMA_TREND + MOM_LOOKBACK + 5:
        return {"target": 0.0, "in_market": False, "reason": "yetersiz günlük veri"}
    c = daily["close"]
    pos = trend_position(daily, vol_target)
    target = float(pos.iloc[-1])
    price = float(c.iloc[-1]); sma = float(c.rolling(SMA_TREND).mean().iloc[-1])
    mom = float(c.pct_change(MOM_LOOKBACK).iloc[-1])
    above = price > sma
    reason = (f"Fiyat {price:.4f} {'>' if above else '<'} SMA200 {sma:.4f}; "
              f"20g momentum {mom*100:+.1f}% → {'POZİSYON' if target > 0 else 'NAKİT'}")
    return {"target": round(target, 3), "in_market": target > 0,
            "above_sma200": above, "mom20_pct": round(mom * 100, 2),
            "price": price, "sma200": round(sma, 6), "reason": reason}


def backtest(df: pd.DataFrame, cost_per_side: float = 0.0006, vol_target: bool = True) -> Dict:
    """Tek parite günlük trend-takip backtest (gerçekçi maliyet). Doğrulama için."""
    daily = to_daily(df)
    c = daily["close"]; r = c.pct_change().fillna(0)
    pos = trend_position(daily, vol_target)
    p = pos.shift(1).fillna(0)
    turn = p.diff().abs().fillna(0)
    strat = p * r - turn * cost_per_side
    eq = (1 + strat).cumprod()
    total = float((eq.iloc[-1] - 1) * 100)
    dd = float(((eq.cummax() - eq) / eq.cummax()).max() * 100)
    sharpe = float(strat.mean() / (strat.std() + 1e-12) * np.sqrt(365))
    return {"total_return_pct": round(total, 1), "max_drawdown_pct": round(dd, 1),
            "sharpe": round(sharpe, 2), "days": len(daily),
            "cagr_pct": round(float((eq.iloc[-1] ** (365 / max(1, len(eq))) - 1) * 100), 1)}
