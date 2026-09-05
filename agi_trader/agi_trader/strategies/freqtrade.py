"""
Popüler Freqtrade topluluk stratejilerinin SADIK uygulamaları.

Mantık, kanonik freqtrade/freqtrade-strategies reposundaki orijinal kodlardan
çevrilmiştir (qtpylib Bollinger bantları tipik fiyat (h+l+c)/3 üzerinde hesaplanır).
Long-only; çıkış Freqtrade semantiğiyle: ROI hedefi · stop-loss · (varsa) satış sinyali.

Eklenen stratejiler:
  • BbandRsi      — RSI(14)<30 & close<BB-alt → al; RSI>70 → sat. ROI %10, stop −%25.
  • BinHV45       — BB(40) delta/tail kırılımı. Çıkış sinyali yok. ROI %1.25, stop −%5.
  • ClucMay72018  — close<EMA100 & close<0.985·BB-alt & düşük hacim → al; close>BB-orta → sat.
                    ROI %1, stop −%5.

Yeni strateji eklemek: bir fonksiyon yaz (df → (buy, sell) bool Series) ve
FREQTRADE_STRATEGIES sözlüğüne (fonksiyon, roi, stoploss, tasarım_tf) olarak ekle.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ..analysis.indicators import ema, rsi, sma

FEE = 0.001        # Freqtrade varsayılan spot fee (~%0.1 / taraf)
SLIP = 0.0005


def _typ(df: pd.DataFrame) -> pd.Series:
    return (df["high"] + df["low"] + df["close"]) / 3.0


def _bb(series: pd.Series, n: int, k: float = 2.0):
    mid = sma(series, n)
    std = series.rolling(n, min_periods=1).std()
    return mid + k * std, mid, mid - k * std


# ---------------------------------------------------------------- stratejiler
def s_bbandrsi(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    c = df["close"]
    r = rsi(c, 14)
    _, _, lower = _bb(_typ(df), 20, 2.0)
    buy = (r < 30) & (c < lower)
    sell = (r > 70)
    return buy.fillna(False), sell.fillna(False)


def s_binhv45(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    c = df["close"]
    _, mid, lower = _bb(_typ(df), 40, 2.0)
    bbdelta = (mid - lower).abs()
    closedelta = (c - c.shift(1)).abs()
    tail = (c - df["low"]).abs()
    buy = (
        (lower.shift(1) > 0)
        & (bbdelta > c * 0.008)
        & (closedelta > c * 0.017)
        & (tail < bbdelta * 0.25)
        & (c < lower.shift(1))
        & (c < c.shift(1))
    )
    sell = pd.Series(False, index=df.index)     # çıkış sinyali yok (ROI/stop)
    return buy.fillna(False), sell


def s_cluc(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    c = df["close"]
    _, mid, lower = _bb(_typ(df), 20, 2.0)
    e100 = ema(c, 100)
    vol_mean = df["volume"].rolling(30, min_periods=1).mean().shift(1)
    buy = (c < e100) & (c < 0.985 * lower) & (df["volume"] < vol_mean * 20)
    sell = (c > mid)
    return buy.fillna(False), sell.fillna(False)


# (fonksiyon, minimal_roi, stoploss, tasarım_timeframe)
FREQTRADE_STRATEGIES: Dict[str, dict] = {
    "BbandRsi":     {"fn": s_bbandrsi, "roi": 0.10,   "stop": -0.25, "tf": "1h"},
    "BinHV45":      {"fn": s_binhv45,  "roi": 0.0125, "stop": -0.05, "tf": "1m"},
    "ClucMay72018": {"fn": s_cluc,     "roi": 0.01,   "stop": -0.05, "tf": "5m"},
}

_TF_MIN = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}


# ---------------------------------------------------------------- simülatör
def simulate_freqtrade(df: pd.DataFrame, buy: pd.Series, sell: pd.Series,
                       roi: float, stoploss: float, tf: str,
                       start: int = 120, end: int = None) -> Dict:
    """Freqtrade-tarzı long-only backtest: ROI hedefi · stop-loss · satış sinyali.
    Giriş sinyal mumunun kapanışında (slippage'li); çıkışlar her mumda kontrol edilir
    (önce stop, sonra ROI, sonra satış sinyali). Bileşik özsermaye, %0.1 fee/taraf.
    [start, end) aralığında işlem açar (end=None → seri sonu)."""
    o = df["open"].values; c = df["close"].values
    h = df["high"].values; l = df["low"].values
    b = buy.values; s = sell.values
    tf_min = _TF_MIN.get(tf, 60)
    if end is None:
        end = len(df)

    equity = 1.0; peak = 1.0; maxdd = 0.0
    trades = []; durations = []
    pos = None
    for i in range(start, end):
        if pos is not None:
            entry = pos["entry"]; mins = (i - pos["bar"]) * tf_min
            sl_price = entry * (1 + stoploss)
            roi_price = entry * (1 + roi)
            exit_price = None
            if l[i] <= sl_price:                       # 1) stop-loss (intrabar)
                exit_price = sl_price
            elif h[i] >= roi_price:                     # 2) ROI hedefi
                exit_price = roi_price
            elif s[i]:                                  # 3) satış sinyali
                exit_price = c[i]
            if exit_price is not None:
                ret = (exit_price - entry) / entry - 2 * FEE
                equity *= (1 + ret); trades.append(ret); durations.append(mins)
                peak = max(peak, equity); maxdd = max(maxdd, (peak - equity) / peak)
                pos = None
        if pos is None and b[i]:
            pos = {"entry": c[i] * (1 + SLIP), "bar": i}
    if pos is not None:
        ret = (c[end - 1] - pos["entry"]) / pos["entry"] - FEE
        equity *= (1 + ret); trades.append(ret)

    wins = [t for t in trades if t > 0]
    gl = abs(sum(t for t in trades if t <= 0))
    pf = sum(wins) / (gl + 1e-9) if gl > 0 else (99.0 if wins else 0.0)
    return {
        "return": round((equity - 1) * 100, 2),
        "trades": len(trades),
        "win_rate": round(100 * len(wins) / len(trades), 1) if trades else 0.0,
        "profit_factor": round(float(pf), 2),
        "max_drawdown": round(maxdd * 100, 2),
        "avg_hold_min": round(float(np.mean(durations)), 0) if durations else 0,
    }


def run_strategy(name: str, df: pd.DataFrame, start: int = 120) -> Dict:
    cfgs = FREQTRADE_STRATEGIES[name]
    buy, sell = cfgs["fn"](df)
    return simulate_freqtrade(df, buy, sell, cfgs["roi"], cfgs["stop"],
                              cfgs.get("tf", "1h"), start)
