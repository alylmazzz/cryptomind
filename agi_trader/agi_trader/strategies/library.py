"""
Kural-tabanlı strateji kütüphanesi.

Stratejiler bir DataFrame alır ve her bar için hedef pozisyon (-1/0/+1) Series'i döndürür.
`simulate_trailing` bu pozisyon serisini ATR iz-süren (trailing) stop ile işler — sabit
stop yerine trend boyunca stop'u takip ettirmek, choppy piyasada erken stop-out + tekrar
giriş "churn" zararını azaltır (son 48 saat testindeki ana kayıp kaynağı buydu).

Tasarım ilkeleri:
  - Rejim filtresi: trend stratejisi yalnız ADX>eşik (gerçek trend) varken işlem açar.
  - Mean-reversion yalnız düşük-ADX (range) rejiminde çalışır.
  - Tüm fiyat girişleri/çıkışları gerçekçi fee + slippage ile.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..analysis.indicators import ema, rsi, atr, adx, bollinger, donchian

FEE = 0.0004
SLIP = 0.0002


def _adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    a = adx(df, n)
    return a[0] if isinstance(a, tuple) else a


# ---------------------------------------------------------------- stratejiler
def strat_trend(df: pd.DataFrame, adx_min: float = 22.0) -> pd.Series:
    """Trend-takip: EMA20/50 dizilimi + ADX trend filtresi."""
    c = df["close"]
    e_f, e_s = ema(c, 20), ema(c, 50)
    a = _adx(df)
    pos = pd.Series(0.0, index=df.index)
    pos[(e_f > e_s) & (c > e_s) & (a > adx_min)] = 1
    pos[(e_f < e_s) & (c < e_s) & (a > adx_min)] = -1
    return pos


def strat_breakout(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Donchian kırılımı: N-bar tepe/dip kırılınca yönde pozisyon (kırılım kalıcı)."""
    up, _, lo = donchian(df, n)
    up, lo = up.shift(1), lo.shift(1)          # look-ahead yok
    c = df["close"]
    pos = pd.Series(np.nan, index=df.index)
    pos[c > up] = 1
    pos[c < lo] = -1
    return pos.ffill().fillna(0.0)


def strat_meanrev(df: pd.DataFrame, adx_max: float = 20.0) -> pd.Series:
    """Ortalamaya dönüş: yalnız düşük-ADX (range) rejiminde BB+RSI aşırılığı."""
    c = df["close"]
    a = _adx(df)
    r = rsi(c, 14)
    up, mid, lo = bollinger(c, 20)
    pos = pd.Series(0.0, index=df.index)
    rng = a < adx_max
    pos[rng & (c < lo) & (r < 35)] = 1
    pos[rng & (c > up) & (r > 65)] = -1
    return pos


def strat_supertrend(df: pd.DataFrame) -> pd.Series:
    """Supertrend yön takibi."""
    from ..analysis.indicators import supertrend
    st = supertrend(df)
    d = st[1] if isinstance(st, tuple) else st
    return pd.Series(np.asarray(d), index=df.index).fillna(0.0)


def strat_macd(df: pd.DataFrame) -> pd.Series:
    """MACD çizgisi sinyal çizgisini kesince yön."""
    from ..analysis.indicators import macd
    m, sig, _ = macd(df["close"])
    pos = pd.Series(0.0, index=df.index)
    pos[m > sig] = 1; pos[m < sig] = -1
    return pos


def strat_ema_cross(df: pd.DataFrame) -> pd.Series:
    """EMA20/50 altın/ölüm kesişimi."""
    f, s = ema(df["close"], 20), ema(df["close"], 50)
    pos = pd.Series(0.0, index=df.index)
    pos[f > s] = 1; pos[f < s] = -1
    return pos


def strat_rsi_rev(df: pd.DataFrame) -> pd.Series:
    """RSI aşırı alım/satım ortalamaya dönüş."""
    r = rsi(df["close"], 14)
    pos = pd.Series(0.0, index=df.index)
    pos[r < 35] = 1; pos[r > 65] = -1
    return pos


def strat_stoch(df: pd.DataFrame) -> pd.Series:
    """Stochastic RSI aşırılık dönüşü."""
    from ..analysis.indicators import stoch_rsi
    st = stoch_rsi(df["close"])
    pos = pd.Series(0.0, index=df.index)
    pos[st < 20] = 1; pos[st > 80] = -1
    return pos


def strat_bb_breakout(df: pd.DataFrame) -> pd.Series:
    """Bollinger bant kırılımı (momentum, dönüşün tersi)."""
    up, _, lo = bollinger(df["close"], 20)
    c = df["close"]
    pos = pd.Series(np.nan, index=df.index)
    pos[c > up] = 1; pos[c < lo] = -1
    return pos.ffill().fillna(0.0)


def strat_ha_trend(df: pd.DataFrame) -> pd.Series:
    """Heikin-Ashi trend yönü."""
    ha_c = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    o = df["open"].values; cc = ha_c.values
    ha_o = np.empty(len(df)); ha_o[0] = o[0]
    for i in range(1, len(df)):
        ha_o[i] = (ha_o[i - 1] + cc[i - 1]) / 2.0
    pos = pd.Series(0.0, index=df.index)
    pos[ha_c.values > ha_o] = 1; pos[ha_c.values < ha_o] = -1
    return pos


STRATEGIES = {
    "trend": strat_trend, "breakout": strat_breakout, "meanrev": strat_meanrev,
    "supertrend": strat_supertrend, "macd": strat_macd, "ema_cross": strat_ema_cross,
    "rsi_rev": strat_rsi_rev, "stoch": strat_stoch, "bb_breakout": strat_bb_breakout,
    "ha_trend": strat_ha_trend,
}


# ---------------------------------------------------------------- simülatör
def simulate_trailing(df: pd.DataFrame, pos: pd.Series, lo: int, hi: int,
                      atr_mult: float = 2.0, trail_mult: float = 2.5) -> Dict:
    """[lo, hi) bar aralığında pozisyon serisini ATR iz-süren stop ile işle.
    Çıkış: stop tetiği VEYA hedef pozisyon ters/sıfır olunca. Aralık sonunda kapanır."""
    C = df["close"].values
    H = df["high"].values
    L = df["low"].values
    A = atr(df, 14).values
    p = pos.values

    equity = 1.0
    peak = 1.0
    maxdd = 0.0
    trades: List[float] = []
    cur = 0
    entry = 0.0
    stop = 0.0

    def _close(ex_price: float, fee_legs: int = 2):
        nonlocal equity, cur, peak, maxdd
        ret = cur * (ex_price - entry) / entry - fee_legs * FEE
        equity *= (1 + ret)
        trades.append(ret)
        peak = max(peak, equity)
        maxdd = max(maxdd, (peak - equity) / peak)

    for i in range(lo, hi):
        if cur != 0:
            if cur == 1:
                stop = max(stop, C[i] - trail_mult * A[i])
            else:
                stop = min(stop, C[i] + trail_mult * A[i])
            hit = (L[i] <= stop) if cur == 1 else (H[i] >= stop)
            flip = (p[i] == -cur) or (p[i] == 0)
            if hit:
                _close(stop)
                cur = 0
            elif flip:
                _close(C[i])
                cur = 0
        if cur == 0 and p[i] != 0 and A[i] > 0:
            cur = int(p[i])
            entry = C[i] * (1 + cur * SLIP)
            stop = entry - cur * atr_mult * A[i]
    if cur != 0:
        _close(C[hi - 1], fee_legs=1)

    wins = [t for t in trades if t > 0]
    gl = abs(sum(t for t in trades if t <= 0))
    pf = sum(wins) / (gl + 1e-9) if gl > 0 else (99.0 if wins else 0.0)
    return {
        "return": round((equity - 1) * 100, 2),
        "trades": len(trades),
        "win_rate": round(100 * len(wins) / len(trades), 1) if trades else 0.0,
        "profit_factor": round(float(pf), 2),
        "max_drawdown": round(maxdd * 100, 2),
        "trade_returns": [round(float(t), 5) for t in trades],
    }


# ---------------------------------------------------------------- tarayıcı
def scan_opportunities(data: Dict[str, pd.DataFrame], window_bars: int,
                       atr_mult: float = 2.0, trail_mult: float = 2.5) -> List[Dict]:
    """Her (parite, strateji) için son `window_bars` penceresinde performans.
    `data` = {sembol: df}. Sonuçları getiriye göre sıralı döndürür."""
    out = []
    for sym, df in data.items():
        n = len(df)
        if n < window_bars + 60:
            continue
        for sname, sfn in STRATEGIES.items():
            try:
                pos = sfn(df)
                res = simulate_trailing(df, pos, n - window_bars, n, atr_mult, trail_mult)
                out.append({"symbol": sym, "strategy": sname, **res})
            except Exception as e:
                out.append({"symbol": sym, "strategy": sname, "error": str(e)})
    out.sort(key=lambda r: r.get("return", -1e9), reverse=True)
    return out
