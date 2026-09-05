"""
120+ Teknik İndikatör Motoru (Teknik Analiz Uzmanı rolü).

Hiçbir dış TA kütüphanesi gerektirmez — saf numpy/pandas. pandas-ta veya TA-Lib
kuruluysa bile bu modül bağımsız çalışır. `compute_all_indicators` her göstergenin
SON değerini döndürür; `technical_vote` bunları -1..+1 boğa/ayı skoruna ve
açıklamalı gerekçelere dönüştürür (Explainable AI).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ..core.models import LayerVote


# ----------------------------------------------------------------------------
# Temel yardımcılar
# ----------------------------------------------------------------------------
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=1).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=1).mean()


def wma(s: pd.Series, n: int) -> pd.Series:
    weights = np.arange(1, n + 1)
    return s.rolling(n).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def hma(s: pd.Series, n: int) -> pd.Series:
    half = max(int(n / 2), 1)
    sqrt_n = max(int(np.sqrt(n)), 1)
    return wma(2 * wma(s, half) - wma(s, n), sqrt_n)


def true_range(df: pd.DataFrame) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    return pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=1).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / n, adjust=False, min_periods=1).mean()
    roll_down = down.ewm(alpha=1 / n, adjust=False, min_periods=1).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))


def stoch(df: pd.DataFrame, n: int = 14, d: int = 3):
    low_n = df["low"].rolling(n, min_periods=1).min()
    high_n = df["high"].rolling(n, min_periods=1).max()
    k = 100 * (df["close"] - low_n) / (high_n - low_n + 1e-12)
    return k, k.rolling(d, min_periods=1).mean()


def macd(s: pd.Series, fast=12, slow=26, signal=9):
    line = ema(s, fast) - ema(s, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def bollinger(s: pd.Series, n=20, k=2.0):
    mid = sma(s, n)
    std = s.rolling(n, min_periods=1).std()
    return mid + k * std, mid, mid - k * std


def keltner(df: pd.DataFrame, n=20, k=2.0):
    mid = ema(df["close"], n)
    rng = atr(df, n)
    return mid + k * rng, mid, mid - k * rng


def donchian(df: pd.DataFrame, n=20):
    upper = df["high"].rolling(n, min_periods=1).max()
    lower = df["low"].rolling(n, min_periods=1).min()
    return upper, (upper + lower) / 2, lower


def adx(df: pd.DataFrame, n=14):
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = true_range(df)
    atr_n = tr.ewm(alpha=1 / n, adjust=False, min_periods=1).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / (atr_n + 1e-12)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / (atr_n + 1e-12)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    return dx.ewm(alpha=1 / n, adjust=False).mean(), plus_di, minus_di


def cci(df: pd.DataFrame, n=20):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = sma(tp, n)
    md = (tp - ma).abs().rolling(n, min_periods=1).mean()
    return (tp - ma) / (0.015 * md + 1e-12)


def mfi(df: pd.DataFrame, n=14):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    rmf = tp * df["volume"]
    pos = rmf.where(tp > tp.shift(1), 0.0).rolling(n, min_periods=1).sum()
    neg = rmf.where(tp < tp.shift(1), 0.0).rolling(n, min_periods=1).sum()
    return 100 - 100 / (1 + pos / (neg + 1e-12))


def obv(df: pd.DataFrame) -> pd.Series:
    sign = np.sign(df["close"].diff().fillna(0))
    return (sign * df["volume"]).cumsum()


def cmf(df: pd.DataFrame, n=20):
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"] + 1e-12)
    mfv = mfm * df["volume"]
    return mfv.rolling(n, min_periods=1).sum() / (df["volume"].rolling(n, min_periods=1).sum() + 1e-12)


def vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return (tp * df["volume"]).cumsum() / (df["volume"].cumsum() + 1e-12)


def williams_r(df: pd.DataFrame, n=14):
    hh = df["high"].rolling(n, min_periods=1).max()
    ll = df["low"].rolling(n, min_periods=1).min()
    return -100 * (hh - df["close"]) / (hh - ll + 1e-12)


def roc(s: pd.Series, n=12):
    return 100 * (s - s.shift(n)) / (s.shift(n) + 1e-12)


def momentum(s: pd.Series, n=10):
    return s - s.shift(n)


def supertrend(df: pd.DataFrame, n=10, mult=3.0):
    hl2 = (df["high"] + df["low"]) / 2
    a = atr(df, n)
    upper = hl2 + mult * a
    lower = hl2 - mult * a
    st = pd.Series(index=df.index, dtype=float)
    dir_ = pd.Series(index=df.index, dtype=float)
    st.iloc[0] = upper.iloc[0]
    dir_.iloc[0] = -1
    for i in range(1, len(df)):
        if df["close"].iloc[i] > st.iloc[i - 1]:
            dir_.iloc[i] = 1
        elif df["close"].iloc[i] < st.iloc[i - 1]:
            dir_.iloc[i] = -1
        else:
            dir_.iloc[i] = dir_.iloc[i - 1]
        if dir_.iloc[i] == 1:
            st.iloc[i] = max(lower.iloc[i], st.iloc[i - 1]) if dir_.iloc[i - 1] == 1 else lower.iloc[i]
        else:
            st.iloc[i] = min(upper.iloc[i], st.iloc[i - 1]) if dir_.iloc[i - 1] == -1 else upper.iloc[i]
    return st, dir_


def psar(df: pd.DataFrame, af_step=0.02, af_max=0.2):
    high, low = df["high"].values, df["low"].values
    n = len(df)
    sar = np.zeros(n)
    trend = 1
    af = af_step
    ep = high[0]
    sar[0] = low[0]
    for i in range(1, n):
        sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
        if trend == 1:
            if low[i] < sar[i]:
                trend = -1
                sar[i] = ep
                ep = low[i]
                af = af_step
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:
            if high[i] > sar[i]:
                trend = 1
                sar[i] = ep
                ep = high[i]
                af = af_step
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)
    return pd.Series(sar, index=df.index)


def ichimoku(df: pd.DataFrame):
    high, low = df["high"], df["low"]
    tenkan = (high.rolling(9, min_periods=1).max() + low.rolling(9, min_periods=1).min()) / 2
    kijun = (high.rolling(26, min_periods=1).max() + low.rolling(26, min_periods=1).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((high.rolling(52, min_periods=1).max() + low.rolling(52, min_periods=1).min()) / 2).shift(26)
    return tenkan, kijun, span_a, span_b


def aroon(df: pd.DataFrame, n=25):
    up = df["high"].rolling(n + 1, min_periods=1).apply(lambda x: 100 * (np.argmax(x) / n), raw=True)
    dn = df["low"].rolling(n + 1, min_periods=1).apply(lambda x: 100 * (np.argmin(x) / n), raw=True)
    return up, dn


def trix(s: pd.Series, n=15):
    e = ema(ema(ema(s, n), n), n)
    return 100 * e.diff() / (e.shift(1) + 1e-12)


def tsi(s: pd.Series, long=25, short=13):
    m = s.diff()
    abs_m = m.abs()
    ds = ema(ema(m, long), short)
    das = ema(ema(abs_m, long), short)
    return 100 * ds / (das + 1e-12)


def ultimate_osc(df: pd.DataFrame):
    bp = df["close"] - pd.concat([df["low"], df["close"].shift(1)], axis=1).min(axis=1)
    tr = true_range(df)
    avg = lambda n: bp.rolling(n, min_periods=1).sum() / (tr.rolling(n, min_periods=1).sum() + 1e-12)
    return 100 * (4 * avg(7) + 2 * avg(14) + avg(28)) / 7


def vortex(df: pd.DataFrame, n=14):
    tr = true_range(df)
    vmp = (df["high"] - df["low"].shift(1)).abs()
    vmm = (df["low"] - df["high"].shift(1)).abs()
    vip = vmp.rolling(n, min_periods=1).sum() / (tr.rolling(n, min_periods=1).sum() + 1e-12)
    vim = vmm.rolling(n, min_periods=1).sum() / (tr.rolling(n, min_periods=1).sum() + 1e-12)
    return vip, vim


def chaikin_osc(df: pd.DataFrame):
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"] + 1e-12)
    adl = (mfm * df["volume"]).cumsum()
    return ema(adl, 3) - ema(adl, 10)


def force_index(df: pd.DataFrame, n=13):
    return ema(df["close"].diff() * df["volume"], n)


def kst(s: pd.Series):
    r1, r2, r3, r4 = roc(s, 10), roc(s, 15), roc(s, 20), roc(s, 30)
    k = sma(r1, 10) + 2 * sma(r2, 10) + 3 * sma(r3, 10) + 4 * sma(r4, 15)
    return k, sma(k, 9)


def stoch_rsi(s: pd.Series, n=14):
    r = rsi(s, n)
    ll = r.rolling(n, min_periods=1).min()
    hh = r.rolling(n, min_periods=1).max()
    return 100 * (r - ll) / (hh - ll + 1e-12)


def awesome_osc(df: pd.DataFrame):
    median = (df["high"] + df["low"]) / 2
    return sma(median, 5) - sma(median, 34)


def dpo(s: pd.Series, n=20):
    return s.shift(int(n / 2) + 1) - sma(s, n)


def elder_ray(df: pd.DataFrame, n=13):
    e = ema(df["close"], n)
    return df["high"] - e, df["low"] - e   # bull power, bear power


def cmo(s: pd.Series, n=14):
    d = s.diff()
    up = d.clip(lower=0).rolling(n, min_periods=1).sum()
    dn = (-d.clip(upper=0)).rolling(n, min_periods=1).sum()
    return 100 * (up - dn) / (up + dn + 1e-12)


def bop(df: pd.DataFrame):
    return (df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-12)


def eom(df: pd.DataFrame, n=14):
    dm = ((df["high"] + df["low"]) / 2).diff()
    box = df["volume"] / (df["high"] - df["low"] + 1e-12)
    return (dm / (box + 1e-12)).rolling(n, min_periods=1).mean()


def vpt(df: pd.DataFrame):
    return (df["volume"] * df["close"].pct_change().fillna(0)).cumsum()


def nvi_pvi(df: pd.DataFrame):
    chg = df["close"].pct_change().fillna(0)
    vol_chg = df["volume"].diff().fillna(0)
    nvi = pd.Series(1000.0, index=df.index)
    pvi = pd.Series(1000.0, index=df.index)
    for i in range(1, len(df)):
        if vol_chg.iloc[i] < 0:
            nvi.iloc[i] = nvi.iloc[i - 1] * (1 + chg.iloc[i])
            pvi.iloc[i] = pvi.iloc[i - 1]
        else:
            pvi.iloc[i] = pvi.iloc[i - 1] * (1 + chg.iloc[i])
            nvi.iloc[i] = nvi.iloc[i - 1]
    return nvi, pvi


def mass_index(df: pd.DataFrame, n=9, sum_n=25):
    rng = df["high"] - df["low"]
    e1 = ema(rng, n)
    e2 = ema(e1, n)
    return (e1 / (e2 + 1e-12)).rolling(sum_n, min_periods=1).sum()


def coppock(s: pd.Series):
    return wma(roc(s, 14) + roc(s, 11), 10)


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Heikin Ashi mumları — gürültü filtrelenmiş fiyat serisi."""
    ha = pd.DataFrame(index=df.index)
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    # HA open'ı numpy dizisinde hesapla (chained-assignment + copy-on-write hatasından kaçın)
    close_vals = ha_close.values
    open_src = df["open"].values
    ha_open = np.empty(len(df), dtype=float)
    if len(df):
        ha_open[0] = open_src[0]
        for i in range(1, len(df)):
            ha_open[i] = (ha_open[i - 1] + close_vals[i - 1]) / 2
    ha["close"] = ha_close
    ha["open"] = pd.Series(ha_open, index=df.index)
    ha["high"] = pd.concat([df["high"], ha["open"], ha["close"]], axis=1).max(axis=1)
    ha["low"] = pd.concat([df["low"], ha["open"], ha["close"]], axis=1).min(axis=1)
    return ha


def heikin_ashi_signal(df: pd.DataFrame) -> float:
    """Heikin Ashi trend sinyali: +1 boğa, -1 ayı, 0 kararsız."""
    ha = heikin_ashi(df)
    last = ha.iloc[-1]
    prev = ha.iloc[-2]
    # Yeşil mum (close > open) ve alt gölge yok → güçlü boğa
    if last["close"] > last["open"] and last["low"] == last["open"]:
        return 1.0
    # Kırmızı mum ve üst gölge yok → güçlü ayı
    if last["close"] < last["open"] and last["high"] == last["open"]:
        return -1.0
    # Boğa trendi: close > open
    if last["close"] > last["open"]:
        return 0.5
    if last["close"] < last["open"]:
        return -0.5
    return 0.0


def fisher_transform(df: pd.DataFrame, n=9):
    med = (df["high"] + df["low"]) / 2
    ll = med.rolling(n, min_periods=1).min()
    hh = med.rolling(n, min_periods=1).max()
    val = 2 * ((med - ll) / (hh - ll + 1e-12) - 0.5)
    val = val.clip(-0.999, 0.999)
    return 0.5 * np.log((1 + val) / (1 - val))


def schaff_trend_cycle(s: pd.Series, fast=23, slow=50, cycle=10):
    m = ema(s, fast) - ema(s, slow)
    ll = m.rolling(cycle, min_periods=1).min()
    hh = m.rolling(cycle, min_periods=1).max()
    k = 100 * (m - ll) / (hh - ll + 1e-12)
    d = k.ewm(span=3, adjust=False).mean()
    ll2 = d.rolling(cycle, min_periods=1).min()
    hh2 = d.rolling(cycle, min_periods=1).max()
    return 100 * (d - ll2) / (hh2 - ll2 + 1e-12)


def rvi(df: pd.DataFrame, n=10):
    num = (df["close"] - df["open"])
    den = (df["high"] - df["low"])
    return num.rolling(n, min_periods=1).mean() / (den.rolling(n, min_periods=1).mean() + 1e-12)


def qstick(df: pd.DataFrame, n=10):
    return sma(df["close"] - df["open"], n)


def accel_osc(df: pd.DataFrame):
    ao = awesome_osc(df)
    return ao - sma(ao, 5)


def pivot_points(df: pd.DataFrame):
    """Standart, Fibonacci ve Camarilla pivotları (önceki bara göre)."""
    h, l, c = df["high"].iloc[-2], df["low"].iloc[-2], df["close"].iloc[-2]
    p = (h + l + c) / 3
    rng = h - l
    return {
        "pivot": p,
        "r1": 2 * p - l, "s1": 2 * p - h,
        "r2": p + rng, "s2": p - rng,
        "r3": h + 2 * (p - l), "s3": l - 2 * (h - p),
        "fib_r1": p + 0.382 * rng, "fib_s1": p - 0.382 * rng,
        "fib_r2": p + 0.618 * rng, "fib_s2": p - 0.618 * rng,
        "cam_r3": c + rng * 1.1 / 4, "cam_s3": c - rng * 1.1 / 4,
    }


def fib_levels(df: pd.DataFrame, lookback=100):
    """Son swing aralığının Fibonacci retracement seviyeleri."""
    seg = df.iloc[-lookback:]
    hi, lo = float(seg["high"].max()), float(seg["low"].min())
    diff = hi - lo
    return {
        "fib_0": hi,
        "fib_236": hi - 0.236 * diff,
        "fib_382": hi - 0.382 * diff,
        "fib_500": hi - 0.500 * diff,
        "fib_618": hi - 0.618 * diff,
        "fib_786": hi - 0.786 * diff,
        "fib_1000": lo,
    }


# ----------------------------------------------------------------------------
# Tüm göstergeleri hesapla (son değerler)
# ----------------------------------------------------------------------------
INDICATOR_NAMES: List[str] = []  # compute_all_indicators ilk çağrıda doldurur


def compute_all_indicators(df: pd.DataFrame) -> Dict[str, float]:
    """120+ göstergenin son barındaki değerlerini döndürür."""
    out: Dict[str, float] = {}
    c = df["close"]

    # --- Hareketli ortalamalar (çok periyot) -> ~30 gösterge
    for n in (5, 8, 9, 10, 13, 20, 21, 34, 50, 55, 100, 144, 200):
        out[f"ema_{n}"] = float(ema(c, n).iloc[-1])
        out[f"sma_{n}"] = float(sma(c, n).iloc[-1])
    out["wma_20"] = float(wma(c, 20).iloc[-1])
    out["hma_20"] = float(hma(c, 20).iloc[-1])
    out["vwap"] = float(vwap(df).iloc[-1])

    # --- Momentum osilatörleri
    for n in (7, 14, 21):
        out[f"rsi_{n}"] = float(rsi(c, n).iloc[-1])
    k, d = stoch(df)
    out["stoch_k"], out["stoch_d"] = float(k.iloc[-1]), float(d.iloc[-1])
    out["stoch_rsi"] = float(stoch_rsi(c).iloc[-1])
    out["cci_20"] = float(cci(df).iloc[-1])
    out["mfi_14"] = float(mfi(df).iloc[-1])
    out["williams_r"] = float(williams_r(df).iloc[-1])
    out["roc_12"] = float(roc(c, 12).iloc[-1])
    out["momentum_10"] = float(momentum(c, 10).iloc[-1])
    out["trix_15"] = float(trix(c).iloc[-1])
    out["tsi"] = float(tsi(c).iloc[-1])
    out["ultimate_osc"] = float(ultimate_osc(df).iloc[-1])
    out["awesome_osc"] = float(awesome_osc(df).iloc[-1])
    out["dpo_20"] = float(dpo(c).iloc[-1])
    kst_line, kst_sig = kst(c)
    out["kst"], out["kst_signal"] = float(kst_line.iloc[-1]), float(kst_sig.iloc[-1])

    # --- MACD aileleri
    for (f, s_, sig) in ((12, 26, 9), (5, 35, 5)):
        line, signal, hist = macd(c, f, s_, sig)
        out[f"macd_{f}_{s_}"] = float(line.iloc[-1])
        out[f"macd_signal_{f}_{s_}"] = float(signal.iloc[-1])
        out[f"macd_hist_{f}_{s_}"] = float(hist.iloc[-1])

    # --- Trend gücü
    adx_v, pdi, mdi = adx(df)
    out["adx_14"] = float(adx_v.iloc[-1])
    out["plus_di"], out["minus_di"] = float(pdi.iloc[-1]), float(mdi.iloc[-1])
    ar_up, ar_dn = aroon(df)
    out["aroon_up"], out["aroon_down"] = float(ar_up.iloc[-1]), float(ar_dn.iloc[-1])
    vip, vim = vortex(df)
    out["vortex_plus"], out["vortex_minus"] = float(vip.iloc[-1]), float(vim.iloc[-1])
    st, st_dir = supertrend(df)
    out["supertrend"], out["supertrend_dir"] = float(st.iloc[-1]), float(st_dir.iloc[-1])
    out["psar"] = float(psar(df).iloc[-1])

    # --- Volatilite / bantlar
    for n in (14, 21):
        out[f"atr_{n}"] = float(atr(df, n).iloc[-1])
    bb_u, bb_m, bb_l = bollinger(c)
    out["bb_upper"], out["bb_mid"], out["bb_lower"] = float(bb_u.iloc[-1]), float(bb_m.iloc[-1]), float(bb_l.iloc[-1])
    out["bb_width"] = float((bb_u.iloc[-1] - bb_l.iloc[-1]) / (bb_m.iloc[-1] + 1e-12))
    out["bb_pct_b"] = float((c.iloc[-1] - bb_l.iloc[-1]) / (bb_u.iloc[-1] - bb_l.iloc[-1] + 1e-12))
    kc_u, kc_m, kc_l = keltner(df)
    out["kc_upper"], out["kc_mid"], out["kc_lower"] = float(kc_u.iloc[-1]), float(kc_m.iloc[-1]), float(kc_l.iloc[-1])
    dc_u, dc_m, dc_l = donchian(df)
    out["dc_upper"], out["dc_mid"], out["dc_lower"] = float(dc_u.iloc[-1]), float(dc_m.iloc[-1]), float(dc_l.iloc[-1])
    # squeeze (BB Keltner içinde mi)
    out["squeeze_on"] = float(1.0 if (bb_u.iloc[-1] < kc_u.iloc[-1] and bb_l.iloc[-1] > kc_l.iloc[-1]) else 0.0)

    # --- Hacim
    out["obv"] = float(obv(df).iloc[-1])
    out["cmf_20"] = float(cmf(df).iloc[-1])
    out["chaikin_osc"] = float(chaikin_osc(df).iloc[-1])
    out["force_index_13"] = float(force_index(df).iloc[-1])
    out["volume"] = float(df["volume"].iloc[-1])
    out["volume_sma_20"] = float(sma(df["volume"], 20).iloc[-1])
    out["rel_volume"] = float(df["volume"].iloc[-1] / (sma(df["volume"], 20).iloc[-1] + 1e-12))

    # --- Ichimoku
    tenkan, kijun, span_a, span_b = ichimoku(df)
    out["ichimoku_tenkan"] = float(tenkan.iloc[-1])
    out["ichimoku_kijun"] = float(kijun.iloc[-1])
    out["ichimoku_span_a"] = float(span_a.iloc[-1]) if not np.isnan(span_a.iloc[-1]) else out["ichimoku_tenkan"]
    out["ichimoku_span_b"] = float(span_b.iloc[-1]) if not np.isnan(span_b.iloc[-1]) else out["ichimoku_kijun"]

    # --- Elder ray
    bull, bear = elder_ray(df)
    out["elder_bull"], out["elder_bear"] = float(bull.iloc[-1]), float(bear.iloc[-1])

    # --- Ek osilatörler / hacim göstergeleri
    out["cmo_14"] = float(cmo(c).iloc[-1])
    out["bop"] = float(bop(df).iloc[-1])
    out["eom_14"] = float(eom(df).iloc[-1])
    out["vpt"] = float(vpt(df).iloc[-1])
    nvi, pvi = nvi_pvi(df)
    out["nvi"], out["pvi"] = float(nvi.iloc[-1]), float(pvi.iloc[-1])
    out["mass_index"] = float(mass_index(df).iloc[-1])
    out["coppock"] = float(coppock(c).iloc[-1])
    out["fisher"] = float(fisher_transform(df).iloc[-1])
    out["stc"] = float(schaff_trend_cycle(c).iloc[-1])
    out["rvi"] = float(rvi(df).iloc[-1])
    out["qstick_10"] = float(qstick(df).iloc[-1])
    out["accel_osc"] = float(accel_osc(df).iloc[-1])
    out["heikin_ashi"] = float(heikin_ashi_signal(df))

    # --- Pivot noktaları (standart + Fibonacci + Camarilla)
    for k, v in pivot_points(df).items():
        out[f"pp_{k}"] = float(v)

    # --- Fibonacci retracement seviyeleri
    for k, v in fib_levels(df).items():
        out[k] = float(v)

    # --- Fiyat & getiri istatistikleri
    out["close"] = float(c.iloc[-1])
    out["return_1"] = float(c.pct_change().iloc[-1])
    out["return_5"] = float(c.pct_change(5).iloc[-1])
    out["realized_vol_20"] = float(c.pct_change().rolling(20).std().iloc[-1])

    # --- Genişletilmiş set (+50 indikatör): gelişmiş MA / momentum / vol / hacim / yapısal
    try:
        from .indicators_ext import extra_indicators
        out.update(extra_indicators(df))
    except Exception:
        pass

    # NaN temizliği
    cleaned = {k: (0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else v) for k, v in out.items()}

    global INDICATOR_NAMES
    if not INDICATOR_NAMES:
        INDICATOR_NAMES = list(cleaned.keys())
    return cleaned


# ----------------------------------------------------------------------------
# Teknik oy (boğa/ayı skoru + gerekçe)
# ----------------------------------------------------------------------------
def technical_vote(df: pd.DataFrame, ind: Dict[str, float]) -> LayerVote:
    price = ind["close"]
    signals: List[float] = []
    reasons: List[str] = []

    def add(cond_bull: bool, cond_bear: bool, text_bull: str, text_bear: str, w: float = 1.0):
        if cond_bull:
            signals.append(w)
            reasons.append(f"↑ {text_bull}")
        elif cond_bear:
            signals.append(-w)
            reasons.append(f"↓ {text_bear}")

    # Trend: EMA dizilimi
    add(price > ind["ema_50"] > ind["ema_200"], price < ind["ema_50"] < ind["ema_200"],
        "Fiyat EMA50>EMA200 üzerinde (yükseliş trendi)", "Fiyat EMA50<EMA200 altında (düşüş trendi)", 1.5)
    add(ind["ema_21"] > ind["ema_55"], ind["ema_21"] < ind["ema_55"],
        "EMA21>EMA55 (kısa vade boğa)", "EMA21<EMA55 (kısa vade ayı)")

    # RSI
    r = ind["rsi_14"]
    if r < 30:
        signals.append(0.8); reasons.append(f"↑ RSI aşırı satım ({r:.0f})")
    elif r > 70:
        signals.append(-0.8); reasons.append(f"↓ RSI aşırı alım ({r:.0f})")
    else:
        add(r > 55, r < 45, f"RSI boğa bölgesi ({r:.0f})", f"RSI ayı bölgesi ({r:.0f})", 0.5)

    # MACD
    add(ind["macd_hist_12_26"] > 0, ind["macd_hist_12_26"] < 0,
        "MACD histogram pozitif", "MACD histogram negatif")

    # ADX yön gücü
    if ind["adx_14"] > 25:
        add(ind["plus_di"] > ind["minus_di"], ind["minus_di"] > ind["plus_di"],
            f"Güçlü trend +DI>-DI (ADX {ind['adx_14']:.0f})", f"Güçlü trend -DI>+DI (ADX {ind['adx_14']:.0f})", 1.2)

    # Supertrend
    add(ind["supertrend_dir"] > 0, ind["supertrend_dir"] < 0, "Supertrend yukarı", "Supertrend aşağı", 1.2)

    # PSAR
    add(price > ind["psar"], price < ind["psar"], "Fiyat PSAR üzerinde", "Fiyat PSAR altında", 0.6)

    # Bollinger %B
    if ind["bb_pct_b"] < 0.05:
        signals.append(0.6); reasons.append("↑ Bollinger alt bandı (aşırı satım)")
    elif ind["bb_pct_b"] > 0.95:
        signals.append(-0.6); reasons.append("↓ Bollinger üst bandı (aşırı alım)")

    # Ichimoku bulutu
    cloud_top = max(ind["ichimoku_span_a"], ind["ichimoku_span_b"])
    cloud_bot = min(ind["ichimoku_span_a"], ind["ichimoku_span_b"])
    add(price > cloud_top, price < cloud_bot, "Fiyat Ichimoku bulutu üstünde", "Fiyat Ichimoku bulutu altında", 1.0)

    # Hacim onayı
    add(ind["cmf_20"] > 0.05, ind["cmf_20"] < -0.05, "CMF pozitif (alım baskısı)", "CMF negatif (satım baskısı)", 0.6)
    if ind["rel_volume"] > 1.5:
        reasons.append(f"• Yüksek bağıl hacim ({ind['rel_volume']:.1f}x)")

    # Stochastic
    add(ind["stoch_k"] < 20, ind["stoch_k"] > 80, "Stochastic aşırı satım", "Stochastic aşırı alım", 0.5)

    # MFI
    add(ind["mfi_14"] < 20, ind["mfi_14"] > 80, "MFI aşırı satım", "MFI aşırı alım", 0.5)

    # VWAP
    add(price > ind["vwap"], price < ind["vwap"], "Fiyat VWAP üzerinde", "Fiyat VWAP altında", 0.6)

    # Heikin Ashi
    ha = ind.get("heikin_ashi", 0)
    if ha > 0.7:
        signals.append(1.0); reasons.append("↑ Heikin Ashi güçlü boğa (alt gölgesiz yeşil)")
    elif ha < -0.7:
        signals.append(-1.0); reasons.append("↓ Heikin Ashi güçlü ayı (üst gölgesiz kırmızı)")
    elif ha > 0:
        signals.append(0.4); reasons.append("↑ Heikin Ashi boğa trendi")
    elif ha < 0:
        signals.append(-0.4); reasons.append("↓ Heikin Ashi ayı trendi")

    # Skor: ağırlıklı toplam normalize
    if signals:
        raw = sum(signals)
        max_possible = sum(abs(s) for s in signals)
        score = raw / (max_possible + 1e-12)
    else:
        score = 0.0

    # Güven: kaç gösterge aynı yönde hizalanmış
    agree = sum(1 for s in signals if (s > 0) == (score > 0))
    confidence = min(1.0, 0.4 + 0.06 * agree)

    return LayerVote(
        name="technical",
        score=float(np.clip(score, -1, 1)),
        confidence=float(confidence),
        reasons=reasons[:12],
        detail={"indicator_count": len(INDICATOR_NAMES), "signals_used": len(signals)},
    )
