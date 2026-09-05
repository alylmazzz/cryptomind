"""
Genişletilmiş İndikatör Seti (+50) — compute_all_indicators'a eklenir.

Mevcut ~60 göstergenin üzerine 50 yeni skalar özellik: gelişmiş hareketli
ortalamalar, momentum/osilatör türevleri, volatilite ölçüleri, hacim akışı ve
yapısal (fiyat-konumu) göstergeler. Hepsi son-bar skaları döndürür; look-ahead yok.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .indicators import ema, sma, rsi, atr, true_range, macd, bollinger, adx


def _last(x) -> float:
    try:
        v = float(x.iloc[-1]) if hasattr(x, "iloc") else float(x)
        return v if np.isfinite(v) else 0.0
    except Exception:
        return 0.0


def extra_indicators(df: pd.DataFrame) -> Dict[str, float]:
    c = df["close"]; h = df["high"]; l = df["low"]; v = df["volume"]
    o = df["open"]
    out: Dict[str, float] = {}
    px = float(c.iloc[-1]) if len(c) else 0.0

    # ---- 1-10: Gelişmiş hareketli ortalamalar + fiyat mesafesi ----
    def dist(ma): return (px / (_last(ma) + 1e-12) - 1) * 100
    out["kama_dist"] = dist(_kama(c))
    out["zlema_dist"] = dist(_zlema(c, 20))
    out["tema_dist"] = dist(_tema(c, 20))
    out["dema_dist"] = dist(_dema(c, 20))
    out["t3_dist"] = dist(_t3(c, 10))
    out["vidya_dist"] = dist(_vidya(c, 14))
    out["mcginley_dist"] = dist(_mcginley(c, 14))
    out["trima_dist"] = dist(c.rolling(20).mean().rolling(20).mean())
    out["guppy_spread"] = (_last(ema(c, 3)) - _last(ema(c, 30))) / (px + 1e-12) * 100
    out["ma_ribbon_align"] = _ribbon_align(c)

    # ---- 11-22: Momentum / osilatör türevleri ----
    out["rsi_2"] = _last(rsi(c, 2))
    out["rsi_50_dist"] = _last(rsi(c, 14)) - 50
    out["connors_rsi"] = _connors_rsi(c)
    out["ppo"] = _ppo(c)
    out["ppo_hist"] = _ppo(c, hist=True)
    out["rmi_14"] = _rmi(c, 14, 5)
    for n in (5, 10, 20, 50):
        out[f"roc_{n}"] = float(c.pct_change(n).iloc[-1] * 100) if len(c) > n else 0.0
    out["momentum_pct_10"] = float((c.iloc[-1] / c.iloc[-11] - 1) * 100) if len(c) > 11 else 0.0
    out["accel_10"] = _last(c.pct_change(10) - c.pct_change(10).shift(5)) * 100

    # ---- 23-34: Volatilite ----
    tr = true_range(df)
    out["natr_14"] = _last(atr(df, 14)) / (px + 1e-12) * 100
    out["hist_vol_20"] = float(c.pct_change().rolling(20).std().iloc[-1] * np.sqrt(365) * 100) if len(c) > 20 else 0.0
    out["hist_vol_50"] = float(c.pct_change().rolling(50).std().iloc[-1] * np.sqrt(365) * 100) if len(c) > 50 else 0.0
    out["vol_ratio_20_50"] = out["hist_vol_20"] / (out["hist_vol_50"] + 1e-9)
    out["chaikin_vol"] = _chaikin_vol(df)
    out["ulcer_index_14"] = _ulcer(c, 14)
    bb_u, bb_m, bb_l = bollinger(c)
    out["bb_bandwidth"] = _last((bb_u - bb_l) / (bb_m + 1e-12)) * 100
    out["true_range_pct"] = _last(tr) / (px + 1e-12) * 100
    out["gap_pct"] = float((o.iloc[-1] / c.iloc[-2] - 1) * 100) if len(c) > 1 else 0.0
    out["intraday_range_pct"] = float((h.iloc[-1] - l.iloc[-1]) / (c.iloc[-1] + 1e-12) * 100) if len(c) else 0.0
    out["atr_expansion"] = _last(atr(df, 5)) / (_last(atr(df, 20)) + 1e-12)
    out["vol_of_vol"] = float(c.pct_change().rolling(10).std().rolling(20).std().iloc[-1]) if len(c) > 30 else 0.0

    # ---- 35-44: Hacim akışı ----
    out["pvo"] = _ppo(v.astype(float))
    out["vroc_14"] = float(v.pct_change(14).iloc[-1] * 100) if len(v) > 14 else 0.0
    out["volume_zscore"] = _zscore(v, 20)
    out["adl_slope"] = _slope(_adl(df), 10)
    out["klinger"] = _klinger(df)
    out["twiggs_mf"] = _twiggs_mf(df, 21)
    out["up_down_vol_ratio"] = _up_down_vol(df, 20)
    out["vwma_dist"] = dist((c * v).rolling(20).sum() / (v.rolling(20).sum() + 1e-12))
    out["clv"] = _last(((c - l) - (h - c)) / ((h - l) + 1e-12))
    out["volume_trend"] = _slope(v.rolling(5).mean(), 10) / (float(v.rolling(20).mean().iloc[-1]) + 1e-9)

    # ---- 45-52: Yapısal / rejim ----
    out["choppiness_14"] = _choppiness(df, 14)
    aroon_up = 100 * (h.rolling(25).apply(lambda x: x.argmax(), raw=True)) / 25
    out["aroon_osc"] = _last(aroon_up) - (100 * _last(l.rolling(25).apply(lambda x: x.argmin(), raw=True)) / 25)
    out["pct_from_high_252"] = float((c.iloc[-1] / c.rolling(min(252, len(c))).max().iloc[-1] - 1) * 100)
    out["pct_from_low_252"] = float((c.iloc[-1] / c.rolling(min(252, len(c))).min().iloc[-1] - 1) * 100)
    out["price_zscore_50"] = _zscore(c, 50)
    out["donchian_pos_20"] = _donchian_pos(df, 20)
    out["adx_slope"] = _slope(adx(df)[0], 5)
    out["ttm_squeeze_mom"] = _squeeze_mom(df)
    return out


# --------------------------------------------------------------------------- MA'lar
def _kama(c, n=10, f=2, s=30):
    ch = c.diff(n).abs()
    vol = c.diff().abs().rolling(n).sum()
    er = (ch / (vol + 1e-12)).fillna(0)
    sc = (er * (2/(f+1) - 2/(s+1)) + 2/(s+1)) ** 2
    kv = np.array(c.values, dtype=float); cv = np.asarray(c.values, float); scv = np.asarray(sc.values, float)
    for i in range(1, len(cv)):
        kv[i] = kv[i-1] + scv[i] * (cv[i] - kv[i-1])
    return pd.Series(kv, index=c.index)

def _zlema(c, n):
    lag = (n - 1) // 2
    return ema(c + (c - c.shift(lag)), n)

def _tema(c, n):
    e1 = ema(c, n); e2 = ema(e1, n); e3 = ema(e2, n)
    return 3*e1 - 3*e2 + e3

def _dema(c, n):
    e1 = ema(c, n); e2 = ema(e1, n)
    return 2*e1 - e2

def _t3(c, n, vf=0.7):
    e1=ema(c,n); e2=ema(e1,n); e3=ema(e2,n); e4=ema(e3,n); e5=ema(e4,n); e6=ema(e5,n)
    a=vf
    c1=-a**3; c2=3*a**2+3*a**3; c3=-6*a**2-3*a-3*a**3; c4=1+3*a+a**3+3*a**2
    return c1*e6 + c2*e5 + c3*e4 + c4*e3

def _vidya(c, n=14):
    cmo = c.diff().rolling(n).apply(lambda x: (x[x>0].sum()+x[x<0].sum())/(np.abs(x).sum()+1e-12), raw=True).abs()
    alpha = 2/(n+1)
    ov = np.array(c.values, dtype=float); cv=np.asarray(c.values,float); k=np.asarray(cmo.fillna(0).values,float)
    for i in range(1,len(cv)):
        a=alpha*k[i]
        ov[i]=a*cv[i]+(1-a)*ov[i-1]
    return pd.Series(ov,index=c.index)

def _mcginley(c, n=14):
    ov=np.array(c.values,dtype=float); cv=np.asarray(c.values,float)
    for i in range(1,len(cv)):
        ov[i]=ov[i-1]+(cv[i]-ov[i-1])/(n*(cv[i]/(ov[i-1]+1e-12))**4+1e-12)
    return pd.Series(ov,index=c.index)

def _ribbon_align(c):
    mas=[ema(c,n).iloc[-1] for n in (5,10,20,50,100) if len(c)>n]
    if len(mas)<2: return 0.0
    desc=all(mas[i]>=mas[i+1] for i in range(len(mas)-1))
    asc=all(mas[i]<=mas[i+1] for i in range(len(mas)-1))
    return 1.0 if asc else (-1.0 if desc else 0.0)

# --------------------------------------------------------------------------- momentum
def _connors_rsi(c):
    r3=rsi(c,3)
    streak=_streak(c); srsi=rsi(streak,2)
    pct=c.pct_change().rolling(100).apply(lambda x:(x[:-1]<x[-1]).mean()*100 if len(x)>1 else 50,raw=True)
    return _last((r3+srsi+pct)/3)

def _streak(c):
    d=np.sign(c.diff().fillna(0)).values; out=np.zeros(len(d))
    for i in range(1,len(d)):
        out[i]=out[i-1]+d[i] if d[i]==np.sign(out[i-1]) or out[i-1]==0 else d[i]
    return pd.Series(out,index=c.index)

def _ppo(c, hist=False):
    line=(ema(c,12)-ema(c,26))/(ema(c,26)+1e-12)*100
    sig=ema(line,9)
    return _last(line-sig) if hist else _last(line)

def _rmi(c, n=14, mom=5):
    ch=c.diff(mom)
    up=ch.clip(lower=0).ewm(alpha=1/n).mean(); dn=(-ch.clip(upper=0)).ewm(alpha=1/n).mean()
    return _last(100-100/(1+up/(dn+1e-12)))

# --------------------------------------------------------------------------- volatilite
def _chaikin_vol(df, n=10):
    hl=(df["high"]-df["low"]).ewm(span=n).mean()
    return _last(hl.pct_change(n)*100)

def _ulcer(c, n=14):
    roll=c.rolling(n).max(); dd=((c-roll)/roll*100)
    return float(np.sqrt((dd**2).rolling(n).mean().iloc[-1])) if len(c)>n else 0.0

# --------------------------------------------------------------------------- hacim
def _adl(df):
    clv=((df["close"]-df["low"])-(df["high"]-df["close"]))/((df["high"]-df["low"])+1e-12)
    return (clv*df["volume"]).cumsum()

def _klinger(df):
    dm=df["high"]-df["low"]; trend=np.sign(df[["high","low","close"]].sum(axis=1).diff().fillna(0))
    vf=df["volume"]*trend
    return _last(ema(vf,34)-ema(vf,55))/(float(df["volume"].rolling(20).mean().iloc[-1])+1e-9)

def _twiggs_mf(df, n=21):
    tr_h=np.maximum(df["high"],df["close"].shift(1)); tr_l=np.minimum(df["low"],df["close"].shift(1))
    ad=(2*df["close"]-tr_l-tr_h)/((tr_h-tr_l)+1e-12)*df["volume"]
    return _last(ad.ewm(span=n).mean()/(df["volume"].ewm(span=n).mean()+1e-12))

def _up_down_vol(df, n=20):
    up=df["volume"].where(df["close"]>df["open"],0).rolling(n).sum()
    dn=df["volume"].where(df["close"]<df["open"],0).rolling(n).sum()
    return _last(up/(dn+1e-12))

# --------------------------------------------------------------------------- yapısal
def _choppiness(df, n=14):
    atr_sum=true_range(df).rolling(n).sum()
    rng=df["high"].rolling(n).max()-df["low"].rolling(n).min()
    return _last(100*np.log10(atr_sum/(rng+1e-12))/np.log10(n))

def _donchian_pos(df, n=20):
    hi=df["high"].rolling(n).max(); lo=df["low"].rolling(n).min()
    return _last((df["close"]-lo)/((hi-lo)+1e-12))

def _squeeze_mom(df, n=20):
    c=df["close"]; m=c.rolling(n).mean()
    hh=(df["high"].rolling(n).max()+df["low"].rolling(n).min())/2
    val=c-(hh+m)/2
    x=np.arange(n)
    return _last(val.rolling(n).apply(lambda y: np.polyfit(x,y,1)[0] if len(y)==n else 0,raw=True))

# --------------------------------------------------------------------------- yardımcı
def _zscore(s, n):
    return _last((s-s.rolling(n).mean())/(s.rolling(n).std()+1e-12))

def _slope(s, n):
    if len(s)<n: return 0.0
    x=np.arange(n); y=s.tail(n).values
    try: return float(np.polyfit(x,y,1)[0])
    except Exception: return 0.0
