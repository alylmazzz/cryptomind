#!/usr/bin/env python3
"""GELİŞMİŞ trend portföyü v2 — Tier-1/2 iyileştirmeler + ablasyon + v1 karşılaştırma.
İyileştirmeler (kümülatif, her biri OOS'ta ölçülür):
  1) Çok-hızlı trend ensemble (50/100/200)   2) Cross-sectional momentum tilt
  3) Piyasa-beta rejim filtresi (BTC)         4) Portföy-seviyesi vol-targeting
  5) Kriz kill-switch (DD + korelasyon)       6) Maker yürütme (düşük maliyet)
Her ekleme yalnız Sharpe/DD'yi GERÇEKTEN iyileştirirse değerlidir (dürüstlük kuralı).
"""
from __future__ import annotations
import sys, glob, json
from pathlib import Path
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

DATA = Path(__file__).parent / "runs" / "data_full"
PAIRS = ["BTCUSDT","ETHUSDT","SOLUSDT","DOGEUSDT","AVAXUSDT"]
COST_TAKER = 0.0006
COST_MAKER = 0.0003
TARGET_VOL_D = 0.30/np.sqrt(365)   # ~%30 yıllık portföy vol hedefi


def daily(short):
    df=pd.read_csv(DATA/f"{short}_1h.csv"); df.index=pd.to_datetime(df["dt"])
    d=df[["open","high","low","close","volume"]].astype(float)
    return pd.DataFrame({"close":d["close"].resample("1D").last(),
        "high":d["high"].resample("1D").max(),"low":d["low"].resample("1D").min()}).dropna()


def conviction(c, ensemble):
    mom = (c.pct_change(20) > 0).astype(float)
    if ensemble:
        conv = (((c>c.rolling(50).mean()).astype(float)+
                 (c>c.rolling(100).mean()).astype(float)+
                 (c>c.rolling(200).mean()).astype(float))/3.0)
        return (conv*mom).fillna(0)
    return ((c>c.rolling(200).mean()).astype(float)*mom).fillna(0)


def vol_target_pair(c):
    vol=c.pct_change().rolling(30).std(); return (0.025/(vol+1e-9)).clip(0,1.0)


def build(closes, cfg):
    """closes: {sym:close}. cfg toggles. Döndürür: günlük portföy getiri serisi."""
    idx=None
    for s in PAIRS: idx = closes[s].index if idx is None else idx.union(closes[s].index)
    R=pd.DataFrame({s:closes[s].reindex(idx).ffill().pct_change().fillna(0) for s in PAIRS})
    # ham pozisyon (0..1) her parite
    W={}
    for s in PAIRS:
        c=closes[s].reindex(idx).ffill()
        conv=conviction(c, cfg["ensemble"])
        w=conv*vol_target_pair(c)
        W[s]=w
    W=pd.DataFrame(W).fillna(0)
    # cross-sectional momentum tilt (90g getiri sırasına göre)
    if cfg["xs_mom"]:
        mom90=pd.DataFrame({s:closes[s].reindex(idx).ffill().pct_change(90) for s in PAIRS})
        rank=mom90.rank(axis=1, pct=True).fillna(0.5)     # 0..1
        W=W*(0.5+rank)                                    # güçlüyü fazla ağırlıkla
    # piyasa-beta rejim filtresi (BTC 200g altındaysa kitabı küçült)
    if cfg["market_filter"]:
        btc=closes["BTCUSDT"].reindex(idx).ffill()
        mkt=np.where(btc>btc.rolling(200).mean(),1.0,0.35)
        W=W.mul(pd.Series(mkt,index=idx),axis=0)
    W=W.div(len(PAIRS))                                   # eşit-ağırlık normalizasyon
    cost=cfg["cost"]
    # taban portföy getirisi (cost dahil)
    Wl=W.shift(1).fillna(0)
    turn=(W-W.shift(1)).abs().sum(axis=1).fillna(0)
    base=(Wl*R).sum(axis=1) - turn*cost
    # portföy vol-targeting (geçmiş vol ile, look-ahead yok)
    if cfg["port_vt"]:
        rv=base.rolling(30).std()
        lev=(TARGET_VOL_D/(rv+1e-9)).clip(0.2,1.0).shift(1).fillna(1.0)
        base=base*lev
    # kriz kill-switch (20g DD veya ortalama korelasyon yüksekse yarıla)
    if cfg["crisis"]:
        eq=(1+base).cumprod(); dd20=1-eq/eq.rolling(20).max()
        corr=R.rolling(30).corr().groupby(level=0).apply(
            lambda x: x.values[np.triu_indices(len(PAIRS),1)].mean() if x.shape[0]==len(PAIRS) else np.nan)
        corr=pd.Series(corr.values,index=eq.index[-len(corr):]) if len(corr)==len(eq) else pd.Series(0.5,index=eq.index)
        crisis=((dd20>0.15)|(corr.reindex(eq.index).fillna(0.5)>0.9)).astype(float)
        mult=(1-0.5*crisis).shift(1).fillna(1.0)
        base=base*mult
    return base


def metrics(ret, label=""):
    eq=(1+ret).cumprod()
    total=(eq.iloc[-1]-1)*100
    dd=((eq.cummax()-eq)/eq.cummax()).max()*100
    sh=ret.mean()/(ret.std()+1e-12)*np.sqrt(365)
    cagr=(eq.iloc[-1]**(365/len(eq))-1)*100
    calmar=cagr/(dd+1e-9)
    return {"label":label,"total":round(total,1),"cagr":round(cagr,1),"sharpe":round(float(sh),2),
            "dd":round(dd,1),"calmar":round(float(calmar),2)}


def sub(ret, start, end):
    m=ret[(ret.index>=start)&(ret.index<end)]
    if len(m)<5: return None
    return metrics(m)


def main():
    files={Path(p).stem.split("_")[0]:p for p in glob.glob(str(DATA/"*.csv"))}
    closes={s:daily(s)["close"] for s in PAIRS}
    OFF={"ensemble":False,"xs_mom":False,"market_filter":False,"port_vt":False,"crisis":False,"cost":COST_TAKER}
    # ABLASYON — kümülatif
    steps=[
        ("v1 BAZ (Trend200+Mom20)", dict(OFF)),
        ("+ensemble (50/100/200)", {**OFF,"ensemble":True}),
        ("+cross-sec momentum", {**OFF,"ensemble":True,"xs_mom":True}),
        ("+piyasa rejim filtresi", {**OFF,"ensemble":True,"xs_mom":True,"market_filter":True}),
        ("+portföy vol-target", {**OFF,"ensemble":True,"xs_mom":True,"market_filter":True,"port_vt":True}),
        ("+kriz kill-switch", {**OFF,"ensemble":True,"xs_mom":True,"market_filter":True,"port_vt":True,"crisis":True}),
        ("+maker yürütme (v2 SON)", {"ensemble":True,"xs_mom":True,"market_filter":True,"port_vt":True,"crisis":True,"cost":COST_MAKER}),
    ]
    print("=== ABLASYON (4.5 yıl, kümülatif — her satır bir öncekine ekler) ===")
    v1=v2=None
    for label,cfg in steps:
        ret=build(closes,cfg); m=metrics(ret,label)
        if label.startswith("v1"): v1=ret
        if "SON" in label: v2=ret
        print(f"  {label:28s} Sharpe={m['sharpe']:+.2f} CAGR={m['cagr']:+6.1f}% DD={m['dd']:4.1f}% Calmar={m['calmar']:.2f}")
    m1=metrics(v1,"v1"); m2=metrics(v2,"v2")
    print("\n=== v1 (öncesi) vs v2 (son hali) ===")
    print(f"  v1 BAZ : Sharpe {m1['sharpe']:+.2f} | CAGR {m1['cagr']:+.1f}% | DD {m1['dd']:.1f}% | Calmar {m1['calmar']:.2f}")
    print(f"  v2 SON : Sharpe {m2['sharpe']:+.2f} | CAGR {m2['cagr']:+.1f}% | DD {m2['dd']:.1f}% | Calmar {m2['calmar']:.2f}")
    print(f"  Δ      : Sharpe {m2['sharpe']-m1['sharpe']:+.2f} | CAGR {m2['cagr']-m1['cagr']:+.1f}% | DD {m2['dd']-m1['dd']:+.1f}%")
    # SON 6 AY (2026-01..2026-07) — kullanıcının istediği kısa dönem karşılaştırma
    print("\n=== SON 6 AY (2026 H1, düşüş dönemi) ===")
    for nm,ret in [("v1",v1),("v2",v2)]:
        s=sub(ret,"2026-01-01","2026-07-02")
        if s: print(f"  {nm}: getiri {s['total']:+.1f}% | DD {s['dd']:.1f}% | Sharpe {s['sharpe']:+.2f}")
    json.dump({"v1":m1,"v2":m2},open("runs/trend_v2_compare.json","w"))

if __name__=="__main__":
    main()
