#!/usr/bin/env python3
"""Hedef-vol kaldıraç + kuyruk hedge overlay'i — diversifiye trend kitabına.
Amaç: CAGR'ı yükselt (hedef-vol'e lever) AMA stres anında (vol-spike/DD) de-risk
ile tail'i kes. Her katman OOS'ta A/B; yalnız Sharpe/Calmar'ı koruyup CAGR'ı
artıran kabul edilir."""
from __future__ import annotations
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

DATA = Path(__file__).parent / "runs" / "data_full"
NCACHE = Path(__file__).parent / "runs" / "data_noncrypto"; NCACHE.mkdir(parents=True, exist_ok=True)
CRYPTO = ["BTCUSDT","ETHUSDT","SOLUSDT","DOGEUSDT","AVAXUSDT"]
NONCRYPTO = ["GLD","SPY","QQQ","TLT","UUP","USO","SLV","DBC"]
COST = 0.0004; START = "2022-01-01"
TARGET_ANN = 0.15; MAX_LEV = 2.5


def crypto_close(s):
    df=pd.read_csv(DATA/f"{s}_1h.csv"); df.index=pd.to_datetime(df["dt"])
    return df["close"].astype(float).resample("1D").last().dropna()

def nc_close(t):
    p=NCACHE/f"{t}.csv"
    if p.exists():
        s=pd.read_csv(p,index_col=0,parse_dates=True)["close"]; return s.astype(float)
    import yfinance as yf
    d=yf.download(t,start="2017-01-01",progress=False,auto_adjust=True)
    if d is None or len(d)==0: return None
    cl=d["Close"];
    if isinstance(cl,pd.DataFrame): cl=cl.iloc[:,0]
    cl.index=pd.to_datetime(cl.index); cl=cl.dropna()
    cl.rename("close").to_frame().to_csv(p)
    return cl.astype(float)

def strat(close):
    c=close.astype(float); r=c.pct_change().fillna(0)
    raw=((c>c.rolling(200).mean())&(c.pct_change(20)>0)).astype(float)
    vol=c.pct_change().rolling(30).std(); pos=(raw*(0.025/(vol+1e-9)).clip(0,1.0)).fillna(0)
    p=pos.shift(1).fillna(0); return p*r - p.diff().abs().fillna(0)*COST

def metrics(ret):
    ret=ret[ret.index>=START]; eq=(1+ret).cumprod()
    dd=((eq.cummax()-eq)/eq.cummax()).max()*100
    sh=ret.mean()/(ret.std()+1e-12)*np.sqrt(365); cagr=(eq.iloc[-1]**(365/len(eq))-1)*100
    # en kötü tek-gün + 2022 (ayı) getirisi (tail göstergesi)
    worst=ret.min()*100
    y2022=(1+ret[(ret.index>='2022-01-01')&(ret.index<'2023-01-01')]).prod()-1
    return dict(sharpe=round(float(sh),2),cagr=round(cagr,1),dd=round(dd,1),
                calmar=round(float(cagr/(dd+1e-9)),2),worst_day=round(float(worst),1),
                y2022=round(float(y2022*100),1))

def main():
    print("Veri (kripto + kripto-dışı cache) yükleniyor...",flush=True)
    S={s:strat(crypto_close(s)) for s in CRYPTO}
    for t in NONCRYPTO:
        cl=nc_close(t)
        if cl is not None and len(cl)>250: S[t]=strat(cl)
    idx=None
    for v in S.values(): idx=v.index if idx is None else idx.union(v.index)
    idx=idx[idx>=pd.Timestamp("2021-06-01")]
    M=pd.DataFrame({k:v.reindex(idx).fillna(0) for k,v in S.items()})
    base=M.mean(axis=1)                                  # eşit-ağırlık diversifiye

    # target-vol kaldıraç
    rv=base.rolling(30).std(); lev=(TARGET_ANN/np.sqrt(365)/(rv+1e-9)).clip(0,MAX_LEV).shift(1).fillna(1.0)
    tv=base*lev

    # kuyruk de-risk: vol-spike VEYA DD → kaldıracı kes
    rv20=base.rolling(20).std(); rv100=base.rolling(100).std()
    eqb=(1+base).cumprod(); dd20=1-eqb/eqb.rolling(40).max()
    stress=((rv20>1.6*rv100)|(dd20>0.06)).astype(float)
    derisk=(1-0.6*stress).shift(1).fillna(1.0)
    tvh=tv*derisk

    print("\n=== OVERLAY A/B (diversifiye kitap, OOS 2022-2026) ===")
    for name,ret in [("BAZ diversifiye",base),("+hedef-vol kaldıraç",tv),("+kuyruk hedge (SON)",tvh)]:
        m=metrics(ret)
        print(f"  {name:24s} Sharpe={m['sharpe']:+.2f} CAGR={m['cagr']:+6.1f}% DD={m['dd']:4.1f}% "
              f"Calmar={m['calmar']:.2f} enKötüGün={m['worst_day']:+.1f}% 2022={m['y2022']:+.1f}%")
    mb=metrics(base); mf=metrics(tvh)
    print(f"\n  BAZ → SON: CAGR {mb['cagr']:+.1f}%→{mf['cagr']:+.1f}% | DD {mb['dd']:.1f}%→{mf['dd']:.1f}% | "
          f"Sharpe {mb['sharpe']:+.2f}→{mf['sharpe']:+.2f} | Calmar {mb['calmar']:.2f}→{mf['calmar']:.2f}")

if __name__=="__main__":
    main()
