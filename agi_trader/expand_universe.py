#!/usr/bin/env python3
"""Bölüm 1+2: Genişletilmiş uncorrelated evren + cross-asset momentum.
Mevcut 13-varlık nihai stratejiye daha fazla tahvil/FX/bölge/emtia ekle;
ayrıca en güçlü trendleri seçen cross-sectional momentum'u test et.
Hepsi hedef-vol kaldıraçla (nihai strateji tutarlılığı), OOS 2022-2026."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0, str(Path(__file__).parent))
from leverage_hedge import crypto_close, nc_close, strat, CRYPTO, START, TARGET_ANN, MAX_LEV

BASE_NC = ["GLD","SPY","QQQ","TLT","UUP","USO","SLV","DBC"]                     # mevcut 8
MORE_NC = ["IEF","SHY","HYG","LQD","FXE","FXY","FXB","EEM","EFA","IWM","VNQ",   # tahvil/FX/bölge
           "UNG","CORN","PPLT","COPX"]                                          # emtia


def target_vol(base):
    rv=base.rolling(30).std()
    lev=(TARGET_ANN/np.sqrt(365)/(rv+1e-9)).clip(0,MAX_LEV).shift(1).fillna(1.0)
    return base*lev

def metrics(ret):
    ret=ret[ret.index>=START]; eq=(1+ret).cumprod()
    dd=((eq.cummax()-eq)/eq.cummax()).max()*100
    sh=ret.mean()/(ret.std()+1e-12)*np.sqrt(365); cagr=(eq.iloc[-1]**(365/len(eq))-1)*100
    return dict(sharpe=round(float(sh),2),cagr=round(cagr,1),dd=round(dd,1),calmar=round(float(cagr/(dd+1e-9)),2))

def build_strats(nc_list):
    S={s:strat(crypto_close(s)) for s in CRYPTO}
    ok=[]
    for t in nc_list:
        try:
            cl=nc_close(t)
            if cl is not None and len(cl)>250: S[t]=strat(cl); ok.append(t)
        except Exception: pass
    return S, ok

def portfolio(S, idx, xs_top=None):
    M=pd.DataFrame({k:v.reindex(idx).fillna(0) for k,v in S.items()})
    if xs_top:
        # cross-sectional momentum: en güçlü N trend. SEÇİM 1 GÜN GECİKTİRİLİR
        # (look-ahead yok: dünkü momentum+pozisyona göre seç, bugün uygula).
        mom=M.rolling(20).sum()
        sel=((mom.rank(axis=1,ascending=False)<=xs_top) & (M!=0))
        sel=sel.shift(1).fillna(False)           # <-- kritik: gecikme
        M=M.where(sel, 0)
    ew=M.replace(0,np.nan).mean(axis=1).fillna(0) if xs_top else M.mean(axis=1)
    return target_vol(ew)

def main():
    print("13-varlık (nihai) yükleniyor...",flush=True)
    S13,ok13=build_strats(BASE_NC)
    print(f"Genişletilmiş ({len(BASE_NC+MORE_NC)} kripto-dışı) çekiliyor/cache...",flush=True)
    Sx,okx=build_strats(BASE_NC+MORE_NC)
    print(f"  eklenen: {[t for t in MORE_NC if t in okx]}",flush=True)
    idx=None
    for v in Sx.values(): idx=v.index if idx is None else idx.union(v.index)
    idx=idx[idx>=pd.Timestamp("2021-06-01")]
    tests=[
        (f"NİHAİ 13-varlık (5+{len(ok13)})", portfolio(S13,idx)),
        (f"GENİŞ {len(Sx)}-varlık", portfolio(Sx,idx)),
        (f"GENİŞ + cross-sec mom (top10)", portfolio(Sx,idx,xs_top=10)),
        (f"GENİŞ + cross-sec mom (top6)", portfolio(Sx,idx,xs_top=6)),
    ]
    print("\n=== A/B (hedef-vol kaldıraçlı, OOS 2022-2026) ===")
    for name,ret in tests:
        m=metrics(ret)
        print(f"  {name:32s} Sharpe={m['sharpe']:+.2f} CAGR={m['cagr']:+6.1f}% DD={m['dd']:4.1f}% Calmar={m['calmar']:.2f}")

if __name__=="__main__":
    main()
