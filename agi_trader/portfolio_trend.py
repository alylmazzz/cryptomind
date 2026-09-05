#!/usr/bin/env python3
"""OOS-sağlam trend-takip PORTFÖYÜ (Trend200+Mom20) + vol-targeting risk overlay.
5 parite eşit-ağırlık, gerçekçi maliyet. Deploy edilebilir çekirdek strateji."""
from __future__ import annotations
import sys, glob, json
from pathlib import Path
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

DATA = Path(__file__).parent / "runs" / "data_full"
COST = 0.0006
TARGET_VOL = 0.025      # günlük hedef vol (~ yıllık %48)
MAX_LEV = 1.0           # kaldıraçsız


def load_daily(path):
    df = pd.read_csv(path); df.index = pd.to_datetime(df["dt"])
    d = df[["open","high","low","close","volume"]].astype(float)
    return pd.DataFrame({"open":d["open"].resample("1D").first(),"high":d["high"].resample("1D").max(),
        "low":d["low"].resample("1D").min(),"close":d["close"].resample("1D").last(),
        "volume":d["volume"].resample("1D").sum()}).dropna()


def pair_strat(d, vol_target=True):
    c=d["close"]; r=c.pct_change().fillna(0)
    raw=((c>c.rolling(200).mean()) & (c.pct_change(20)>0)).astype(float)   # Trend200+Mom20
    if vol_target:
        vol=r.rolling(30).std()
        scale=(TARGET_VOL/(vol+1e-9)).clip(0,MAX_LEV)
        pos=(raw*scale).fillna(0)
    else:
        pos=raw
    p=pos.shift(1).fillna(0)
    turn=p.diff().abs().fillna(0)
    strat=p*r - turn*COST
    return strat


def summarize(port):
    eq=(1+port).cumprod()
    total=(eq.iloc[-1]-1)*100
    dd=((eq.cummax()-eq)/eq.cummax()).max()*100
    sh=port.mean()/(port.std()+1e-12)*np.sqrt(365)
    calmar=(total/ (len(eq)/365))/ (dd+1e-9)
    yr={str(y):round((g.iloc[-1]/g.iloc[0]-1)*100,1) for y,g in eq.groupby(eq.index.year)}
    # kâr aylarının oranı
    monthly=eq.resample("1ME").last().pct_change().dropna()
    posm=(monthly>0).mean()*100
    return {"total":round(total,1),"cagr":round(((eq.iloc[-1])**(365/len(eq))-1)*100,1),
            "dd":round(dd,1),"sharpe":round(float(sh),2),"calmar":round(float(calmar),2),
            "pos_months_pct":round(float(posm),0),"yearly":yr}


def main():
    files={Path(p).stem.split("_")[0]:p for p in sorted(glob.glob(str(DATA/"*.csv")))}
    pairs=["BTCUSDT","ETHUSDT","SOLUSDT","DOGEUSDT","AVAXUSDT"]
    for label,vt in [("VOL-TARGETING",True),("SABİT (vol yok)",False)]:
        strats={}
        for s in pairs:
            d=load_daily(files[s]); strats[s]=pair_strat(d,vol_target=vt)
        idx=None
        for s in pairs: idx = strats[s].index if idx is None else idx.union(strats[s].index)
        mat=pd.DataFrame({s:strats[s].reindex(idx).fillna(0) for s in pairs})
        port=mat.mean(axis=1)   # eşit ağırlık
        m=summarize(port)
        print(f"\n=== TREND PORTFÖY ({label}) — 5 parite eşit-ağırlık, gerçekçi maliyet ===")
        print(f"  4.5y getiri: {m['total']:+.1f}%  | CAGR: {m['cagr']:+.1f}%  | Sharpe: {m['sharpe']:+.2f}  "
              f"| Calmar: {m['calmar']:.2f}  | MaxDD: {m['dd']:.1f}%  | kâr-ay: %{m['pos_months_pct']:.0f}")
        print(f"  yıllık: {m['yearly']}")
        if vt:
            Path(__file__).parent.joinpath("runs/portfolio_trend.json").write_text(
                json.dumps({"metrics":m},indent=2,ensure_ascii=False),encoding="utf-8")

if __name__=="__main__":
    main()
