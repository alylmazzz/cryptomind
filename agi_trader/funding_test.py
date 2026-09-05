#!/usr/bin/env python3
"""FAZ1 — funding'in trend stratejisine OOS ALFA katıp katmadığı testi.
Hipotez: aşırı-yüksek funding (froth/aşırı-kaldıraç) → trend-long crowded →
pozisyonu azalt → risk-ayarlı iyileşme. Kanıt yoksa ÇIKAR (dürüstlük kuralı)."""
from __future__ import annotations
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

PRICE = Path(__file__).parent / "runs" / "data_full"
FUND = Path(__file__).parent / "runs" / "data_funding"
COST = 0.0006
PAIRS = ["BTC","ETH","SOL","DOGE","AVAX"]


def daily_price(short):
    df = pd.read_csv(PRICE/f"{short}USDT_1h.csv"); df.index=pd.to_datetime(df["dt"])
    d=df[["close","high","low","open","volume"]].astype(float)
    return pd.DataFrame({"close":d["close"].resample("1D").last()}).dropna()

def daily_funding(short):
    f=pd.read_csv(FUND/f"{short}_funding.csv"); f.index=pd.to_datetime(f["dt"])
    fr=f["funding_rate"].astype(float).resample("1D").mean()
    return fr

def trend_pos(c, vt=True):
    raw=((c>c.rolling(200).mean())&(c.pct_change(20)>0)).astype(float)
    if not vt: return raw.fillna(0)
    vol=c.pct_change().rolling(30).std(); scale=(0.025/(vol+1e-9)).clip(0,1.0)
    return (raw*scale).fillna(0)

def sharpe(strat): return float(strat.mean()/(strat.std()+1e-12)*np.sqrt(365))
def dd(strat): eq=(1+strat).cumprod(); return float(((eq.cummax()-eq)/eq.cummax()).max()*100)
def tot(strat): return float(((1+strat).cumprod().iloc[-1]-1)*100)

def run(pos, r):
    p=pos.shift(1).fillna(0); turn=p.diff().abs().fillna(0)
    return p*r - turn*COST

def main():
    base_port=[]; filt_port=[]; froth_only=[]
    print("Parite bazında (base trend vs +funding-froth filtresi):")
    for s in PAIRS:
        c=daily_price(s)["close"]; fr=daily_funding(s).reindex(c.index).ffill()
        r=c.pct_change().fillna(0)
        base=trend_pos(c)
        # funding z-score (60g); yüksek z = froth
        fz=(fr - fr.rolling(60).mean())/(fr.rolling(60).std()+1e-12)
        # froth filtresi: z>1.5 iken pozisyonu yarıla, z>2.5 iken sıfırla
        mult=pd.Series(1.0,index=c.index)
        mult[fz>1.5]=0.5; mult[fz>2.5]=0.0
        filt=base*mult
        sb=run(base,r); sf=run(filt,r)
        base_port.append(sb); filt_port.append(sf)
        print(f"  {s:5s} BASE Sharpe={sharpe(sb):+.2f} tot={tot(sb):+6.1f}% DD={dd(sb):4.1f}%  |  "
              f"+FROTH Sharpe={sharpe(sf):+.2f} tot={tot(sf):+6.1f}% DD={dd(sf):4.1f}%", flush=True)
    # portföy
    def port(lst):
        idx=lst[0].index
        for x in lst: idx=idx.union(x.index)
        m=pd.DataFrame({i:lst[i].reindex(idx).fillna(0) for i in range(len(lst))})
        return m.mean(axis=1)
    pb=port(base_port); pf=port(filt_port)
    print("\n=== PORTFÖY KARŞILAŞTIRMA ===")
    print(f"  BASE trend    : Sharpe {sharpe(pb):+.2f}  getiri {tot(pb):+.1f}%  MaxDD {dd(pb):.1f}%")
    print(f"  +funding froth: Sharpe {sharpe(pf):+.2f}  getiri {tot(pf):+.1f}%  MaxDD {dd(pf):.1f}%")
    dS=sharpe(pf)-sharpe(pb); dD=dd(pf)-dd(pb)
    verdict = "KABUL (alfa/risk iyileşti)" if (dS>0.03 or dD<-1.0) else "RET (anlamlı katkı yok → çıkar)"
    print(f"  → ΔSharpe {dS:+.2f}, ΔMaxDD {dD:+.1f}%  ⇒  {verdict}")

if __name__=="__main__":
    main()
