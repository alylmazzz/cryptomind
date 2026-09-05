#!/usr/bin/env python3
"""Basit/sağlam BAZ stratejiler — SABİT parametre (fit yok → tüm 4.5y doğal OOS).
Gerçekçi maliyetle, günlük bazda, buy&hold'a karşı. Amaç: karmaşık TA-konsensüs
edge'siz çıktı; BASİT bir şeyin OOS'ta gerçek edge'i var mı?"""
from __future__ import annotations
import sys, glob, json
from pathlib import Path
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

DATA = Path(__file__).parent / "runs" / "data_full"
COST = 0.0006   # ~%0.06 tek yön (fee+slip, günlük düşük turnover)


def load_daily(path):
    df = pd.read_csv(path)
    df.index = pd.to_datetime(df["dt"])
    d = df[["open","high","low","close","volume"]].astype(float)
    o=d["open"].resample("1D").first(); h=d["high"].resample("1D").max()
    l=d["low"].resample("1D").min(); c=d["close"].resample("1D").last()
    v=d["volume"].resample("1D").sum()
    return pd.DataFrame({"open":o,"high":h,"low":l,"close":c,"volume":v}).dropna()


def equity_from_pos(close, pos):
    """pos: her gün hedef pozisyon (0/1 veya -1..1), t-1'den uygulanır; maliyet turnover'da."""
    r = close.pct_change().fillna(0).values
    p = pos.shift(1).fillna(0).values          # look-ahead yok
    turn = np.abs(np.diff(np.concatenate([[0], p])))
    strat = p * r - turn * COST
    eq = np.cumprod(1 + strat)
    return pd.Series(eq, index=close.index), pd.Series(strat, index=close.index)


def metrics(eq, strat, close):
    total = (eq.iloc[-1]-1)*100
    peak = eq.cummax(); dd = ((peak-eq)/peak).max()*100
    sh = strat.mean()/(strat.std()+1e-12)*np.sqrt(365)
    # yıllık
    yr = {}
    for y,g in eq.groupby(eq.index.year):
        yr[str(y)] = round((g.iloc[-1]/g.iloc[0]-1)*100,1)
    return {"total":round(total,1),"dd":round(dd,1),"sharpe":round(float(sh),2),"yearly":yr}


def strategies(d):
    c=d["close"]; out={}
    out["BuyHold"] = pd.Series(1.0, index=c.index)
    out["SMA200_trend"] = (c > c.rolling(200).mean()).astype(float)
    out["EMA50x200"] = (c.ewm(span=50).mean() > c.ewm(span=200).mean()).astype(float)
    # Donchian 20/10 breakout (Turtle-lite), long/flat
    hi=d["high"].rolling(20).max().shift(1); lo=d["low"].rolling(10).min().shift(1)
    pos=pd.Series(np.nan,index=c.index); pos[c>hi]=1.0; pos[c<lo]=0.0
    out["Donchian20_10"]=pos.ffill().fillna(0)
    # TS-momentum 90g
    out["TSmom90"] = (c.pct_change(90) > 0).astype(float)
    # Trend + üstünde momentum filtresi (200 trend VE 20g momentum)
    out["Trend200+Mom20"] = ((c>c.rolling(200).mean()) & (c.pct_change(20)>0)).astype(float)
    return out


def main():
    files={Path(p).stem.split("_")[0]:p for p in sorted(glob.glob(str(DATA/"*.csv")))}
    pairs=["BTCUSDT","ETHUSDT","SOLUSDT","DOGEUSDT","AVAXUSDT"]
    allres={}
    for short in pairs:
        d=load_daily(files[short])
        print(f"\n{short} ({d.index[0].date()} → {d.index[-1].date()}, {len(d)} gün):", flush=True)
        strat_res={}
        for name,pos in strategies(d).items():
            eq,st=equity_from_pos(d["close"],pos)
            m=metrics(eq,st,d["close"]); strat_res[name]=m
            flag = "★" if (name!="BuyHold" and m["sharpe"]>strat_res.get("BuyHold",{}).get("sharpe",0) and m["total"]>0) else " "
            print(f"  {flag}{name:18s} tot={m['total']:+7.1f}% Sharpe={m['sharpe']:+.2f} DD={m['dd']:4.1f}%  {m['yearly']}", flush=True)
        allres[short]=strat_res
    # strateji bazında ortalama Sharpe (parite-üstü genelleme)
    print("\n=== STRATEJİ GENELLEMESİ (parite-ort Sharpe / ort getiri) ===", flush=True)
    names=list(next(iter(allres.values())).keys())
    rank=[]
    for nm in names:
        shs=[allres[p][nm]["sharpe"] for p in pairs]; tots=[allres[p][nm]["total"] for p in pairs]
        rank.append((nm, round(float(np.mean(shs)),2), round(float(np.mean(tots)),1)))
    for nm,sh,to in sorted(rank,key=lambda x:-x[1]):
        print(f"  {nm:18s} ort_Sharpe={sh:+.2f}  ort_getiri={to:+.1f}%", flush=True)
    Path(__file__).parent.joinpath("runs/baselines.json").write_text(
        json.dumps(allres,indent=2,ensure_ascii=False),encoding="utf-8")

if __name__=="__main__":
    main()
