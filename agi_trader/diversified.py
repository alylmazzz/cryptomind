#!/usr/bin/env python3
"""Kripto-DIŞI diversifikasyon — en büyük Sharpe kaldıracı testi.
Aynı trend-takip (Trend200+Mom20) altın/endeks/tahvil/USD/petrol/gümüşe uygulanır;
risk-parity (inverse-vol) portföy: kripto-only vs diversifiye OOS karşılaştırma.
Hipotez: uncorrelated varlıklar Sharpe'ı ↑, DD'yi ↓ (eşit portföy-vol'de daha iyi)."""
from __future__ import annotations
import sys, glob, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

DATA = Path(__file__).parent / "runs" / "data_full"
CRYPTO = ["BTCUSDT","ETHUSDT","SOLUSDT","DOGEUSDT","AVAXUSDT"]
NONCRYPTO = {"GLD":"altın","SPY":"S&P500","QQQ":"Nasdaq","TLT":"20y-tahvil",
             "UUP":"USD","USO":"petrol","SLV":"gümüş","DBC":"emtia"}
COST = 0.0004
START = "2022-01-01"


def crypto_close(short):
    df=pd.read_csv(DATA/f"{short}_1h.csv"); df.index=pd.to_datetime(df["dt"])
    return df["close"].astype(float).resample("1D").last().dropna()


def yf_close(ticker):
    import yfinance as yf
    d=yf.download(ticker, start="2017-01-01", progress=False, auto_adjust=True)
    if d is None or len(d)==0: return None
    cl=d["Close"]
    if isinstance(cl,pd.DataFrame): cl=cl.iloc[:,0]
    cl.index=pd.to_datetime(cl.index)
    return cl.dropna()


def strat_returns(close):
    """Trend200+Mom20 (vol-targeted) günlük strateji getirisi (maliyet dahil)."""
    c=close.astype(float); r=c.pct_change().fillna(0)
    raw=((c>c.rolling(200).mean())&(c.pct_change(20)>0)).astype(float)
    vol=c.pct_change().rolling(30).std(); pos=(raw*(0.025/(vol+1e-9)).clip(0,1.0)).fillna(0)
    p=pos.shift(1).fillna(0); turn=p.diff().abs().fillna(0)
    return p*r - turn*COST


def risk_parity(strats: dict, idx):
    """Inverse-vol ağırlıklı portföy getirisi (her varlık ~eşit risk katkısı)."""
    S=pd.DataFrame({k:v.reindex(idx).fillna(0) for k,v in strats.items()})
    vol=S.rolling(60).std()
    iv=(1/(vol+1e-9)); w=iv.div(iv.sum(axis=1),axis=0).shift(1).fillna(0)
    return (w*S).sum(axis=1)


def metrics(ret):
    ret=ret[ret.index>=START]
    eq=(1+ret).cumprod()
    if len(eq)<10: return None
    total=(eq.iloc[-1]-1)*100; dd=((eq.cummax()-eq)/eq.cummax()).max()*100
    sh=ret.mean()/(ret.std()+1e-12)*np.sqrt(365)
    cagr=(eq.iloc[-1]**(365/len(eq))-1)*100
    return {"total":round(total,1),"cagr":round(cagr,1),"sharpe":round(float(sh),2),
            "dd":round(dd,1),"calmar":round(float(cagr/(dd+1e-9)),2)}


def main():
    print("Kripto verisi yükleniyor...", flush=True)
    strats={s: strat_returns(crypto_close(s)) for s in CRYPTO}
    print("Kripto-dışı (yfinance) çekiliyor...", flush=True)
    nc={}
    for t,name in NONCRYPTO.items():
        try:
            cl=yf_close(t)
            if cl is not None and len(cl)>250:
                nc[t]=strat_returns(cl); print(f"  {t} ({name}): {len(cl)} gün", flush=True)
        except Exception as e:
            print(f"  {t}: HATA {type(e).__name__}", flush=True)
    # ortak indeks
    allidx=None
    for v in list(strats.values())+list(nc.values()):
        allidx = v.index if allidx is None else allidx.union(v.index)
    allidx=allidx[allidx>=pd.Timestamp("2021-01-01")]
    # korelasyon (strateji getirileri, kripto vs kripto-dışı ortalama)
    cport=risk_parity(strats, allidx)
    div=risk_parity({**strats,**nc}, allidx)
    mc=metrics(cport); md=metrics(div)
    # kripto-dışı sepetin kripto ile korelasyonu
    if nc:
        ncport=risk_parity(nc, allidx)
        corr=float(cport.reindex(allidx).fillna(0).corr(ncport.reindex(allidx).fillna(0)))
    else:
        corr=float("nan")
    print("\n=== RISK-PARITY PORTFÖY KARŞILAŞTIRMA (OOS 2022-2026) ===")
    print(f"  KRİPTO-ONLY (5)     : Sharpe {mc['sharpe']:+.2f} | CAGR {mc['cagr']:+.1f}% | DD {mc['dd']:.1f}% | Calmar {mc['calmar']:.2f}")
    print(f"  DİVERSİFİYE (5+{len(nc)})   : Sharpe {md['sharpe']:+.2f} | CAGR {md['cagr']:+.1f}% | DD {md['dd']:.1f}% | Calmar {md['calmar']:.2f}")
    print(f"  Δ                   : Sharpe {md['sharpe']-mc['sharpe']:+.2f} | DD {md['dd']-mc['dd']:+.1f}% | Calmar {md['calmar']-mc['calmar']:+.2f}")
    print(f"  Kripto ↔ kripto-dışı sepet korelasyonu: {corr:+.2f} (düşük = iyi diversifikasyon)")
    verdict="KABUL — diversifikasyon Sharpe/Calmar'ı iyileştirdi" if (md['sharpe']>mc['sharpe']+0.05 or md['calmar']>mc['calmar']+0.1) else "NÖTR/RET — anlamlı katkı yok"
    print(f"  ⇒ {verdict}")

if __name__=="__main__":
    main()
