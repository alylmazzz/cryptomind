#!/usr/bin/env python3
"""Veri-temelli evren SEÇİMİ (forward selection). 13-varlık nihai kitaba,
FX/emtia/tahvil aday havuzundan yalnız Calmar'ı İYİLEŞTİRENLERİ ekle.
Körlemesine genişletme zarar veriyordu (28→Sharpe düştü); bu disiplinli seçim."""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0, str(Path(__file__).parent))
from leverage_hedge import crypto_close, nc_close, strat, CRYPTO, START, TARGET_ANN, MAX_LEV

BASE_NC = ["GLD","SPY","QQQ","TLT","UUP","USO","SLV","DBC"]
CANDIDATES = ["IEF","SHY","HYG","LQD","TIP","EMB","BNDX",      # tahvil (vade/kredi/EM/intl)
              "FXE","FXY","FXB","FXA","FXC","FXF",              # FX (euro/yen/pound/aud/cad/chf)
              "PPLT","PALL","CPER","WEAT","DBA","UNG","CORN"]   # emtia (platin/paladyum/bakır/tarım/gaz)


def tv(base):
    rv=base.rolling(30).std(); lev=(TARGET_ANN/np.sqrt(365)/(rv+1e-9)).clip(0,MAX_LEV).shift(1).fillna(1.0)
    return base*lev

def port_metrics(strats, idx):
    M=pd.DataFrame({k:v.reindex(idx).fillna(0) for k,v in strats.items()})
    ret=tv(M.mean(axis=1)); ret=ret[ret.index>=START]; eq=(1+ret).cumprod()
    dd=((eq.cummax()-eq)/eq.cummax()).max()*100
    sh=ret.mean()/(ret.std()+1e-12)*np.sqrt(365); cagr=(eq.iloc[-1]**(365/len(eq))-1)*100
    return dict(sharpe=round(float(sh),2),cagr=round(cagr,1),dd=round(dd,1),calmar=round(float(cagr/(dd+1e-9)),2))

def main():
    print("Temel 13 + aday havuzu yükleniyor/cache...",flush=True)
    S={s:strat(crypto_close(s)) for s in CRYPTO}
    for t in BASE_NC: S[t]=strat(nc_close(t))
    cand={}
    for t in CANDIDATES:
        try:
            cl=nc_close(t)
            if cl is not None and len(cl)>250: cand[t]=strat(cl)
        except Exception: pass
    print(f"  aday sayısı: {len(cand)} ({list(cand.keys())})",flush=True)
    idx=None
    for v in list(S.values())+list(cand.values()): idx=v.index if idx is None else idx.union(v.index)
    idx=idx[idx>=pd.Timestamp("2021-06-01")]

    cur=dict(S); base_m=port_metrics(cur,idx)
    print(f"\nBAŞLANGIÇ (13 varlık): {base_m}")
    added=[]
    remaining=dict(cand)
    print("\n=== FORWARD SELECTION (Calmar'ı iyileştiren eklenir) ===",flush=True)
    while remaining:
        best=None; best_m=None
        for t,s in remaining.items():
            m=port_metrics({**cur,t:s},idx)
            if best is None or m["calmar"]>best_m["calmar"]:
                best,best_m=t,m
        cur_m=port_metrics(cur,idx)
        if best_m["calmar"]>cur_m["calmar"]+0.01:   # anlamlı iyileşme
            cur[best]=remaining.pop(best); added.append(best)
            print(f"  + {best:5s} → Calmar {cur_m['calmar']:.2f}→{best_m['calmar']:.2f} "
                  f"Sharpe {best_m['sharpe']:.2f} CAGR {best_m['cagr']:.1f}% DD {best_m['dd']:.1f}%",flush=True)
        else:
            break
    final_m=port_metrics(cur,idx)
    print(f"\n=== SEÇİLEN NİHAİ EVREN ({len(cur)} varlık) ===")
    print(f"  eklenenler: {added}")
    print(f"  metrikler: {final_m}")
    print(f"  13-varlık → seçilmiş: Calmar {base_m['calmar']:.2f}→{final_m['calmar']:.2f}, "
          f"Sharpe {base_m['sharpe']:.2f}→{final_m['sharpe']:.2f}, DD {base_m['dd']:.1f}%→{final_m['dd']:.1f}%")
    noncrypto_final=[k for k in cur if k not in CRYPTO]
    Path("runs/selected_universe.json").write_text(json.dumps(
        {"noncrypto":noncrypto_final,"added":added,"metrics":final_m},indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"\n  NONCRYPTO listesi (daemon'a): {noncrypto_final}")

if __name__=="__main__":
    main()
