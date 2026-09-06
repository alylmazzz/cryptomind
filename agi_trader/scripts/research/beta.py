# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, os, glob, re, collections, json, math, statistics as st
HIST=os.path.expanduser("~/cmbench/runs/history")
def son():
    d={}
    for f in glob.glob(f"{HIST}/*_1m_*.csv"):
        m=re.match(r".*/mexc_(.+?)_1m_(\d+)_(\d+)\.csv",f)
        if m and (m.group(1) not in d or int(m.group(3))>d[m.group(1)][1]): d[m.group(1)]=(f,int(m.group(3)))
    return {k:v[0] for k,v in d.items()}
print("="*88); print("KONTROL 1 — iki yarının PİYASA YÖNÜ (al-tut getirisi)"); print("="*88)
a_r, b_r = [], []
for sym,f in sorted(son().items()):
    if sym=="BNB-USDT": continue
    c=pd.read_csv(f, usecols=["close"])["close"].values
    if len(c)<3000: continue
    m=len(c)//2
    a_r.append((c[m-1]/c[0]-1)*100); b_r.append((c[-1]/c[m]-1)*100)
print(f"  1.yarı (Ağu 23 – Ağu 30): ortalama parite getirisi {st.mean(a_r):+.2f}%  (medyan {st.median(a_r):+.2f}%)")
print(f"  2.yarı (Ağu 30 – Eyl 6) : ortalama parite getirisi {st.mean(b_r):+.2f}%  (medyan {st.median(b_r):+.2f}%)")
print(f"  → Beklentinin işareti PİYASA YÖNÜNÜ izliyor. Kural değil, BETA.")

print("\n"+"="*88); print("KONTROL 2 — saat profili iki yarıda TUTARLI mı?"); print("="*88)
D=pd.read_csv(os.path.expanduser("~/cmwatch/mine5.csv")); D=D[D["sym"]!="BNB-USDT"]
k=D["i"].quantile(0.5); A,B=D[D["i"]<=k],D[D["i"]>k]
ha=A.groupby("saat")["kazanan"].mean()/A["kazanan"].mean()
hb=B.groupby("saat")["kazanan"].mean()/B["kazanan"].mean()
ort=pd.DataFrame({"1.yarı_kaldırma":ha.round(2),"2.yarı_kaldırma":hb.round(2)})
rho=ha.corr(hb, method="spearman")
print(ort.T.to_string())
print(f"\n  iki yarı arasında SIRA korelasyonu (Spearman) = {rho:.3f}")
print(f"  → {'saat profili TUTARLI' if rho>0.5 else 'saat profili TUTARSIZ — tekrarlamıyor'}")

print("\n"+"="*88); print("KONTROL 3 — CANLI DEFTER: şu anki rejimde (son 100 işlem) ne kaybettiriyor?"); print("="*88)
d=json.load(open(os.path.expanduser("~/cmwatch/runner_0_mexc.json"),encoding="utf-8"))
son100=d["trades"][-100:]
def grup(v, ad, key):
    g=collections.defaultdict(list)
    for t in v: g[key(t)].append(t)
    rows=[]
    for kk,x in g.items():
        n=[float(t["net_pnl"]) for t in x]; p=[float(t.get("net_pct_realized") or 0) for t in x]
        sd=st.pstdev(p) or 1e-9
        rows.append((kk,len(x),sum(n),st.mean(p), st.mean(p)/(sd/math.sqrt(len(x)))))
    rows.sort(key=lambda r:r[2])
    print(f"\n  ### {ad}")
    print(f"  {'':22s} {'n':>4s} {'NET $':>8s} {'ort %':>8s} {'t':>7s}")
    for kk,n,s,m,t_ in rows:
        print(f"  {str(kk)[:22]:22s} {n:4d} {s:+8.3f} {m:+8.4f} {t_:+7.2f}{'  ⚠' if t_< -1.8 else ''}")
grup(son100,"SLEEVE (son 100)",lambda t:t.get("sleeve") or t.get("trigger"))
grup(son100,"EMİR TİPİ (son 100)",lambda t:t.get("order_type"))
grup(son100,"ÇIKIŞ MODU (son 100)",lambda t:t.get("exit_mode"))
