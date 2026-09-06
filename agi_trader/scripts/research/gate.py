# -*- coding: utf-8 -*-
import json, os, statistics as st, math, collections
d=json.load(open(os.path.expanduser("~/cmwatch/runner_0_mexc.json"),encoding="utf-8"))
tr=d["trades"]
def t_(p):
    if len(p)<4: return float("nan")
    sd=st.pstdev(p); return float("nan") if sd<1e-12 else st.mean(p)/(sd/math.sqrt(len(p)))
for t in tr:
    t["_cost"]=float(t.get("cost_pct_roundtrip") or (float(t.get("peak_pnl_pct") or 0)-float(t.get("peak_net_pct") or 0)) or 0.07)
    t["_tgt"]=float(t.get("target_pct") or 0) or ((float(t["target"])/float(t["entry"])-1)*100 if t.get("target") else 0)
    t["_oran"]=t["_tgt"]/max(1e-9,t["_cost"])
    t["_p"]=float(t.get("net_pct_realized") or 0)

print("="*86)
print("HEDEF/MALİYET ORANINA GÖRE SONUÇ — maliyet kapısını sıkmak işe yarar mı?")
print("(mevcut komite kapısı: brüt hedef / maliyet ≥ 2,0)")
print("="*86)
kova=[(0,10),(10,20),(20,30),(30,45),(45,1e9)]
print(f"{'hedef/maliyet':>16s} {'n':>4s} {'NET $':>8s} {'ort %':>8s} {'t':>7s} {'kazanma%':>9s} {'ort tepe%':>10s}")
for a,b in kova:
    x=[t for t in tr if a<=t["_oran"]<b]
    if len(x)<4: continue
    lbl = f"{a}-{b}" if b<1e9 else f"{a}+"
    p=[t["_p"] for t in x]
    print(f"{lbl:>16s} {len(x):4d} {sum(float(t['net_pnl']) for t in x):+8.3f} "
          f"{st.mean(p):+8.4f} {t_(p):+7.2f} {100*sum(1 for t in x if float(t['net_pnl'])>0)/len(x):9.1f} "
          f"{st.mean([float(t.get('peak_net_pct') or 0) for t in x]):+10.3f}")

print("\n"+"="*86)
print("STOP MESAFESİNE GÖRE (dar stop = daha çok stop yeme?)")
print("="*86)
for a,b in [(0,0.6),(0.6,1.0),(1.0,1.5),(1.5,2.5),(2.5,99)]:
    x=[t for t in tr if a<=float(t.get("stop_pct") or (100/max(1,float(t.get("notional") or 100))))<b]
    if len(x)<4: continue
    p=[t["_p"] for t in x]
    print(f"  stop %{a}-{b}: n={len(x):3d} net {sum(float(t['net_pnl']) for t in x):+7.3f}$ ort {st.mean(p):+.4f}% t={t_(p):+.2f}")

print("\n"+"="*86)
print("POZİTİF ALT KÜME VAR MI? (catalyst + donchian_breakout = tek pozitif sleeve'ler)")
print("="*86)
iyi=[t for t in tr if (t.get("sleeve") or t.get("trigger")) in ("catalyst","donchian_breakout")]
p=[t["_p"] for t in iyi]
print(f"  n={len(iyi)} net {sum(float(t['net_pnl']) for t in iyi):+.3f}$ ort {st.mean(p):+.4f}% t={t_(p):+.2f} "
      f"kazanma %{100*sum(1 for t in iyi if float(t['net_pnl'])>0)/len(iyi):.1f}")
print(f"  → t {'> 2 (ANLAMLI)' if t_(p)>2 else '< 2 → henüz KANIT DEĞİL, örneklem yetersiz'}")
dp=[t for t in tr if t.get("exit_mode")=="DYNAMIC_PEAK"]
p=[t["_p"] for t in dp]
print(f"  DYNAMIC_PEAK tümü: n={len(dp)} net {sum(float(t['net_pnl']) for t in dp):+.3f}$ ort {st.mean(p):+.4f}% t={t_(p):+.2f}")

print("\n"+"="*86)
print("GÜNLÜK İŞLEM SAYISI × MALİYET — bleed'in aritmetiği")
print("="*86)
g=collections.defaultdict(list)
import time
for t in tr: g[time.strftime('%m-%d',time.gmtime(t['closed_ts']))].append(t)
for k in sorted(g):
    x=g[k]; f=sum(float(t['fees']) for t in x); n=sum(float(t['net_pnl']) for t in x); gr=sum(float(t['gross_pnl']) for t in x)
    print(f"  {k}: {len(x):3d} işlem · brüt {gr:+7.3f}$ · komisyon {f:6.3f}$ · net {n:+7.3f}$ "
          f"→ komisyon net zararın %{100*f/max(1e-9,abs(n)):.0f}'i")
