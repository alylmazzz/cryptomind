# -*- coding: utf-8 -*-
import json, collections, statistics as st, math, os
d=json.load(open(os.path.expanduser("~/cmwatch/runner_0_mexc.json"),encoding="utf-8"))
tr=d["trades"]; son=tr[-120:]
def t_stat(p):
    if len(p)<3: return float("nan")
    sd=st.pstdev(p); return float("nan") if sd<1e-12 else st.mean(p)/(sd/math.sqrt(len(p)))

print("="*94)
print("KARIŞTIRICI TESTİ — çıkış modu mu, yoksa onu kullanan sleeve mi kaybettiriyor?")
print("="*94)
g=collections.defaultdict(list)
for t in son: g[(t.get("sleeve") or t.get("trigger"), t.get("exit_mode"))].append(t)
print(f"{'sleeve':20s} {'çıkış modu':18s} {'n':>4s} {'NET $':>8s} {'ort %':>8s} {'t':>7s}")
for (s,m),x in sorted(g.items(), key=lambda kv: sum(float(t['net_pnl']) for t in kv[1])):
    if len(x)<3: continue
    p=[float(t.get("net_pct_realized") or 0) for t in x]
    print(f"{str(s)[:20]:20s} {str(m)[:18]:18s} {len(x):4d} {sum(float(t['net_pnl']) for t in x):+8.3f} {st.mean(p):+8.4f} {t_stat(p):+7.2f}")

print("\n### SLEEVE→ÇIKIŞ MODU eşlemesi tek mi? (karıştırıcı var mı)")
m=collections.defaultdict(set)
for t in son: m[t.get("sleeve") or t.get("trigger")].add(t.get("exit_mode"))
cok=[k for k,v in m.items() if len(v)>1]
print(f"  birden fazla moda giren sleeve: {cok if cok else 'YOK → mod ile sleeve TAM KARIŞIK (ayrıştırılamaz)'}")
for k in cok: print(f"    {k}: {m[k]}")

print("\n### AYNI SLEEVE, FARKLI MOD (varsa doğrudan kıyas)")
for k in cok:
    sub=collections.defaultdict(list)
    for t in son:
        if (t.get("sleeve") or t.get("trigger"))==k: sub[t.get("exit_mode")].append(float(t.get("net_pct_realized") or 0))
    for mm,p in sub.items():
        if len(p)>=3: print(f"  {k:18s} {mm:18s} n={len(p):3d} ort %{st.mean(p):+.4f} t={t_stat(p):+.2f}")

print("\n"+"="*94)
print("TAHSİSÇİ (allocator) — dip_moderate neden hâlâ tam boyutta işlem açıyor?")
print("="*94)
try:
    a=json.load(open(os.path.expanduser("~/cmwatch/allocator_0_mexc.json"),encoding="utf-8"))
    st_=a.get("sleeves") or a.get("state") or a
    for k in ("dip_moderate","obi_momentum","dip","catalyst","donchian_breakout"):
        v=st_.get(k)
        if isinstance(v,dict):
            print(f"  {k:20s} " + " ".join(f"{kk}={v[kk]}" for kk in list(v)[:9] if not isinstance(v[kk],(list,dict))))
        else:
            print(f"  {k:20s} {str(v)[:120]}")
    print(f"\n  allocator anahtarları: {list(a)[:12]}")
except Exception as e:
    print("  okunamadı:", e)
