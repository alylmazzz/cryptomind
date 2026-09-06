import json, collections, statistics as st, time, pathlib
d = json.load(open(pathlib.Path.home()/"cmwatch/runner_0_mexc.json", encoding="utf-8"))
tr = d["trades"]

print("### GÜNE GÖRE (ve notional rejimi)")
g=collections.defaultdict(list)
for t in tr: g[time.strftime('%m-%d', time.gmtime(t['closed_ts']))].append(t)
print(f"{'gün':>6s} {'n':>4s} {'NET $':>9s} {'BRÜT $':>9s} {'kom $':>7s} {'ort notional':>13s} {'EA n':>5s} {'EA $':>8s}")
for k in sorted(g):
    x=g[k]; n=[float(t['net_pnl']) for t in x]; gr=[float(t['gross_pnl']) for t in x]
    f=[float(t['fees']) for t in x]; no=[float(t['notional']) for t in x]
    ea=[t for t in x if t['reason']=='EARLY_ABORT']
    print(f"{k:>6s} {len(x):4d} {sum(n):+9.3f} {sum(gr):+9.3f} {sum(f):7.3f} {st.mean(no):13.2f} "
          f"{len(ea):5d} {sum(float(t['net_pnl']) for t in ea):+8.3f}")

print("\n### EARLY_ABORT ayrıntı — büyük mü küçük mü pozisyonlarda?")
ea=[t for t in tr if t['reason']=='EARLY_ABORT']
buyuk=[t for t in ea if float(t['notional'])>30]; kucuk=[t for t in ea if float(t['notional'])<=30]
for ad,v in (("notional > 30 $", buyuk), ("notional ≤ 30 $", kucuk)):
    if not v: continue
    n=[float(t['net_pnl']) for t in v]
    print(f"  {ad:16s} n={len(v):3d}  net {sum(n):+7.3f} $  ort {st.mean(n):+.4f} $  "
          f"ort %{st.mean([float(t['net_pct_realized']) for t in v]):+.3f}  "
          f"ort notional {st.mean([float(t['notional']) for t in v]):.1f}")
print(f"  EARLY_ABORT'un {sum(1 for t in ea if t['closed_ts']<1788566400)}/{len(ea)}'i 09-05 ÖNCESİ")

print("\n### ŞU ANKİ REJİM (son 100 işlem — hepsi ≤25 $ kanıt tavanında)")
son=tr[-100:]
n=[float(t['net_pnl']) for t in son]; gr=[float(t['gross_pnl']) for t in son]; f=[float(t['fees']) for t in son]
print(f"  n={len(son)}  NET {sum(n):+.3f} $  BRÜT {sum(gr):+.3f} $  KOMİSYON {sum(f):.3f} $")
print(f"  ort notional {st.mean([float(t['notional']) for t in son]):.2f} $")
by=collections.defaultdict(list)
for t in son: by[t['reason']].append(float(t['net_pnl']))
print("  sebep:", {k: f"{len(v)}·{sum(v):+.2f}$" for k,v in sorted(by.items(), key=lambda kv:sum(kv[1]))})
by=collections.defaultdict(list)
for t in son: by[t.get('order_type')].append(float(t['net_pnl']))
print("  emir :", {k: f"{len(v)}·{sum(v):+.2f}$" for k,v in by.items()})

print("\n### KOMİSYON / BRÜT DENGESİ — asıl soru")
w=[t for t in tr if float(t['gross_pnl'])>0]; l=[t for t in tr if float(t['gross_pnl'])<=0]
print(f"  BRÜT kazanan {len(w)} işlem: +{sum(float(t['gross_pnl']) for t in w):.3f} $")
print(f"  BRÜT kaybeden {len(l)} işlem: {sum(float(t['gross_pnl']) for t in l):.3f} $")
print(f"  → brüt beklenti {st.mean([float(t['pnl_pct']) for t in tr]):+.4f} %/işlem")
print(f"  → maliyet       {st.mean([float(t['fees'])/max(1e-9,float(t['notional']))*100 for t in tr]):.4f} %/işlem")
print(f"  KARAR: brüt beklenti maliyeti {'KARŞILIYOR' if st.mean([float(t['pnl_pct']) for t in tr])>st.mean([float(t['fees'])/max(1e-9,float(t['notional']))*100 for t in tr]) else 'KARŞILAMIYOR'}")
