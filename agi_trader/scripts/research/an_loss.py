import json, collections, statistics as st, time, pathlib
d = json.load(open(pathlib.Path.home()/"cmwatch/runner_0_mexc.json", encoding="utf-8"))
tr = d["trades"]; cfg = d["config"]
N = lambda v: sum(float(t.get("net_pnl") or 0) for t in v)

print("="*88)
print(f"DEFTER: {len(tr)} işlem · {time.strftime('%m-%d %H:%M', time.gmtime(tr[0]['opened_ts']))} → "
      f"{time.strftime('%m-%d %H:%M', time.gmtime(tr[-1]['closed_ts']))} UTC")
net=[float(t['net_pnl']) for t in tr]; gross=[float(t['gross_pnl']) for t in tr]; fee=[float(t['fees']) for t in tr]
print(f"NET {sum(net):+.3f} $  =  BRÜT {sum(gross):+.3f} $  −  KOMİSYON {sum(fee):.3f} $")
print("="*88)

def blok(v, ad, key, minn=1):
    g = collections.defaultdict(list)
    for t in v: g[key(t)].append(t)
    rows=[]
    for k, x in g.items():
        if len(x) < minn: continue
        n=[float(t['net_pnl']) for t in x]
        rows.append((k, len(x), sum(n), st.mean(n), 100*sum(1 for y in n if y>0)/len(n),
                     st.mean([float(t.get('net_pct_realized') or 0) for t in x])))
    rows.sort(key=lambda r: r[2])
    print(f"\n### {ad}")
    print(f"{'':26s} {'n':>4s} {'NET $':>9s} {'ort $':>8s} {'kazanma%':>9s} {'ort %':>8s}")
    for k,n,s,m,w,p in rows:
        print(f"{str(k)[:26]:26s} {n:4d} {s:+9.3f} {m:+8.4f} {w:9.1f} {p:+8.4f}")

blok(tr, "ÇIKIŞ SEBEBİNE GÖRE", lambda t: t.get('reason'))
blok(tr, "SLEEVE'E GÖRE (n≥3)", lambda t: t.get('sleeve') or t.get('trigger'), minn=3)
blok(tr, "PARİTEYE GÖRE (n≥3)", lambda t: t.get('symbol'), minn=3)
blok(tr, "EMİR TİPİNE GÖRE", lambda t: t.get('order_type'))
blok(tr, "YÖNE GÖRE", lambda t: t.get('direction'))
blok(tr, "TUTMA SÜRESİNE GÖRE", lambda t: t.get('hold_bucket'))
blok(tr, "ÇIKIŞ MODUNA GÖRE", lambda t: t.get('exit_mode'))

# ---- SEANS (UTC 4 saatlik blok + saat) ----
print("\n### SEANS — UTC 4 SAATLİK BLOK")
def blok4(ts): 
    h=time.gmtime(ts).tm_hour; return f"{h//4*4:02d}-{h//4*4+4:02d}"
blok(tr, "  (açılış saatine göre)", lambda t: blok4(t['opened_ts']))

print("\n### SEANS — UTC SAAT (açılış)")
g=collections.defaultdict(list)
for t in tr: g[time.gmtime(t['opened_ts']).tm_hour].append(t)
print(f"{'saat':>5s} {'n':>4s} {'NET $':>9s} {'ort %':>8s} {'kazanma%':>9s}")
for h in sorted(g):
    x=g[h]; n=[float(t['net_pnl']) for t in x]
    print(f"{h:5d} {len(x):4d} {sum(n):+9.3f} {st.mean([float(t.get('net_pct_realized') or 0) for t in x]):+8.4f} "
          f"{100*sum(1 for y in n if y>0)/len(n):9.1f}")

# ---- EN BÜYÜK 15 KAYIP ----
print("\n### EN BÜYÜK 15 KAYIP")
worst = sorted(tr, key=lambda t: float(t['net_pnl']))[:15]
print(f"{'parite':12s} {'sebep':13s} {'sleeve':18s} {'net$':>8s} {'net%':>8s} {'tepe%':>7s} {'notional':>9s} {'dk':>5s}")
for t in worst:
    print(f"{t['symbol']:12s} {str(t['reason'])[:13]:13s} {str(t.get('sleeve'))[:18]:18s} "
          f"{float(t['net_pnl']):+8.3f} {float(t.get('net_pct_realized') or 0):+8.3f} "
          f"{float(t.get('peak_net_pct') or 0):+7.3f} {float(t.get('notional') or 0):9.2f} {float(t['hold_sec'])/60:5.0f}")
print(f"\n  → bu 15 işlem toplam {N(worst):+.3f} $ (defterin %{100*N(worst)/sum(net):.0f}'i)")

# ---- KOMİSYON YÜKÜ ----
print("\n### KOMİSYON")
print(f"  toplam {sum(fee):.3f} $ · işlem başına {st.mean(fee):.4f} $ · brüt hareketin %{100*sum(fee)/max(1e-9,sum(abs(g) for g in gross)):.1f}'i")
byot=collections.defaultdict(list)
for t in tr: byot[t.get('order_type')].append(float(t['fees'])/max(1e-9,float(t['notional']))*100)
for k,v in byot.items(): print(f"  {k}: ort %{st.mean(v):.4f} gidiş-dönüş (n={len(v)})")
