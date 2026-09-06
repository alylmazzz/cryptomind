# -*- coding: utf-8 -*-
"""DOĞRU karşılaştırma: kuralı KENDİ döneminin taban oranıyla kıyasla (kaldırma).
   Ayrıca +%5 hedefi SABİTken en iyi stop nedir — taranır."""
import pandas as pd, numpy as np, os, math, glob, re, collections
D = pd.read_csv(os.path.expanduser("~/cmwatch/mine5.csv"))
D = D[D["sym"] != "BNB-USDT"].sort_values(["sym","i"])
kesim = D["i"].quantile(0.5)
A, B = D[D["i"]<=kesim], D[D["i"]>kesim]
bA, bB = A["kazanan"].mean(), B["kazanan"].mean()
BE = 2/7
print("="*96)
print(f"KALDIRMA = kural isabeti / O DÖNEMİN taban oranı   (taban: 1.yarı %{100*bA:.2f} · 2.yarı %{100*bB:.2f})")
print("Bir kuralın işe yaraması için HER İKİ yarıda da kaldırma > 1,4 olmalı (başabaş/taban ≈ 1,4)")
print("="*96)
print(f"{'kural':40s} {'1.yarı':>16s} {'2.yarı':>16s}   {'karar':>10s}")
def test(ad, fn):
    a, b = A[fn(A)], B[fn(B)]
    if len(a)<300 or len(b)<300: print(f"{ad:40s} örneklem yetersiz"); return
    la, lb = a["kazanan"].mean()/bA, b["kazanan"].mean()/bB
    ok = (la>1.4 and lb>1.4)
    print(f"{ad:40s} %{100*a['kazanan'].mean():5.1f} ({la:4.2f}×) %{100*b['kazanan'].mean():5.1f} ({lb:4.2f}×)   "
          f"{'✅ TUTARLI' if ok else ('~' if min(la,lb)>1.15 else '❌')}")
test("hacim patlaması (üst %10)",       lambda d: d["hacim_orani"]>=d["hacim_orani"].quantile(0.90))
test("hacim üst %10 + 1sa getiri > 0",  lambda d: (d["hacim_orani"]>=d["hacim_orani"].quantile(0.90))&(d["ret_1h"]>0))
test("sıkışma (alt %20)",               lambda d: d["sikisma"]<=d["sikisma"].quantile(0.20))
test("aralık üst %20",                  lambda d: d["aralik_konum"]>=0.8)
test("aralık alt %20",                  lambda d: d["aralik_konum"]<=0.2)
test("yüksek ATR (üst %20)",            lambda d: d["atr_pct"]>=d["atr_pct"].quantile(0.80))
test("yüksek ATR + hacim üst %20",      lambda d: (d["atr_pct"]>=d["atr_pct"].quantile(0.80))&(d["hacim_orani"]>=d["hacim_orani"].quantile(0.80)))
test("4sa getiri üst %10",              lambda d: d["ret_4h"]>=d["ret_4h"].quantile(0.90))
test("24sa getiri alt %10 (dip)",       lambda d: d["ret_24h"]<=d["ret_24h"].quantile(0.10))
test("24sa getiri üst %10",             lambda d: d["ret_24h"]>=d["ret_24h"].quantile(0.90))
test("oynaklık üst %20",                lambda d: d["oynaklik_1h"]>=d["oynaklik_1h"].quantile(0.80))
test("tepeye uzaklık üst %20",          lambda d: d["tepe_uzaklik"]>=d["tepe_uzaklik"].quantile(0.80))
test("EN İYİ SAATLER (17-23 UTC)",      lambda d: d["saat"].between(17,23))
test("saat 17-23 + hacim üst %20",      lambda d: d["saat"].between(17,23)&(d["hacim_orani"]>=d["hacim_orani"].quantile(0.80)))
test("saat 17-23 + ATR üst %20",        lambda d: d["saat"].between(17,23)&(d["atr_pct"]>=d["atr_pct"].quantile(0.80)))

# ---- STOP TARAMASI: +%5 hedef SABİT, stop değişken ----
print("\n"+"="*96)
print("STOP TARAMASI — hedef +%5 SABİT. Hangi stop mesafesi beklentiyi en iyi yapar?")
print("(maliyet %0,08 gidiş-dönüş düşülür · ilk-geçiş, 24 sa ufuk, aynı bar → kötümser)")
print("="*96)
HIST = os.path.expanduser("~/cmbench/runs/history")
def son():
    d={}
    for f in glob.glob(f"{HIST}/*_1m_*.csv"):
        m=re.match(r".*/mexc_(.+?)_1m_(\d+)_(\d+)\.csv", f)
        if m and (m.group(1) not in d or int(m.group(3))>d[m.group(1)][1]): d[m.group(1)]=(f,int(m.group(3)))
    return {k:v[0] for k,v in d.items()}
TP, H, STEP, MAL = 0.05, 1440, 15, 0.08
files = son(); res = collections.defaultdict(lambda: collections.defaultdict(list))
for sym,f in sorted(files.items()):
    if sym=="BNB-USDT": continue
    df = pd.read_csv(f, usecols=["ts","high","low","close"], parse_dates=["ts"])
    if len(df)<3000: continue
    h,l,c = df["high"].values, df["low"].values, df["close"].values
    yari = len(c)//2
    for i in range(1440, len(c)-H-1, STEP):
        e=c[i]; up=e*(1+TP); j0,j1=i+1,i+1+H
        hh,ll = h[j0:j1], l[j0:j1]
        iu = np.argmax(hh>=up) if (hh>=up).any() else 10**9
        d2 = "1.yarı" if i<=yari else "2.yarı"
        for SL in (0.01,0.015,0.02,0.03,0.04,0.05):
            dn=e*(1-SL)
            idn = np.argmax(ll<=dn) if (ll<=dn).any() else 10**9
            if iu==10**9 and idn==10**9: r=(c[j1-1]/e-1)*100
            elif iu<idn: r=TP*100
            else: r=-SL*100
            res[SL][d2].append(r)
print(f"{'stop':>6s} {'başabaş isabet':>15s} | {'1.yarı beklenti':>16s} {'n':>7s} | {'2.yarı beklenti':>16s} {'n':>7s} | karar")
for SL in (0.01,0.015,0.02,0.03,0.04,0.05):
    be = 100*SL/(TP+SL)
    a=np.array(res[SL]["1.yarı"]); b=np.array(res[SL]["2.yarı"])
    ea, eb = a.mean()-MAL, b.mean()-MAL
    print(f"%{100*SL:5.1f} {be:14.1f}% | {ea:+15.4f}% {len(a):7,d} | {eb:+15.4f}% {len(b):7,d} | "
          f"{'✅ İKİ YARIDA DA +' if (ea>0 and eb>0) else ('yalnız 2.yarı' if eb>0 else 'HAYIR')}")
