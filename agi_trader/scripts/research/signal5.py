# -*- coding: utf-8 -*-
"""Sinyal analizi: +%5 öncesi özellikler ayırt ediyor mu? Örtüşme ve rejim dürüstçe ele alınır."""
import pandas as pd, numpy as np, os, math
D = pd.read_csv(os.path.expanduser("~/cmwatch/mine5.csv"))
D = D[D["sym"] != "BNB-USDT"]                      # n=40, veri eksik
FEAT = ["ret_15m","ret_1h","ret_4h","ret_24h","hacim_orani","oynaklik_1h","atr_pct",
        "rsi","aralik_konum","tepe_uzaklik","ema60_sapma","sikisma"]
TABAN = D["kazanan"].mean()
BE = 2/7
print(f"taban oran %{100*TABAN:.2f} · başabaş %{100*BE:.1f} · gereken kaldırma {BE/TABAN:.2f}×")
print(f"karar noktası {len(D):,} · ama ÖRTÜŞME var: 24 sa ufuk / 5 dk adım = her olay ~288 kez sayılıyor")
print(f"→ ETKİN bağımsız örneklem ≈ {len(D)//288:,} (istatistik bunun üstünden okunmalı)\n")

print("="*92)
print("ÖZELLİK BAŞINA DESİL ANALİZİ — en düşük %10 → en yüksek %10 arasında isabet oranı")
print("="*92)
print(f"{'özellik':16s} {'D1':>7s} {'D2':>7s} {'D3':>7s} {'D5':>7s} {'D8':>7s} {'D9':>7s} {'D10':>7s}   {'D10/D1':>7s}")
sonuc = {}
for f in FEAT:
    x = D[f].replace([np.inf,-np.inf], np.nan)
    q = pd.qcut(x, 10, labels=False, duplicates="drop")
    r = D.groupby(q)["kazanan"].mean()
    if len(r) < 10: continue
    sonuc[f] = r
    print(f"{f:16s} " + " ".join(f"{100*r.get(i,np.nan):7.1f}" for i in [0,1,2,4,7,8,9]) +
          f"   {r.get(9,np.nan)/max(1e-9,r.get(0,np.nan)):7.2f}")

print("\n" + "="*92)
print("SAAT (UTC) — hangi seansta +%5 daha sık?")
print("="*92)
g = D.groupby("saat").agg(n=("kazanan","size"), isabet=("kazanan","mean"),
                          stop=("sonuc", lambda s: (s=="stop").mean()))
g["isabet%"] = 100*g["isabet"]; g["stop%"] = 100*g["stop"]
g["kaldirma"] = g["isabet"]/TABAN
print(g[["n","isabet%","stop%","kaldirma"]].round(3).to_string())
en = g.sort_values("isabet%", ascending=False)
print(f"\n  EN İYİ 4 saat: {list(en.index[:4])} → isabet %{en['isabet%'].iloc[:4].mean():.1f}")
print(f"  EN KÖTÜ 4 saat: {list(en.index[-4:])} → isabet %{en['isabet%'].iloc[-4:].mean():.1f}")
print(f"  hiçbiri başabaşın (%{100*BE:.1f}) üstünde mi? "
      f"{'EVET: ' + str([int(h) for h in en.index[en['isabet%']>100*BE]]) if (en['isabet%']>100*BE).any() else 'HAYIR'}")

print("\n" + "="*92)
print("ZAMANDA BÖLÜNMÜŞ TEST — ilk 7 gün EĞİTİM, son 7 gün TEST (uydurma freni)")
print("="*92)
D = D.sort_values(["sym","i"])
kesim = D["i"].quantile(0.5)
tr_, te_ = D[D["i"] <= kesim], D[D["i"] > kesim]
print(f"eğitim {len(tr_):,} · test {len(te_):,} · eğitim taban %{100*tr_['kazanan'].mean():.2f} · test taban %{100*te_['kazanan'].mean():.2f}")

def kural_test(ad, maske_fn):
    a = tr_[maske_fn(tr_)]; b = te_[maske_fn(te_)]
    if len(a) < 200 or len(b) < 200:
        print(f"{ad:44s} örneklem yetersiz (eğitim {len(a)}, test {len(b)})"); return
    ia, ib = a["kazanan"].mean(), b["kazanan"].mean()
    # örtüşme düzeltilmiş standart hata: etkin n = n/288
    ne = max(1, len(b)//288)
    se = math.sqrt(ib*(1-ib)/ne)
    print(f"{ad:44s} eğitim %{100*ia:5.1f} → TEST %{100*ib:5.1f}  (n={len(b):6,d}, etkin {ne:3d}, ±%{100*1.96*se:4.1f})  "
          f"{'BAŞABAŞ ÜSTÜ' if ib > BE else ''}")

print()
kural_test("hacim patlaması (üst %10)",        lambda d: d["hacim_orani"] >= d["hacim_orani"].quantile(0.90))
kural_test("hacim üst %10 + 1sa getiri > 0",   lambda d: (d["hacim_orani"]>=d["hacim_orani"].quantile(0.90)) & (d["ret_1h"]>0))
kural_test("sıkışma (alt %20) = düşük aralık", lambda d: d["sikisma"] <= d["sikisma"].quantile(0.20))
kural_test("24sa aralığın üst %20'sinde",      lambda d: d["aralik_konum"] >= 0.8)
kural_test("24sa aralığın alt %20'sinde",      lambda d: d["aralik_konum"] <= 0.2)
kural_test("yüksek ATR (üst %20)",             lambda d: d["atr_pct"] >= d["atr_pct"].quantile(0.80))
kural_test("yüksek ATR + hacim üst %20",       lambda d: (d["atr_pct"]>=d["atr_pct"].quantile(0.80)) & (d["hacim_orani"]>=d["hacim_orani"].quantile(0.80)))
kural_test("RSI < 30 (aşırı satım)",           lambda d: d["rsi"] < 30)
kural_test("RSI > 70 (momentum)",              lambda d: d["rsi"] > 70)
kural_test("4sa getiri üst %10 (momentum)",    lambda d: d["ret_4h"] >= d["ret_4h"].quantile(0.90))
kural_test("24sa getiri alt %10 (dip)",        lambda d: d["ret_24h"] <= d["ret_24h"].quantile(0.10))
kural_test("EMA60 üstü + hacim üst %20",       lambda d: (d["ema60_sapma"]>0) & (d["hacim_orani"]>=d["hacim_orani"].quantile(0.80)))
