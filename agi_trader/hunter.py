#!/usr/bin/env python3
"""%1 GÜNLÜK-HEDEF AVCISI — 8-katman spesifikasyonunun çekirdek entegratörü.
Her gün: (2) öngörü [trend+momentum+rejim] + (3) işlem-bulma [beklenen-vol≥%1
filtresi + ortak-karar kapısı] → nitelikli varlıklarda gün-içi +%1 hedefli işlem;
fırsat yoksa NAKİT (5) risk [sıkı stop]. Gerçek saatlik veri (data_full, 4,5 yıl).
Amaç: 'günde %1 yakalanabilir mi, ne sıklıkla?' sorusunu DÜRÜSTÇE ölçmek."""
from __future__ import annotations
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

DATA = Path(__file__).parent / "runs" / "data_full"
PAIRS = ["BTCUSDT","ETHUSDT","SOLUSDT","DOGEUSDT","AVAXUSDT"]
TARGET = 0.01          # gün-içi +%1 hedef
STOP = 0.007           # −%0,7 stop (R/R ~1,43)
COST = 0.001           # giriş+çıkış round-trip (~%0,1)
MIN_EVOL = 0.012       # beklenen günlük vol ≥ %1,2 değilse AVLAMA (%1 matematiksel ön-koşul)
START = "2022-01-01"


def load_hourly(short):
    df = pd.read_csv(DATA/f"{short}_1h.csv"); df.index = pd.to_datetime(df["dt"])
    return df[["open","high","low","close"]].astype(float)


def signals_daily(h):
    """Günlük öngörü sinyalleri (look-ahead yok: karar önceki günle)."""
    d = h["close"].resample("1D").last().dropna()
    evol = d.pct_change().rolling(20).std()          # beklenen günlük hareket
    trend = d > d.rolling(50).mean()                 # trend yukarı
    mom = d.pct_change(5) > 0                         # kısa momentum
    breakout = d > d.rolling(20).max().shift(1)*0.995
    # ortak-karar: trend VE momentum VE yeterli vol
    qualify = (trend & mom & (evol >= MIN_EVOL)).shift(1).fillna(False)  # dünkü sinyalle bugün avla
    strength = (mom.astype(int) + trend.astype(int) + breakout.astype(int)).shift(1).fillna(0)
    return pd.DataFrame({"qualify": qualify, "strength": strength, "evol": evol.shift(1)}, index=d.index)


def hunt_day(day_bars, entry):
    """Gün-içi: +%1 hedef mi, −%0,7 stop mu, gün-sonu mu? (stop önce, konservatif)."""
    for _, row in day_bars.iterrows():
        hi, lo = row["high"], row["low"]
        if lo <= entry*(1-STOP):    # stop önce (konservatif)
            return -STOP - COST
        if hi >= entry*(1+TARGET):  # hedef
            return TARGET - COST
    close = day_bars["close"].iloc[-1]
    return (close/entry - 1) - COST   # gün-sonu kapanış


def main():
    print(f"Av parametreleri: hedef +%{TARGET*100:.0f}, stop −%{STOP*100:.1f}, min beklenen-vol %{MIN_EVOL*100:.1f}\n", flush=True)
    hs = {s: load_hourly(s) for s in PAIRS}
    sigs = {s: signals_daily(hs[s]) for s in PAIRS}
    # ortak gün indeksi
    days = None
    for s in PAIRS:
        dd = pd.Index(sorted(set(hs[s].index.date)))
        days = dd if days is None else days.union(dd)
    days = [d for d in days if pd.Timestamp(d) >= pd.Timestamp(START)]

    daily_ret = []; trade_days = 0; hits = 0; n_trades = 0
    for d in days:
        ts = pd.Timestamp(d)
        picks = [s for s in PAIRS if s in sigs and ts in sigs[s].index and bool(sigs[s].loc[ts,"qualify"])]
        if not picks:
            daily_ret.append((d, 0.0)); continue    # fırsat yok → NAKİT
        rets = []
        for s in picks:
            db = hs[s][hs[s].index.date == d]
            if len(db) < 2: continue
            r = hunt_day(db, db["open"].iloc[0])
            rets.append(r); n_trades += 1
            if r >= TARGET - COST - 1e-9: hits += 1
        if rets:
            daily_ret.append((d, float(np.mean(rets)))); trade_days += 1
        else:
            daily_ret.append((d, 0.0))

    dr = pd.Series({pd.Timestamp(d): r for d, r in daily_ret}).sort_index()
    eq = (1+dr).cumprod()
    total = (eq.iloc[-1]-1)*100
    cagr = (eq.iloc[-1]**(365/len(eq))-1)*100
    dd = ((eq.cummax()-eq)/eq.cummax()).max()*100
    sh = dr.mean()/(dr.std()+1e-12)*np.sqrt(365)
    pos_days = (dr > 0).mean()*100
    ge1 = (dr >= 0.01).mean()*100     # portföyün +%1 yaptığı gün oranı
    yr = {str(y): round((g.iloc[-1]/g.iloc[0]-1)*100,1) for y,g in eq.groupby(eq.index.year)}

    print("=== AV İSTATİSTİKLERİ (4,5 yıl, gerçek saatlik veri) ===")
    print(f"  Toplam gün: {len(dr)} | işlem-günü: {trade_days} (%{trade_days/len(dr)*100:.0f}) | nakit-günü: %{(1-trade_days/len(dr))*100:.0f}")
    print(f"  İşlem-günü İSABET (+%1 hedefe ulaşma): %{hits/max(1,n_trades)*100:.1f} ({hits}/{n_trades})")
    print(f"  Portföyün +%1'i yakaladığı gün oranı: %{ge1:.1f}")
    print(f"  Ortalama/gün: {dr.mean()*100:+.3f}% | işlem-günü ort: {dr[dr!=0].mean()*100:+.3f}%")
    print(f"\n=== GETİRİ ===")
    print(f"  Toplam(4,5y): {total:+.1f}% | CAGR: {cagr:+.1f}% | Sharpe: {sh:+.2f} | MaxDD: {dd:.1f}% | pozitif-gün: %{pos_days:.0f}")
    print(f"  yıl-yıl: {yr}")
    # son 6 ay
    l6 = dr[dr.index >= pd.Timestamp('2026-01-01')]
    if len(l6) > 5:
        e6 = (1+l6).cumprod()
        print(f"\n  SON 6 AY (2026 H1): {(e6.iloc[-1]-1)*100:+.2f}% | +%1-gün oranı %{(l6>=0.01).mean()*100:.0f}")
    print(f"\n  >> Hedef %0.5–1/gün ile: ortalama {dr.mean()*100:+.3f}%/gün. Avcı, %1'i")
    print(f"     yalnız FIRSAT olan günlerde (yüksek-vol+trend) yakalar; günlerin ~%{(1-trade_days/len(dr))*100:.0f}'i nakit.")

if __name__ == "__main__":
    main()
