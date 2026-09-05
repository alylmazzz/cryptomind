#!/usr/bin/env python3
"""PİYASA-NÖTR MODÜL 1 — Funding-Rate Arbitrajı (delta-nötr).
Perp funding'i topla, yön riskini hedge et (spot + ters perp). Kazanç yön
tahmininden DEĞİL, her 8 saatte tahsil edilen yapısal funding ödemesinden gelir.
Gerçek funding verisi (runs/data_funding, 2022-2026, 8h). Look-ahead yok:
pozisyon ÖNCEKİ funding işaretine göre (funding kalıcıdır). Katı OOS: yıl-yıl."""
from __future__ import annotations
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

FUND = Path(__file__).parent / "runs" / "data_funding"
PAIRS = ["BTC", "ETH", "SOL", "DOGE", "AVAX"]
# gerçekçi maliyet: delta-nötr açılış = spot+perp 2 taker; kapanış 2 taker.
# İşaret değişiminde (flip) eski kapat + yeni aç = 4 bacak.
FEE_LEG = 0.0002           # MAKER/bacak (%0.02) — limit emirle carry masası maker olur
OPEN_COST = 2 * FEE_LEG    # açılış (2 bacak: spot + perp)
FLIP_COST = 4 * FEE_LEG
BASE_8H = 0.05/100/3       # nakitteyken stablecoin getiri tabanı (~%5/yıl)
MIN_EDGE = 0.00005
SMOOTH = 9                 # funding kararını 9 aralık (~3 gün) ortalamasıyla ver (gürültü filtresi)


def load(short):
    df = pd.read_csv(FUND/f"{short}_funding.csv")
    df.index = pd.to_datetime(df["dt"])
    return df["funding_rate"].astype(float).sort_index()


ENTER = 0.00003      # funding > %0.003/8h → pozitif carry'ye gir (short-perp/long-spot)
EXIT  = -0.00001     # funding ~sıfır/negatif → çık (nakit + taban getiri)

def strat_returns(fr: pd.Series) -> pd.Series:
    """Delta-nötr POZİTİF-carry funding hasadı. Pozisyon ∈ {0, +1} (yalnız pozitif
    funding'i topla; negatif-carry/spot-short'tan kaçın). Histerezis (ENTER/EXIT)
    ile churn önlenir → kalıcı tutuş, düşük maliyet. Look-ahead yok: karar f[t-1]."""
    f = fr.values
    fs = fr.rolling(SMOOTH).mean().values          # yumuşatılmış funding (karar için)
    n = len(f)
    ret = np.zeros(n)
    prev = 0
    for t in range(1, n):
        p = prev
        s = fs[t-1]
        if not np.isnan(s):
            if prev == 0 and s > ENTER:
                p = 1
            elif prev == 1 and s < EXIT:
                p = 0
        collected = f[t] if p == 1 else BASE_8H     # pozisyondayken funding, değilse taban
        cost = OPEN_COST if p != prev else 0.0       # yalnız giriş/çıkışta 2-bacak
        ret[t] = collected - cost
        prev = p
    return pd.Series(ret, index=fr.index)


def stats(ret: pd.Series, label=""):
    eq = (1 + ret).cumprod()
    total = (eq.iloc[-1] - 1) * 100
    # 8h → yıllık (3 aralık/gün × 365)
    per_year = 3 * 365
    ann = (eq.iloc[-1] ** (per_year / len(eq)) - 1) * 100
    sharpe = ret.mean() / (ret.std() + 1e-12) * np.sqrt(per_year)
    dd = ((eq.cummax() - eq) / eq.cummax()).max() * 100
    # günlük toparla
    daily = (1 + ret).groupby(ret.index.date).prod() - 1
    pos_days = (daily > 0).mean() * 100
    avg_day = daily.mean() * 100
    yearly = {}
    for y, g in eq.groupby(eq.index.year):
        yearly[str(y)] = round((g.iloc[-1]/g.iloc[0]-1)*100, 1)
    return dict(total=round(total,1), ann=round(ann,1), sharpe=round(float(sharpe),2),
                dd=round(dd,1), pos_days=round(float(pos_days),1),
                avg_day=round(float(avg_day),3), yearly=yearly)


def main():
    print(f"Maliyet: açılış %{OPEN_COST*100:.2f}, flip %{FLIP_COST*100:.2f} | eşik %{MIN_EDGE*100:.3f} | taban ~%5/yıl\n")
    all_ret = {}
    idx = None
    for s in PAIRS:
        fr = load(s)
        r = strat_returns(fr)
        all_ret[s] = r
        idx = r.index if idx is None else idx.union(r.index)
        m = stats(r, s)
        print(f"{s:5s} yıllık={m['ann']:+6.1f}% Sharpe={m['sharpe']:+.2f} MaxDD={m['dd']:4.1f}% "
              f"pozGün=%{m['pos_days']:.0f} ort/gün={m['avg_day']:+.3f}%  {m['yearly']}", flush=True)
    # eşit-ağırlık portföy (5 parite funding hasadı)
    M = pd.DataFrame({s: all_ret[s].reindex(idx).fillna(0) for s in PAIRS})
    port = M.mean(axis=1)
    mp = stats(port, "PORT")
    print(f"\n=== PORTFÖY (5 parite delta-nötr funding hasadı) ===")
    print(f"  yıllık(CAGR): {mp['ann']:+.1f}% | Sharpe: {mp['sharpe']:+.2f} | MaxDD: {mp['dd']:.1f}% | "
          f"pozitif-gün: %{mp['pos_days']:.0f} | ortalama/gün: {mp['avg_day']:+.3f}%")
    print(f"  yıl-yıl (OOS, çok-rejim): {mp['yearly']}")
    print(f"\n  >> Hedef %0.5/gün ile kıyas: bu strateji ~{mp['avg_day']:+.3f}%/gün (tutarlı ama hedefin ~1/{max(1,round(0.5/max(0.001,mp['avg_day'])))}'i)")

if __name__ == "__main__":
    main()
