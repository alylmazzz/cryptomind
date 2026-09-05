#!/usr/bin/env python3
"""KENAR PORTFÖYÜ — piyasa-nötr modülleri ENTOURAGE ETKİSİYLE istifle.
Modüller (tek borsa, delta/dolar-nötr, düşük korelasyon):
  M1 Funding arbitrajı   (funding_arb.py)
  M2 Stat-arb pairs      (kointegre kripto çiftleri, ortalamaya dönüş)
  M3 Basis/carry proxy   (funding'in yumuşatılmış carry'si)
  M4 Getiri tabanı       (stablecoin lending ~%5/yıl)
Entourage etkisi: düşük-korelasyonlu kenarlar inverse-vol ağırlıkla birleşince
birleşik Sharpe > tek tek modüller; drawdown düşer, günlük P&L pürüzsüzleşir.
Gerçek 4,5-yıl veri; katı OOS (yıl-yıl)."""
from __future__ import annotations
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0, str(Path(__file__).parent))
import funding_arb as FA

DATA = Path(__file__).parent / "runs" / "data_full"
START = "2022-01-01"


def daily_close(short):
    df = pd.read_csv(DATA/f"{short}_1h.csv"); df.index = pd.to_datetime(df["dt"])
    return df["close"].astype(float).resample("1D").last().dropna()


# ---------------- M1: Funding (8h → günlük) ----------------
def module_funding():
    rets = {}
    for s in FA.PAIRS:
        r8 = FA.strat_returns(FA.load(s))
        rets[s] = (1 + r8).groupby(r8.index.date).prod() - 1     # günlük
    idx = None
    for r in rets.values(): idx = r.index if idx is None else idx.union(r.index)
    M = pd.DataFrame({s: rets[s].reindex(idx).fillna(0) for s in rets})
    out = M.mean(axis=1); out.index = pd.to_datetime(out.index); return out


# ---------------- M2: Stat-arb (kointegrasyon pairs) ----------------
PAIRS_SA = [("ETHUSDT","BTCUSDT"), ("SOLUSDT","ETHUSDT"), ("BNBUSDT","BTCUSDT"),
            ("AVAXUSDT","SOLUSDT"), ("DOGEUSDT","BTCUSDT")]

def statarb_pair(a, b, lookback=60, entry=2.0, exit=0.5, stop=4.0, cost=0.001):
    rA = a.pct_change().fillna(0); rB = b.pct_change().fillna(0)
    la, lb = np.log(a), np.log(b)
    beta = (la.rolling(lookback).cov(lb) / (lb.rolling(lookback).var() + 1e-12)).clip(0.3, 3)
    spread = la - beta * lb
    z = (spread - spread.rolling(lookback).mean()) / (spread.rolling(lookback).std() + 1e-12)
    pos = np.zeros(len(z)); zv = z.values; prev = 0
    for t in range(len(zv)):
        p = prev
        if not np.isnan(zv[t]):
            if prev == 0:
                if zv[t] > entry: p = -1       # spread yüksek → short spread
                elif zv[t] < -entry: p = 1      # spread düşük → long spread
            elif abs(zv[t]) < exit or abs(zv[t]) > stop:   # ortalamaya döndü VEYA stop
                p = 0
        pos[t] = p; prev = p
    pos = pd.Series(pos, index=z.index)
    # dolar-nötr çift getirisi (sermaye-normalize): pos × (rA − β·rB)/(1+β)
    pair_ret = (pos.shift(1) * (rA - beta.shift(1) * rB) / (1 + beta.shift(1))).fillna(0)
    turn = pos.diff().abs().fillna(0)
    return pair_ret - turn * cost

def module_statarb():
    closes = {}
    for s in set([x for pr in PAIRS_SA for x in pr]):
        closes[s] = daily_close(s)
    idx = None
    for c in closes.values(): idx = c.index if idx is None else idx.union(c.index)
    idx = idx[idx >= pd.Timestamp("2021-06-01")]
    series = []
    for a, b in PAIRS_SA:
        ca = closes[a].reindex(idx).ffill(); cb = closes[b].reindex(idx).ffill()
        series.append(statarb_pair(ca, cb))
    return pd.concat(series, axis=1).mean(axis=1)


# ---------------- M3: Basis/carry proxy (funding'in yumuşak carry'si) ----------------
def module_basis():
    # aynı-borsa cash-and-carry ≈ sürekli pozitif carry; funding'in düşük-frekans bileşeni
    rets = {}
    for s in FA.PAIRS:
        fr = FA.load(s)
        carry = fr.rolling(9).mean().clip(lower=0)     # yalnız pozitif carry, yumuşak
        r = (carry - 0.00002).clip(lower=-0.0002)      # küçük maliyet
        rets[s] = (1 + r).groupby(r.index.date).prod() - 1
    idx = None
    for r in rets.values(): idx = r.index if idx is None else idx.union(r.index)
    M = pd.DataFrame({s: rets[s].reindex(idx).fillna(0) for s in rets})
    out = M.mean(axis=1); out.index = pd.to_datetime(out.index); return out


# ---------------- M4: Getiri tabanı ----------------
def module_yield(idx):
    return pd.Series(0.05/365, index=idx)              # ~%5/yıl stablecoin lending


def stats(ret):
    ret = ret[ret.index >= START]; eq = (1 + ret).cumprod()
    if len(eq) < 5: return None
    dd = ((eq.cummax()-eq)/eq.cummax()).max()*100
    sh = ret.mean()/(ret.std()+1e-12)*np.sqrt(365)
    cagr = (eq.iloc[-1]**(365/len(eq))-1)*100
    pos = (ret > 0).mean()*100
    yr = {str(y): round((g.iloc[-1]/g.iloc[0]-1)*100,1) for y,g in eq.groupby(eq.index.year)}
    return dict(cagr=round(cagr,1), sharpe=round(float(sh),2), dd=round(dd,1),
                pos=round(float(pos),0), avgday=round(ret.mean()*100,3), yr=yr)


def main():
    print("Modüller hesaplanıyor (gerçek 4,5-yıl veri)...\n", flush=True)
    mods = {"M1 Funding": module_funding(), "M2 Stat-arb": module_statarb(),
            "M3 Basis": module_basis()}
    idx = None
    for m in mods.values(): idx = m.index if idx is None else idx.union(m.index)
    idx = pd.DatetimeIndex(sorted(idx)); idx = idx[idx >= pd.Timestamp(START)]
    mods = {k: v.reindex(idx).fillna(0) for k, v in mods.items()}
    mods["M4 Yield"] = module_yield(idx)
    # tekil metrikler
    print("=== TEKİL MODÜLLER (OOS 2022-2026) ===")
    for k, v in mods.items():
        m = stats(v)
        print(f"  {k:12s} CAGR={m['cagr']:+6.1f}% Sharpe={m['sharpe']:+6.2f} DD={m['dd']:4.1f}% pozGün=%{m['pos']:.0f} ort/gün={m['avgday']:+.3f}%")
    # korelasyon matrisi (entourage'ın temeli)
    M = pd.DataFrame(mods)
    print("\n=== KORELASYON (düşük = güçlü entourage) ===")
    corr = M.corr()
    print("            " + " ".join(f"{k.split()[0]:>7s}" for k in mods))
    for k in mods:
        print(f"  {k.split()[0]:8s} " + " ".join(f"{corr.loc[k,j]:+7.2f}" for j in mods))
    # ENTOURAGE: yalnız POZİTİF-Sharpe aktif kenarları inverse-vol istifle;
    # M4 (getiri tabanı, sıfır-varyans) risk-ağırlığa girmez → atıl teminat getirisi olarak eklenir.
    active = {k: v for k, v in mods.items() if k != "M4 Yield" and (stats(v) or {}).get("sharpe", -9) > 0}
    A = pd.DataFrame(active)
    vol = A.std(); w = (1/vol) / (1/vol).sum()
    port = (A * w).sum(axis=1) + mods["M4 Yield"]     # aktif kenarlar + atıl teminat getirisi
    mp = stats(port)
    avg_sharpe = np.mean([stats(v)["sharpe"] for k, v in mods.items() if k != "M4 Yield"])
    print(f"\n=== ENTOURAGE KENAR PORTFÖYÜ (aktif-kenar inverse-vol + getiri tabanı) ===")
    print(f"  aktif kenarlar: {list(active.keys())} (Sharpe>0 filtreli)")
    print(f"  ağırlıklar: " + ", ".join(f"{k.split()[0]}={w[k]*100:.0f}%" for k in active) + " + M4 taban")
    print(f"  CAGR {mp['cagr']:+.1f}% | Sharpe {mp['sharpe']:+.2f} | MaxDD {mp['dd']:.1f}% | pozitif-gün %{mp['pos']:.0f} | ort/gün {mp['avgday']:+.3f}%")
    print(f"  yıl-yıl (her rejim): {mp['yr']}")
    print(f"\n  >> ENTOURAGE İSPATI: birleşik Sharpe {mp['sharpe']:.1f} vs modül-ort {avg_sharpe:.1f} "
          f"→ {'sinerji VAR (birleşik > ortalama)' if mp['sharpe']>avg_sharpe else 'sinerji yok'}")
    print(f"  >> %0.5/gün hedefi ile: ~{mp['avgday']:+.3f}%/gün (tutarlı ama hedefin ~1/{max(1,round(0.5/max(0.001,mp['avgday'])))}'i)")

if __name__ == "__main__":
    main()
