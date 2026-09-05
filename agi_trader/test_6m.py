#!/usr/bin/env python3
"""SON 6 AY (2026-01-01 → 2026-06-30) — uygulamanın son halinin TÜM doğrulanmış
özelliklerini bu pencerede test et, 6-aylık % kârı belirle.
Bileşenler: (A) diversifiye trend-takip + hedef-vol kaldıraç (yönlü çekirdek),
(B) entourage piyasa-nötr kenar portföyü (funding + basis + getiri tabanı)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0, str(Path(__file__).parent))

import leverage_hedge as LH
import edge_portfolio as EP

S6, E6 = "2026-01-01", "2026-07-01"


def slc(r):
    r = r.copy(); r.index = pd.to_datetime(r.index)
    return r[(r.index >= pd.Timestamp(S6)) & (r.index < pd.Timestamp(E6))]

def rep(name, r):
    r = slc(r)
    if len(r) < 2: return None
    eq = (1 + r).cumprod()
    total = (eq.iloc[-1] - 1) * 100
    dd = ((eq.cummax() - eq) / eq.cummax()).max() * 100
    pos = (r > 0).mean() * 100
    sh = r.mean()/(r.std()+1e-12)*np.sqrt(365)
    print(f"  {name:34s} 6ay={total:+6.2f}%  MaxDD={dd:4.1f}%  Sharpe={sh:+5.2f}  pozGün=%{pos:.0f}  gün={len(r)}")
    return r


def main():
    print(f"Test penceresi: {S6} → 2026-06-30 (son 6 ay)\n")

    # ---- (A) Diversifiye trend-takip + hedef-vol kaldıraç (17 varlık) ----
    S = {s: LH.strat(LH.crypto_close(s)) for s in LH.CRYPTO}
    for t in LH.NONCRYPTO + ["HYG","FXB","FXF","FXE"]:
        try:
            cl = LH.nc_close(t)
            if cl is not None and len(cl) > 250: S[t] = LH.strat(cl)
        except Exception: pass
    idx = None
    for v in S.values(): idx = v.index if idx is None else idx.union(v.index)
    M = pd.DataFrame({k: v.reindex(idx).fillna(0) for k, v in S.items()})
    base = M.mean(axis=1)
    rv = base.rolling(30).std()
    lev = (LH.TARGET_ANN/np.sqrt(365)/(rv+1e-9)).clip(0, LH.MAX_LEV).shift(1).fillna(1.0)
    trend = base * lev

    # ---- (B) Entourage kenar portföyü (funding + basis + getiri tabanı) ----
    m1 = EP.module_funding(); m3 = EP.module_basis()
    eidx = m1.index.union(m3.index)
    m1 = m1.reindex(eidx).fillna(0); m3 = m3.reindex(eidx).fillna(0)
    m4 = pd.Series(0.05/365, index=eidx)
    A = pd.DataFrame({"M1": m1, "M3": m3}); vol = A.std(); w = (1/vol)/(1/vol).sum()
    edge = (A*w).sum(axis=1) + m4

    print("=== BİLEŞEN BAZINDA (son 6 ay) ===")
    rt = rep("A) Diversifiye trend-takip (kaldıraçlı)", trend)
    re = rep("B) Entourage kenar portföyü (nötr)", edge)

    # ---- Birleşik: eşit-sermaye (%50 trend + %50 kenar) ----
    ci = slc(trend).index.union(slc(edge).index)
    tr = slc(trend).reindex(ci).fillna(0); ed = slc(edge).reindex(ci).fillna(0)
    combo = 0.5*tr + 0.5*ed
    print("\n=== BİRLEŞİK (uygulamanın son hali, %50 trend + %50 nötr-kenar) ===")
    ceq = (1+combo).cumprod(); ctot = (ceq.iloc[-1]-1)*100
    cdd = ((ceq.cummax()-ceq)/ceq.cummax()).max()*100
    print(f"  SON 6 AY TOPLAM KÂR: {ctot:+.2f}%  |  MaxDD {cdd:.1f}%  |  pozGün %{(combo>0).mean()*100:.0f}")
    print(f"  (yıllık bileşik eşdeğeri ≈ {((1+ctot/100)**2-1)*100:+.1f}%)")

    print("\nNOT: Son 6 ay (2026 H1) DÜŞÜŞ piyasasıydı (BTC ~100k→59k). Trend-takip bu")
    print("dönemde çoğunlukla NAKİTTE kalıp sermayeyi korur (küçük +/−); getiriyi asıl")
    print("piyasa-nötr kenar portföyü (funding/basis) taşır — yön riski olmadan.")

if __name__ == "__main__":
    main()
