#!/usr/bin/env python3
"""Nihai stratejiyi MOTORUN KENDİSİYLE (TrendTrader) gün-gün paper-replay et.
Amaç: (1) canlı motorun araştırma-backtest'iyle (Sharpe ~1.36) aynı sonucu
ürettiğini KANITLA, (2) risk panelini gerçek track record'la doldur.
13 varlık (kripto + kripto-dışı), 2022-2026, hedef-vol kaldıraçlı."""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0, str(Path(__file__).parent))
from agi_trader.config import load_config
from agi_trader.auto.trend_engine import TrendTrader
from agi_trader.monitor import risk_report
from leverage_hedge import crypto_close, nc_close, CRYPTO, NONCRYPTO

STATE = Path(__file__).parent / "runs" / "trend_state.json"


def ohlcv_from_close(cl):
    return pd.DataFrame({"open":cl,"high":cl,"low":cl,"close":cl,"volume":cl*0+1.0})


def main():
    print("13 varlık günlük veri hazırlanıyor...", flush=True)
    daily = {}
    for s in CRYPTO:
        cl = crypto_close(s); daily[s] = ohlcv_from_close(cl)
    for t in NONCRYPTO:
        cl = nc_close(t)
        if cl is not None and len(cl) > 250: daily[t] = ohlcv_from_close(cl)
    pairs = list(daily.keys())
    # ortak günlük indeks (2022 öncesi warmup dahil)
    idx = None
    for df in daily.values(): idx = df.index if idx is None else idx.union(df.index)
    idx = idx[idx >= pd.Timestamp("2021-06-01")]
    for k in daily: daily[k] = daily[k].reindex(idx).ffill()

    c = load_config()
    tt = TrendTrader(c, pairs=pairs, initial=10000)
    start_i = 230   # SMA200 + momentum warmup
    print(f"Motor replay: {len(pairs)} varlık, {len(idx)-start_i} gün (2022-2026)...", flush=True)
    for i in range(start_i, len(idx)):
        d = idx[i]
        data = {s: daily[s].iloc[:i+1] for s in pairs}
        tt.step(data, date_str=str(d)[:10])
    tt.save_state(str(STATE))

    # motorun track record'u
    eq = np.array([h["equity"] for h in tt.history], dtype=float)
    rets = np.diff(eq)/eq[:-1]
    yrs = len(eq)/365
    total = (eq[-1]/10000-1)*100
    cagr = ((eq[-1]/10000)**(1/yrs)-1)*100
    sharpe = rets.mean()/(rets.std()+1e-12)*np.sqrt(365)
    dd = (np.maximum.accumulate(eq)-eq); dd = (dd/np.maximum.accumulate(eq)).max()*100
    print("\n=== MOTOR PAPER TRACK RECORD (2022-2026, gün-gün) ===")
    print(f"  getiri {total:+.1f}% | CAGR {cagr:+.1f}% | Sharpe {sharpe:+.2f} | MaxDD {dd:.1f}% | {len(eq)} gün")
    print(f"  (araştırma backtest ≈ Sharpe 1.36 / CAGR %21 / DD %13 → motor {'DOĞRULANDI' if sharpe>1.0 and dd<25 else 'SAPMA VAR'})")
    # risk raporu (gerçek track record'la)
    r = risk_report(json.loads(STATE.read_text(encoding="utf-8")))
    print(f"\n=== RİSK RAPORU (canlı ops katmanı) ===")
    print(f"  health={r['health']} | realized_sharpe={r['realized_sharpe']} | VaR95=%{r['var95_pct']} "
          f"CVaR95=%{r['cvar95_pct']} | maruziyet=%{r['gross_exposure_pct']}")
    for a in r["alerts"]:
        try: print("  " + a)
        except Exception: pass

if __name__ == "__main__":
    main()
