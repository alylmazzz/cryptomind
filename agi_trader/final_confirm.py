#!/usr/bin/env python3
"""Kalibre edilmiş NİHAİ konfigi (config.yaml) allowlist paritelerde doğrula."""
from __future__ import annotations
import sys, glob
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0, str(Path(__file__).parent))
from agi_trader.config import load_config
from agi_trader.backtest.real_engine import load_csv, run_real_backtest

DATA = Path(__file__).parent / "runs" / "data_6m"

def main():
    c = load_config()
    gate = c.get("decision.pair_trend_gate") or {}
    allow = [s for s in c.symbols]
    print(f"symbols(allowlist)={allow}")
    print(f"pair_trend_gate={gate}\n", flush=True)
    files = {Path(f).stem.split("_")[0]: f for f in sorted(glob.glob(str(DATA / "*.csv")))}
    short = {"BTC/USDT":"BTCUSDT","ETH/USDT":"ETHUSDT","SOL/USDT":"SOLUSDT",
             "DOGE/USDT":"DOGEUSDT","AVAX/USDT":"AVAXUSDT"}
    eq = 1.0; rows = []
    for sym in allow:
        df = load_csv(files[short[sym]])
        r = run_real_backtest(df, c, symbol=sym)
        eq *= (1 + r["total_return_pct"]/100); rows.append(r)
        print(f"{sym:9s} 6ay={r['total_return_pct']:+7.2f}% ay={r['avg_monthly_pct']:+5.2f}% "
              f"tr={r['trades']:3d} win={r['win_rate']:4.1f}% DD={r['max_drawdown_pct']:4.1f}%", flush=True)
    n=len(rows)
    print(f"\n=== KALİBRE PORTFÖY (eşit-ağırlık {n} parite) ===")
    print(f"6ay bileşik: {(eq-1)*100:+.2f}%  | ~aylık bileşik: {((eq**(1/6))-1)*100:+.2f}%")
    print(f"parite-ort 6ay: {sum(r['total_return_pct'] for r in rows)/n:+.2f}% | "
          f"ort win {sum(r['win_rate'] for r in rows)/n:.1f}% | "
          f"ort DD {sum(r['max_drawdown_pct'] for r in rows)/n:.1f}% | "
          f"toplam işlem {sum(r['trades'] for r in rows)}")

if __name__ == "__main__":
    main()
