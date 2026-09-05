#!/usr/bin/env python3
"""GERÇEK OUT-OF-SAMPLE testi: mevcut konfig (Dec2025-Jun2026'ya kalibre) 2022-2025
verisinde (HİÇ GÖRÜLMEMİŞ) nasıl davranıyor? +%61.8 in-sample mıydı, gerçek mi?
Her parite full 4.5y koşturulur; aylık getiriler yıllara toplanır (rejim kırılımı)."""
from __future__ import annotations
import sys, glob, json
from pathlib import Path
from collections import defaultdict
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0, str(Path(__file__).parent))
from agi_trader.config import load_config
from agi_trader.backtest.real_engine import load_csv, run_real_backtest

DATA = Path(__file__).parent / "runs" / "data_full"

def yearly(monthly: dict) -> dict:
    y = defaultdict(lambda: 1.0)
    for ym, pct in monthly.items():
        y[ym[:4]] *= (1 + pct/100)
    return {k: round((v-1)*100, 1) for k, v in sorted(y.items())}

def main():
    c = load_config()
    allow = list(c.symbols)
    short = {s: s.replace("/","") for s in allow}
    files = {Path(p).stem.split("_")[0]: p for p in glob.glob(str(DATA/"*.csv"))}
    print(f"OOS konfig: tp_mode={c.get('risk.tp_mode')} fee={c.get('risk.fee_taker')} "
          f"conf={c.get('decision.min_confidence')} | pariteler={allow}\n", flush=True)
    res = {}
    for sym in allow:
        df = load_csv(files[short[sym]])
        r = run_real_backtest(df, c, symbol=sym)
        yb = yearly(r.get("monthly_returns", {}))
        res[sym] = {"full": r["total_return_pct"], "trades": r["trades"],
                    "win": r["win_rate"], "dd": r["max_drawdown_pct"], "yearly": yb}
        print(f"{sym:9s} 4.5y={r['total_return_pct']:+8.1f}% tr={r['trades']:4d} "
              f"win={r['win_rate']:4.1f}% DD={r['max_drawdown_pct']:4.1f}%", flush=True)
        print(f"          yıllık: {yb}", flush=True)
    # yıl bazında portföy (eşit ağırlık)
    years = ["2022","2023","2024","2025","2026"]
    print("\n=== YIL BAZINDA PORTFÖY (eşit-ağırlık, OOS=2022-2025) ===", flush=True)
    for y in years:
        vals = [res[s]["yearly"].get(y) for s in allow if y in res[s]["yearly"]]
        if vals:
            print(f"  {y}: ort {sum(vals)/len(vals):+6.1f}%  (pariteler: {[res[s]['yearly'].get(y) for s in allow]})", flush=True)
    Path(__file__).parent.joinpath("runs/oos_test.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nYazıldı: runs/oos_test.json", flush=True)

if __name__ == "__main__":
    main()
