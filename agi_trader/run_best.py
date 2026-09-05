#!/usr/bin/env python3
"""Kazanan konfigi (ATR-seçici) rr-tolerans düzeltmesiyle 7 paritede koştur."""
from __future__ import annotations
import sys, glob, json
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0, str(Path(__file__).parent))
from agi_trader.config import load_config
from agi_trader.backtest.real_engine import load_csv, run_real_backtest

DATA = Path(__file__).parent / "runs" / "data_6m"
DEC = {"min_confidence": 0.74, "consensus_min_agreement": 0.85, "min_risk_reward": 1.5}
RISK = {"tp_mode": "atr", "atr_stop_mult": 2.5, "tp_r_multiples": [1.5, 2.5, 4.0]}


def main():
    c = load_config()
    c.data["decision"].update(DEC); c.data["risk"].update(RISK)
    rows, eq = [], 1.0
    for f in sorted(glob.glob(str(DATA / "*.csv"))):
        sym = Path(f).stem.split("_")[0]
        r = run_real_backtest(load_csv(f), c, symbol=sym)
        rows.append((sym, r)); eq *= (1 + r["total_return_pct"] / 100)
        print(f"{sym:9s} tr={r['trades']:3d} win={r['win_rate']:5.1f}% PF={r['profit_factor']:4.2f} "
              f"ret={r['total_return_pct']:+7.2f}% ay={r['avg_monthly_pct']:+5.2f}% "
              f"gun={r['est_daily_pct']:+.3f}% DD={r['max_drawdown_pct']:4.1f}% sig={r['actionable_signals']}", flush=True)
    valid = [r for _, r in rows if r["trades"] > 0]
    n = len(valid) or 1
    print(f"\nPORTFOY(esit-agirlik bilesik 6ay): {(eq-1)*100:+.2f}%")
    print(f"Parite-ort: 6ay {sum(r['total_return_pct'] for r in valid)/n:+.2f}% | "
          f"ay {sum(r['avg_monthly_pct'] for r in valid)/n:+.2f}% | "
          f"win {sum(r['win_rate'] for r in valid)/n:.1f}% | "
          f"DD {sum(r['max_drawdown_pct'] for r in valid)/n:.1f}% | "
          f"toplam islem {sum(r['trades'] for r in valid)}")
    Path(__file__).parent.joinpath("runs/best_atr_7pair.json").write_text(
        json.dumps({"config": {**DEC, **RISK},
                    "portfolio_6m_pct": round((eq-1)*100, 2),
                    "pairs": {s: r for s, r in rows}}, indent=2, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    main()
