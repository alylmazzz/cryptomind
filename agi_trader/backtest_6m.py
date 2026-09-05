#!/usr/bin/env python3
"""
Gerçek-pipeline 6 aylık backtest koşucusu.

runs/data_6m/*.csv dosyalarını gerçek karar zinciriyle (konsensüs kapısı +
scalp 3-hedef) bar-bar test eder ve aylık getiri / win-rate / drawdown çıkarır.

Kullanım:
  python backtest_6m.py                 # tüm pariteler
  python backtest_6m.py --pairs BTCUSDT ETHUSDT
  python backtest_6m.py --pos 0.5       # işlem başına equity'nin %50'si
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent))

from agi_trader.config import load_config
from agi_trader.backtest.real_engine import load_csv, run_real_backtest

DATA_DIR = Path(__file__).parent / "runs" / "data_6m"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", nargs="*", help="örn: BTCUSDT ETHUSDT")
    p.add_argument("--pos", type=float, default=1.0, help="işlem başına equity oranı (0-1)")
    p.add_argument("--lookback", type=int, default=400)
    p.add_argument("--max-hold", type=int, default=48)
    p.add_argument("--out", default="runs/real_sixmonth_results.json")
    args = p.parse_args(argv)

    cfg = load_config()
    files = sorted(glob.glob(str(DATA_DIR / "*.csv")))
    if args.pairs:
        want = {x.upper() for x in args.pairs}
        files = [f for f in files if Path(f).stem.split("_")[0].upper() in want]
    if not files:
        print("Veri bulunamadı:", DATA_DIR); return 1

    results = {}
    agg_trades = agg_wins = 0
    eq_product = 1.0
    print(f"Mod: {cfg.get('risk.tp_mode')} | konsensüs ≥%{cfg.get('decision.consensus_min_agreement')*100:.0f} "
          f"| min katman {cfg.get('decision.consensus_min_layers')} | min güven %{cfg.get('decision.min_confidence')*100:.0f}\n")
    for f in files:
        sym = Path(f).stem.split("_")[0]
        df = load_csv(f)
        r = run_real_backtest(df, cfg, symbol=sym, tf="1h",
                              lookback=args.lookback, position_fraction=args.pos,
                              max_hold=args.max_hold)
        results[sym] = r
        if "error" in r:
            print(f"{sym:10s} HATA: {r['error']}"); continue
        agg_trades += r["trades"]; agg_wins += round(r["trades"] * r["win_rate"] / 100)
        eq_product *= (1 + r["total_return_pct"] / 100)
        print(f"{sym:10s} işlem={r['trades']:3d}  win={r['win_rate']:4.1f}%  "
              f"PF={r['profit_factor']:4.2f}  toplam={r['total_return_pct']:+6.2f}%  "
              f"ay/ort={r['avg_monthly_pct']:+5.2f}%  DD={r['max_drawdown_pct']:4.1f}%  "
              f"sinyal={r['actionable_signals']}")

    portfolio_ret = (eq_product - 1) * 100 if results else 0.0
    valid = [r for r in results.values() if "error" not in r]
    summary = {
        "config": {
            "tp_mode": cfg.get("risk.tp_mode"),
            "scalp_tp_pct": cfg.get("risk.scalp_tp_pct"),
            "scalp_sl_pct": cfg.get("risk.scalp_sl_pct"),
            "consensus_min_agreement": cfg.get("decision.consensus_min_agreement"),
            "consensus_min_layers": cfg.get("decision.consensus_min_layers"),
            "min_confidence": cfg.get("decision.min_confidence"),
            "position_fraction": args.pos,
        },
        "pairs": list(results.keys()),
        "total_trades": agg_trades,
        "overall_win_rate": round(100 * agg_wins / agg_trades, 1) if agg_trades else 0.0,
        "avg_total_return_pct": round(sum(r["total_return_pct"] for r in valid) / len(valid), 2) if valid else 0.0,
        "avg_monthly_pct": round(sum(r["avg_monthly_pct"] for r in valid) / len(valid), 2) if valid else 0.0,
        "results": results,
    }
    print(f"\nTOPLAM: {agg_trades} işlem · genel win %{summary['overall_win_rate']} · "
          f"parite-ort. aylık %{summary['avg_monthly_pct']:+.2f} · "
          f"parite-ort. 6ay %{summary['avg_total_return_pct']:+.2f}")

    out = Path(__file__).parent / args.out
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Yazıldı:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
