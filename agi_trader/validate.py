#!/usr/bin/env python3
"""Hibrit + trend-kapısı konfigini doğrula: 7 paritede tam dönem + 3-kat walk-forward.
config.yaml'daki YENİ varsayılan konfigi kullanır (hybrid, trend_gate, seçici)."""
from __future__ import annotations
import sys, glob, json
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0, str(Path(__file__).parent))
from agi_trader.config import load_config
from agi_trader.backtest.real_engine import load_csv, run_real_backtest, walk_forward

DATA = Path(__file__).parent / "runs" / "data_6m"
MAJORS = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"}


def main():
    c = load_config()
    print(f"KONFIG: tp_mode={c.get('risk.tp_mode')} adx_trend={c.get('risk.hybrid_adx_trend')} "
          f"trend_gate={c.get('decision.trend_gate_min_adx')} conf={c.get('decision.min_confidence')} "
          f"agree={c.get('decision.consensus_min_agreement')}\n", flush=True)
    rows = {}
    for f in sorted(glob.glob(str(DATA / "*.csv"))):
        sym = Path(f).stem.split("_")[0]
        df = load_csv(f)
        full = run_real_backtest(df, c, symbol=sym)
        wf = walk_forward(df, c, symbol=sym, folds=3)
        rows[sym] = {"full": full, "wf": wf}
        folds_str = " ".join(f"{x['return_pct']:+.1f}%" for x in wf["folds"])
        tag = "MAJOR" if sym in MAJORS else "alt"
        print(f"{sym:9s}[{tag:5s}] 6ay={full['total_return_pct']:+7.2f}% ay={full['avg_monthly_pct']:+5.2f}% "
              f"tr={full['trades']:3d} win={full['win_rate']:4.1f}% DD={full['max_drawdown_pct']:4.1f}% "
              f"| WF[{wf['positive_folds']}/3 poz]: {folds_str}", flush=True)

    def agg(keys):
        v = [rows[k]["full"] for k in keys if rows[k]["full"].get("trades", 0) > 0]
        if not v: return None
        eq = 1.0
        for r in v: eq *= (1 + r["total_return_pct"] / 100)
        return {"pairs": len(v), "compound_6m": round((eq - 1) * 100, 2),
                "avg_6m": round(sum(r["total_return_pct"] for r in v) / len(v), 2),
                "avg_monthly": round(sum(r["avg_monthly_pct"] for r in v) / len(v), 2),
                "avg_win": round(sum(r["win_rate"] for r in v) / len(v), 1),
                "avg_dd": round(sum(r["max_drawdown_pct"] for r in v) / len(v), 1),
                "trades": sum(r["trades"] for r in v)}
    majors = agg([k for k in rows if k in MAJORS])
    allp = agg(list(rows))
    print("\n=== ÖZET ===")
    print(f"MAJÖRLER(4): {majors}")
    print(f"TÜM(7):      {allp}")
    Path(__file__).parent.joinpath("runs/validate_hybrid.json").write_text(
        json.dumps({"majors": majors, "all": allp, "rows": rows}, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print("\nYazıldı: runs/validate_hybrid.json")

if __name__ == "__main__":
    main()
