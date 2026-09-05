#!/usr/bin/env python3
"""Scalp/seçicilik parametre taraması — hangi konfig pozitif getiriyor?"""
from __future__ import annotations
import sys, glob, json, copy
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0, str(Path(__file__).parent))
from agi_trader.config import load_config
from agi_trader.backtest.real_engine import load_csv, run_real_backtest

PAIRS = ["BTCUSDT", "ETHUSDT"]
DATA = Path(__file__).parent / "runs" / "data_6m"

# (etiket, decision override, risk override)
CONFIGS = [
    ("scalp-baz 0.5/0.6", {"min_confidence":0.62,"consensus_min_agreement":0.70,"min_risk_reward":0.8},
     {"tp_mode":"scalp","scalp_tp_pct":[0.005,0.01,0.015],"scalp_sl_pct":0.006}),
    ("scalp-seçici 0.5/0.6", {"min_confidence":0.72,"consensus_min_agreement":0.82,"min_risk_reward":0.8},
     {"tp_mode":"scalp","scalp_tp_pct":[0.005,0.01,0.015],"scalp_sl_pct":0.006}),
    ("scalp-geniş 1/1", {"min_confidence":0.66,"consensus_min_agreement":0.78,"min_risk_reward":1.0},
     {"tp_mode":"scalp","scalp_tp_pct":[0.01,0.02,0.03],"scalp_sl_pct":0.01}),
    ("scalp-seçici-geniş 1.2/0.8", {"min_confidence":0.74,"consensus_min_agreement":0.85,"min_risk_reward":1.3},
     {"tp_mode":"scalp","scalp_tp_pct":[0.012,0.022,0.035],"scalp_sl_pct":0.008}),
    ("ATR-R 1.5/2.5/4", {"min_confidence":0.66,"consensus_min_agreement":0.78,"min_risk_reward":1.5},
     {"tp_mode":"atr","atr_stop_mult":2.0,"tp_r_multiples":[1.5,2.5,4.0]}),
    ("ATR-seçici", {"min_confidence":0.74,"consensus_min_agreement":0.85,"min_risk_reward":1.5},
     {"tp_mode":"atr","atr_stop_mult":2.5,"tp_r_multiples":[1.5,2.5,4.0]}),
]


def main():
    base = load_config()
    files = {Path(f).stem.split("_")[0]: f for f in glob.glob(str(DATA / "*.csv"))}
    out = []
    for label, dov, rov in CONFIGS:
        c = load_config()
        c.data = copy.deepcopy(base.data)
        c.data["decision"].update(dov)
        c.data["risk"].update(rov)
        rets, trs, wins, dds = [], 0, [], []
        for p in PAIRS:
            df = load_csv(files[p])
            r = run_real_backtest(df, c, symbol=p)
            rets.append(r["total_return_pct"]); trs += r["trades"]
            wins.append(r["win_rate"]); dds.append(r["max_drawdown_pct"])
            print(f"  {label:28s} {p}: tr={r['trades']:3d} win={r['win_rate']:4.1f}% "
                  f"PF={r['profit_factor']:.2f} ret={r['total_return_pct']:+6.2f}% "
                  f"ay/ort={r['avg_monthly_pct']:+.2f}% DD={r['max_drawdown_pct']:.1f}%", flush=True)
        avg = sum(rets)/len(rets)
        out.append({"config": label, "avg_6m_return": round(avg,2), "trades": trs,
                    "avg_win": round(sum(wins)/len(wins),1), "avg_dd": round(sum(dds)/len(dds),1)})
        print(f"  → {label}: ORT 6ay {avg:+.2f}%\n", flush=True)
    out.sort(key=lambda x: -x["avg_6m_return"])
    print("=== SIRALAMA (ort 6ay getiri) ===")
    for o in out:
        print(f"  {o['avg_6m_return']:+7.2f}%  {o['config']:30s} (tr={o['trades']}, win={o['avg_win']}%, DD={o['avg_dd']}%)")
    Path(__file__).parent.joinpath("runs/scalp_sweep.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    main()
