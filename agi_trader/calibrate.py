#!/usr/bin/env python3
"""Parite-bazlı trend-kapısı kalibrasyonu.

Her parite için trend_gate_min_adx ∈ {0,20,28} dener; tam 6-ay backtest +
AYLIK robustluk (pozitif ay oranı) ölçer. Her pariteye EN İYİ eşiği atar;
hiçbir ayar (toplam>0 VE pozitif-ay≥%50) sağlamıyorsa pariteyi ELER (999).
Çıktı: veri-temelli pair_trend_gate haritası + allowlist → config.yaml'a uygula.
"""
from __future__ import annotations
import sys, glob, json, math, copy
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0, str(Path(__file__).parent))
from agi_trader.config import load_config
from agi_trader.backtest.real_engine import load_csv, run_real_backtest

DATA = Path(__file__).parent / "runs" / "data_6m"
ADX_GRID = [0, 20, 28]          # 0 = kapı kapalı
MIN_TRADES = 12


def pos_month_ratio(r):
    mr = r.get("monthly_returns") or {}
    if not mr: return 0.0
    return sum(1 for v in mr.values() if v > 0) / len(mr)


def main():
    base = load_config()
    files = {Path(f).stem.split("_")[0]: f for f in sorted(glob.glob(str(DATA / "*.csv")))}
    sym_map = {"BTCUSDT": "BTC/USDT", "ETHUSDT": "ETH/USDT", "SOLUSDT": "SOL/USDT",
               "BNBUSDT": "BNB/USDT", "XRPUSDT": "XRP/USDT", "ADAUSDT": "ADA/USDT",
               "DOGEUSDT": "DOGE/USDT", "AVAXUSDT": "AVAX/USDT"}
    rec, allow = {}, []
    for short, f in files.items():
        full = sym_map.get(short, short)
        df = load_csv(f)
        trials = []
        for adx in ADX_GRID:
            c = load_config(); c.data = copy.deepcopy(base.data)
            c.data["decision"]["pair_trend_gate"] = {full: adx}
            r = run_real_backtest(df, c, symbol=full)
            pr = pos_month_ratio(r)
            trials.append({"adx": adx, "ret": r["total_return_pct"], "tr": r["trades"],
                           "pos_ratio": round(pr, 2), "dd": r["max_drawdown_pct"]})
        # geçerli = yeterli işlem + toplam>0 + pozitif-ay≥%50
        valid = [t for t in trials if t["tr"] >= MIN_TRADES and t["ret"] > 0 and t["pos_ratio"] >= 0.5]
        if valid:
            best = max(valid, key=lambda t: (t["pos_ratio"], t["ret"]))
            rec[full] = best["adx"]; allow.append(full)
            verdict = f"AL  → adx={best['adx']} (ret {best['ret']:+.1f}%, pozAy %{best['pos_ratio']*100:.0f}, DD {best['dd']:.0f}%)"
        else:
            rec[full] = 999
            b = max(trials, key=lambda t: t["ret"])
            verdict = f"ELE → hiç sağlam ayar yok (en iyi {b['ret']:+.1f}%, pozAy %{b['pos_ratio']*100:.0f})"
        print(f"{short:9s} {verdict}", flush=True)
        print(f"          " + " | ".join(f"adx{t['adx']}:{t['ret']:+.1f}%/pos%{t['pos_ratio']*100:.0f}/tr{t['tr']}" for t in trials), flush=True)

    out = {"pair_trend_gate": rec, "allowlist": allow, "adx_grid": ADX_GRID}
    Path(__file__).parent.joinpath("runs/pair_calibration.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== ÖNERİ ===")
    print("allowlist (işlenecek):", allow)
    print("pair_trend_gate:", json.dumps(rec, ensure_ascii=False))
    print("Yazıldı: runs/pair_calibration.json")

if __name__ == "__main__":
    main()
