#!/usr/bin/env python3
"""
"%1 hareket adayı" modelinin dürüst doğrulaması.

Soru: model TABAN ORANINI geçiyor mu? Geçmiyorsa panele konmamalıdır.
Eğitim 2022-2024 · doğrulama 2025 · KİLİTLİ TEST (2026) AÇILMAZ.

  python research_mover.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).parent))

from agi_trader.research import dataset as ds
from agi_trader.research.validation import trial_log
from agi_trader.analysis.mover import (
    MoverModel, build_mover_features, move_labels, rank_movers,
    auc_score, brier_skill, calibration_table, MOVE_THRESHOLD,
)

SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "AVAXUSDT"]
OUT = Path(__file__).parent / "runs" / "mover_research.json"


def main() -> None:
    print("=" * 78)
    print("'%1 HAREKET ADAYI' — ÖRNEKLEM DIŞI DOĞRULAMA")
    print("eğitim 2022-2024 · doğrulama 2025 · test 2026 KİLİTLİ")
    print("=" * 78)

    rows, models, panel = [], {}, {}
    print(f"\n{'parite':10s} {'taban':>7s} {'ort tahmin':>11s} {'AUC':>7s} "
          f"{'Brier beceri':>13s} {'ust %20 isabet':>15s}")
    for sym in SYMS:
        d = ds.load_crypto_ohlcv_daily(sym)
        panel[sym] = ds.train_val(d)
        X = build_mover_features(d)
        y = move_labels(d, MOVE_THRESHOLD)

        tr = (d.index >= "2022-01-01") & (d.index <= "2024-12-31")
        va = (d.index >= "2025-01-01") & (d.index <= "2025-12-31")

        m = MoverModel().fit(X[tr], y[tr])
        models[sym] = m
        p = m.predict_proba(X[va])
        yv = y[va].values

        base = float(yv.mean())
        a = auc_score(yv, p)
        bs = brier_skill(yv, p, base)
        # en yüksek olasılıklı %20 günde gerçekleşme oranı
        k = max(5, int(len(p) * 0.2))
        top_idx = np.argsort(p)[-k:]
        top_hit = float(yv[top_idx].mean())

        print(f"{sym:10s} {base*100:6.1f}% {p.mean()*100:10.1f}% {a:7.3f} "
              f"{bs:+13.3f} {top_hit*100:14.1f}%")
        rows.append({"symbol": sym, "base_rate": round(base, 4),
                     "mean_pred": round(float(p.mean()), 4), "auc": round(a, 4),
                     "brier_skill": round(bs, 4), "top20_hit": round(top_hit, 4),
                     "n_val": int(len(yv)),
                     "calibration": calibration_table(yv, p)})

    aucs = [r["auc"] for r in rows if np.isfinite(r["auc"])]
    bss = [r["brier_skill"] for r in rows if np.isfinite(r["brier_skill"])]
    lift = [r["top20_hit"] / r["base_rate"] for r in rows if r["base_rate"] > 0]
    print(f"\n{'ORTALAMA':10s} {'':7s} {'':11s} {np.mean(aucs):7.3f} "
          f"{np.mean(bss):+13.3f} lift {np.mean(lift):.3f}×")

    print("\n=== KALİBRASYON (BTC, doğrulama 2025) ===")
    for c in rows[0]["calibration"]:
        print(f"  dilim {c['bin']}: tahmin %{c['tahmin']*100:5.1f} → "
              f"gerçekleşen %{c['gerceklesen']*100:5.1f}  (n={c['n']})")

    print("\n=== BUGÜNÜN SIRALAMASI (doğrulama sonu itibarıyla) ===")
    rk = rank_movers(panel, models)
    for p in rk["picks"]:
        ev = " · ".join(f"{e['tr']} ({e['contribution']:+.2f})" for e in p["evidence"][:2])
        print(f"  {p['rank']}. {p['symbol']:9s} P={p['probability']*100:5.1f}% "
              f"(taban %{p['base_rate']*100:4.1f}, lift {p['lift']:.2f}×) "
              f"beklenen |hareket| %{p['expected_move_pct']:.2f}")
        print(f"       kanıt: {ev}")

    print("\n" + "=" * 78)
    ok = np.mean(aucs) > 0.55 and np.mean(bss) > 0
    if ok:
        print(f"HÜKÜM: model taban oranı GEÇİYOR (AUC {np.mean(aucs):.3f} > 0,55; "
              f"Brier beceri {np.mean(bss):+.3f} > 0).")
        print("Panele konabilir — AMA yalnız BÜYÜKLÜK sıralaması olarak.")
        print(rk["direction_warning"])
    else:
        print(f"HÜKÜM: model taban oranı GEÇEMİYOR (AUC {np.mean(aucs):.3f}, "
              f"Brier beceri {np.mean(bss):+.3f}). Panele KONMAZ.")

    trial_log("mover_1pct", {"period": "train2022-24_val2025"},
              {"sharpe": float(np.mean(aucs))})
    OUT.write_text(json.dumps({"rows": rows, "mean_auc": round(float(np.mean(aucs)), 4),
                               "mean_brier_skill": round(float(np.mean(bss)), 4),
                               "ranking": rk, "passes": bool(ok)},
                              indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
