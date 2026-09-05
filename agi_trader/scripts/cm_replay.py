#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CryptoMind REPLAY — komiteyi gerçek geçmiş 1 dk veride oynat, sleeve başına bilimsel kabul kanıtı üret.

  python scripts/cm_replay.py --days 2 --symbols BTC/USDT,ETH/USDT,SOL/USDT --venue mexc
  python scripts/cm_replay.py --days 3 --all --evidence          # 40 parite, lifecycle'a kanıt yaz
  python scripts/cm_replay.py --result runs/replay/replay_*.json  # kayıtlı sonucu yeniden analiz et

Çıktı: tablo (sleeve · n · kazanma · beklenti · CI · maliyet×2 · tutarlı · DSR · PBO · KAPI) + JSON dosyası.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
def _utf8_stdout():
    if hasattr(sys.stdout, "buffer") and getattr(sys.stdout, "encoding", "").lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=2.0)
    ap.add_argument("--symbols", default="")
    ap.add_argument("--all", action="store_true", help="simülatörün 40 paritesi")
    ap.add_argument("--venue", default="mexc")
    ap.add_argument("--step", type=int, default=60, help="döngü adımı (sn)")
    ap.add_argument("--max-cycles", type=int, default=0)
    ap.add_argument("--evidence", action="store_true", help="lifecycle kayıt defterine kanıt yaz")
    ap.add_argument("--result", default="", help="kayıtlı replay JSON'unu yeniden analiz et")
    ap.add_argument("--out", default=str(ROOT / "runs" / "replay"))
    ap.add_argument("--trials", type=int, default=20, help="DSR için denenen strateji/param sayısı")
    a = ap.parse_args()

    from agi_trader.auto import replay as RP
    from agi_trader.strategies.lifecycle import Lifecycle

    if a.result:
        d = json.loads(Path(a.result).read_text(encoding="utf-8"))
        result = d["result"]
    else:
        from agi_trader.auto import simulator as SIM
        syms = SIM.HEAVY_SYMBOLS + SIM.LIGHT_SYMBOLS if a.all else [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
        if not syms:
            syms = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        t0 = time.time()
        hf = RP.HistoryFetcher(a.venue)
        print(f"» geçmiş veri: {len(syms)} parite × {a.days} gün ({a.venue})")
        hist = hf.bundle(syms, a.days, progress=lambda m: print("   ", m))
        print(f"» veri {time.time() - t0:.0f} sn · replay başlıyor (adım {a.step} sn)")
        t1 = time.time()
        result = RP.run_replay(hist, syms, Path(a.out) / "tmp", step_sec=a.step, progress=lambda m: print("   ", m),
                               max_cycles=(a.max_cycles or None))
        print(f"» replay {time.time() - t1:.0f} sn · {result['n_cycles']} döngü · {len(result['trades'])} işlem")
    n_trials = max(int(a.trials), RP.trials_count() + 1)      # DSR: gerçek deneme kaydı (silinmez) ile deflate
    an = RP.analyze(result, n_trials=n_trials)
    an["n_trials_registry"] = RP.trials_count() + 1
    p = RP.save_result(result, an, Path(a.out))
    print(f"\n=== GENEL ===  işlem {an['n_trades']} · net {an.get('net_pnl')} $ ({an.get('return_pct')}%) · kazanma {an.get('win_rate')} · "
          f"beklenti %{an.get('expectancy_pct')} CI{an.get('ci95')} · maliyet×2 %{an.get('expectancy_cost_x2_pct')} · "
          f"Sharpe/işlem {an.get('sharpe_per_trade')} · PSR {an.get('psr')} · DSR {an.get('dsr')} · PBO {an.get('pbo')} · "
          f"maks DD %{an.get('max_dd_pct', an.get('max_drawdown_pct'))} · ücret/brüt %{an.get('fee_share_of_gross_pct')} · PCR {an.get('avg_peak_capture')}")
    print("çıkışlar:", an.get("exit_reasons"), "| alt dönem:", an.get("subperiod"))
    lc = Lifecycle(ROOT / "runs" / "live" / "lifecycle.json") if a.evidence else Lifecycle()
    rows = RP.write_evidence(lc, an, source=f"replay {a.days}g") if an.get("per_sleeve") else []
    if rows:
        print(f"\n{'sleeve':18s} {'n':>4s} {'kazanma':>8s} {'beklenti%':>10s} {'CI alt':>8s} {'×2 mal.':>8s} {'tutarlı':>8s} {'DSR':>6s} {'PBO':>5s}  KAPI")
        for r in sorted(rows, key=lambda x: -x["net_pnl"]):
            print(f"{r['sleeve']:18s} {r['n_trades']:4d} {r['win_rate']:8.2f} {r['oos_expectancy']:10.3f} {r['ci_lower']:8.3f} {r['expectancy_cost_x2']:8.3f} "
                  f"{str(r['subperiod_consistent']):>8s} {str(r['dsr']):>6s} {str(r['pbo']):>5s}  {'GEÇTİ' if r['passed'] else 'geçmedi: ' + ', '.join(k for k, v in r['checks'].items() if not v)}")
    m = result.get("missed") or {}
    print(f"\nkaçırılan motoru (replay): kaçırılan {m.get('n_missed')} · kaçınılan {m.get('n_avoided')} · en çok kapı: {(m.get('gates') or [{}])[0].get('gate_tr')}")
    print("sınırlar:", " · ".join(result.get("limits") or []))
    print(f"kayıt: {p}")
    return 0


if __name__ == "__main__":
    _utf8_stdout()
    sys.exit(main())
