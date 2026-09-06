#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ÇIKIŞ MOTORU A/B — v1 (başabaş kilidi, merdiven yok) vs v2 (kâr kilidi + merdiven).

AYNI geçmiş veri, AYNI komite, AYNI giriş mantığı; farklı YALNIZ çıkış parametreleri.
Böylece "kârı daha çok koruyor mu?" sorusu, giriş gürültüsünden ayrılarak ölçülür.

  python scripts/cm_exit_ab.py --days 3 --symbols BTC/USDT,ETH/USDT,SOL/USDT --venue mexc
  python scripts/cm_exit_ab.py --days 3 --all --step 60

Kollar:
  v1        lock_mode=breakeven · protect_before_min_hold=False · ladder_enabled=False
  v2        (varsayılan) kâr kilidi + T1..Tn merdiven + koruma asgari tutmadan muaf
  v2-lock   yalnız kâr kilidi (merdiven kapalı) — hangi parçanın ne kattığını ayırmak için
  v2-ladder yalnız merdiven (kilit v1) — aynı sebeple

ÇIKTI dürüsttür: örneklem küçükse fark GÜRÜLTÜ olabilir. Bu yüzden işlem sayısı,
bootstrap %95 güven aralığı ve maliyet×2 dayanıklılığı da yazılır.
"""
from __future__ import annotations

import argparse
import io
import json
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ARMS = {
    "v1": {"lock_mode": "breakeven", "protect_before_min_hold": False, "ladder_enabled": False},
    "v2": {},
    "v2-lock": {"ladder_enabled": False},
    "v2-ladder": {"lock_mode": "breakeven", "protect_before_min_hold": False},
}


def _utf8_stdout():
    if hasattr(sys.stdout, "buffer") and getattr(sys.stdout, "encoding", "").lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def summarize(trades, capital: float) -> dict:
    if not trades:
        return {"n": 0}
    net = [float(t.get("net_pnl") or 0.0) for t in trades]
    gross = [float(t.get("gross_pnl") or 0.0) for t in trades]
    fees = [float(t.get("fees") or 0.0) for t in trades]
    w = [x for x in net if x > 0]
    l = [x for x in net if x <= 0]
    pcr = [t["peak_capture"] for t in trades if t.get("peak_capture") is not None]
    pk = [float(t.get("peak_net_pct") or 0.0) for t in trades]
    npc = [float(t.get("net_pct_realized") or 0.0) for t in trades]
    reasons = {}
    for t in trades:
        reasons[t.get("reason", "?")] = reasons.get(t.get("reason", "?"), 0) + 1
    # bootstrap %95 CI (işlem başına net %)
    try:
        import numpy as np
        a = np.array(npc, dtype=float)
        rng = np.random.default_rng(7)
        bs = np.array([rng.choice(a, len(a), replace=True).mean() for _ in range(2000)])
        ci = (round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4))
    except Exception:
        ci = (None, None)
    # maliyet×2 dayanıklılığı: her işlemin maliyeti iki katına çıksa beklenti ne olur
    cost_pct = [(float(t.get("fees") or 0.0) / max(1e-9, float(t.get("notional") or 0.0))) * 100.0 for t in trades]
    exp_x2 = st.mean([n - c for n, c in zip(npc, cost_pct)]) if npc else 0.0
    return {
        "n": len(trades),
        "net": round(sum(net), 4), "gross": round(sum(gross), 4), "fees": round(sum(fees), 4),
        "return_pct": round(sum(net) / max(1e-9, capital) * 100.0, 4),
        "win_rate": round(len(w) / len(net), 3),
        "avg_win": round(st.mean(w), 4) if w else 0.0,
        "avg_loss": round(st.mean(l), 4) if l else 0.0,
        "payoff": round((st.mean(w) / abs(st.mean(l))) if (w and l and st.mean(l) != 0) else 0.0, 3),
        "pf": round(sum(w) / abs(sum(l)), 3) if l and sum(l) < 0 else None,
        "expectancy_pct": round(st.mean(npc), 4),
        "expectancy_cost_x2_pct": round(exp_x2, 4),
        "ci95_expectancy_pct": ci,
        "peak_net_avg": round(st.mean(pk), 4),
        "pcr_avg": round(st.mean(pcr), 3) if pcr else None,
        "levels_hit_total": sum(int(t.get("levels_hit") or 0) for t in trades),
        "trades_with_partial": sum(1 for t in trades if int(t.get("levels_hit") or 0) > 0 or t.get("partial_done")),
        "fee_share_of_gross": round(sum(fees) / max(1e-9, sum(abs(g) for g in gross)), 3),
        "reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=3.0)
    ap.add_argument("--symbols", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--venue", default="mexc")
    ap.add_argument("--step", type=int, default=60)
    ap.add_argument("--arms", default="v1,v2,v2-lock,v2-ladder")
    ap.add_argument("--out", default=str(ROOT / "runs" / "exit_ab"))
    a = ap.parse_args()

    from agi_trader.auto import replay as RP
    from agi_trader.auto import simulator as SIM

    syms = SIM.HEAVY_SYMBOLS + SIM.LIGHT_SYMBOLS if a.all else \
        [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    if not syms:
        syms = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    print(f"» geçmiş veri: {len(syms)} parite × {a.days} gün ({a.venue})")
    t0 = time.time()
    hf = RP.HistoryFetcher(a.venue)
    hist = hf.bundle(syms, a.days, progress=lambda m: print("   ", m))
    print(f"» veri {time.time() - t0:.0f} sn")

    arms = [x.strip() for x in a.arms.split(",") if x.strip() in ARMS]
    res = {}
    for name in arms:
        t1 = time.time()
        print(f"\n» KOL {name}: {ARMS[name] or 'varsayılan (v2)'}")
        r = RP.run_replay(hist, syms, out / name, cfg_overrides={"exit": ARMS[name]},
                          step_sec=a.step, progress=lambda m: print("   ", m))
        s = summarize(r["trades"], float(r.get("capital") or 1000.0))
        s["seconds"] = round(time.time() - t1)
        res[name] = s
        print(f"   → {s['n']} işlem · net {s.get('net')} $ · PCR {s.get('pcr_avg')} · PF {s.get('pf')}")

    p = out / f"exit_ab_{int(time.time())}.json"
    p.write_text(json.dumps({"symbols": syms, "days": a.days, "step": a.step, "arms": res},
                            ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 108)
    hdr = f"{'kol':10s} {'n':>4s} {'net $':>9s} {'getiri%':>8s} {'kazanma':>8s} {'öd.oranı':>9s} {'PF':>6s} " \
          f"{'beklenti%':>10s} {'CI alt':>8s} {'×2mal.%':>8s} {'PCR':>6s} {'tepe%':>7s} {'kısmi':>6s}"
    print(hdr)
    print("-" * 108)
    for name in arms:
        s = res[name]
        if not s.get("n"):
            print(f"{name:10s}    0  (işlem yok)")
            continue
        ci_lo = (s["ci95_expectancy_pct"] or [None])[0]
        print(f"{name:10s} {s['n']:4d} {s['net']:9.3f} {s['return_pct']:8.3f} {s['win_rate']:8.3f} "
              f"{s['payoff']:9.3f} {(s['pf'] if s['pf'] is not None else float('nan')):6.3f} "
              f"{s['expectancy_pct']:10.4f} {(ci_lo if ci_lo is not None else float('nan')):8.4f} "
              f"{s['expectancy_cost_x2_pct']:8.4f} {(s['pcr_avg'] or 0):6.3f} {s['peak_net_avg']:7.3f} "
              f"{s['trades_with_partial']:6d}")
    print("=" * 108)
    for name in arms:
        if res[name].get("n"):
            print(f"{name:10s} çıkışlar: {res[name]['reasons']}")
    if "v1" in res and "v2" in res and res["v1"].get("n") and res["v2"].get("n"):
        d = res["v2"]["net"] - res["v1"]["net"]
        dp = (res["v2"]["pcr_avg"] or 0) - (res["v1"]["pcr_avg"] or 0)
        print(f"\nFARK (v2 − v1): net {d:+.3f} $ · PCR {dp:+.3f} · beklenti "
              f"{res['v2']['expectancy_pct'] - res['v1']['expectancy_pct']:+.4f} puan")
        print("UYARI: tek dönem tek örneklemdir. CI alt sınırı 0'ın altındaysa fark KANITLANMAMIŞTIR.")
    print(f"\nkayıt: {p}")
    return 0


if __name__ == "__main__":
    _utf8_stdout()
    sys.exit(main())
