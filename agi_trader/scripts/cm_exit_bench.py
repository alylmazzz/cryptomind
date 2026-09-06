#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ÇIKIŞ MOTORU TEZGÂHI — v1 ve v2'yi AYNI fiyat yollarında, AYNI girişlerle karşılaştırır.

NEDEN AYRI BİR TEZGÂH
─────────────────────
Tam replay (scripts/cm_exit_ab.py) komiteyi de çalıştırır; ağır bağlam replay'de
olmadığı için 2 günde yalnız ~10 işlem üretiyor — çıkış motorundaki farkı ölçmek
için ÇOK ZAYIF bir örneklem. Oysa sorulan soru dar ve nettir:

    "Aynı giriş, aynı fiyat yolu verildiğinde yeni çıkış motoru kârın daha
     büyük bir bölümünü tutuyor mu?"

Bu tezgâh girişi TARAFSIZ üretir (sabit aralıklarla, her iki yönde, sinyal
kullanmadan) ve yalnız ÇIKIŞ mantığını değiştirir. Böylece giriş alfası (canlıda
ölçülen brüt ≈ 0) sonucu kirletmez ve binlerce yol üzerinde güç kazanılır.

DÜRÜSTLÜK SINIRLARI (rapora aynen yazılmalı)
  • Girişler sinyal DEĞİL — sonuç "sistem kârlı olur" demez, yalnız çıkış
    motorlarını kıyaslar. Mutlak getiriler anlamsızdır; FARK anlamlıdır.
  • Dolum modeli iki kol için AYNIDIR: stop bar içinde delinirse seviyeden
    (boşlukta bardaki en kötü fiyattan), merdiven basamağı seviyeden dolar.
  • Merdiven ek emir üretir; her ek kısmi çıkış için `--partial-penalty-bps`
    kadar EK maliyet yazılır (varsayılan 1 bps) — merdiven kayırılmasın diye.
  • Barlar 1 dakikalıktır; bar İÇİ sıra (önce tepe mi dip mi) gözlenemez, bu
    yüzden aynı barda hem koruma seviyesi hem stop delinirse STOP kazanır
    (kötümser taraf, iki kolda da aynı).

KULLANIM
  python scripts/cm_exit_bench.py --days 5 --symbols BTC/USDT,ETH/USDT,SOL/USDT --every 20
  python scripts/cm_exit_bench.py --days 5 --all --every 30 --cost-pct 0.07
"""
from __future__ import annotations

import argparse
import io
import json
import math
import statistics as st
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agi_trader.strategies import exit_engine as XE  # noqa: E402

# A/B kolları — YALNIZ çıkış parametreleri değişir.
ARMS: Dict[str, Dict] = {
    "v1":         {"lock_mode": "breakeven", "protect_before_min_hold": False, "ladder_enabled": False},
    "v2":         {},                                                            # kilit + merdiven + asgari-tutma
    "v2-lock":    {"ladder_enabled": False},                                     # kilit + asgari-tutma
    "v2-ladder":  {"lock_mode": "breakeven", "protect_before_min_hold": False},   # yalnız merdiven
    "v2-nomin":   {"lock_mode": "breakeven", "ladder_enabled": False},            # yalnız asgari-tutma
    "lock-only":  {"protect_before_min_hold": False, "ladder_enabled": False},    # YALNIZ kâr kilidi
    # retain ızgarası — "kârın yarısı" gerçekten en iyisi mi? (ölçülmeden sabitlenmez)
    "lock-r35":   {"protect_before_min_hold": False, "ladder_enabled": False, "retain_fraction": 0.35},
    "lock-r65":   {"protect_before_min_hold": False, "ladder_enabled": False, "retain_fraction": 0.65},
    "lock-r70":   {"protect_before_min_hold": False, "ladder_enabled": False, "retain_fraction": 0.70},
    # merdiven varyantları — 1R yerine daha geç başlayıp daha az dilim
    "ladder-late": {"ladder_enabled": True, "ladder_levels": 2, "ladder_r": [2.0, 3.5],
                    "ladder_fracs": [0.30, 0.25]},
    "ladder-1lvl": {"ladder_enabled": True, "ladder_levels": 1, "ladder_r": [2.5],
                    "ladder_fracs": [0.33]},
}


def _utf8_stdout():
    if hasattr(sys.stdout, "buffer") and getattr(sys.stdout, "encoding", "").lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def entry_ok(kind: str, highs, lows, closes, i: int, direction: str) -> bool:
    """Giriş KOŞULLAMASI — sinyal değil, POPÜLASYON seçimi.

    Tarafsız (`all`) girişlerde ortalama tepe hareketi küçüktür; canlı sistemin
    aldığı kurulumlar ise seçilmiş kurulumlardır (dip/kırılım). Çıkış motorları
    aynı girişleri gördüğü için kıyas yine adildir; değişen yalnız hangi fiyat
    yolları ailesinde ölçtüğümüzdür. Canlıya benzeyen ailede ölçmek daha bilgilendirici."""
    if kind == "all":
        return True
    n = 20
    if i < n + 2:
        return False
    win = closes[i - n:i]
    m = sum(win) / n
    sd = (sum((x - m) ** 2 for x in win) / n) ** 0.5 or 1e-12
    z = (closes[i] - m) / sd
    if kind == "dip":         # ortalamanın altına sarkma + son bar yeşil (canlı 'dip' sleeve'i)
        return (z <= -1.2 and closes[i] > closes[i - 1]) if direction == "LONG" else \
               (z >= 1.2 and closes[i] < closes[i - 1])
    if kind == "breakout":    # 20 barın tepesini/dibini kırma
        return closes[i] >= max(highs[i - n:i]) if direction == "LONG" else \
               closes[i] <= min(lows[i - n:i])
    if kind == "momentum":    # z ile aynı yönde
        return z >= 1.0 if direction == "LONG" else z <= -1.0
    return True


def atr_pct_at(highs, lows, closes, i: int, n: int = 14) -> float:
    """Basit ATR% (yüzde). Geçmiş barlardan — ileriye bakış YOK."""
    a, j = 0.0, 0
    for k in range(max(1, i - n + 1), i + 1):
        tr = max(highs[k] - lows[k], abs(highs[k] - closes[k - 1]), abs(lows[k] - closes[k - 1]))
        a += tr; j += 1
    if not j or not closes[i]:
        return 0.3
    return max(0.02, a / j / closes[i] * 100.0)


def run_one(arm_over: Dict, direction: str, bars, i0: int, stop_pct: float, rr: float,
            cost_pct: float, atr_pct: float, horizon_bars: int, min_hold_sec: int,
            partial_penalty_bps: float, exit_mode: str) -> Optional[Dict]:
    """Tek girişi tek kolla oynat. Dönen: gerçekleşen NET % (başlangıç notional'ının),
    ulaşılan tepe NET %, çıkış sebebi, merdiven basamak sayısı."""
    ts, o, h, l, c = bars
    n = len(c)
    entry = float(c[i0])
    if not (entry > 0):
        return None
    s = 1.0 if direction == "LONG" else -1.0
    p = XE.ExitParams(min_hold_sec=min_hold_sec, time_stop_sec=horizon_bars * 60, **arm_over).validated()
    t = XE.PositionTrack(direction, entry, entry * (1 - s * stop_pct / 100.0),
                         entry * (1 + s * rr * stop_pct / 100.0), float(ts[i0]),
                         exit_mode, stop_pct, cost_pct, atr_pct)
    t.highest_high = t.lowest_low = entry
    t.build_ladder(p)
    weight = 1.0
    realized = 0.0
    n_partials = 0
    reason = "TIME_STOP"
    end = min(n - 1, i0 + horizon_bars)
    for i in range(i0 + 1, end + 1):
        d = XE.decide_exit(t, float(c[i]), float(h[i]), float(l[i]), p, float(ts[i]))
        if not d:
            continue
        if d.get("partial"):
            f = min(float(d.get("fraction") or 0.0), weight)
            if f <= 0:
                continue
            fill = float(d.get("exit_price") or c[i])
            realized += f * (t.net_pct(fill) - partial_penalty_bps / 100.0)
            weight -= f
            n_partials += 1
            if weight <= 1e-9:
                reason = "LADDER_SON"
                break
            continue
        fill = float(d.get("exit_price") or c[i])
        realized += weight * t.net_pct(fill)
        weight = 0.0
        reason = d["reason"]
        break
    if weight > 1e-9:                                   # ufuk doldu, kalanı kapanıştan çık
        realized += weight * t.net_pct(float(c[end]))
    return {"net_pct": realized, "peak_net_pct": t.peak_net_pct, "reason": reason,
            "levels_hit": t.levels_hit, "n_partials": n_partials}


def summarize(rows: List[Dict]) -> Dict:
    if not rows:
        return {"n": 0}
    net = [r["net_pct"] for r in rows]
    pk = [r["peak_net_pct"] for r in rows]
    pcr = [r["net_pct"] / r["peak_net_pct"] for r in rows if r["peak_net_pct"] > 0]
    w = [x for x in net if x > 0]
    lo = [x for x in net if x <= 0]
    reasons: Dict[str, int] = {}
    for r in rows:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    mean = st.mean(net)
    sd = st.pstdev(net) or 1e-12
    return {
        "n": len(rows),
        "expectancy_pct": round(mean, 5),
        "stderr": round(sd / math.sqrt(len(net)), 5),
        "t_stat": round(mean / (sd / math.sqrt(len(net))), 2),
        "total_pct": round(sum(net), 3),
        "win_rate": round(len(w) / len(net), 4),
        "avg_win": round(st.mean(w), 5) if w else 0.0,
        "avg_loss": round(st.mean(lo), 5) if lo else 0.0,
        "payoff": round(st.mean(w) / abs(st.mean(lo)), 3) if (w and lo and st.mean(lo) != 0) else 0.0,
        "pf": round(sum(w) / abs(sum(lo)), 3) if lo and sum(lo) < 0 else None,
        "peak_net_avg": round(st.mean(pk), 4),
        "pcr_avg": round(st.mean(pcr), 4) if pcr else None,
        "partials": sum(r["n_partials"] for r in rows),
        "reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
    }


def paired_test(a: List[Dict], b: List[Dict]) -> Dict:
    """EŞLEŞTİRİLMİŞ fark testi: aynı giriş, aynı yol → fark dizisi üzerinde t.
    Eşleştirme, piyasa gürültüsünü ortadan kaldırır; kalan fark ÇIKIŞ motorudur."""
    d = [x["net_pct"] - y["net_pct"] for x, y in zip(a, b)]
    if not d:
        return {}
    m = st.mean(d)
    sd = st.pstdev(d) or 1e-12
    se = sd / math.sqrt(len(d))
    return {"n": len(d), "mean_diff_pct": round(m, 5), "stderr": round(se, 5),
            "t_stat": round(m / se, 2), "ci95": (round(m - 1.96 * se, 5), round(m + 1.96 * se, 5)),
            "better": sum(1 for x in d if x > 1e-9), "worse": sum(1 for x in d if x < -1e-9),
            "same": sum(1 for x in d if abs(x) <= 1e-9)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=5.0)
    ap.add_argument("--symbols", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--venue", default="mexc")
    ap.add_argument("--every", type=int, default=20, help="kaç dakikada bir giriş üretilsin")
    ap.add_argument("--stops", default="0.4,0.7,1.0,1.5,2.5", help="stop mesafesi ızgarası (%)")
    ap.add_argument("--rr", type=float, default=1.6)
    ap.add_argument("--cost-pct", type=float, default=0.07, help="gidiş-dönüş maliyet (canlı medyan %0,07)")
    ap.add_argument("--horizon-min", type=int, default=60)
    ap.add_argument("--min-hold-sec", type=int, default=900)
    ap.add_argument("--partial-penalty-bps", type=float, default=1.0)
    ap.add_argument("--exit-mode", default=XE.PARTIAL_AND_RUN, choices=list(XE.MODES))
    ap.add_argument("--entry", default="all", choices=("all", "dip", "breakout", "momentum"),
                    help="giriş POPÜLASYONU (sinyal değil) — canlıya benzeyen aile için dip/breakout")
    ap.add_argument("--arms", default="v1,v2,v2-lock,v2-ladder,v2-nomin")
    ap.add_argument("--out", default=str(ROOT / "runs" / "exit_bench"))
    a = ap.parse_args()

    from agi_trader.auto import replay as RP
    from agi_trader.auto import simulator as SIM

    syms = SIM.HEAVY_SYMBOLS + SIM.LIGHT_SYMBOLS if a.all else \
        [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    if not syms:
        syms = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    stops = [float(x) for x in a.stops.split(",") if x.strip()]
    arms = [x.strip() for x in a.arms.split(",") if x.strip() in ARMS]

    print(f"» geçmiş veri: {len(syms)} parite × {a.days} gün ({a.venue})")
    hf = RP.HistoryFetcher(a.venue)
    hist = hf.bundle(syms, a.days, progress=lambda m: print("   ", m))

    rows: Dict[str, List[Dict]] = {k: [] for k in arms}
    per_stop: Dict[float, Dict[str, List[Dict]]] = {s: {k: [] for k in arms} for s in stops}
    n_entries = 0
    t0 = time.time()
    for sym in syms:
        df = hist.get((sym, "1m"))
        if df is None or len(df) < 200:
            continue
        ts = [x.timestamp() for x in df.index]
        o, h, l, c = (list(df["open"]), list(df["high"]), list(df["low"]), list(df["close"]))
        bars = (ts, o, h, l, c)
        for i0 in range(60, len(c) - a.horizon_min - 1, a.every):
            atr = atr_pct_at(h, l, c, i0)
            for direction in ("LONG", "SHORT"):
                if not entry_ok(a.entry, h, l, c, i0, direction):
                    continue
                for sp in stops:
                    res = {}
                    ok = True
                    for arm in arms:
                        r = run_one(ARMS[arm], direction, bars, i0, sp, a.rr, a.cost_pct, atr,
                                    a.horizon_min, a.min_hold_sec, a.partial_penalty_bps, a.exit_mode)
                        if r is None:
                            ok = False
                            break
                        res[arm] = r
                    if not ok:
                        continue
                    n_entries += 1
                    for arm in arms:
                        rows[arm].append(res[arm])
                        per_stop[sp][arm].append(res[arm])
        print(f"   {sym}: toplam {n_entries} giriş · {time.time() - t0:.0f} sn")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    summ = {k: summarize(v) for k, v in rows.items()}
    pairs = {k: paired_test(rows[k], rows["v1"]) for k in arms if k != "v1"} if "v1" in arms else {}
    per_stop_s = {str(s): {k: summarize(v) for k, v in d.items()} for s, d in per_stop.items()}
    p = out / f"exit_bench_{int(time.time())}.json"
    p.write_text(json.dumps({"symbols": syms, "days": a.days, "every": a.every, "stops": stops,
                             "entry": a.entry, "min_hold_sec": a.min_hold_sec,
                             "cost_pct": a.cost_pct, "rr": a.rr, "horizon_min": a.horizon_min,
                             "n_entries": n_entries, "summary": summ, "paired_vs_v1": pairs,
                             "per_stop": per_stop_s}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 104)
    print(f"{'kol':11s} {'n':>6s} {'beklenti%':>10s} {'t':>7s} {'kazanma':>8s} {'öd.oranı':>9s} "
          f"{'PF':>6s} {'PCR':>7s} {'tepe%':>7s} {'kısmi':>7s}")
    print("-" * 104)
    for k in arms:
        s = summ[k]
        if not s.get("n"):
            continue
        print(f"{k:11s} {s['n']:6d} {s['expectancy_pct']:10.5f} {s['t_stat']:7.2f} {s['win_rate']:8.4f} "
              f"{s['payoff']:9.3f} {(s['pf'] if s['pf'] is not None else float('nan')):6.3f} "
              f"{(s['pcr_avg'] if s['pcr_avg'] is not None else float('nan')):7.4f} "
              f"{s['peak_net_avg']:7.4f} {s['partials']:7d}")
    print("=" * 104)
    if pairs:
        print("\nEŞLEŞTİRİLMİŞ FARK (kol − v1) — aynı giriş, aynı yol; piyasa gürültüsü elenir")
        print(f"{'kol':11s} {'n':>6s} {'ort fark%':>10s} {'t':>8s} {'%95 GA':>26s} {'iyi':>7s} {'kötü':>7s} {'aynı':>7s}")
        for k, d in pairs.items():
            if not d:
                continue
            print(f"{k:11s} {d['n']:6d} {d['mean_diff_pct']:10.5f} {d['t_stat']:8.2f} "
                  f"{str(d['ci95']):>26s} {d['better']:7d} {d['worse']:7d} {d['same']:7d}")
        print("\nKARAR KURALI: %95 güven aralığının ALT sınırı 0'ın üstünde değilse fark KANITLANMAMIŞTIR.")
    print("\nstop mesafesine göre beklenti% (v1 → v2):")
    for s in stops:
        a1 = per_stop_s[str(s)].get("v1", {}); a2 = per_stop_s[str(s)].get("v2", {})
        if a1.get("n") and a2.get("n"):
            print(f"   stop %{s:<5g} n={a1['n']:6d}  v1 {a1['expectancy_pct']:+.5f}  →  v2 {a2['expectancy_pct']:+.5f}  "
                  f"(PCR {a1.get('pcr_avg')} → {a2.get('pcr_avg')})")
    for k in arms:
        if summ[k].get("n"):
            print(f"\n{k:11s} çıkışlar: {summ[k]['reasons']}")
    print(f"\nkayıt: {p}")
    print("SINIR: girişler sinyal değildir — bu tezgâh yalnız ÇIKIŞ motorlarını kıyaslar, "
          "sistemin kârlı olacağını göstermez.")
    return 0


if __name__ == "__main__":
    _utf8_stdout()
    sys.exit(main())
