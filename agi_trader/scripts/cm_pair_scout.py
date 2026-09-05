#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PARİTE KEŞFİ — borsanın gerçek verisiyle "en işlevsel" 10 yeni parite (statik liste değil, ölçüm).

Ölçüt (hepsi ölçülür, uydurulmaz): 24 sa USDT hacmi · üst-defter spread (bps) · ±%2 derinlik (USDT) ·
1 dk σ · 15 dk pencerelerde |hareket| ≥ %0,5 sıklığı (fırsat yoğunluğu) · asgari notional.
Elenir: stablecoin/kaldıraçlı token, mevcut evren, spread > 10 bps, derinlik < 25k $, hacim < 5 M $.

  python scripts/cm_pair_scout.py --venue mexc --top 10 --write runs/live/universe_extra.json
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
STABLE = {"USDC", "USDT", "DAI", "TUSD", "FDUSD", "USDE", "USD1", "BUSD", "PYUSD", "USDD", "EURC", "EUR", "TRY", "BRL"}


def _utf8_stdout():
    if hasattr(sys.stdout, "buffer") and getattr(sys.stdout, "encoding", "").lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def scout(venue: str = "mexc", top: int = 10, candidates: int = 60, exclude=None, client=None, progress=None):
    import numpy as np
    from agi_trader.auto import simulator as SIM
    exclude = set(exclude or (SIM.HEAVY_SYMBOLS + SIM.LIGHT_SYMBOLS))
    if client is None:
        import ccxt
        client = getattr(ccxt, venue)({"enableRateLimit": True})
    markets = client.load_markets()
    tickers = client.fetch_tickers()
    rows = []
    for sym, t in tickers.items():
        m = markets.get(sym) or {}
        if not m.get("active", True) or m.get("spot") is False or not sym.endswith("/USDT") or ":" in sym:
            continue
        base = sym.split("/")[0]
        if base in STABLE or base.endswith(("3L", "3S", "5L", "5S", "UP", "DOWN")) or sym in exclude:
            continue
        qv = float(t.get("quoteVolume") or 0.0)
        if qv < 5_000_000:
            continue
        rows.append({"symbol": sym, "quote_volume_usd": qv})
    rows.sort(key=lambda r: -r["quote_volume_usd"])
    rows = rows[:candidates]
    out = []
    for r in rows:
        s = r["symbol"]
        try:
            ob = client.fetch_order_book(s, limit=20)
            bb, ba = float(ob["bids"][0][0]), float(ob["asks"][0][0]); mid = (bb + ba) / 2
            spread = (ba - bb) / mid * 1e4
            depth = min(sum(float(p) * float(q) for p, q in ob["bids"] if float(p) >= mid * 0.98),
                        sum(float(p) * float(q) for p, q in ob["asks"] if float(p) <= mid * 1.02))
            o = client.fetch_ohlcv(s, "1m", limit=300)
            c = np.array([x[4] for x in o], dtype=float)
            rets = np.diff(np.log(c))
            sigma = float(rets.std() * 100.0)
            win = c[::15]
            moves = np.abs(np.diff(win) / win[:-1]) * 100.0
            move_freq = float((moves >= 0.5).mean()) if len(moves) else 0.0
            minc = float(((markets.get(s) or {}).get("limits") or {}).get("cost", {}).get("min") or 0.0)
            ok = spread <= 10.0 and depth >= 25_000 and minc <= 10.0
            out.append({**r, "spread_bps": round(spread, 2), "depth_2pct_usd": round(depth), "sigma_1m_pct": round(sigma, 4),
                        "move_freq_15m": round(move_freq, 3), "min_cost": minc, "eligible": ok})
            if progress:
                progress(f"{s:12s} hacim {r['quote_volume_usd']/1e6:7.1f}M · spread {spread:5.2f} bps · derinlik {depth/1e3:7.0f}k · σ {sigma:.3f} · hareket {move_freq:.2f} · {'✓' if ok else '✗'}")
            time.sleep(0.15)
        except Exception as e:
            if progress:
                progress(f"{s}: HATA {type(e).__name__}")
    el = [r for r in out if r["eligible"]]
    if not el:
        return {"ts": time.time(), "venue": venue, "symbols": [], "rows": out}
    def z(key, sign=1.0):
        v = np.array([r[key] for r in el], dtype=float)
        sd = v.std() or 1.0
        return sign * (v - v.mean()) / sd
    score = 0.35 * z("quote_volume_usd") + 0.25 * z("depth_2pct_usd") + 0.3 * z("move_freq_15m") - 0.1 * z("spread_bps")
    for r, sc in zip(el, score):
        r["score"] = round(float(sc), 3)
    el.sort(key=lambda r: -r["score"])
    return {"ts": time.time(), "venue": venue, "symbols": [r["symbol"] for r in el[:top]], "rows": el[:top],
            "note": "ölçüt: hacim/derinlik/hareket sıklığı/spread z-skoru; spread ≤ 10 bps, derinlik ≥ 25k $ (emir ≤ 200 $), hacim ≥ 5 M $"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue", default="mexc"); ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--write", default=""); a = ap.parse_args()
    res = scout(a.venue, a.top, progress=lambda m: print("  ", m))
    print("\nSEÇİLEN:", ", ".join(res["symbols"]) or "yok")
    if a.write:
        p = Path(a.write); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8"); print("yazıldı:", p)
    return 0


if __name__ == "__main__":
    _utf8_stdout()
    sys.exit(main())
