#!/usr/bin/env python3
"""Çok-rejimli uzun geçmiş indirici (2022 bear → 2024 bull → 2025 chop → 2026).
1h OHLCV, runs/data_full/SYMBOL_1h.csv (ts,open,high,low,close,volume,dt)."""
from __future__ import annotations
import sys, time, csv
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import ccxt

OUT = Path(__file__).parent / "runs" / "data_full"
OUT.mkdir(parents=True, exist_ok=True)
PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "AVAX/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT"]
START = "2022-01-01T00:00:00Z"
TF = "1h"


def fetch_all(ex, symbol, since_ms, tf="1h"):
    out, ms = [], since_ms
    step = ex.parse_timeframe(tf) * 1000
    while True:
        try:
            batch = ex.fetch_ohlcv(symbol, tf, since=ms, limit=1000)
        except Exception as e:
            print(f"  retry {symbol} @ {ms}: {type(e).__name__}", flush=True); time.sleep(2); continue
        if not batch:
            break
        out += batch
        ms = batch[-1][0] + step
        if len(batch) < 1000:
            break
        if ms > ex.milliseconds():
            break
    # dedupe + sort
    seen, rows = set(), []
    for r in sorted(out, key=lambda x: x[0]):
        if r[0] in seen: continue
        seen.add(r[0]); rows.append(r)
    return rows


def main():
    ex = ccxt.binance({"enableRateLimit": True})
    since = ex.parse8601(START)
    import datetime
    for sym in PAIRS:
        short = sym.replace("/", "")
        rows = fetch_all(ex, sym, since, TF)
        if not rows:
            print(f"{short}: VERİ YOK", flush=True); continue
        p = OUT / f"{short}_{TF}.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["ts","open","high","low","close","volume","dt"])
            for ts,o,h,l,c,v in rows:
                dt = datetime.datetime.utcfromtimestamp(ts/1000).strftime("%Y-%m-%d %H:%M:%S")
                w.writerow([ts,o,h,l,c,v,dt])
        d0 = rows[0][0]; d1 = rows[-1][0]
        f0 = datetime.datetime.utcfromtimestamp(d0/1000).strftime("%Y-%m-%d")
        f1 = datetime.datetime.utcfromtimestamp(d1/1000).strftime("%Y-%m-%d")
        print(f"{short}: {len(rows)} bar  {f0} → {f1}  → {p.name}", flush=True)
    print("INDIRME TAMAM", flush=True)

if __name__ == "__main__":
    main()
