#!/usr/bin/env python3
"""
Perp + spot günlük kapanış indirici → BAZ (basis) serisi.

NEDEN KRİTİK: carry sleeve'inin ilk modeli yalnız funding TAHAKKUKUNU sayıyordu.
Gerçek delta-nötr pozisyonun günlük P&L'i:

    long spot + short perp,  b = perp/spot − 1  (baz)
    P&L ≈ funding − Δb

Δb terimi atlanınca getiri yapay olarak pürüzsüzleşir; 2023-2025'te yıllık
volatilite %0,4 gibi imkânsız değerler çıkar ve Sharpe 15-18 görünür. Baz,
perp priminin günlük dalgalanmasıdır ve gerçek risktir — funding tam olarak
bu riskin ücretidir.

  python fetch_basis.py
"""
from __future__ import annotations

import csv
import datetime as dt
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = Path(__file__).parent / "runs" / "data_basis"
OUT.mkdir(parents=True, exist_ok=True)
FUND_DIR = Path(__file__).parent / "runs" / "data_funding"
START = "2019-01-01T00:00:00Z"


def fetch_daily(ex, symbol: str, since_ms: int) -> dict:
    """{tarih: kapanış} — sayfalı günlük kline."""
    out, ms, guard = {}, since_ms, 0
    while guard < 200:
        guard += 1
        try:
            b = ex.fetch_ohlcv(symbol, "1d", since=ms, limit=1000)
        except Exception as e:
            print(f"    yeniden deneme {symbol} ({type(e).__name__})", flush=True)
            time.sleep(2)
            if out:
                break
            return out
        if not b:
            break
        for row in b:
            d = dt.datetime.fromtimestamp(row[0] / 1000, dt.UTC).date()
            out[d] = float(row[4])
        ms = b[-1][0] + 86400000
        if len(b) < 1000 or ms > ex.milliseconds():
            break
    return out


def main() -> None:
    import ccxt
    perp_ex = ccxt.binanceusdm({"enableRateLimit": True})
    spot_ex = ccxt.binance({"enableRateLimit": True})
    pm, sm = perp_ex.load_markets(), spot_ex.load_markets()
    since = perp_ex.parse8601(START)

    coins = sorted(p.stem.replace("_funding", "") for p in FUND_DIR.glob("*_funding.csv"))
    print(f"{len(coins)} parite için perp+spot günlük kapanış indiriliyor…\n", flush=True)

    ok = 0
    for c in coins:
        psym, ssym = f"{c}/USDT:USDT", f"{c}/USDT"
        if psym not in pm or ssym not in sm:
            print(f"  {c:6s} — piyasa eksik, atlandı", flush=True)
            continue
        perp = fetch_daily(perp_ex, psym, since)
        spot = fetch_daily(spot_ex, ssym, since)
        common = sorted(set(perp) & set(spot))
        if len(common) < 400:
            print(f"  {c:6s} — ortak gün {len(common)}, atlandı", flush=True)
            continue
        path = OUT / f"{c}_basis.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Date", "perp", "spot", "basis"])
            for d in common:
                s = spot[d]
                w.writerow([d, perp[d], s, (perp[d] / s - 1.0) if s else ""])
        import statistics as st
        bs = [perp[d] / spot[d] - 1.0 for d in common if spot[d]]
        print(f"  {c:6s} {len(common):>5} gün {common[0]} → {common[-1]} | "
              f"ort baz {st.mean(bs)*1e4:+6.1f} bps | std {st.pstdev(bs)*1e4:6.1f} bps",
              flush=True)
        ok += 1

    print(f"\nTAMAM — {ok} paritede baz serisi → {OUT}", flush=True)


if __name__ == "__main__":
    main()
