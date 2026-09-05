#!/usr/bin/env python3
"""FAZ1 — fiyat-dışı veri: funding rate + open interest geçmişi (perp futures).
Funding aşırılıkları = aşırı-kaldıraç → kontrarian sinyal (gerçek off-price alfa umudu)."""
from __future__ import annotations
import sys, time, csv
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import ccxt, datetime

OUT = Path(__file__).parent / "runs" / "data_funding"
OUT.mkdir(parents=True, exist_ok=True)
PAIRS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "DOGE/USDT:USDT", "AVAX/USDT:USDT"]
START = "2022-01-01T00:00:00Z"


def fetch_funding(ex, sym, since):
    out, ms = [], since
    while True:
        try:
            b = ex.fetch_funding_rate_history(sym, since=ms, limit=1000)
        except Exception as e:
            print(f"  retry funding {sym}: {type(e).__name__}", flush=True); time.sleep(2);
            if len(out)>0: break
            else: return out
        if not b: break
        out += b; ms = b[-1]["timestamp"] + 1
        if len(b) < 1000: break
        if ms > ex.milliseconds(): break
    return out


def main():
    ex = ccxt.binanceusdm({"enableRateLimit": True})
    since = ex.parse8601(START)
    for sym in PAIRS:
        short = sym.split("/")[0]
        fr = fetch_funding(ex, sym, since)
        if not fr:
            print(f"{short}: funding YOK", flush=True); continue
        p = OUT / f"{short}_funding.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["ts","dt","funding_rate"])
            for r in fr:
                ts = r["timestamp"]
                dt = datetime.datetime.fromtimestamp(ts/1000, datetime.UTC).strftime("%Y-%m-%d %H:%M:%S")
                w.writerow([ts, dt, r["fundingRate"]])
        f0 = datetime.datetime.fromtimestamp(fr[0]["timestamp"]/1000, datetime.UTC).strftime("%Y-%m-%d")
        f1 = datetime.datetime.fromtimestamp(fr[-1]["timestamp"]/1000, datetime.UTC).strftime("%Y-%m-%d")
        print(f"{short}: {len(fr)} funding kaydı  {f0} → {f1}", flush=True)
    print("FUNDING İNDİRME TAMAM", flush=True)

if __name__ == "__main__":
    main()
