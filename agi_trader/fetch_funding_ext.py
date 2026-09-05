#!/usr/bin/env python3
"""
Genişletilmiş funding geçmişi indirici (FAZ 3b).

NEDEN: carry sleeve'i ölçüldüğünde Sharpe 0,80 / kitapla korelasyon +0,01 çıktı —
ekonomik olarak tam istenen profil— ama DSR 0,66'da kaldı. Sorun parametre
DEĞİL, ÖRNEKLEM: yalnız 5 parite × 4 yıl. Çözüm eşik oynatmak değil, veri:
  • 5 → ~24 parite (carry'nin kendi içinde çeşitlenmesi)
  • 2022 → 2019 (BTC funding 2019-09-10'da başlıyor)

Bu, aşırı uyum değildir: hipotez değişmiyor, sadece daha çok kanıt toplanıyor.

  python fetch_funding_ext.py            # varsayılan 24 majör
  python fetch_funding_ext.py --top 40   # en likit 40 perp
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = Path(__file__).parent / "runs" / "data_funding"
OUT.mkdir(parents=True, exist_ok=True)
START = "2019-01-01T00:00:00Z"

# Uzun geçmişi ve derin likiditesi olan majörler (küratörlü — rastgele altcoin
# eklemek carry'yi iyileştirmez, yalnız tasfiye/kuyruk riskini artırır)
MAJORS = ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX", "DOT", "LINK",
          "LTC", "UNI", "ATOM", "ETC", "FIL", "NEAR", "ICP", "TRX", "BCH", "XLM",
          "EOS", "AAVE", "ALGO", "VET"]


def fetch_all(ex, symbol: str, since_ms: int) -> list:
    """Sayfalı tam funding geçmişi."""
    out, ms, guard = [], since_ms, 0
    while guard < 500:
        guard += 1
        try:
            batch = ex.fetch_funding_rate_history(symbol, since=ms, limit=1000)
        except Exception as e:
            print(f"    yeniden deneme ({type(e).__name__})", flush=True)
            time.sleep(2)
            if out:
                break
            return out
        if not batch:
            break
        out += batch
        ms = batch[-1]["timestamp"] + 1
        if len(batch) < 1000 or ms > ex.milliseconds():
            break
    # tekilleştir + sırala
    seen, uniq = set(), []
    for r in sorted(out, key=lambda x: x["timestamp"]):
        if r["timestamp"] in seen:
            continue
        seen.add(r["timestamp"])
        uniq.append(r)
    return uniq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=0,
                    help="küratörlü liste yerine hacme göre en likit N perp")
    args = ap.parse_args()

    import ccxt
    ex = ccxt.binanceusdm({"enableRateLimit": True})
    markets = ex.load_markets()

    if args.top:
        tickers = ex.fetch_tickers()
        cand = [(s, t.get("quoteVolume") or 0) for s, t in tickers.items()
                if markets.get(s, {}).get("swap") and markets[s].get("quote") == "USDT"]
        cand.sort(key=lambda x: -x[1])
        symbols = [s for s, _ in cand[:args.top]]
    else:
        symbols = [f"{c}/USDT:USDT" for c in MAJORS]

    since = ex.parse8601(START)
    ok = 0
    print(f"{len(symbols)} parite için funding geçmişi indiriliyor "
          f"({START[:10]}'dan itibaren)…\n", flush=True)

    for sym in symbols:
        base = sym.split("/")[0]
        if sym not in markets:
            print(f"  {base:6s} — piyasa yok, atlandı", flush=True)
            continue
        rows = fetch_all(ex, sym, since)
        if len(rows) < 500:
            print(f"  {base:6s} — yalnız {len(rows)} kayıt, atlandı (geçmiş kısa)",
                  flush=True)
            continue
        path = OUT / f"{base}_funding.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ts", "dt", "funding_rate"])
            for r in rows:
                ts = r["timestamp"]
                w.writerow([ts,
                            dt.datetime.fromtimestamp(ts / 1000, dt.UTC)
                              .strftime("%Y-%m-%d %H:%M:%S"),
                            r["fundingRate"]])
        d0 = dt.datetime.fromtimestamp(rows[0]["timestamp"] / 1000, dt.UTC).date()
        d1 = dt.datetime.fromtimestamp(rows[-1]["timestamp"] / 1000, dt.UTC).date()
        yil = (d1 - d0).days / 365.25
        print(f"  {base:6s} {len(rows):>6} kayıt  {d0} → {d1}  ({yil:.1f} yıl)", flush=True)
        ok += 1

    print(f"\nTAMAM — {ok} paritede funding geçmişi → {OUT}", flush=True)


if __name__ == "__main__":
    main()
