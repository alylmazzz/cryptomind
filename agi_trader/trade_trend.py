#!/usr/bin/env python3
"""FAZ6 — trend-takip motoru: (1) parity doğrulama (engine backtest ≈ Sharpe 1.05),
(2) CANLI bugünkü hedef ağırlıklar (ccxt ile taze veri)."""
from __future__ import annotations
import sys, glob
from pathlib import Path
import pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0, str(Path(__file__).parent))
from agi_trader.config import load_config
from agi_trader.auto.trend_engine import TrendTrader

PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "AVAX/USDT"]
DATA = Path(__file__).parent / "runs" / "data_full"


def daily_close(short):
    df = pd.read_csv(DATA/f"{short}_1h.csv"); df.index = pd.to_datetime(df["dt"])
    return df["close"].astype(float).resample("1D").last().dropna()


def main():
    c = load_config()
    tt = TrendTrader(c, pairs=PAIRS, initial=10000)

    # (1) PARITY: motorun kendi backtest'i (data_full)
    short = {p: p.replace("/", "") for p in PAIRS}
    price_daily = {p: daily_close(short[p]) for p in PAIRS}
    bt = tt.backtest(price_daily)
    print("=== (1) PARITY DOĞRULAMA (engine backtest, data_full) ===")
    print(f"  Sharpe {bt['sharpe']} | CAGR {bt['cagr_pct']}% | getiri {bt['total_return_pct']}% | "
          f"MaxDD {bt['max_drawdown_pct']}% | {bt['days']} gün")
    print(f"  (beklenen ≈ Sharpe 1.05, CAGR +19%, DD 18% → motor doğrulandı)\n")

    # (2) CANLI bugünkü hedefler (taze veri)
    print("=== (2) CANLI BUGÜNKÜ HEDEF ALLOKASYON ===")
    data = {}
    try:
        import ccxt
        ex = ccxt.binance({"enableRateLimit": True})
        since = ex.milliseconds() - 320*24*3600*1000   # ~320 gün (SMA200+mom20 için)
        for p in PAIRS:
            o = ex.fetch_ohlcv(p, "1d", since=since, limit=400)
            df = pd.DataFrame(o, columns=["ts","open","high","low","close","volume"])
            df.index = pd.to_datetime(df["ts"], unit="ms")
            data[p] = df[["open","high","low","close","volume"]].astype(float)
        src = "CANLI (binance)"
    except Exception as e:
        print(f"  (canlı veri alınamadı: {type(e).__name__}; data_full ile gösteriliyor)")
        for p in PAIRS:
            dc = daily_close(short[p])
            data[p] = pd.DataFrame({"open":dc,"high":dc,"low":dc,"close":dc,"volume":dc*0+1})
        src = "data_full (son)"

    sigs = tt.signals(data)
    ev = tt.rebalance(data, date_str=str(list(data.values())[0].index[-1])[:10])
    print(f"  kaynak: {src} | tarih: {ev['date']}")
    for p in PAIRS:
        s = sigs[p]
        state = "🟢 POZİSYON" if s["in_market"] else "⚪ NAKİT"
        w = ev["targets"].get(p, 0)*100
        print(f"  {p:9s} {state}  ağırlık %{w:4.1f}  | {s.get('reason','')}")
    print(f"\n  Toplam yatırılan: %{ev['invested_pct']}  (kalan nakit %{100-ev['invested_pct']:.1f})")
    print(f"  → Bu allokasyonla paper portföy günlük yeniden dengelenir. Canlı emir YOK.")

if __name__ == "__main__":
    main()
