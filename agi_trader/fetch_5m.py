#!/usr/bin/env python3
"""5 dakikalık USD-M vadeli kline indirici — nitelendirme katmanının TABAN serisi.

NEDEN 5 DAKİKA
  1h barla 5m/15m/30m ufukları ÖLÇÜLEMEZ; "hedef mi stop mu önce geldi" sorusu
     bir 1h barın içinde cevapsızdır.
  2 Uzun ufuklarda da aynı-bar belirsizliğini büyük ölçüde çözer: ±%1'lik iki
     bariyerin AYNI 5 dakikalık barda vurulması nadirdir; olduğunda örnek
     AMBIGUOUS sayılır ve ölçümden düşer (varsayım yapılmaz).

NEDEN VADELİ (futures/um), SPOT DEĞİL
  Kaydedici (recorder.py) funding, mark/index, açık pozisyon ve L2 defterini
  Binance USD-M vadeli borsasından alıyor. Fiyat serisi başka bir venue'den
  gelirse maliyet ve mikroyapı katmanı fiyat serisiyle tutarsız olur.
  Bu yüzden nitelendirme katmanının TEK fiyat kaynağı vadeli 5m'dir.

Kaynak: data.binance.vision aylık zip arşivi (anahtarsız, ücretsiz, rate-limit yok).
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

BASE = "https://data.binance.vision/data/futures/um/monthly/klines"
OUT = Path(__file__).parent / "runs" / "data_5m"
UNIVERSE = Path(__file__).parent / "runs" / "qualification" / "universe_5m.json"
FALLBACK_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
                    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT"]
WORKERS = 4          # arşiv sunucusuna saygılı paralellik
TF = "5m"
START = (2022, 1)
COLS = ["ts", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]


def _months(start, end):
    y, m = start
    ey, em = end
    while (y, m) <= (ey, em):
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def _fetch_month(sym: str, y: int, m: int, tries: int = 3):
    url = f"{BASE}/{sym}/{TF}/{sym}-{TF}-{y:04d}-{m:02d}.zip"
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                raw = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None            # o ay yok (listelenmemiş / gelecek)
            if k == tries - 1:
                raise
            time.sleep(2 + 3 * k)
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 + 3 * k)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            head = f.read(64)
        # 2025 sonrası dosyalar başlık satırı taşıyor — otomatik algıla
        skip = 1 if head[:9] == b"open_time" else 0
        with z.open(name) as f:
            d = pd.read_csv(f, header=None, names=COLS, skiprows=skip)
    return d


def _symbols(arg: str = "") -> list:
    """Evren: --symbols · universe_5m.json · yedek liste (bu sırayla)."""
    if arg:
        return [s for s in arg.split(",") if s]
    if UNIVERSE.exists():
        try:
            u = json.loads(UNIVERSE.read_text(encoding="utf-8"))
            syms = [x["symbol"] for x in u.get("selected", [])]
            if syms:
                return syms
        except Exception:
            pass
    return list(FALLBACK_SYMBOLS)


def _one(sym: str, end, force: bool) -> str:
    hedef = OUT / f"{sym}_5m.parquet"
    if hedef.exists() and not force:
        try:
            import pyarrow.parquet as pq
            return f"{sym}: atlandı ({pq.ParquetFile(hedef).metadata.num_rows:,} bar var)"
        except Exception:
            pass
    parcalar, eksik = [], 0
    for y, m in _months(START, end):
        try:
            d = _fetch_month(sym, y, m)
        except Exception as e:
            return f"{sym}: HATA {type(e).__name__} ({y}-{m:02d})"
        if d is None:
            eksik += 1
            continue
        parcalar.append(d)
    if not parcalar:
        return f"{sym}: HİÇ VERİ YOK"
    d = pd.concat(parcalar, ignore_index=True)
    # Binance bazı aylarda ts'i mikrosaniye yazıyor — birim tespiti ZORUNLU
    if d["ts"].max() > 1e14:
        d["ts"] = d["ts"] // 1000
    d = d[["ts", "open", "high", "low", "close", "volume", "trades",
           "taker_buy_base", "quote_volume"]].astype(
        {"ts": "int64", "open": "float64", "high": "float64",
         "low": "float64", "close": "float64", "volume": "float64",
         "trades": "float64", "taker_buy_base": "float64",
         "quote_volume": "float64"})
    d = d.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    d.to_parquet(hedef, index=False)
    ilk = pd.to_datetime(d.ts.iloc[0], unit="ms")
    son = pd.to_datetime(d.ts.iloc[-1], unit="ms")
    # boşluk denetimi — 5 dakikalık ızgarada kaç bar eksik
    beklenen = int((d.ts.iloc[-1] - d.ts.iloc[0]) / 300_000) + 1
    return (f"{sym}: {len(d):,} bar  {ilk:%Y-%m-%d} → {son:%Y-%m-%d}  "
            f"boşluk {beklenen - len(d):,}  eksik ay {eksik}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="5 dakikalık vadeli kline indirici")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--force", action="store_true",
                    help="var olan parquet'i yeniden indir")
    a = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    now = time.gmtime()
    end = (now.tm_year, now.tm_mon)
    syms = _symbols(a.symbols)
    print(f"{len(syms)} parite · {a.workers} paralel indirici", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        isler = {ex.submit(_one, s, end, a.force): s for s in syms}
        for k, f in enumerate(as_completed(isler), 1):
            print(f"[{k}/{len(syms)}] {f.result()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
