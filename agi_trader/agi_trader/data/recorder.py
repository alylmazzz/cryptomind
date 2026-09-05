"""
Fiyat-dışı özellik kaydedicisi (FAZ 3a).

SORUN: `onchain/flow_engine.py` funding, open interest, order-book dengesizliği ve
CVD'yi CANLI okuyor ama hiçbir yere YAZMIYOR. Bu yüzden bu sinyallerin hiçbiri
backtest edilemiyor — fiyat-dışı alfa arayışının önündeki tek gerçek engel budur.
Geçmiş veri satın alınamıyorsa tek yol bugünden itibaren kaydetmektir; kayıt
bugün başlamazsa altı ay sonra da başlanamaz.

Kaydedilenler (5 dakikada bir, Binance public — API ANAHTARI GEREKMEZ):
  funding_rate      son funding oranı + mark/index fiyatı
  open_interest     açık pozisyon miktarı
  book_imb          ±%0,5 derinlikte alış/satış dengesizliği
  taker_buy_ratio   5 dk'lık taker alış hacmi oranı (CVD vekili)
  ls_account_ratio  global long/short hesap oranı
  top_trader_ratio  büyük hesapların pozisyon oranı

Depolama: aylık dosya, `runs/features/YYYY-MM.parquet` (pyarrow yoksa .csv.gz).
Günlük ~4 MB (24 parite).

Likidasyon akışı BİLİNÇLİ OLARAK DIŞARIDA: Binance açık likidasyon REST ucunu
kaldırdı; yalnız WebSocket (`!forceOrder@arr`) veya ücretli CoinGlass ile alınır.
BYOK panelinden CoinGlass anahtarı girilirse `liquidation.py` devreye girer.
"""
from __future__ import annotations

import gzip
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

FAPI = "https://fapi.binance.com"
# Nitelendirme evreni ile AYNI liste (runs/qualification/universe_5m.json).
# Kaydedici L2 defterini yalnız izlediği paritelerde toplar; araştırma
# evreni bundan genişse o paritelerde maliyet ESTIMATED kalır ve
# HIGH_CONFIDENCE üretilemez. İki liste bu yüzden birlikte büyür.
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ZECUSDT", "ACEUSDT",
                   "XRPUSDT", "BNBUSDT", "DOGEUSDT", "WLDUSDT", "LINKUSDT",
                   "1000PEPEUSDT", "ADAUSDT", "SUIUSDT", "ONGUSDT",
                   "NEARUSDT", "AVAXUSDT", "TRXUSDT", "AAVEUSDT",
                   "UNIUSDT", "BICOUSDT", "FILUSDT", "XMRUSDT", "LTCUSDT",
                   "FETUSDT", "XLMUSDT", "BCHUSDT", "DOTUSDT"]
INTERVAL_SEC = 300
DEPTH_PCT = 0.005

# --- L2 MERDİVEN (kümülatif derinlik eğrisi) --------------------------------
# Ham 500 seviyeyi kaydetmek 10 paritede günde ~2,9 milyon satır demek ve
# çoğu gürültü. Yürütme maliyeti için YETERLİ İSTATİSTİK, mid'den X baz puan
# uzaklığa kadar BİRİKMİŞ nominal'dir: VWAP bu eğriden TAM olarak hesaplanır.
# Bu yüzden ham merdiven yerine eğri kaydedilir — kayıpsız ve ~20× küçük.
LADDER_BPS = (1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 50.0, 75.0, 100.0)
LADDER_LIMIT = 1000        # Binance fapi depth üst sınırı


def _out_dir() -> Path:
    p = Path(__file__).resolve().parents[2] / "runs" / "features"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ===========================================================================
# HTTP
# ===========================================================================
def _get(path: str, params: Optional[Dict] = None, timeout: int = 10):
    """Anahtarsız public GET. Hata durumunda None — kayıt döngüsü ASLA ölmemeli."""
    import requests
    try:
        r = requests.get(FAPI + path, params=params or {}, timeout=timeout)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


# ===========================================================================
# Tek çevrim toplayıcılar
# ===========================================================================
def collect_premium_index() -> Dict[str, Dict]:
    """Tüm semboller için funding + mark/index — TEK çağrı (ağırlık 10)."""
    data = _get("/fapi/v1/premiumIndex")
    if not isinstance(data, list):
        return {}
    out = {}
    for d in data:
        try:
            out[d["symbol"]] = {
                "funding_rate": float(d.get("lastFundingRate", 0) or 0),
                "mark_price": float(d.get("markPrice", 0) or 0),
                "index_price": float(d.get("indexPrice", 0) or 0),
                "next_funding_ms": int(d.get("nextFundingTime", 0) or 0),
            }
        except (TypeError, ValueError):
            continue
    return out


def collect_open_interest(symbol: str) -> Optional[float]:
    d = _get("/fapi/v1/openInterest", {"symbol": symbol})
    try:
        return float(d["openInterest"])
    except (TypeError, KeyError, ValueError):
        return None


def collect_book_imbalance(symbol: str, depth_pct: float = DEPTH_PCT) -> Optional[Dict]:
    """Mid fiyatın ±depth_pct bandındaki alış/satış derinliği dengesizliği.

    imbalance ∈ [-1, 1]; +1 = yalnız alış tarafı, -1 = yalnız satış."""
    d = _get("/fapi/v1/depth", {"symbol": symbol, "limit": 500})
    if not d or not d.get("bids") or not d.get("asks"):
        return None
    try:
        bids = np.array(d["bids"], dtype=float)
        asks = np.array(d["asks"], dtype=float)
    except (TypeError, ValueError):
        return None
    if len(bids) == 0 or len(asks) == 0:
        return None
    mid = (bids[0, 0] + asks[0, 0]) / 2.0
    lo, hi = mid * (1 - depth_pct), mid * (1 + depth_pct)
    bid_qty = float(bids[bids[:, 0] >= lo][:, 1].sum())
    ask_qty = float(asks[asks[:, 0] <= hi][:, 1].sum())
    tot = bid_qty + ask_qty
    spread = float(asks[0, 0] - bids[0, 0])
    # `bid_depth`/`ask_depth` BAZ VARLIK miktarıdır (BTC, ETH…), dolar DEĞİL.
    # Maliyet motoru dolar cinsi nominal ister; birim karışıklığı sessiz ve
    # ölümcül bir hataydı (BTC derinliği "245 $" görünüp 10.000 $'lık emir
    # "derinliği 34× aşıyor" sanılıyordu). Dolar karşılığı da yazılır.
    return {"book_imb": (bid_qty - ask_qty) / tot if tot > 0 else 0.0,
            "bid_depth": bid_qty, "ask_depth": ask_qty,
            "bid_depth_usd": bid_qty * mid, "ask_depth_usd": ask_qty * mid,
            "spread_bps": (spread / mid * 1e4) if mid > 0 else None,
            "mid": mid}


def collect_book_ladder(symbol: str) -> Optional[Dict]:
    """L2 KÜMÜLATİF DERİNLİK EĞRİSİ — gerçek VWAP'ın girdisi.

    Her taraf için mid'den `LADDER_BPS` uzaklığa kadar birikmiş DOLAR nominali
    döner (`bid_cum_5bps`, `ask_cum_20bps` …).

    Neden kümülatif eğri, neden ham merdiven değil: bir emrin ortalama dolum
    fiyatı yalnız "şu uzaklığa kadar ne kadar nominal var" bilgisine bağlıdır.
    Eğri bu sorunun tam cevabıdır; ham seviyeler ek bilgi taşımaz ama 20× yer
    kaplar. Fiyat-seviyesi mikro yapısı gerekirse `best_bid_qty`/`best_ask_qty`
    ve ilk beş seviyenin nominali ayrıca yazılır.

    TRUNCATION DÜRÜSTLÜĞÜ: `depth` çağrısı en fazla `LADDER_LIMIT` seviye verir.
    Defter o seviyede 100 bps'e ulaşmıyorsa son kovalar SANSÜRLÜDÜR; bunu
    gizlememek için `bid_max_bps`/`ask_max_bps` (kapsanan en uzak nokta) ve
    `bid_truncated`/`ask_truncated` bayrakları yazılır. Sansürlü kovayı dolu
    sanmak, likiditeyi olduğundan çok göstermek demektir.
    """
    d = _get("/fapi/v1/depth", {"symbol": symbol, "limit": LADDER_LIMIT})
    if not d or not d.get("bids") or not d.get("asks"):
        return None
    try:
        bids = np.asarray(d["bids"], dtype=float)     # [[price, qty], ...] azalan
        asks = np.asarray(d["asks"], dtype=float)     # artan
    except Exception:
        return None
    if bids.size == 0 or asks.size == 0:
        return None

    mid = (bids[0, 0] + asks[0, 0]) / 2.0
    if mid <= 0:
        return None

    out: Dict[str, object] = {"ladder_mid": mid, "ladder_levels": int(len(bids))}

    for taraf, arr, isaret in (("bid", bids, -1.0), ("ask", asks, +1.0)):
        # her seviyenin mid'den uzaklığı (bps) ve dolar nominali
        off_bps = isaret * (arr[:, 0] - mid) / mid * 1e4
        notional = arr[:, 0] * arr[:, 1]
        # uzaklığa göre sırala (defter zaten sıralı ama garanti altına al)
        sira = np.argsort(off_bps)
        off_bps, notional = off_bps[sira], notional[sira]
        kum = np.cumsum(notional)

        kapsanan = float(off_bps[-1]) if len(off_bps) else 0.0
        out[f"{taraf}_max_bps"] = round(kapsanan, 2)
        out[f"{taraf}_total_usd"] = float(kum[-1]) if len(kum) else 0.0
        out[f"{taraf}_truncated"] = bool(kapsanan < LADDER_BPS[-1])

        for b in LADDER_BPS:
            if b > kapsanan:
                out[f"{taraf}_cum_{b:g}bps"] = None      # SANSÜRLÜ — 0 değil
            else:
                idx = np.searchsorted(off_bps, b, side="right")
                out[f"{taraf}_cum_{b:g}bps"] = float(kum[idx - 1]) if idx > 0 else 0.0

        out[f"best_{taraf}_qty"] = float(arr[0, 1])
        out[f"{taraf}_top5_usd"] = float(notional[:5].sum())

    return out


def collect_taker_ratio(symbol: str) -> Optional[float]:
    """Son 5 dk taker alış hacmi / toplam (CVD eğim vekili)."""
    d = _get("/fapi/v1/klines", {"symbol": symbol, "interval": "5m", "limit": 1})
    try:
        k = d[0]
        vol = float(k[5])
        taker_buy = float(k[9])
        return taker_buy / vol if vol > 0 else None
    except (TypeError, IndexError, ValueError, ZeroDivisionError):
        return None


def collect_ls_ratios(symbol: str) -> Dict[str, Optional[float]]:
    """Perakende (global hesap) ve büyük hesap long/short oranları."""
    out: Dict[str, Optional[float]] = {"ls_account_ratio": None,
                                       "top_trader_ratio": None}
    d = _get("/futures/data/globalLongShortAccountRatio",
             {"symbol": symbol, "period": "5m", "limit": 1})
    try:
        out["ls_account_ratio"] = float(d[0]["longShortRatio"])
    except (TypeError, IndexError, KeyError, ValueError):
        pass
    d = _get("/futures/data/topLongShortPositionRatio",
             {"symbol": symbol, "period": "5m", "limit": 1})
    try:
        out["top_trader_ratio"] = float(d[0]["longShortRatio"])
    except (TypeError, IndexError, KeyError, ValueError):
        pass
    return out


def snapshot(symbols: List[str]) -> pd.DataFrame:
    """Tüm semboller için tek anlık görüntü."""
    ts = pd.Timestamp.now("UTC").floor("s")
    prem = collect_premium_index()
    rows = []
    for sym in symbols:
        row: Dict[str, object] = {"ts": ts, "symbol": sym}
        row.update(prem.get(sym, {}))
        row["open_interest"] = collect_open_interest(sym)
        book = collect_book_imbalance(sym)
        if book:
            row.update(book)
        ladder = collect_book_ladder(sym)
        if ladder:
            row.update(ladder)
        row["taker_buy_ratio"] = collect_taker_ratio(sym)
        row.update(collect_ls_ratios(sym))
        rows.append(row)
    return pd.DataFrame(rows)


# ===========================================================================
# Depolama
# ===========================================================================
def _month_path(ts: pd.Timestamp) -> Path:
    return _out_dir() / f"{ts.strftime('%Y-%m')}"


def append(df: pd.DataFrame) -> Path:
    """Aylık dosyaya ekle. pyarrow varsa parquet, yoksa gzip'li CSV.

    Parquet tercih edilir (10× küçük, tip korunur) ama sunucuda pyarrow kurulu
    olmayabilir; kayıt kaybetmemek için sessizce CSV'ye düşer."""
    if df.empty:
        return Path()
    base = _month_path(pd.Timestamp(df["ts"].iloc[0]))
    try:
        import pyarrow  # noqa: F401
        path = base.with_suffix(".parquet")
        if path.exists():
            old = pd.read_parquet(path)
            df = pd.concat([old, df], ignore_index=True)
        df.to_parquet(path, index=False)
        return path
    except Exception:
        path = base.with_suffix(".csv.gz")
        header = not path.exists()
        with gzip.open(path, "at", encoding="utf-8", newline="") as f:
            df.to_csv(f, index=False, header=header)
        return path


def load_features(month: Optional[str] = None) -> pd.DataFrame:
    """Kaydedilmiş özellikleri oku (month='2026-08' veya tümü)."""
    d = _out_dir()
    files = sorted(list(d.glob("*.parquet")) + list(d.glob("*.csv.gz")))
    if month:
        files = [p for p in files if p.name.startswith(month)]
    frames = []
    for p in files:
        try:
            frames.append(pd.read_parquet(p) if p.suffix == ".parquet"
                          else pd.read_csv(p))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    return out.dropna(subset=["ts"]).sort_values("ts")


# ===========================================================================
# Döngü
# ===========================================================================
def run(symbols: Optional[List[str]] = None, interval: int = INTERVAL_SEC,
        once: bool = False) -> None:
    syms = symbols or DEFAULT_SYMBOLS
    print(f"[recorder] {len(syms)} parite · her {interval}s · → {_out_dir()}",
          flush=True)
    while True:
        t0 = time.time()
        try:
            df = snapshot(syms)
            got = int(df["funding_rate"].notna().sum()) if "funding_rate" in df else 0
            path = append(df)
            print(f"[recorder] {pd.Timestamp.now("UTC"):%Y-%m-%d %H:%M:%S} "
                  f"{got}/{len(syms)} kayıt → {path.name} "
                  f"({time.time() - t0:.1f}s)", flush=True)
        except Exception as e:                      # döngü ASLA ölmemeli
            print(f"[recorder] hata: {type(e).__name__}: {e}", flush=True)
        if once:
            return
        time.sleep(max(5.0, interval - (time.time() - t0)))
