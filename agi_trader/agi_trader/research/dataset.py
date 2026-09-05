"""
Araştırma veri katmanı — tüm sleeve'lerin ortak veri kaynağı.

Yerel önbellekten okur (ağ gerekmez):
  runs/data_full/<PARITE>_1h.csv      — kripto, saatlik, 2022→  (ts,open,high,low,close,volume,dt)
  runs/data_noncrypto/<TICKER>.csv    — ETF/FX/emtia, günlük, 2017→ (Date,close)
  runs/data_funding/<COIN>_funding.csv— funding geçmişi

Neden ayrı modül: `diversified.py`, `portfolio_trend.py`, `select_universe.py` ve
`leverage_hedge.py` aynı yükleme/dönüştürme kodunu dört kez kopyalamıştı. Yeni
sleeve'ler bunu tekrar kopyalamasın diye tek yerde toplandı.

BÖLME DİSİPLİNİ: `split_periods()` runs/SPLIT.md ile sabitlenen train/val/test
sınırlarını döndürür. Kilitli test dönemi ancak açıkça istenirse verilir.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_CRYPTO = ROOT / "runs" / "data_full"
DATA_NONCRYPTO = ROOT / "runs" / "data_noncrypto"
DATA_FUNDING = ROOT / "runs" / "data_funding"
DATA_BASIS = ROOT / "runs" / "data_basis"

# runs/SPLIT.md ile aynı — orayı değiştirmeden burayı değiştirme
TRAIN = ("2022-01-01", "2024-12-31")
VALIDATION = ("2025-01-01", "2025-12-31")
LOCKED_TEST = ("2026-01-01", "2099-12-31")


# ===========================================================================
# Yükleyiciler
# ===========================================================================
def load_crypto_daily(symbol: str) -> pd.Series:
    """Saatlik kripto CSV'sinden günlük kapanış serisi. symbol: 'BTCUSDT' veya 'BTC/USDT'."""
    short = symbol.replace("/", "").upper()
    path = DATA_CRYPTO / f"{short}_1h.csv"
    if not path.exists():
        raise FileNotFoundError(f"kripto verisi yok: {path}")
    df = pd.read_csv(path)
    df.index = pd.to_datetime(df["dt"])
    return df["close"].astype(float).resample("1D").last().dropna()


def load_crypto_ohlcv_daily(symbol: str) -> pd.DataFrame:
    """Günlük OHLCV (bazı sleeve'ler yüksek/düşük ister)."""
    short = symbol.replace("/", "").upper()
    df = pd.read_csv(DATA_CRYPTO / f"{short}_1h.csv")
    df.index = pd.to_datetime(df["dt"])
    return pd.DataFrame({
        "open": df["open"].astype(float).resample("1D").first(),
        "high": df["high"].astype(float).resample("1D").max(),
        "low": df["low"].astype(float).resample("1D").min(),
        "close": df["close"].astype(float).resample("1D").last(),
        "volume": df["volume"].astype(float).resample("1D").sum(),
    }).dropna()


def load_noncrypto_daily(ticker: str) -> pd.Series:
    """Kripto-dışı günlük kapanış (yerel önbellek)."""
    path = DATA_NONCRYPTO / f"{ticker.upper()}.csv"
    if not path.exists():
        raise FileNotFoundError(f"kripto-dışı verisi yok: {path}")
    df = pd.read_csv(path)
    df.index = pd.to_datetime(df["Date"])
    return df["close"].astype(float).dropna()


def load_funding(coin: str) -> Optional[pd.Series]:
    """Funding oranı geçmişi (8 saatlik). coin: 'BTC'."""
    path = DATA_FUNDING / f"{coin.upper()}_funding.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    tcol = next((c for c in df.columns if c.lower() in
                 ("dt", "date", "time", "fundingtime", "ts")), df.columns[0])
    vcol = next((c for c in df.columns if "rate" in c.lower() or "funding" in c.lower()),
                df.columns[-1])
    idx = pd.to_datetime(df[tcol], unit="ms" if np.issubdtype(df[tcol].dtype, np.number)
                         and df[tcol].max() > 1e11 else None)
    return pd.Series(df[vcol].astype(float).values, index=idx).sort_index()


def load_basis(coin: str) -> Optional[pd.Series]:
    """Günlük baz serisi (perp/spot − 1). `fetch_basis.py` üretir.

    Carry sleeve'i için ZORUNLU: baz olmadan delta-nötr P&L eksik modellenir."""
    path = DATA_BASIS / f"{coin.upper()}_basis.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.index = pd.to_datetime(df["Date"])
    return df["basis"].astype(float).dropna()


def available_crypto() -> List[str]:
    return sorted(Path(p).stem.replace("_1h", "")
                  for p in glob.glob(str(DATA_CRYPTO / "*_1h.csv")))


def available_noncrypto() -> List[str]:
    return sorted(Path(p).stem for p in glob.glob(str(DATA_NONCRYPTO / "*.csv")))


def load_universe(symbols: List[str]) -> Dict[str, pd.Series]:
    """Karışık evren yükle — 'BTCUSDT'/'BTC/USDT' kripto, diğerleri ETF sayılır.
    Bulunamayan semboller sessizce atlanmaz, uyarı basılır (sessiz eksik veri
    portföy ağırlıklarını sessizce bozar)."""
    out: Dict[str, pd.Series] = {}
    for s in symbols:
        try:
            if "USDT" in s.upper() or "/" in s:
                out[s] = load_crypto_daily(s)
            else:
                out[s] = load_noncrypto_daily(s)
        except FileNotFoundError as e:
            print(f"  ⚠️ atlandı: {s} ({e.__class__.__name__})")
    return out


# ===========================================================================
# Bölme
# ===========================================================================
def split_periods(include_locked_test: bool = False) -> Dict[str, Tuple[str, str]]:
    """runs/SPLIT.md sözleşmesi. Kilitli test yalnız açıkça istenirse döner."""
    d = {"train": TRAIN, "validation": VALIDATION}
    if include_locked_test:
        d["test"] = LOCKED_TEST
    return d


def slice_period(s: "pd.Series | pd.DataFrame", period: Tuple[str, str]):
    lo, hi = period
    return s[(s.index >= pd.Timestamp(lo)) & (s.index <= pd.Timestamp(hi))]


def train_val(s: "pd.Series | pd.DataFrame"):
    """Train + validation birleşik (2022-01-01 → 2025-12-31). Geliştirme burada yapılır."""
    return slice_period(s, (TRAIN[0], VALIDATION[1]))


def locked_test(s: "pd.Series | pd.DataFrame"):
    """KİLİTLİ test dilimi — çağırmadan önce runs/SPLIT.md kurallarını oku."""
    return slice_period(s, LOCKED_TEST)


# ===========================================================================
# Ortak yardımcılar
# ===========================================================================
def align(series_map: Dict[str, pd.Series], start: Optional[str] = None) -> pd.DataFrame:
    """Farklı takvimlerdeki serileri tek indekste hizala (ffill, ETF hafta sonu boşluğu)."""
    idx = None
    for v in series_map.values():
        idx = v.index if idx is None else idx.union(v.index)
    if idx is None:
        return pd.DataFrame()
    if start:
        idx = idx[idx >= pd.Timestamp(start)]
    return pd.DataFrame({k: v.reindex(idx).ffill() for k, v in series_map.items()})


def to_returns(prices: "pd.Series | pd.DataFrame") -> "pd.Series | pd.DataFrame":
    return prices.pct_change().fillna(0.0)


def annualized(ret: pd.Series, periods_per_year: float = 365.0) -> Dict[str, float]:
    """Standart metrik seti — her sleeve aynı biçimde raporlasın."""
    r = ret.dropna()
    if len(r) < 10:
        return {"sharpe": 0.0, "cagr": 0.0, "dd": 0.0, "calmar": 0.0, "vol": 0.0, "n": len(r)}
    eq = (1 + r).cumprod()
    dd = float(((eq.cummax() - eq) / eq.cummax()).max() * 100)
    vol = float(r.std() * np.sqrt(periods_per_year) * 100)
    sh = float(r.mean() / (r.std() + 1e-12) * np.sqrt(periods_per_year))
    cagr = float((eq.iloc[-1] ** (periods_per_year / len(eq)) - 1) * 100)
    return {"sharpe": round(sh, 3), "cagr": round(cagr, 2), "dd": round(dd, 2),
            "calmar": round(cagr / (dd + 1e-9), 2), "vol": round(vol, 2), "n": len(r)}
