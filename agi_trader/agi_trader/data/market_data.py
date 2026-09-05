"""
Ücretsiz HTTP piyasa-verisi sağlayıcıları (ccxt alternatifleri / yedekleri).

ccxt borsa istemcileri birincil kaynaktır (anahtarsız public OHLCV). Bu modül,
ccxt yüklü değilse YA DA tüm borsalar başarısız/coğrafi-engelli ise devreye giren
ÜCRETSİZ REST alternatiflerini sağlar:

  • Binance public REST  — anahtarsız, çok güvenilir (/api/v3/klines)
  • CryptoCompare        — ücretsiz API anahtarı (catalog: CRYPTOCOMPARE_API_KEY)
  • CoinGecko            — ücretsiz API anahtarı (catalog: COINGECKO_API_KEY)

Hepsi pandas DataFrame (timestamp index, open/high/low/close/volume) döndürür ya
da başarısızlıkta None. Hata fırlatmaz — çağıran zincirleme yedeğe geçer.

NOT: TradingView'in ücretsiz herkese-açık grafik-verisi REST API'si yoktur
(ücretsiz parçası "Lightweight Charts" görüntüleyicisidir, veri değil). Bu yüzden
gerçek "ücretsiz alternatifler" yukarıdaki borsa/aggregator API'leridir.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

try:
    import requests
    _HAS_REQUESTS = True
except Exception:  # pragma: no cover
    requests = None  # type: ignore
    _HAS_REQUESTS = False


_UA = {"User-Agent": "agi-trader/1.0"}

try:
    import yfinance as _yf
    _HAS_YF = True
except Exception:  # pragma: no cover
    _yf = None  # type: ignore
    _HAS_YF = False


def is_crypto_symbol(symbol: str) -> bool:
    """'BTC/USDT' gibi pariteler kripto (ccxt/REST); 'AAPL', 'THYAO.IS', '^GSPC',
    'EURUSD=X', 'GC=F' gibi semboller hisse/endeks/forex/emtia (yfinance)."""
    return "/" in symbol


# --------------------------------------------------------------------------- #
# yfinance — hisse senedi / endeks / forex / emtia (BIST, ABD, küresel)
# --------------------------------------------------------------------------- #
# (yf_interval, period) — 4h yok → 60m çekip yeniden örnekle
_YF_TF = {
    "15m": ("15m", "60d"), "1h": ("60m", "730d"), "4h": ("60m", "730d"),
    "1d": ("1d", "5y"), "1w": ("1wk", "10y"), "1M": ("1mo", "max"),
}


def fetch_yfinance(symbol: str, timeframe: str, limit: int = 400) -> Optional[pd.DataFrame]:
    """Hisse/endeks/forex/emtia OHLCV (yfinance). 4h için 60m → 4h resample."""
    if not _HAS_YF or timeframe not in _YF_TF:
        return None
    yf_int, period = _YF_TF[timeframe]
    try:
        hist = _yf.Ticker(symbol).history(period=period, interval=yf_int, auto_adjust=False)
        if hist is None or hist.empty:
            return None
        df = hist.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                  "Close": "close", "Volume": "volume"})
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.index = pd.to_datetime(df.index, utc=True)
        if timeframe == "4h":  # 60m -> 4h
            df = df.resample("4h").agg({"open": "first", "high": "max", "low": "min",
                                        "close": "last", "volume": "sum"}).dropna()
        df = df.dropna()
        return df.tail(limit).astype(float) if len(df) > 20 else None
    except Exception:
        return None


def _split(symbol: str):
    """'BTC/USDT' -> ('BTC','USDT'). '/' yoksa kabaca böler."""
    if "/" in symbol:
        base, quote = symbol.split("/", 1)
    else:
        quote = "USDT"
        base = symbol[:-4] if symbol.upper().endswith("USDT") else symbol
    return base.upper(), quote.upper()


def _df(rows) -> Optional[pd.DataFrame]:
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp").astype(float)


# --------------------------------------------------------------------------- #
# Binance public REST — anahtarsız
# --------------------------------------------------------------------------- #
_BINANCE_TF = {"15m", "1m", "3m", "5m", "30m", "1h", "2h", "4h", "6h", "8h",
               "12h", "1d", "3d", "1w", "1M"}


def fetch_binance_rest(symbol: str, timeframe: str, limit: int = 400) -> Optional[pd.DataFrame]:
    if not _HAS_REQUESTS or timeframe not in _BINANCE_TF:
        return None
    base, quote = _split(symbol)
    pair = f"{base}{quote}"
    try:
        r = requests.get("https://api.binance.com/api/v3/klines",
                         params={"symbol": pair, "interval": timeframe, "limit": min(limit, 1000)},
                         headers=_UA, timeout=12)
        if r.status_code != 200:
            return None
        rows = [[k[0], k[1], k[2], k[3], k[4], k[5]] for k in r.json()]
        df = _df(rows)
        return df if df is not None and len(df) > 20 else None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# CryptoCompare — ücretsiz API anahtarı (opsiyonel; anahtarsız da sınırlı çalışır)
# --------------------------------------------------------------------------- #
_CC_TF = {
    "15m": ("histominute", 15), "1h": ("histohour", 1), "4h": ("histohour", 4),
    "1d": ("histoday", 1), "1w": ("histoday", 7), "1M": ("histoday", 30),
}


def fetch_cryptocompare(symbol: str, timeframe: str, limit: int = 400,
                        api_key: Optional[str] = None) -> Optional[pd.DataFrame]:
    if not _HAS_REQUESTS or timeframe not in _CC_TF:
        return None
    base, quote = _split(symbol)
    endpoint, aggregate = _CC_TF[timeframe]
    params = {"fsym": base, "tsym": quote, "limit": min(limit, 2000), "aggregate": aggregate}
    headers = dict(_UA)
    if api_key:
        headers["authorization"] = f"Apikey {api_key}"
    try:
        r = requests.get(f"https://min-api.cryptocompare.com/data/v2/{endpoint}",
                         params=params, headers=headers, timeout=12)
        if r.status_code != 200:
            return None
        data = (r.json() or {}).get("Data", {}).get("Data", [])
        rows = [[d["time"] * 1000, d["open"], d["high"], d["low"], d["close"], d.get("volumefrom", 0)]
                for d in data if d.get("close")]
        df = _df(rows)
        return df if df is not None and len(df) > 20 else None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# CoinGecko — ücretsiz API anahtarı (günlük/saatlik granülerlik; OHLC sınırlı)
# --------------------------------------------------------------------------- #
# CoinGecko OHLC yalnızca belirli gün aralıkları verir; burada market_chart'tan
# kapanışları çekip OHLC'ye yaklaşık dönüştürmek yerine sade close serisi sunarız.
_CG_IDS = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "BNB": "binancecoin",
           "XRP": "ripple", "ADA": "cardano", "DOGE": "dogecoin", "AVAX": "avalanche-2",
           "LINK": "chainlink", "MATIC": "matic-network", "DOT": "polkadot"}


def fetch_coingecko(symbol: str, timeframe: str, limit: int = 400,
                    api_key: Optional[str] = None) -> Optional[pd.DataFrame]:
    if not _HAS_REQUESTS:
        return None
    base, _ = _split(symbol)
    coin_id = _CG_IDS.get(base)
    if not coin_id:
        return None
    # gün sayısı: TF'ye göre kaba pencere
    days = {"15m": 1, "1h": 7, "4h": 30, "1d": 200, "1w": 365, "1M": 365}.get(timeframe, 30)
    params = {"vs_currency": "usd", "days": days}
    headers = dict(_UA)
    if api_key:
        headers["x-cg-demo-api-key"] = api_key
    try:
        r = requests.get(f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc",
                         params=params, headers=headers, timeout=12)
        if r.status_code != 200:
            return None
        rows = [[c[0], c[1], c[2], c[3], c[4], 0] for c in (r.json() or [])]
        df = _df(rows)
        return df.tail(limit) if df is not None and len(df) > 20 else None
    except Exception:
        return None
