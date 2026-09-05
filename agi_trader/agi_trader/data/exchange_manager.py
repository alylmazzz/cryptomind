"""
Borsa veri yöneticisi (Veri Mühendisi + Backend rolü).

Spec: Binance/Bybit/OKX/Coinbase/Kraken/KuCoin/Hyperliquid/Gate/MEXC/Bitget ...
ccxt kuruluysa bu borsalardan ANAHTARSIZ public OHLCV çeker (çoklu borsa,
çoklu zaman dilimi), birden çok borsadan gelen veriyi kalite/öncelik ile
birleştirir. ccxt yoksa veya ağ erişimi başarısızsa otomatik sentetik veriye
düşer (graceful degradation) — böylece sistem her koşulda çalışır.

Senkron arayüz kullanılır (CLI'dan kolay çağrı için); ccxt'nin senkron istemcisi
yeterlidir. Yüksek-frekanslı WS akışı production'da ccxt.pro / websockets ile
eklenebilir; iskelet `stream_supported` ile işaretlenmiştir.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from . import synthetic

try:  # ccxt opsiyonel
    import ccxt  # type: ignore
    _HAS_CCXT = True
except Exception:  # pragma: no cover
    ccxt = None  # type: ignore
    _HAS_CCXT = False


# Borsa öncelik puanları (veri birleştirme ağırlığı) — TEK KAYNAK: providers.py
from ..providers import exchange_priority as _exchange_priority

EXCHANGE_PRIORITY = _exchange_priority()


class ExchangeManager:
    stream_supported = False  # bu iskelette REST tabanlı; WS için genişletilebilir

    def __init__(self, config):
        self.config = config
        self.source_mode = config.get("data_source", "auto")
        self.exchanges: Dict[str, object] = {}
        self._init_exchanges()

    # ------------------------------------------------------------------ setup
    def _init_exchanges(self) -> None:
        if not _HAS_CCXT or self.source_mode == "synthetic":
            return
        # Anahtarlar (api/secret/passphrase) katalogdan otomatik toplanır
        from ..providers import exchange_credentials
        creds = exchange_credentials(self.config)
        for ex_id in self.config.get("exchanges", []):
            if not hasattr(ccxt, ex_id):
                continue
            try:
                klass = getattr(ccxt, ex_id)
                params = {"enableRateLimit": True, "timeout": 15000}
                # Anahtar varsa ekle (yine de public veri için gerekmez).
                # Pasaparola gerektiren borsalar (OKX/KuCoin/Bitget) password ile.
                if ex_id in creds:
                    params.update(creds[ex_id])
                self.exchanges[ex_id] = klass(params)
            except Exception:
                continue

    @property
    def live_enabled(self) -> bool:
        return bool(self.exchanges) and self.source_mode != "synthetic"

    # ------------------------------------------------------------------- fetch
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: Optional[int] = None) -> pd.DataFrame:
        """Tek parite/zaman dilimi için OHLCV (geriye-uyumlu — yalnız DataFrame)."""
        return self.fetch_ohlcv_with_meta(symbol, timeframe, limit)[0]

    def fetch_ohlcv_with_meta(self, symbol: str, timeframe: str,
                              limit: Optional[int] = None):
        """OHLCV + veri KÖKENİ (provenance). Zincir: ccxt borsalar → ücretsiz REST
        alternatifleri (Binance public / CryptoCompare / CoinGecko) → sentetik.
        Döndürür: (DataFrame, meta) — meta: {live, provider, exchange, bars, last_ts}."""
        limit = limit or self.config.get("ohlcv_limit", 400)

        # 0) zorla sentetik
        if self.source_mode == "synthetic":
            df = synthetic.generate_ohlcv(symbol, timeframe, limit)
            return df, self._meta(False, "synthetic", None, df)

        # 0b) kripto OLMAYAN sembol (hisse/endeks/forex/emtia) -> yfinance
        from . import market_data as md
        if not md.is_crypto_symbol(symbol):
            ydf = md.fetch_yfinance(symbol, timeframe, limit)
            if ydf is not None:
                return ydf, self._meta(True, "yfinance", None, ydf)
            sdf = synthetic.generate_ohlcv(symbol, timeframe, limit)
            return sdf, self._meta(False, "synthetic", None, sdf)

        # 1) ccxt borsalar (anahtarsız public) — öncelik + uzunluğa göre seç
        if self.live_enabled:
            frames: List[tuple] = []  # (priority, len, df, ex_id)
            for ex_id, ex in self.exchanges.items():
                df = self._safe_fetch(ex, ex_id, symbol, timeframe, limit)
                if df is not None and len(df) > 20:
                    frames.append((EXCHANGE_PRIORITY.get(ex_id, 0.5), len(df), df, ex_id))
            if frames:
                frames.sort(key=lambda t: (t[0], t[1]), reverse=True)
                _, _, df, ex_id = frames[0]
                meta = self._meta(True, f"ccxt:{ex_id}", ex_id, df)
                meta["sources_ok"] = [f[3] for f in frames]
                return df, meta

        # 2) ücretsiz HTTP alternatifleri (ccxt yoksa veya tüm borsalar başarısızsa)
        df, prov = self._free_fallback(symbol, timeframe, limit)
        if df is not None:
            return df, self._meta(True, prov, None, df)

        # 3) son çare: sentetik (sistem her koşulda çalışsın)
        df = synthetic.generate_ohlcv(symbol, timeframe, limit)
        return df, self._meta(False, "synthetic", None, df)

    def _free_fallback(self, symbol, timeframe, limit):
        """Ücretsiz REST sağlayıcı zinciri. (df, provider_adı) veya (None, None)."""
        from . import market_data as md
        df = md.fetch_binance_rest(symbol, timeframe, limit)
        if df is not None:
            return df, "binance-rest"
        cck = self.config.secret("CRYPTOCOMPARE_API_KEY")
        df = md.fetch_cryptocompare(symbol, timeframe, limit, cck)
        if df is not None:
            return df, "cryptocompare" + ("+key" if cck else "")
        cgk = self.config.secret("COINGECKO_API_KEY")
        df = md.fetch_coingecko(symbol, timeframe, limit, cgk)
        if df is not None:
            return df, "coingecko" + ("+key" if cgk else "")
        return None, None

    @staticmethod
    def _meta(live: bool, provider: str, exchange, df: pd.DataFrame) -> Dict[str, object]:
        last = df.index[-1] if len(df) else None
        return {
            "live": bool(live),
            "provider": provider,
            "exchange": exchange,
            "bars": int(len(df)),
            "last_ts": (last.isoformat() if hasattr(last, "isoformat") else str(last)),
        }

    def _safe_fetch(self, ex, ex_id, symbol, timeframe, limit) -> Optional[pd.DataFrame]:
        try:
            if timeframe not in getattr(ex, "timeframes", {}) and getattr(ex, "timeframes", None):
                return None
            raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if not raw:
                return None
            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp")
            return df.astype(float)
        except Exception:
            return None

    def fetch_multi_timeframe(
        self, symbol: str, timeframes: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """Bir parite için tüm zaman dilimlerinin OHLCV'sini topla."""
        tfs = timeframes or self.config.timeframes
        out: Dict[str, pd.DataFrame] = {}
        for tf in tfs:
            df = self.fetch_ohlcv(symbol, tf)
            if df is not None and len(df) > 20:
                out[tf] = df
        return out

    def fetch_price(self, symbol: str):
        """Güncel fiyat + 24s değişim % — kripto için ccxt ticker (anlık), diğer
        varlıklar için kısa OHLCV. Döndürür: (price, change_pct) veya (None, None)."""
        from . import market_data as md
        # kripto + canlı ccxt: ticker en günceli verir
        if md.is_crypto_symbol(symbol) and self.live_enabled:
            for ex_id, ex in self.exchanges.items():
                try:
                    t = ex.fetch_ticker(symbol)
                    last = t.get("last") or t.get("close")
                    if last:
                        pct = t.get("percentage")
                        return float(last), (float(pct) if pct is not None else None)
                except Exception:
                    continue
        # genel: kısa OHLCV (kripto fallback + hisse/forex/endeks)
        try:
            df = self.fetch_ohlcv(symbol, "1h", limit=30)
            if df is not None and len(df) > 2:
                last = float(df["close"].iloc[-1])
                ref = float(df["close"].iloc[-25]) if len(df) >= 25 else float(df["close"].iloc[0])
                return last, (last / ref - 1) * 100 if ref else None
        except Exception:
            pass
        return None, None

    def describe(self) -> str:
        if self.source_mode == "synthetic":
            return "SENTETİK veri (synthetic mod seçili)"
        if self.live_enabled:
            return f"CANLI (ccxt) — borsalar: {', '.join(self.exchanges)}"
        # ccxt yok/kapalı ama ücretsiz REST alternatifleri devrede olabilir
        try:
            from .market_data import _HAS_REQUESTS
        except Exception:
            _HAS_REQUESTS = False
        if _HAS_REQUESTS:
            return "CANLI (ücretsiz REST: Binance public / CryptoCompare / CoinGecko)"
        return "SENTETİK veri (ccxt yok + requests yok)"
