"""
MarketStateStore + RateLimitCoordinator — "her veri BİR kez çekilir, herkes paylaşır".

Ölçülen problem: aynı döngüde aynı parite/zaman dilimi birden çok modülce (koşucu,
hafif bağlam, haber doğrulaması, venue kıyası) ayrı ayrı çekiliyordu. Bu depo:
  • (borsa, parite, tf) → OHLCV tamponu, TTL'li; eş zamanlı isteklerde tek fetch (coalescing)
  • (borsa, parite) → en iyi kotasyon/derinlik, 10 sn TTL
  • ucuz artımlı özellikler: yalnız YENİ bar geldiğinde yeniden hesaplanır
  • tazelik: LIVE / DELAYED / STALE (bar aralığına göre) — STALE → işlem yok
  • RateLimitCoordinator: borsa başına token bucket + devre kesici (art arda hata → aç)
  • bellek: en fazla N tampon, en eski atılır; istatistik: hit/miss/fetch sayıları

WebSocket akışı bu sürümde YOK (REST + coalescing). Bir sonraki adım: ccxt.pro ile
watch_ohlcv/watch_order_book → aynı depoyu besler; tüketiciler değişmez.
"""
from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TF_SEC = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600,
          "2h": 7200, "4h": 14400, "1d": 86400}
DEFAULT_TTL = {"1m": 20.0, "5m": 60.0, "15m": 120.0, "1h": 300.0, "4h": 900.0, "1d": 3600.0}
BOOK_TTL = 10.0
MAX_BUFFERS = 400


class RateLimited(RuntimeError):
    pass


class CircuitOpen(RuntimeError):
    pass


class RateLimitCoordinator:
    """Borsa başına token bucket (rps, burst) + devre kesici."""

    def __init__(self, rps: float = 6.0, burst: int = 12, max_wait_sec: float = 2.0,
                 breaker_errors: int = 5, breaker_open_sec: float = 30.0):
        self.rps, self.burst, self.max_wait = float(rps), int(burst), float(max_wait_sec)
        self.breaker_errors, self.breaker_open = int(breaker_errors), float(breaker_open_sec)
        self._tokens: Dict[str, float] = {}
        self._last: Dict[str, float] = {}
        self._errors: Dict[str, int] = {}
        self._open_until: Dict[str, float] = {}
        self._lock = threading.Lock()
        self.stats: Dict[str, Dict[str, int]] = {}

    def _st(self, ex: str) -> Dict[str, int]:
        return self.stats.setdefault(ex, {"acquired": 0, "waited": 0, "rejected": 0,
                                          "errors": 0, "breaker_trips": 0})

    def acquire(self, exchange: str, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        with self._lock:
            st = self._st(exchange)
            if self._open_until.get(exchange, 0.0) > now:
                st["rejected"] += 1
                raise CircuitOpen(f"{exchange}: devre kesici açık "
                                  f"({self._open_until[exchange] - now:.0f} sn)")
            tok = self._tokens.get(exchange, float(self.burst))
            last = self._last.get(exchange, now)
            tok = min(float(self.burst), tok + (now - last) * self.rps)
            wait = 0.0
            if tok < 1.0:
                wait = (1.0 - tok) / self.rps
                if wait > self.max_wait:
                    st["rejected"] += 1
                    raise RateLimited(f"{exchange}: hız sınırı ({wait:.1f} sn bekleme)")
                st["waited"] += 1
            self._tokens[exchange] = tok - 1.0 + (wait * self.rps)
            self._last[exchange] = now + wait
            st["acquired"] += 1
        if wait > 0:
            time.sleep(wait)

    def report(self, exchange: str, ok: bool, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        with self._lock:
            st = self._st(exchange)
            if ok:
                self._errors[exchange] = 0
                return
            st["errors"] += 1
            self._errors[exchange] = self._errors.get(exchange, 0) + 1
            if self._errors[exchange] >= self.breaker_errors:
                self._open_until[exchange] = now + self.breaker_open
                self._errors[exchange] = 0
                st["breaker_trips"] += 1

    def state(self, exchange: str, now: Optional[float] = None) -> str:
        now = time.time() if now is None else now
        return "OPEN" if self._open_until.get(exchange, 0.0) > now else "CLOSED"


class MarketStateStore:
    def __init__(self, fetch_ohlcv: Callable[[str, str, str, int], pd.DataFrame],
                 fetch_book: Optional[Callable[[str, str], Dict]] = None,
                 rate: Optional[RateLimitCoordinator] = None,
                 ttl: Optional[Dict[str, float]] = None, max_buffers: int = MAX_BUFFERS):
        self._fetch_ohlcv = fetch_ohlcv
        self._fetch_book = fetch_book
        self.rate = rate or RateLimitCoordinator()
        self.ttl = {**DEFAULT_TTL, **(ttl or {})}
        self.max_buffers = int(max_buffers)
        self._ohlcv: "OrderedDict[Tuple[str, str, str], Dict]" = OrderedDict()
        self._book: Dict[Tuple[str, str], Dict] = {}
        self._feat: Dict[Tuple[str, str, str], Dict] = {}
        self._locks: Dict[Tuple, threading.Lock] = {}
        self._glock = threading.Lock()
        self.stats = {"hits": 0, "misses": 0, "fetches": 0, "errors": 0,
                      "book_hits": 0, "book_misses": 0, "feature_recomputes": 0}

    # ------------------------------------------------------------ yardımcı
    def _lock_for(self, key: Tuple) -> threading.Lock:
        with self._glock:
            lk = self._locks.get(key)
            if lk is None:
                lk = threading.Lock()
                self._locks[key] = lk
            return lk

    def _evict(self) -> None:
        while len(self._ohlcv) > self.max_buffers:
            k, _ = self._ohlcv.popitem(last=False)
            self._feat.pop(k, None)

    # ------------------------------------------------------------ OHLCV
    def get_ohlcv(self, exchange: str, symbol: str, tf: str = "1m", limit: int = 150,
                  now: Optional[float] = None, max_age: Optional[float] = None) -> pd.DataFrame:
        """TTL içindeyse önbellek; değilse TEK fetch (aynı anahtar için eş zamanlı istekler bekler).
        Zaman geriye gittiyse (replay/simülasyon saati) önbellek geçersiz sayılır."""
        now = time.time() if now is None else now
        key = (exchange, symbol, tf)
        ttl = self.ttl.get(tf, 60.0) if max_age is None else max_age
        hit = self._ohlcv.get(key)
        if hit and 0.0 <= now - hit["ts"] < ttl and len(hit["df"]) >= min(limit, len(hit["df"])):
            self.stats["hits"] += 1
            self._ohlcv.move_to_end(key)
            return hit["df"]
        with self._lock_for(key):
            hit = self._ohlcv.get(key)                     # bir başkası az önce çekmiş olabilir
            if hit and 0.0 <= now - hit["ts"] < ttl:
                self.stats["hits"] += 1
                return hit["df"]
            self.stats["misses"] += 1
            self.rate.acquire(exchange)
            try:
                df = self._fetch_ohlcv(exchange, symbol, tf, limit)
                self.rate.report(exchange, True)
            except Exception:
                self.stats["errors"] += 1
                self.rate.report(exchange, False)
                if hit is not None:
                    return hit["df"]                       # bayat ama var: tüketici tazeliğe bakar
                raise
            self.stats["fetches"] += 1
            self._ohlcv[key] = {"df": df, "ts": now}
            self._ohlcv.move_to_end(key)
            self._evict()
            return df

    def freshness(self, exchange: str, symbol: str, tf: str = "1m",
                  now: Optional[float] = None) -> Dict:
        """LIVE: son bar ≤ 2 aralık · DELAYED: ≤ 5 aralık · STALE: daha eski / yok."""
        now = time.time() if now is None else now
        hit = self._ohlcv.get((exchange, symbol, tf))
        if not hit or hit["df"] is None or not len(hit["df"]):
            return {"state": "STALE", "age_sec": None, "note": "veri yok"}
        try:
            last_bar = float(hit["df"].index[-1].timestamp())
        except Exception:
            last_bar = hit["ts"]
        age = now - last_bar
        sec = TF_SEC.get(tf, 60)
        st = "LIVE" if age <= 2 * sec else "DELAYED" if age <= 5 * sec else "STALE"
        return {"state": st, "age_sec": round(age, 1), "fetched_age_sec": round(now - hit["ts"], 1)}

    # ------------------------------------------------------------ defter
    def get_book(self, exchange: str, symbol: str, now: Optional[float] = None) -> Dict:
        now = time.time() if now is None else now
        key = (exchange, symbol)
        hit = self._book.get(key)
        if hit and 0.0 <= now - hit["ts"] < BOOK_TTL:
            self.stats["book_hits"] += 1
            return hit["book"]
        if self._fetch_book is None:
            return {"spread_bps": 0.0, "bid_depth_usd": 0.0, "ask_depth_usd": 0.0}
        with self._lock_for(key):
            hit = self._book.get(key)
            if hit and 0.0 <= now - hit["ts"] < BOOK_TTL:
                return hit["book"]
            self.stats["book_misses"] += 1
            self.rate.acquire(exchange)
            try:
                b = self._fetch_book(exchange, symbol)
                self.rate.report(exchange, True)
            except Exception:
                self.stats["errors"] += 1
                self.rate.report(exchange, False)
                return (hit or {}).get("book") or {"spread_bps": 0.0, "bid_depth_usd": 0.0, "ask_depth_usd": 0.0}
            self._book[key] = {"book": b, "ts": now}
            return b

    # ------------------------------------------------------------ ucuz özellikler (artımlı)
    def cheap_features(self, exchange: str, symbol: str, tf: str = "1m") -> Optional[Dict]:
        """Tier-A tarayıcı özellikleri. Yalnız YENİ bar gelince hesaplanır (bar damgasıyla önbellek)."""
        hit = self._ohlcv.get((exchange, symbol, tf))
        if not hit or hit["df"] is None or len(hit["df"]) < 60:
            return None
        df = hit["df"]
        try:
            stamp = float(df.index[-1].timestamp())
        except Exception:
            stamp = hit["ts"]
        key = (exchange, symbol, tf)
        f = self._feat.get(key)
        if f and f.get("_stamp") == stamp:
            return f
        self.stats["feature_recomputes"] += 1
        c = df["close"].astype(float)
        h = df["high"].astype(float) if "high" in df else c
        l = df["low"].astype(float) if "low" in df else c
        v = df["volume"].astype(float) if "volume" in df else None
        price = float(c.iloc[-1])
        e20 = float(c.ewm(span=20, adjust=False).mean().iloc[-1])
        e50 = float(c.ewm(span=50, adjust=False).mean().iloc[-1])
        tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        ret = np.log(c).diff().dropna()
        sig = float(ret.tail(60).std(ddof=0) * 100.0) if len(ret) >= 20 else float("nan")
        sma = c.rolling(20).mean(); sd = c.rolling(20).std(ddof=0)
        z = float((price - sma.iloc[-1]) / sd.iloc[-1]) if float(sd.iloc[-1]) > 0 else 0.0
        hi20, lo20 = float(h.tail(20).max()), float(l.tail(20).min())
        hi120, lo120 = float(h.tail(120).max()), float(l.tail(120).min())
        vr = (float(v.tail(3).mean() / max(1e-9, v.tail(63).head(60).mean()))
              if v is not None and len(v) >= 63 else None)
        r1h = float(price / c.iloc[-61] - 1.0) * 100.0 if len(c) > 61 else None
        f = {"_stamp": stamp, "price": price, "ema20": e20, "ema50": e50,
             "above_ema20": price > e20, "above_ema50": price > e50, "trend_up": e20 > e50,
             "atr_pct": atr / price * 100.0 if price else None, "sigma_1m_pct": sig, "z20": z,
             "dist_hi20_pct": (hi20 - price) / price * 100.0, "dist_lo20_pct": (price - lo20) / price * 100.0,
             "range_pos_120": (price - lo120) / max(1e-12, hi120 - lo120),
             "vol_ratio": vr, "ret_1h_pct": r1h, "bar_up": bool(price > float(c.iloc[-2])),
             "bars": int(len(df))}
        # JSON uyumu: NaN/inf asla dışarı sızmaz (σ=0, kısa tampon vb.) — tüketiciler None'a dayanıklı
        f = {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in f.items()}
        self._feat[key] = f
        return f

    def cross_section(self, exchange: str, symbols: List[str], tf: str = "1m") -> Dict[str, Dict]:
        """Genişlik/göreli güç için: yalnız önbellekten (fetch YOK)."""
        out = {}
        for s in symbols:
            f = self.cheap_features(exchange, s, tf)
            if f:
                out[s] = f
        return out

    def returns_matrix(self, exchange: str, symbols: List[str], tf: str = "1m",
                       bars: int = 120) -> Optional[pd.DataFrame]:
        cols = {}
        for s in symbols:
            hit = self._ohlcv.get((exchange, s, tf))
            if hit and hit["df"] is not None and len(hit["df"]) > bars:
                cols[s] = np.log(hit["df"]["close"].astype(float)).diff().tail(bars).reset_index(drop=True)
        if len(cols) < 3:
            return None
        return pd.DataFrame(cols)

    def status(self) -> Dict:
        return {**self.stats, "buffers": len(self._ohlcv), "books": len(self._book),
                "features": len(self._feat), "rate": self.rate.stats}
