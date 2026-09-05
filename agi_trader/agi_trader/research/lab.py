"""
ARAŞTIRMA LABORATUVARI — gölge modüllerin (çiftler / carry / üçgen / piyasa yapıcı) düşük frekanslı
orkestrasyonu. Hiçbir zaman emir vermez; yalnız kaynak durumu GREEN iken çalışır; hata = sessiz atla.

  pairs        6 saatte bir kointegrasyon taraması (4h kapanış, depo üzerinden), her 4h barda z adımı
  carry        30 dk'da bir funding (opt-in CRYPTOMIND_RESEARCH_CARRY=1; tek fetch_funding_rates çağrısı)
  triangular   10 dk'da bir fetch_tickers (mevcut borsa istemcisi) → üçgen tarama
  market_making her döngü BTC/ETH için A–S kotasyon (defter varsa)
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from . import carry as CR
from . import library as LIB
from . import pairs as PR
from . import triangular as TR
from . import market_making as MM

PAIRS_SCAN_SEC = 6 * 3600
PAIRS_STEP_SEC = 4 * 3600
CARRY_SEC = 30 * 60
TRI_SEC = 10 * 60


class ResearchLab:
    def __init__(self, live_dir: Path, tag: str, exchange_id: str, fetch_tickers: Optional[Callable] = None,
                 client_factory: Optional[Callable] = None, taker_bps: float = 5.0, maker_bps: float = 0.0):
        live_dir = Path(live_dir)
        self.exchange_id = exchange_id
        self.pairs = PR.PairsShadow(live_dir / f"research_pairs_{tag}.json")
        self.carry = CR.CarryShadow(live_dir / f"research_carry_{tag}.json")
        self.tri = TR.TriangularShadow(live_dir / f"research_tri_{tag}.json")
        self.mm = MM.MMShadow(live_dir / f"research_mm_{tag}.json", maker_fee_bps=maker_bps)
        self._fetch_tickers = fetch_tickers
        self._client_factory = client_factory
        self.taker_bps = taker_bps
        self.carry_enabled = os.environ.get("CRYPTOMIND_RESEARCH_CARRY", "0") == "1"
        self._last = {"pairs_step": 0.0, "carry": 0.0, "tri": 0.0}
        self.errors: Dict[str, str] = {}
        self.events: List[Dict] = []

    def step(self, now: float, store, symbols: List[str], frames: Dict, books_fn: Callable, resource_state: str) -> None:
        if resource_state != "GREEN":
            return
        try:
            self._pairs(now, store, symbols)
        except Exception as e:
            self.errors["pairs"] = f"{type(e).__name__}: {str(e)[:80]}"
        try:
            self._carry(now, symbols)
        except Exception as e:
            self.errors["carry"] = f"{type(e).__name__}: {str(e)[:80]}"
        try:
            self._tri(now)
        except Exception as e:
            self.errors["tri"] = f"{type(e).__name__}: {str(e)[:80]}"
        try:
            self._mm(now, frames, books_fn)
        except Exception as e:
            self.errors["mm"] = f"{type(e).__name__}: {str(e)[:80]}"
        self.events = self.events[-50:]

    def _closes_4h(self, store, symbols: List[str]) -> Dict[str, np.ndarray]:
        out = {}
        for s in symbols[:20]:
            try:
                df = store.get_ohlcv(self.exchange_id, s, "4h", 300)
                if df is not None and len(df) >= 150:
                    out[s] = df["close"].astype(float).to_numpy()
            except Exception:
                continue
        return out

    def _pairs(self, now: float, store, symbols: List[str]) -> None:
        need_scan = self.pairs.last_scan_ts is None or now - float(self.pairs.last_scan_ts) >= PAIRS_SCAN_SEC
        need_step = now - self._last["pairs_step"] >= PAIRS_STEP_SEC
        if not (need_scan or need_step):
            return
        closes = self._closes_4h(store, symbols)
        if len(closes) < 3:
            return
        if need_scan:
            found = self.pairs.rescan(closes, now)
            self.events.append({"ts": now, "mod": "pairs", "msg": f"kointegrasyon taraması: {len(found)} çift"})
        for ev in self.pairs.step(closes, now):
            self.events.append({"ts": now, "mod": "pairs", "msg": str(ev)[:120]})
        self._last["pairs_step"] = now

    def _carry(self, now: float, symbols: List[str]) -> None:
        if not self.carry_enabled or now - self._last["carry"] < CARRY_SEC:
            return
        self._last["carry"] = now
        rates = CR.fetch_funding_ccxt("binance", symbols, self._client_factory)
        if not rates:
            self.carry.error = "funding çekilemedi"
            return
        self.carry.error = None
        for ev in self.carry.step(rates, now, self.taker_bps, 5.0):
            self.events.append({"ts": now, "mod": "carry", "msg": str(ev)[:120]})

    def _tri(self, now: float) -> None:
        if self._fetch_tickers is None or now - self._last["tri"] < TRI_SEC:
            return
        self._last["tri"] = now
        tickers = self._fetch_tickers()
        if not tickers:
            return
        found = self.tri.scan(tickers, now, self.taker_bps, 5.0)
        if found:
            self.events.append({"ts": now, "mod": "triangular", "msg": f"{len(found)} üçgen R>1: {found[0]['path']} +{found[0]['net_bps']} bps"})

    def _mm(self, now: float, frames: Dict, books_fn: Callable) -> None:
        for sym in MM.ELIGIBLE:
            df = frames.get(sym)
            if df is None or len(df) < 30:
                continue
            c = df["close"].astype(float)
            mid = float(c.iloc[-1])
            sigma_pct = float(c.pct_change().tail(60).std(ddof=0) * 100.0)
            book = books_fn(sym)
            obi = None
            if book and (book.get("bid_depth_usd") or 0) + (book.get("ask_depth_usd") or 0) > 0:
                obi = float(book["bid_depth_usd"]) / (float(book["bid_depth_usd"]) + float(book["ask_depth_usd"]))
            self.mm.step(sym, mid, float(df["high"].iloc[-1]), float(df["low"].iloc[-1]), sigma_pct, book, now, obi)

    def status(self, lifecycle_status=None) -> Dict:
        return {"pairs": self.pairs.status(), "carry": {**self.carry.status(), "enabled": self.carry_enabled},
                "triangular": self.tri.status(), "market_making": self.mm.status(),
                "errors": self.errors, "events": self.events[-12:][::-1],
                "library": LIB.summary(lifecycle_status)}
