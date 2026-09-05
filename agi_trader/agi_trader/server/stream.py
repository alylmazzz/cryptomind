"""
Canlı Fiyat Akışı (#1) — SSE için arka plan polling.

ccxt.pro (ücretli WS) gerektirmeden, REST ticker'ı düzenli aralıkla çekip
abone olan dashboard istemcilerine Server-Sent Events ile push eder. Canlı
borsa yoksa sentetik son fiyat üzerinde küçük rastgele yürüyüş yayınlar.
"""
from __future__ import annotations

import queue
import random
import threading
import time
from typing import Dict, List, Set

from ..data import synthetic


class LiveStreamer:
    def __init__(self, exchange_manager, config, interval: float = 5.0):
        self.em = exchange_manager
        self.cfg = config
        self.interval = interval
        self._subs: List[tuple] = []        # (queue, set(symbols))
        self._lock = threading.Lock()
        self._last: Dict[str, float] = {}
        self._open_ref: Dict[str, float] = {}
        self._thread: threading.Thread = None
        self._running = False

    def subscribe(self, symbols: List[str]) -> "queue.Queue":
        q: queue.Queue = queue.Queue(maxsize=10)
        with self._lock:
            self._subs.append((q, set(symbols)))
        return q

    def unsubscribe(self, q: "queue.Queue"):
        with self._lock:
            self._subs = [(qq, s) for qq, s in self._subs if qq is not q]

    def _all_symbols(self) -> Set[str]:
        with self._lock:
            out: Set[str] = set()
            for _, s in self._subs:
                out |= s
            return out

    def _price(self, symbol: str) -> float:
        ex = None
        if getattr(self.em, "live_enabled", False):
            for v in self.em.exchanges.values():
                ex = v
                break
        if ex is not None:
            try:
                t = ex.fetch_ticker(symbol)
                return float(t["last"])
            except Exception:
                pass
        # sentetik fallback
        base = self._last.get(symbol) or float(synthetic.generate_ohlcv(symbol, "1m", 50)["close"].iloc[-1])
        return base * (1 + random.uniform(-0.0008, 0.0008))

    def _loop(self):
        while self._running:
            syms = self._all_symbols()
            snapshot: Dict[str, dict] = {}
            for sym in syms:
                p = self._price(sym)
                self._open_ref.setdefault(sym, p)
                chg = (p - self._open_ref[sym]) / (self._open_ref[sym] + 1e-9) * 100
                self._last[sym] = p
                snapshot[sym] = {"price": p, "chg": chg}
            with self._lock:
                for q, s in self._subs:
                    data = {k: v for k, v in snapshot.items() if k in s}
                    if data:
                        try:
                            q.put_nowait(data)
                        except queue.Full:
                            pass
            time.sleep(self.interval)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
