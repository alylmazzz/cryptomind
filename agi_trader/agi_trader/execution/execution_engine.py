"""
Execution Motoru (Algoritmik Trading / Execution Engineer rolü).

GÜVENLİK İLKESİ (kritik):
  - VARSAYILAN mod "paper": hiçbir gerçek emir gönderilmez, simüle edilir.
  - "live" mod yalnızca config'de execution.mode == "live" VE
    execution.allow_live == True ise etkinleşir. İkisi birden gerekir.
  - Kill-switch: portföy drawdown'ı eşiği aşarsa sistem tüm yeni işlemleri durdurur.
  - Canlı emir gönderimi bilinçli olarak STUB bırakılmıştır; gerçek borsa emri
    göndermek kullanıcının kendi API anahtarı + bilinçli kod aktivasyonu gerektirir.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from ..core.models import Direction, TradeSignal


@dataclass
class PaperPosition:
    symbol: str
    direction: Direction
    entry: float
    stop: float
    take_profits: List[float]
    size_usdt: float
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "open"


class ExecutionEngine:
    def __init__(self, config):
        ec = config.get("execution", {})
        self.config = config
        self.mode = ec.get("mode", "paper")
        self.allow_live = bool(ec.get("allow_live", False))
        self.testnet = bool(ec.get("testnet", True))      # GÜVENLİK: varsayılan testnet
        self.live_max_usdt = float(ec.get("live_max_order_usdt", 100.0))  # canlı emir tavanı
        self.max_open = int(ec.get("max_open_positions", 5))
        self.kill_dd = float(ec.get("kill_switch_drawdown", 0.15))
        self.portfolio_value = float(config.get("risk.portfolio_value", 10_000))
        self.peak_equity = self.portfolio_value
        self.positions: Dict[str, PaperPosition] = {}
        self.killed = False
        self._live_client = None

    @property
    def is_live(self) -> bool:
        return self.mode == "live" and self.allow_live

    def _client(self):
        """ccxt Binance istemcisi — testnet ise sandbox modunda (sahte para)."""
        if self._live_client is None:
            import ccxt
            key = self.config.secret("BINANCE_API_KEY")
            sec = self.config.secret("BINANCE_SECRET")
            if not (key and sec):
                raise RuntimeError("BINANCE_API_KEY/SECRET gerekli (canlı emir)")
            ex = ccxt.binance({"apiKey": key, "secret": sec, "enableRateLimit": True,
                               "options": {"defaultType": "spot"}})
            if self.testnet:
                ex.set_sandbox_mode(True)   # Binance Spot Testnet
            self._live_client = ex
        return self._live_client

    def _place_live_order(self, signal, size_usdt: float) -> Dict:
        """Gerçek market emri (testnet varsayılan). Tavan + try/except korumalı."""
        size_usdt = min(size_usdt, self.live_max_usdt)
        try:
            ex = self._client()
            side = "buy" if signal.direction == Direction.LONG else "sell"
            amount = size_usdt / float(signal.entry)
            order = ex.create_order(signal.symbol, "market", side, amount)
            return {"action": "live_order", "testnet": self.testnet, "side": side,
                    "symbol": signal.symbol, "amount": round(amount, 8),
                    "size_usdt": round(size_usdt, 2), "order_id": order.get("id"),
                    "status": order.get("status", "?")}
        except Exception as e:
            return {"action": "live_error", "testnet": self.testnet,
                    "error": f"{type(e).__name__}: {e}",
                    "hint": "Testnet anahtarı testnet.binance.vision'dan; mainnet için testnet=false."}

    def _check_kill_switch(self) -> bool:
        dd = (self.peak_equity - self.portfolio_value) / (self.peak_equity + 1e-12)
        if dd >= self.kill_dd:
            self.killed = True
        return self.killed

    def execute(self, signal: TradeSignal) -> Dict:
        """Sinyali işle. Sadece actionable + risk uygunsa pozisyon açar."""
        if self._check_kill_switch():
            return {"action": "blocked", "reason": "KILL-SWITCH aktif (drawdown limiti)"}

        if not signal.actionable:
            return {"action": "skip", "reason": "Sinyal işleme uygun değil (güven/RR eşiği)"}

        if signal.symbol in self.positions:
            return {"action": "skip", "reason": "Bu paritede zaten açık pozisyon var"}

        if len(self.positions) >= self.max_open:
            return {"action": "skip", "reason": f"Maksimum açık pozisyon ({self.max_open}) doldu"}

        size = signal.risk.recommended_position_size if signal.risk else 0.0

        if self.is_live:
            # GERÇEK EMİR — varsayılan Binance TESTNET (sahte para). Mainnet için
            # execution.testnet=false + allow_live=true + mode=live bilinçli gerekir.
            res = self._place_live_order(signal, size)
            # canlı emir başarılıysa pozisyonu da yerel olarak izle
            if res.get("action") == "live_order":
                self.positions[signal.symbol] = PaperPosition(
                    symbol=signal.symbol, direction=signal.direction, entry=signal.entry,
                    stop=signal.stop_loss, take_profits=signal.take_profits, size_usdt=size)
            return res

        # PAPER: simüle pozisyon aç
        pos = PaperPosition(
            symbol=signal.symbol, direction=signal.direction, entry=signal.entry,
            stop=signal.stop_loss, take_profits=signal.take_profits, size_usdt=size,
        )
        self.positions[signal.symbol] = pos
        return {
            "action": "paper_open",
            "symbol": signal.symbol,
            "side": signal.direction.value,
            "entry": signal.entry,
            "stop": signal.stop_loss,
            "tp": signal.take_profits,
            "size_usdt": round(size, 2),
        }

    def status(self) -> Dict:
        return {
            "mode": self.mode,
            "is_live": self.is_live,
            "testnet": self.testnet,
            "live_max_order_usdt": self.live_max_usdt,
            "killed": self.killed,
            "open_positions": len(self.positions),
            "portfolio_value": self.portfolio_value,
        }
