"""
Kağıt Portföy + Kâr/Zarar (PnL) defteri — otonom işlemcinin "kazanım" katmanı.

Pozisyon yaşam döngüsünü TAM yönetir: aç → mark-to-market → kademeli kâr al
(TP1/TP2/TP3'te 1/3) → stop / iptal → kapat → realize PnL. Özsermaye (equity)
eğrisi, gerçekleşen/gerçekleşmeyen PnL, kazanma oranı ve drawdown tutulur.

Hepsi USDT-notional yüzde-hareket bazlıdır (kaldıraçsız spot varsayımı):
  pnl_pct = (çıkış/giriş - 1) * (LONG:+1 / SHORT:-1)
  pnl_usdt = notional * pnl_pct
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..core.models import Direction


@dataclass
class Position:
    symbol: str
    direction: str               # "LONG" | "SHORT"
    entry: float
    stop: float
    take_profits: List[float]
    notional: float              # açılış USDT büyüklüğü
    opened_ts: float = field(default_factory=time.time)
    remaining: float = 1.0       # kalan oran (kademeli çıkışta azalır)
    tp_hits: int = 0
    realized: float = 0.0        # bu pozisyondan şimdiye dek realize PnL (USDT)
    last_price: float = 0.0
    invalidation: Optional[float] = None

    def dir_sign(self) -> int:
        return 1 if self.direction == "LONG" else -1

    def unrealized(self, price: float) -> float:
        return self.notional * self.remaining * (price / self.entry - 1) * self.dir_sign()

    def to_dict(self, price: Optional[float] = None) -> Dict:
        p = price if price is not None else self.last_price
        upnl = self.unrealized(p) if p else 0.0
        return {
            "symbol": self.symbol, "direction": self.direction, "entry": round(self.entry, 6),
            "stop": round(self.stop, 6), "take_profits": [round(t, 6) for t in self.take_profits],
            "notional": round(self.notional, 2), "remaining": round(self.remaining, 3),
            "tp_hits": self.tp_hits, "last_price": round(p, 6) if p else 0.0,
            "unrealized": round(upnl, 2), "realized": round(self.realized, 2),
            "pnl_pct": round((p / self.entry - 1) * 100 * self.dir_sign(), 2) if p else 0.0,
            "age_min": round((time.time() - self.opened_ts) / 60, 1),
        }


class Portfolio:
    def __init__(self, initial_equity: float = 10_000.0):
        self.initial = float(initial_equity)
        self.realized = 0.0
        self.positions: Dict[str, Position] = {}
        self.trades: List[Dict] = []            # kapanan işlemler
        self.equity_curve: List[Dict] = []      # {ts, equity}
        self.peak = self.initial
        self._mark()

    # ----------------------------------------------------------------- equity
    def unrealized_total(self) -> float:
        return sum(p.unrealized(p.last_price) for p in self.positions.values() if p.last_price)

    def equity(self) -> float:
        return self.initial + self.realized + self.unrealized_total()

    def drawdown(self) -> float:
        eq = self.equity()
        self.peak = max(self.peak, eq)
        return (self.peak - eq) / (self.peak + 1e-12)

    def _mark(self):
        self.equity_curve.append({"ts": time.time(), "equity": round(self.equity(), 2)})
        if len(self.equity_curve) > 5000:
            self.equity_curve = self.equity_curve[-5000:]

    # ------------------------------------------------------------------- open
    def open(self, symbol, direction, entry, stop, take_profits, notional,
             invalidation=None) -> Optional[Position]:
        if symbol in self.positions or notional <= 0:
            return None
        pos = Position(symbol=symbol, direction=direction, entry=float(entry),
                       stop=float(stop), take_profits=[float(t) for t in take_profits],
                       notional=float(notional), last_price=float(entry),
                       invalidation=invalidation)
        self.positions[symbol] = pos
        return pos

    # ------------------------------------------------ mark-to-market + exitler
    def update(self, prices: Dict[str, float]) -> List[Dict]:
        """Güncel fiyatlarla pozisyonları işaretle; stop/TP/iptal tetikleyenleri kapat.
        Döndürür: bu turda gerçekleşen kapanış/çıkış olayları."""
        events: List[Dict] = []
        for sym, pos in list(self.positions.items()):
            price = prices.get(sym)
            if price is None or price <= 0:
                continue
            pos.last_price = price
            sign = pos.dir_sign()

            # STOP / iptal -> tamamını kapat
            stop_hit = (price <= pos.stop) if sign > 0 else (price >= pos.stop)
            inval_hit = (pos.invalidation is not None and
                         ((price <= pos.invalidation) if sign > 0 else (price >= pos.invalidation)))
            if stop_hit or inval_hit:
                events.append(self._close(pos, price, "STOP" if stop_hit else "İPTAL"))
                continue

            # TP'ler -> kademeli çıkış (her TP'de kalanın 1/3'ü; sonuncuda tümü)
            for k, tp in enumerate(pos.take_profits):
                if k < pos.tp_hits:
                    continue
                tp_hit = (price >= tp) if sign > 0 else (price <= tp)
                if tp_hit:
                    pos.tp_hits = k + 1
                    if k + 1 >= len(pos.take_profits):
                        events.append(self._close(pos, price, f"TP{k+1}"))
                        break
                    # kademeli: notional'ın 1/3'ünü realize et, stop'u girişe çek
                    portion = 1.0 / 3.0
                    pnl = pos.notional * portion * (price / pos.entry - 1) * sign
                    pos.realized += pnl
                    self.realized += pnl
                    pos.remaining = max(0.0, pos.remaining - portion)
                    pos.stop = pos.entry  # breakeven
                    events.append({"symbol": sym, "event": f"TP{k+1}", "price": round(price, 6),
                                   "pnl": round(pnl, 2), "remaining": round(pos.remaining, 3),
                                   "kind": "partial"})
        self._mark()
        return events

    def _close(self, pos: Position, price: float, reason: str) -> Dict:
        sign = pos.dir_sign()
        pnl = pos.notional * pos.remaining * (price / pos.entry - 1) * sign
        pos.realized += pnl
        self.realized += pnl
        rec = {
            "symbol": pos.symbol, "direction": pos.direction, "entry": round(pos.entry, 6),
            "exit": round(price, 6), "reason": reason, "notional": round(pos.notional, 2),
            "pnl": round(pos.realized, 2),
            "pnl_pct": round((price / pos.entry - 1) * 100 * sign, 2),
            "opened_ts": pos.opened_ts, "closed_ts": time.time(),
            "win": pos.realized > 0,
        }
        self.trades.append(rec)
        self.positions.pop(pos.symbol, None)
        return {**rec, "event": "CLOSE", "kind": "close"}

    def close_symbol(self, symbol: str, price: float, reason: str = "HARİCİ") -> Optional[Dict]:
        pos = self.positions.get(symbol)
        if pos is None:
            return None
        ev = self._close(pos, float(price) or pos.last_price or pos.entry, reason)
        self._mark()
        return ev

    def force_close_all(self, prices: Dict[str, float], reason: str = "MANUEL") -> List[Dict]:
        out = []
        for sym, pos in list(self.positions.items()):
            price = prices.get(sym, pos.last_price or pos.entry)
            out.append(self._close(pos, price, reason))
        self._mark()
        return out

    # ------------------------------------------------------------------ stats
    def stats(self) -> Dict:
        wins = [t for t in self.trades if t["win"]]
        losses = [t for t in self.trades if not t["win"]]
        gross_win = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        return {
            "initial": round(self.initial, 2),
            "equity": round(self.equity(), 2),
            "realized_pnl": round(self.realized, 2),
            "unrealized_pnl": round(self.unrealized_total(), 2),
            "return_pct": round((self.equity() / self.initial - 1) * 100, 2),
            "open_positions": len(self.positions),
            "closed_trades": len(self.trades),
            "win_rate": round(100 * len(wins) / len(self.trades), 1) if self.trades else 0.0,
            "profit_factor": round(gross_win / (gross_loss + 1e-9), 2) if gross_loss else (gross_win and 99.9 or 0.0),
            "max_drawdown_pct": round(self.drawdown() * 100, 2),
            "peak_equity": round(self.peak, 2),
        }
