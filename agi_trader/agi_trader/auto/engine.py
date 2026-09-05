"""
Otonom İşlem Motoru (AutoTrader) — sürekli tarayan + karar veren + kâğıt-emir
açan/kapatan arka-plan döngüsü. "Doğru zamanlarda doğru miktarlarda al-sat".

Her döngüde:
  1) Tüm pariteler için tam analiz pipeline'ı çalışır (orchestrator).
  2) Açık pozisyonlar güncel fiyatla işaretlenir; stop/TP/iptal tetikleyenler kapanır.
  3) Kill-switch: drawdown limiti aşılırsa yeni işlem durur (mevcutlar yönetilir).
  4) İşlem adayı (actionable) sinyaller için RİSK + KONVİKSİYON + maruziyet tavanı ile
     pozisyon boyutlanır ve KÂĞIT emir açılır.
  5) "Overall" anlık görüntü (her paritenin sinyali) komuta merkezine yayılır.

GÜVENLİK: Otonom motor YALNIZ paper (simülasyon) modunda emir açar. Canlı emir
mevcut güvenlik kapısının (execution.mode=live + allow_live) arkasındadır ve bu
motor canlıyı bilinçli olarak açmaz.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Dict, List, Optional

from .portfolio import Portfolio
from ..learn.online import OnlineLearner


class AutoTrader:
    def __init__(self, orchestrator, config):
        self.orch = orchestrator
        self.config = config
        self.portfolio = Portfolio(float(config.get("risk.portfolio_value", 10_000)))
        # online öğrenme: kapanan işlemlerden katman ağırlıklarını güncelle
        self.learner = OnlineLearner(dict(orchestrator.decision.base_weights),
                                     config.get("output_dir", "runs"))
        self.entry_context: Dict[str, Dict] = {}   # symbol -> {layer: contribution}
        self.symbols: List[str] = list(config.symbols)
        self.interval = 60
        self.max_open = int(config.get("execution.max_open_positions", 5))
        self.kill_dd = float(config.get("execution.kill_switch_drawdown", 0.15))
        self.max_exposure = 0.60      # equity'nin en fazla %60'ı açık notional
        self.mode = config.get("execution.mode", "paper")
        self.running = False
        self.killed = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self.last_cycle_ts: Optional[float] = None
        self.cycle_count = 0
        self.overview: Dict[str, Dict] = {}
        self.events: Deque[Dict] = deque(maxlen=200)

    # ----------------------------------------------------------- kontrol
    def start(self, interval: Optional[int] = None, symbols: Optional[List[str]] = None) -> Dict:
        with self._lock:
            if self.running:
                return {"ok": False, "reason": "otonom motor zaten çalışıyor"}
            if interval:
                self.interval = max(10, int(interval))
            if symbols:
                self.symbols = symbols
            self.killed = False
            self.running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        self._log("system", "▶️ Otonom motor başladı", {"interval": self.interval, "symbols": self.symbols})
        return {"ok": True, "running": True, "interval": self.interval, "symbols": self.symbols}

    def stop(self) -> Dict:
        with self._lock:
            self.running = False
        self._log("system", "⏹️ Otonom motor durduruldu", {})
        return {"ok": True, "running": False}

    def reset(self) -> Dict:
        with self._lock:
            if self.running:
                return {"ok": False, "reason": "önce durdur"}
            self.portfolio = Portfolio(float(self.config.get("risk.portfolio_value", 10_000)))
            self.killed = False
            self.events.clear()
            self.overview.clear()
            self.cycle_count = 0
        return {"ok": True}

    def external_signal(self, symbol: str, action: str, price: Optional[float] = None,
                        source: str = "harici") -> Dict:
        """Dış kaynaktan (TradingView webhook, bot komutu) gelen sinyali işle.
        action: buy/long · sell/short · close/exit. PAPER pozisyon açar/kapatır.
        Boyut: equity'nin %5'i (harici sinyalde risk analizi yok)."""
        with self._lock:
            act = (action or "").lower().strip()
            symbol = symbol.strip()
            px = float(price) if price else (self.overview.get(symbol, {}) or {}).get("price")
            if act in ("buy", "long", "sell", "short"):
                if self.killed:
                    return {"ok": False, "reason": "kill-switch aktif"}
                if symbol in self.portfolio.positions:
                    return {"ok": False, "reason": "bu paritede zaten açık pozisyon var"}
                if not px:
                    return {"ok": False, "reason": "fiyat yok (price gönderin)"}
                direction = "LONG" if act in ("buy", "long") else "SHORT"
                s = 1 if direction == "LONG" else -1
                notional = round(0.05 * self.portfolio.equity(), 2)
                stop = px * (1 - s * 0.03)
                tps = [px * (1 + s * 0.02), px * (1 + s * 0.04), px * (1 + s * 0.06)]
                pos = self.portfolio.open(symbol, direction, px, stop, tps, notional)
                if not pos:
                    return {"ok": False, "reason": "pozisyon açılamadı"}
                self._log("entry", f"📡 {source}: {symbol} {direction} AÇILDI · {notional:.0f} USDT @ {px}",
                          {"symbol": symbol, "direction": direction, "notional": notional, "source": source})
                self._alarm(symbol, f"HARİCİ ({source})", f"{symbol} {direction} açıldı @ {px}")
                return {"ok": True, "action": "open", "symbol": symbol, "direction": direction, "notional": notional}
            if act in ("close", "exit", "flat"):
                ev = self.portfolio.close_symbol(symbol, px or 0, f"HARİCİ {source}")
                if ev:
                    self._log("exit", f"📡 {source}: {symbol} KAPATILDI · PnL {ev.get('pnl',0):+.2f}", ev)
                    self._alarm(symbol, f"HARİCİ ({source})", f"{symbol} kapatıldı · PnL {ev.get('pnl',0):+.2f}")
                    return {"ok": True, "action": "close", "pnl": ev.get("pnl")}
                return {"ok": False, "reason": "kapatılacak açık pozisyon yok"}
            return {"ok": False, "reason": f"bilinmeyen aksiyon: {action}"}

    def close_all(self) -> Dict:
        with self._lock:
            prices = {s: o.get("price") for s, o in self.overview.items() if o.get("price")}
            closed = self.portfolio.force_close_all(prices, "MANUEL KAPAT")
        for c in closed:
            self._log("exit", f"✋ {c['symbol']} elle kapatıldı · PnL {c['pnl']:+.2f}", c)
        return {"ok": True, "closed": len(closed)}

    # ----------------------------------------------------------- döngü
    def _loop(self):
        while self.running:
            try:
                self.run_cycle()
            except Exception as e:
                self._log("error", f"Döngü hatası: {type(e).__name__}: {e}", {})
            slept = 0
            while self.running and slept < self.interval:
                time.sleep(1)
                slept += 1

    def run_cycle(self) -> Dict:
        # --- ağır analiz: kilit DIŞINDA (API yanıtını bloklamasın) ---
        signals = []
        prices: Dict[str, float] = {}
        snapshots: Dict[str, Dict] = {}
        for sym in list(self.symbols):
            try:
                sig = self.orch.analyze_symbol(sym)
                signals.append(sig)
                prices[sym] = float(sig.entry)
                snapshots[sym] = self._snapshot(sig)
            except Exception as e:
                self._log("error", f"{sym} analiz hatası: {type(e).__name__}", {})

        # --- paylaşılan durumu güncelle: kilit İÇİNDE (kısa) ---
        with self._lock:
            self.cycle_count += 1
            # 1) açık pozisyonları yönet
            for ev in self.portfolio.update(prices):
                emoji = "🟢" if ev.get("pnl", 0) >= 0 else "🔴"
                self._log("exit", f"{emoji} {ev['symbol']} {ev['event']} @ {ev.get('exit', ev.get('price'))} · PnL {ev.get('pnl',0):+.2f}", ev)
                self._alarm(ev["symbol"], "KAPANIŞ", f"{ev['symbol']} {ev['event']} · PnL {ev.get('pnl',0):+.2f} USDT")
                # online öğrenme: tam kapanışta sinyal katmanlarından öğren
                if ev.get("kind") == "close":
                    ctx = self.entry_context.pop(ev["symbol"], None)
                    if ctx:
                        new_w = self.learner.record(ctx, ev.get("direction", "LONG"), bool(ev.get("win")))
                        if new_w:
                            self.orch.decision.base_weights = new_w
                            best = max(self.learner.accuracy(), key=self.learner.accuracy().get)
                            self._log("system", f"🧠 Online öğrenme: {self.learner.total_trades} işlemden ağırlıklar güncellendi (en isabetli: {best})", {})
            # 2) kill-switch
            dd = self.portfolio.drawdown()
            if dd >= self.kill_dd and not self.killed:
                self.killed = True
                self._log("system", f"🛑 KILL-SWITCH — drawdown %{dd*100:.1f} ≥ %{self.kill_dd*100:.0f}", {})
                self._alarm("PORTFÖY", "KILL-SWITCH", f"Drawdown %{dd*100:.1f} — yeni işlem durduruldu")
            # 3) yeni pozisyon aç (paper, kill yoksa)
            if not self.killed and self.mode == "paper":
                for sig in signals:
                    if not sig.actionable or sig.symbol in self.portfolio.positions:
                        continue
                    if len(self.portfolio.positions) >= self.max_open:
                        break
                    notional = self._size(sig)
                    if notional <= 0:
                        continue
                    pos = self.portfolio.open(
                        sig.symbol, sig.direction.value, sig.entry, sig.stop_loss,
                        sig.take_profits, notional, sig.invalidation)
                    if pos:
                        # öğrenme için sinyal katman katkılarını sakla
                        self.entry_context[sig.symbol] = {
                            l["layer"]: l.get("contribution", 0.0)
                            for l in sig.layer_breakdown if not l["layer"].startswith("_")}
                        self._log("entry", f"✅ {sig.symbol} {sig.direction.value} AÇILDI · {notional:.0f} USDT · güven %{sig.confidence*100:.0f}",
                                  {"symbol": sig.symbol, "direction": sig.direction.value,
                                   "notional": round(notional, 2), "entry": sig.entry})
                        self._alarm(sig.symbol, "POZİSYON AÇ",
                                    f"{sig.direction.value} {notional:.0f} USDT @ {sig.entry:.4f} · R/R {sig.risk_reward}")
            self.overview = snapshots
            self.last_cycle_ts = time.time()
        return {"cycle": self.cycle_count, "open": len(self.portfolio.positions)}

    # ----------------------------------------------------------- boyutlama
    def _size(self, sig) -> float:
        """Risk-temelli (Kelly) öneri × güncel equity ölçeği × REJİM çarpanı,
        maruziyet tavanıyla. Dinamik: trend→tam, range→0.6, volatil→0.4, karşı-trend→0.5×."""
        base = sig.risk.recommended_position_size if getattr(sig, "risk", None) else 0.0
        if base <= 0:
            return 0.0
        equity = self.portfolio.equity()
        base *= max(0.1, equity / self.portfolio.initial)        # equity büyüdükçe ölçekle
        # rejim-temelli dinamik çarpan
        regime = next((l["detail"] for l in sig.layer_breakdown if l["layer"] == "_regime"), None)
        try:
            from ..analysis.regime import position_multiplier
            base *= position_multiplier(regime, sig.direction.value, sig.volatility)
        except Exception:
            pass
        open_notional = sum(p.notional * p.remaining for p in self.portfolio.positions.values())
        room = max(0.0, self.max_exposure * equity - open_notional)
        return float(min(base, room))

    # ----------------------------------------------------------- overall
    def _snapshot(self, sig) -> Dict:
        layers = {l["layer"]: round(l.get("contribution", 0), 4)
                  for l in sig.layer_breakdown if not l["layer"].startswith("_")}
        formations = next((l["detail"] for l in sig.layer_breakdown if l["layer"] == "_formations"), {})
        regime = next((l["detail"] for l in sig.layer_breakdown if l["layer"] == "_regime"), {})
        return {
            "symbol": sig.symbol, "direction": sig.direction.value, "bias": sig.bias.value,
            "confidence": round(sig.confidence, 3), "actionable": bool(sig.actionable),
            "price": float(sig.entry), "buy_pct": sig.buy_pressure_pct, "sell_pct": sig.sell_pressure_pct,
            "risk_reward": sig.risk_reward, "layers": layers,
            "formations": len(formations.get("patterns", [])),
            "confluence": len(formations.get("confluence", [])),
            "regime": regime.get("label", ""), "regime_emoji": regime.get("emoji", ""),
            "volatility": sig.volatility,
            "top_reason": sig.reasons[0] if sig.reasons else "",
        }

    # ----------------------------------------------------------- yardımcı
    def _log(self, typ: str, msg: str, extra: Dict):
        self.events.appendleft({"ts": time.time(), "type": typ, "msg": msg, **(extra or {})})

    def _alarm(self, symbol: str, kind: str, msg: str):
        try:
            self.orch.alarms._emit(symbol, kind, "high", msg)
        except Exception:
            pass

    def status(self) -> Dict:
        with self._lock:
            return {
                "running": self.running, "killed": self.killed, "mode": self.mode,
                "interval": self.interval, "symbols": self.symbols,
                "cycle_count": self.cycle_count,
                "last_cycle_ts": self.last_cycle_ts,
                "max_open": self.max_open, "max_exposure_pct": round(self.max_exposure * 100),
                "portfolio": self.portfolio.stats(),
            }

    def full_state(self, events_limit: int = 40) -> Dict:
        with self._lock:
            return {
                **self.status(),
                "positions": [p.to_dict() for p in self.portfolio.positions.values()],
                "overview": list(self.overview.values()),
                "recent_trades": self.portfolio.trades[-20:][::-1],
                "events": list(self.events)[:events_limit],
                "equity_curve": self.portfolio.equity_curve[-300:],
                "learning": self.learner.status(),
            }
