"""
Bildirim Sistemi (#4) — Telegram + Discord.

İşlem adayı (actionable) bir sinyal oluştuğunda anlık uyarı gönderir.
Env anahtarları:
  TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID   → Telegram
  DISCORD_WEBHOOK_URL                     → Discord
Anahtar yoksa sessizce no-op (sistem çalışmaya devam eder).
"""
from __future__ import annotations

from typing import List

from ..core.models import TradeSignal

try:
    import requests
    _HAS_REQUESTS = True
except Exception:
    _HAS_REQUESTS = False


class Notifier:
    def __init__(self, config):
        self.tg_token = config.secret("TELEGRAM_BOT_TOKEN")
        self.tg_chat = config.secret("TELEGRAM_CHAT_ID")
        self.discord = config.secret("DISCORD_WEBHOOK_URL")

    @property
    def enabled(self) -> bool:
        return bool((self.tg_token and self.tg_chat) or self.discord)

    def _format(self, s: TradeSignal) -> str:
        tps = " / ".join(f"{t:.4f}" for t in s.take_profits)
        lines = [
            f"🤖 AGI TRADER — {s.symbol} [{s.timeframe}]",
            f"{'🟢' if s.direction.value=='LONG' else '🔴'} {s.direction.value} ({s.bias})  güven %{s.confidence*100:.0f}",
            f"Giriş: {s.entry:.4f}",
            f"Stop: {s.stop_loss:.4f}  |  TP: {tps}",
            f"R/R: {s.risk_reward}  |  Alış %{s.buy_pressure_pct} / Satış %{s.sell_pressure_pct}",
        ]
        if s.forecast:
            f = s.forecast
            lines.append(f"Sonraki MAKS≈{f.get('expected_high'):.4f} / MİN≈{f.get('expected_low'):.4f}")
        if s.reasons:
            lines.append("• " + s.reasons[-1])
        return "\n".join(lines)

    def send_signal(self, s: TradeSignal) -> dict:
        if not (_HAS_REQUESTS and self.enabled):
            return {"sent": False, "reason": "bildirim anahtarı yok veya requests yok"}
        msg = self._format(s)
        results = {}
        if self.tg_token and self.tg_chat:
            try:
                r = requests.post(
                    f"https://api.telegram.org/bot{self.tg_token}/sendMessage",
                    json={"chat_id": self.tg_chat, "text": msg}, timeout=10)
                results["telegram"] = r.status_code == 200
            except Exception as e:
                results["telegram"] = f"hata: {e}"
        if self.discord:
            try:
                r = requests.post(self.discord, json={"content": "```\n" + msg + "\n```"}, timeout=10)
                results["discord"] = r.status_code in (200, 204)
            except Exception as e:
                results["discord"] = f"hata: {e}"
        return {"sent": True, "channels": results}

    def send_text(self, title: str, body: str) -> dict:
        """Genel metin bildirimi (alarmlar için)."""
        if not (_HAS_REQUESTS and self.enabled):
            return {"sent": False, "reason": "anahtar yok"}
        msg = f"{title}\n{body}"
        results = {}
        if self.tg_token and self.tg_chat:
            try:
                r = requests.post(f"https://api.telegram.org/bot{self.tg_token}/sendMessage",
                                  json={"chat_id": self.tg_chat, "text": msg}, timeout=10)
                results["telegram"] = r.status_code == 200
            except Exception as e:
                results["telegram"] = f"hata: {e}"
        if self.discord:
            try:
                r = requests.post(self.discord, json={"content": msg}, timeout=10)
                results["discord"] = r.status_code in (200, 204)
            except Exception as e:
                results["discord"] = f"hata: {e}"
        return {"sent": True, "channels": results}

    def notify_actionable(self, signals: List[TradeSignal]) -> dict:
        out = {}
        for s in signals:
            if s.actionable:
                out[s.symbol] = self.send_signal(s)
        return out
