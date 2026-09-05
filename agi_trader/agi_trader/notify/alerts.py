"""
UYARI KANALI — kullanıcıya giden bildirimler (madde 76). Sırla çalışmaz: token/URL yalnız ortamdan.

  CRYPTOMIND_TG_TOKEN + CRYPTOMIND_TG_CHAT   Telegram (yoksa TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID)
  CRYPTOMIND_ALERT_WEBHOOK                    genel JSON webhook (POST {key, level, text, ts})

Yapılandırılmamışsa sessizdir; yine de son 100 uyarıyı bellekte tutar (panel/rapor okur).
Anahtar başına en az 60 sn aralık (spam koruması); gönderim arka plan iş parçacığında, hot-loop bloklanmaz.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from collections import deque
from typing import Deque, Dict, Optional

MIN_INTERVAL = 60.0


class AlertBus:
    def __init__(self, env: Optional[Dict[str, str]] = None, min_interval: float = MIN_INTERVAL):
        e = env if env is not None else os.environ
        self.tg_token = e.get("CRYPTOMIND_TG_TOKEN") or e.get("TELEGRAM_BOT_TOKEN")
        self.tg_chat = e.get("CRYPTOMIND_TG_CHAT") or e.get("TELEGRAM_CHAT_ID")
        self.webhook = e.get("CRYPTOMIND_ALERT_WEBHOOK")
        self.min_interval = float(min_interval)
        self.log: Deque[Dict] = deque(maxlen=100)
        self._last: Dict[str, float] = {}
        self._lock = threading.Lock()
        self.sent = 0
        self.errors = 0
        self._sender = None                      # test için enjekte edilebilir

    @property
    def configured(self) -> bool:
        return bool((self.tg_token and self.tg_chat) or self.webhook)

    def send(self, key: str, text: str, level: str = "info", now: Optional[float] = None, force: bool = False) -> bool:
        now = time.time() if now is None else now
        with self._lock:
            if not force and now - self._last.get(key, -1e9) < self.min_interval:
                return False
            self._last[key] = now
            rec = {"ts": now, "key": key, "level": level, "text": text[:900], "delivered": None}
            self.log.appendleft(rec)
        if not self.configured and self._sender is None:
            rec["delivered"] = False
            return True
        threading.Thread(target=self._deliver, args=(rec,), daemon=True).start()
        return True

    def _deliver(self, rec: Dict) -> None:
        ok = False
        try:
            if self._sender is not None:
                ok = bool(self._sender(rec))
            else:
                if self.tg_token and self.tg_chat:
                    data = urllib.parse.urlencode({"chat_id": self.tg_chat, "text": f"[CryptoMind {rec['level'].upper()}] {rec['text']}"}).encode()
                    req = urllib.request.Request(f"https://api.telegram.org/bot{self.tg_token}/sendMessage", data=data)
                    with urllib.request.urlopen(req, timeout=10) as r:
                        ok = r.status == 200
                if self.webhook:
                    body = json.dumps({"key": rec["key"], "level": rec["level"], "text": rec["text"], "ts": rec["ts"]}).encode()
                    req = urllib.request.Request(self.webhook, data=body, headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=10) as r:
                        ok = ok or (200 <= r.status < 300)
        except Exception:
            ok = False
        rec["delivered"] = ok
        if ok:
            self.sent += 1
        else:
            self.errors += 1

    def status(self) -> Dict:
        return {"configured": self.configured, "telegram": bool(self.tg_token and self.tg_chat), "webhook": bool(self.webhook),
                "sent": self.sent, "errors": self.errors, "recent": list(self.log)[:10]}


import urllib.parse  # noqa: E402  (urlencode)
