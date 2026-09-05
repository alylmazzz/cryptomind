"""
Alarm Motoru — fiyat + formasyon alarmları.

Her analiz turunda sinyalleri inceler ve şu durumlarda alarm üretir:
  • 🎯 Formasyon birleşimi (confluence) — birden çok formasyon aynı noktada
  • 🔺 Yeni formasyon tamamlandı (son barlarda)
  • 🎢 Fiyat kritik seviyeyi kırdı (giriş / stop / TP) — turlar arası takip
  • ✅ İşlem adayı (actionable, ≥ güven eşiği)

Alarmlar bellekte halka-tampon (ring buffer) olarak tutulur ve (yapılandırılmışsa)
Notifier ile Telegram/Discord'a iletilir. Kullanıcı ayrıca manuel fiyat alarmı
ekleyebilir (add_price_alarm) — fiyat seviyeyi geçince tetiklenir.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, List, Optional


class AlarmEngine:
    def __init__(self, notifier=None, maxlen: int = 100):
        self.notifier = notifier
        self.alarms: Deque[Dict] = deque(maxlen=maxlen)
        self._last_price: Dict[str, float] = {}
        self._seen_formations: Dict[str, set] = {}    # symbol -> {formasyon imzaları}
        self._user_alarms: List[Dict] = []            # manuel fiyat alarmları

    # ----------------------------------------------------------------- kayıt
    def _emit(self, symbol: str, kind: str, level: str, msg: str, extra: Optional[Dict] = None):
        a = {"ts": time.time(), "symbol": symbol, "kind": kind, "level": level,
             "message": msg, **(extra or {})}
        self.alarms.appendleft(a)
        # dış bildirim (anahtar varsa)
        if self.notifier and getattr(self.notifier, "enabled", False):
            try:
                self.notifier.send_text(f"🔔 {symbol} · {kind}", msg)
            except Exception:
                pass
        return a

    # --------------------------------------------------- manuel fiyat alarmı
    def add_price_alarm(self, symbol: str, price: float, direction: str = "cross") -> Dict:
        rec = {"symbol": symbol.upper(), "price": float(price),
               "direction": direction, "id": int(time.time() * 1000)}
        self._user_alarms.append(rec)
        return rec

    def list_price_alarms(self) -> List[Dict]:
        return list(self._user_alarms)

    def remove_price_alarm(self, alarm_id: int) -> bool:
        n = len(self._user_alarms)
        self._user_alarms = [a for a in self._user_alarms if a["id"] != alarm_id]
        return len(self._user_alarms) < n

    # ------------------------------------------------------------- ana kontrol
    def check_signal(self, signal) -> List[Dict]:
        """Bir TradeSignal'i incele, tetiklenen alarmları üret + döndür."""
        fired: List[Dict] = []
        sym = signal.symbol
        price = float(signal.entry)
        prev = self._last_price.get(sym)

        # _formations özetini layer_breakdown'dan çek
        formations = {}
        for lb in signal.layer_breakdown:
            if lb.get("layer") == "_formations":
                formations = lb.get("detail", {})
                break
        confluence = formations.get("confluence", [])
        patterns = formations.get("patterns", [])

        # 1) confluence alarmı
        for z in confluence:
            sig_key = f"conf:{z['bias']}:{round(z['price'], 2)}"
            seen = self._seen_formations.setdefault(sym, set())
            if sig_key not in seen:
                seen.add(sig_key)
                tag = "aynı yön ✓" if z.get("agree") else "karışık yön ⚠"
                fired.append(self._emit(sym, "BİRLEŞİM", "high",
                    f"🎯 {z['count']} formasyon aynı noktada (~{z['price']:.4f}, {z['bias']}, {tag}): "
                    f"{', '.join(z['members'])}", {"price": z["price"], "bias": z["bias"]}))

        # 2) yeni formasyon tamamlandı (son ~2 barda)
        for p in patterns:
            if p.get("near_last"):
                sig_key = f"pat:{p['name']}:{p['direction']}:{round(p.get('apex_price', 0), 2)}"
                seen = self._seen_formations.setdefault(sym, set())
                if sig_key not in seen:
                    seen.add(sig_key)
                    fired.append(self._emit(sym, "FORMASYON", "mid",
                        f"🔺 {p['name']} tamamlandı ({p['direction']}, kalite {p.get('quality')})",
                        {"direction": p["direction"]}))

        # 3) kritik seviye kırılımı — yalnız YÖNLÜ sinyalde (FLAT'ta stop/TP anlamsız,
        #    girişe yapışık olur → sahte alarm üretmesin)
        is_directional = getattr(signal, "direction", None) and signal.direction.value != "FLAT"
        if prev is not None and prev != price and is_directional:
            eps = max(abs(price) * 0.0005, 1e-9)  # girişe çok yakın seviyeleri ele
            levels = [("STOP", signal.stop_loss)]
            levels += [(f"TP{k+1}", t) for k, t in enumerate(signal.take_profits or [])]
            lo, hi = min(prev, price), max(prev, price)
            for name, lv in levels:
                if lv and abs(lv - price) > eps and lo <= lv <= hi:
                    arrow = "▲" if price > prev else "▼"
                    fired.append(self._emit(sym, "SEVİYE", "high",
                        f"🎢 Fiyat {name} seviyesini kırdı {arrow} ({lv:.4f}) — şu an {price:.4f}",
                        {"level": name, "value": lv}))

        # 4) manuel fiyat alarmları
        if prev is not None and prev != price:
            lo, hi = min(prev, price), max(prev, price)
            for ua in [a for a in self._user_alarms if a["symbol"] == sym.upper()]:
                if lo <= ua["price"] <= hi:
                    arrow = "▲" if price > prev else "▼"
                    fired.append(self._emit(sym, "FİYAT ALARMI", "high",
                        f"🔔 {sym} {ua['price']:.4f} seviyesini geçti {arrow} (şu an {price:.4f})",
                        {"alarm_id": ua["id"]}))

        # 5) işlem adayı
        if getattr(signal, "actionable", False):
            sig_key = f"act:{signal.direction.value}:{round(price, 2)}"
            seen = self._seen_formations.setdefault(sym, set())
            if sig_key not in seen:
                seen.add(sig_key)
                fired.append(self._emit(sym, "İŞLEM ADAYI", "high",
                    f"✅ {signal.direction.value} işlem adayı — güven %{signal.confidence*100:.0f}, "
                    f"R/R {signal.risk_reward}", {"direction": signal.direction.value}))

        self._last_price[sym] = price
        return fired

    def recent(self, limit: int = 30) -> List[Dict]:
        return list(self.alarms)[:limit]
