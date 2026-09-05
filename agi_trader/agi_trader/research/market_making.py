"""
AVELLANEDA–STOIKOV PİYASA YAPICILIK (uyarlanmış) — gölge kotasyon simülasyonu.

  rezervasyon fiyatı  r = mid − q·γ·σ²·(T−t)            (envanter cezası)
  optimal spread      δ = γ·σ²·(T−t) + (2/γ)·ln(1 + γ/k)
  bid = r − δ/2 · ask = r + δ/2 ; CryptoMind alfa (OBI/momentum) merkezi HAFİF kaydırır (≤ %25 δ),
  risk kontrolünü asla ezmez.

Yalnız derin/dar-spread paritelerde (BTC, ETH). Dolum: bar low ≤ bid → alım, bar high ≥ ask → satım.
Ters seçim ölçümü: dolumdan 1 bar sonra mid hareketi. Emir YOK.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

ELIGIBLE = ("BTC/USDT", "ETH/USDT")
MAX_SPREAD_BPS = 3.0
MIN_DEPTH_USD = 200_000.0


def as_quotes(mid: float, sigma_pct: float, inventory_q: float, gamma: float = 0.01, k: float = 1.0,
              t_remaining: float = 1.0, alpha_shift_bps: float = 0.0, min_spread_bps: float = 1.0) -> Dict:
    """Bps ölçeğinde A–S: σ (bps/bar), γ (1/bps), k (1/bps). Varsayılanlar dar-spread majörler için;
    kalibre EDİLMEDİ (k emir-akışı yoğunluğundan tahmin edilmeli) — gölge ölçümü bunu ortaya koyar."""
    sigma_bps = sigma_pct * 100.0
    var = sigma_bps * sigma_bps * max(1e-6, t_remaining)
    skew_bps = -inventory_q * gamma * var                       # uzun envanter → kotasyon aşağı
    delta_bps = gamma * var + (2.0 / gamma) * math.log(1.0 + gamma / k)
    delta_bps = max(delta_bps, min_spread_bps)
    shift = max(-0.25 * delta_bps, min(0.25 * delta_bps, alpha_shift_bps))
    r = mid * (1.0 + (skew_bps + shift) / 1e4)
    delta = mid * delta_bps / 1e4
    return {"reservation": r, "spread": delta, "bid": r - delta / 2.0, "ask": r + delta / 2.0,
            "spread_bps": delta_bps, "skew_bps": skew_bps + shift}


class MMShadow:
    def __init__(self, path: Optional[Path] = None, quote_usd: float = 100.0, max_inventory_usd: float = 300.0,
                 maker_fee_bps: float = 0.0):
        self.path = Path(path) if path else None
        self.quote_usd = quote_usd
        self.max_inv = max_inventory_usd
        self.fee = maker_fee_bps / 1e4
        self.state: Dict[str, Dict] = {}
        self.fills: List[Dict] = []
        self.load()

    def load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            self.state, self.fills = d.get("state", {}), d.get("fills", [])
        except Exception:
            pass

    def save(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"state": self.state, "fills": self.fills[-500:]}, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def step(self, symbol: str, mid: float, bar_high: float, bar_low: float, sigma_pct: float, book: Optional[Dict],
             now: float, obi: Optional[float] = None) -> Optional[Dict]:
        if symbol not in ELIGIBLE:
            return None
        b = book or {}
        if (b.get("spread_bps") or 99.0) > MAX_SPREAD_BPS or min(float(b.get("bid_depth_usd") or 0), float(b.get("ask_depth_usd") or 0)) < MIN_DEPTH_USD:
            return {"symbol": symbol, "skipped": "likidite/spread uygun değil"}
        st = self.state.setdefault(symbol, {"inv_usd": 0.0, "inv_qty": 0.0, "cash": 0.0, "pnl": 0.0, "n_fills": 0, "adverse_bps_sum": 0.0, "last_quote": None})
        # önceki kotasyondan ters seçim ölçümü
        lq = st.get("last_quote")
        if lq and lq.get("filled_side"):
            move = (mid / lq["fill_px"] - 1.0) * 1e4 * (1.0 if lq["filled_side"] == "BUY" else -1.0)
            st["adverse_bps_sum"] += -move            # alımdan sonra düşüş = ters seçim
        q = st["inv_usd"] / self.quote_usd            # envanter (kota birimi)
        alpha = 0.0 if obi is None else (obi - 0.5) * 4.0    # OBI 0,75 → +1 bps
        qt = as_quotes(mid, sigma_pct, q, alpha_shift_bps=alpha)
        filled = None
        if bar_low <= qt["bid"] and st["inv_usd"] < self.max_inv:
            qty = self.quote_usd / qt["bid"]
            st["inv_qty"] += qty; st["inv_usd"] += self.quote_usd; st["cash"] -= self.quote_usd * (1 + self.fee)
            filled = ("BUY", qt["bid"])
        elif bar_high >= qt["ask"] and st["inv_usd"] > -self.max_inv:
            qty = self.quote_usd / qt["ask"]
            st["inv_qty"] -= qty; st["inv_usd"] -= self.quote_usd; st["cash"] += self.quote_usd * (1 - self.fee)
            filled = ("SELL", qt["ask"])
        if filled:
            st["n_fills"] += 1
            self.fills.append({"ts": now, "symbol": symbol, "side": filled[0], "px": filled[1], "spread_bps": round(qt["spread_bps"], 3)})
        st["pnl"] = round(st["cash"] + st["inv_qty"] * mid, 4)
        st["last_quote"] = {"bid": qt["bid"], "ask": qt["ask"], "filled_side": filled[0] if filled else None, "fill_px": filled[1] if filled else None, "ts": now}
        st["last_ts"] = now
        if filled or st["n_fills"] % 10 == 0:
            self.save()
        return {"symbol": symbol, "bid": qt["bid"], "ask": qt["ask"], "spread_bps": round(qt["spread_bps"], 3),
                "skew_bps": round(qt["skew_bps"], 3), "filled": filled, "inv_usd": round(st["inv_usd"], 2), "pnl": st["pnl"]}

    def status(self) -> Dict:
        rows = []
        for s, st in self.state.items():
            n = st.get("n_fills", 0)
            rows.append({"symbol": s, "n_fills": n, "pnl_usd": st.get("pnl"), "inv_usd": round(st.get("inv_usd", 0.0), 2),
                         "adverse_selection_bps_avg": (round(st["adverse_bps_sum"] / n, 3) if n else None), "last_ts": st.get("last_ts")})
        return {"rows": rows, "quote_usd": self.quote_usd, "eligible": list(ELIGIBLE), "stage": "SHADOW",
                "note": "A–S kotasyon; bar high/low ile dolum varsayımı iyimserdir (kuyruk önceliği yok); emir yok"}
