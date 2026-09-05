"""
GİRİŞ OPTİMİZASYONU — "şu anki fiyat otomatik giriş değildir".

Aday girişler E ∈ {best ask (marketable), mikro-fiyat/mid, best bid (maker), VWAP, EMA20,
yapısal destek}. Her E için:
  P(fill | E)      — mesafeye göre (ATR biriminde) azalan; maker öncülü kaldıraçlanır
  P(target önce)   — E'den ölçülen stop/hedef mesafesiyle p_win küçük düzeltme (daha iyi R/R)
  cost(E)          — maker/taker maliyet
  EV(E)            — p_win·(hedef−maliyet) − (1−p_win)·(stop+maliyet)  [E'den ölçülür]
  Fayda(E)         = P(fill)·EV(E) + (1−P(fill))·q·EV(taker)   [maker dolmazsa q≈0,85 olasılıkla
                     koşucu edge'i yeniden hesaplayıp taker'a kovalar; kalan = vazgeçilir]
  Fayda(taker)     = 0,97·EV(taker)
E* = argmax Fayda → maker-öncelikli, ama EV kıyasıyla (maker farkı küçük + dolum düşükse taker). Çıktı: ENTRY ZONE (alt/üst), OPTİMAL, MAX CHASE (aşılırsa giriş geçersiz).
Bölge/chase için mevcut qualification.robust.build_entry_plan yeniden kullanılır
(dolum olasılığı ölçülmedi notu korunur — burada kalibre EDİLMEMİŞ sezgisel P(fill) kullanılır
ve bu açıkça yazılır).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from ..qualification.robust import build_entry_plan

CHASE_Q = 0.85      # maker dolmayınca taker'a kovalama olasılığı (sezgisel; koşucu edge'i yeniden hesaplar)


def _p_fill(dist_atr: float, order_type: str, p_maker_prior: float) -> float:
    if order_type == "taker":
        return 0.97
    # limit: fiyatın o seviyeye gelmesi + kuyruk önceliği; ATR uzaklığıyla azalır
    base = max(0.05, min(0.95, p_maker_prior))
    return float(max(0.02, min(0.95, base * math.exp(-max(0.0, dist_atr) / 0.75))))


def optimize_entry(direction: str, price: float, book: Dict, levels: Dict, atr_pct: float,
                   stop_pct: float, target_pct: float, maker_bps: float, taker_bps: float,
                   p_win: float, p_maker_prior: float = 0.5, spread_bps: float = 0.0) -> Dict:
    s = 1.0 if direction == "LONG" else -1.0
    mid = price
    bid = float(book.get("bid") or price * (1 - (book.get("spread_bps") or spread_bps) / 2e4))
    ask = float(book.get("ask") or price * (1 + (book.get("spread_bps") or spread_bps) / 2e4))
    plan = build_entry_plan(direction, mid, bid, ask, atr_pct, None, False)
    atr_abs = max(1e-12, atr_pct / 100.0 * price)
    stop_lvl = price * (1.0 - s * stop_pct / 100.0)
    tgt_lvl = price * (1.0 + s * target_pct / 100.0)
    cost_maker = (2.0 * maker_bps + 4.0) / 100.0
    cost_taker = (2.0 * taker_bps + (book.get("spread_bps") or spread_bps or 0.0) + 4.0) / 100.0
    cands = []

    def add(name: str, e: Optional[float], order_type: str):
        if e is None or not math.isfinite(e) or e <= 0:
            return
        # yalnız fiyatın "iyi tarafında" ya da yakınında (LONG: e ≤ ask×(1+chase))
        dist = (price - e) * s / atr_abs             # + = daha iyi (ucuz) giriş
        if dist < -0.6:                              # kovalamak yok
            return
        stop_d = (e - stop_lvl) * s / e * 100.0
        tgt_d = (tgt_lvl - e) * s / e * 100.0
        if stop_d <= 0 or tgt_d <= 0:
            return
        cost = cost_maker if order_type == "maker" else cost_taker
        pf = _p_fill(dist, order_type, p_maker_prior)
        # daha iyi giriş → aynı stop/hedef seviyesinde R/R iyileşir; p_win küçük ölçekli düzeltme
        rr = tgt_d / stop_d
        pw = float(min(0.9, max(0.1, p_win + 0.03 * (dist))))
        ev = pw * (tgt_d - cost) - (1.0 - pw) * (stop_d + cost)
        cands.append({"name": name, "price": round(e, 8), "order_type": order_type,
                      "dist_atr": round(dist, 3), "p_fill": round(pf, 3), "cost_pct": round(cost, 4),
                      "stop_pct": round(stop_d, 4), "target_pct": round(tgt_d, 4), "rr": round(rr, 3),
                      "p_win": round(pw, 3), "ev_pct": round(ev, 4), "utility": round(ev * pf, 4)})

    add("best_ask_taker" if s > 0 else "best_bid_taker", ask if s > 0 else bid, "taker")
    add("mid_maker", mid, "maker")
    add("best_bid_maker" if s > 0 else "best_ask_maker", bid if s > 0 else ask, "maker")
    for k in ("vwap", "ema_fast", "support" if s > 0 else "resistance"):
        add(f"{k}_maker", levels.get(k), "maker")
    # maker adaylarının faydası: dolmazsa taker'a kovalama seçeneği (q) hesaba katılır
    ev_taker = next((c["ev_pct"] for c in cands if c["order_type"] == "taker"), None)
    if ev_taker is not None:
        for c in cands:
            if c["order_type"] == "maker":
                c["utility"] = round(c["p_fill"] * c["ev_pct"] + (1.0 - c["p_fill"]) * CHASE_Q * ev_taker, 4)
    cands.sort(key=lambda c: -c["utility"])
    best = cands[0] if cands else None
    chase = plan.max_chase_price
    return {"candidates": cands[:6], "optimal": best,
            "entry_low": plan.entry_low, "entry_high": plan.entry_high, "max_chase": chase,
            "order_type": (best["order_type"] if best else plan.order_type),
            "invalidated": bool(chase is not None and (price - chase) * s > 0),
            "note": "P(fill) sezgisel, kalibre EDİLMEDİ (kuyruk/akış geçmişi yok); " + plan.reason}
