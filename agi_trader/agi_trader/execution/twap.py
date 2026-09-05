"""
TWAP dilimleme ve yürütme planlaması (FAZ 8).

NEDEN: Günlük yeniden dengeleme tek bir market emriyle yapılırsa, emir defterini
tek seferde yer ve kayma (slippage) emrin büyüklüğüyle ORANTISIZ artar. Aynı
miktarı N dilime bölüp aralıklarla göndermek ortalama fiyatı iyileştirir.

Ölçülen bağlam (bu proje): maker/limit yürütme Sharpe'ı 1,05 → 1,08 yapmıştı.
Kayma kazancı küçük ama GARANTİLİdir — alfa aramaktan farklı olarak burada
belirsizlik yok, sadece uygulama disiplini var.

Bu modül emir GÖNDERMEZ; yürütme PLANI üretir. Gerçek gönderim
`execution_engine` üzerinden ve `risk/live_guard` onayıyla yapılır.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np

# Plan parametreleri — plan tablosundan
DEFAULT_SLICES = 6
DEFAULT_INTERVAL_SEC = 300          # 6 × 5 dk = 30 dk
MAX_SLIPPAGE_BPS = 25.0             # aşılırsa dilim iptal
MIN_NOTIONAL_USDT = 25.0            # ücret sürtünmesi altında işlem yapma
POST_ONLY_RETRIES = 3               # 3 deneme sonra market'e düş


@dataclass
class Slice:
    index: int
    delay_sec: int
    qty: float
    notional: float
    order_type: str                 # "post_only_limit" | "market"
    note: str = ""


@dataclass
class ExecutionPlan:
    symbol: str
    side: str                       # "BUY" | "SELL"
    total_qty: float
    total_notional: float
    ref_price: float
    slices: List[Slice]
    max_slippage_bps: float
    est_cost_bps: float
    reason: str = ""

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["slices"] = [asdict(s) for s in self.slices]
        return d


def plan_twap(symbol: str, side: str, qty: float, ref_price: float,
              slices: int = DEFAULT_SLICES,
              interval_sec: int = DEFAULT_INTERVAL_SEC,
              adv_notional: Optional[float] = None,
              max_participation: float = 0.05,
              taker_fee: float = 0.0004, maker_fee: float = 0.0002,
              min_notional: float = MIN_NOTIONAL_USDT) -> ExecutionPlan:
    """Emri TWAP dilimlerine böl.

    adv_notional      : günlük ortalama hacim (USDT). Verilirse dilim boyu
                        hacmin `max_participation` oranını aşmayacak şekilde
                        dilim sayısı artırılır.
    max_participation : tek dilimin günlük hacme oranı tavanı (%5).

    Küçük emirler bölünmez: 6 dilime bölünen 60 $'lık emir, dilim başına 10 $
    ile minimum tutarın altında kalır ve ücret sürtünmesi kazancı yer."""
    qty = abs(float(qty))
    notional = qty * float(ref_price)
    side = side.upper()

    if notional < min_notional:
        return ExecutionPlan(symbol, side, qty, notional, ref_price, [],
                             MAX_SLIPPAGE_BPS, 0.0,
                             reason=f"tutar {notional:.2f} $ < minimum {min_notional} $ "
                                    f"— işlem yapılmaz (ücret sürtünmesi)")

    n = max(1, int(slices))
    # hacme göre katılım tavanı
    if adv_notional and adv_notional > 0:
        need = notional / (adv_notional * max_participation)
        n = max(n, int(math.ceil(need)))
        n = min(n, 48)                      # üst sınır: 4 saat

    per_qty = qty / n
    per_notional = notional / n
    # dilim çok küçülürse dilim sayısını azalt
    while n > 1 and per_notional < min_notional:
        n -= 1
        per_qty, per_notional = qty / n, notional / n

    out: List[Slice] = []
    for i in range(n):
        # son dilim: kalanı market ile kapat (kuyrukta kalmasın)
        is_last = (i == n - 1)
        out.append(Slice(
            index=i, delay_sec=i * int(interval_sec),
            qty=round(per_qty, 10), notional=round(per_notional, 4),
            order_type="market" if is_last and n > 1 else "post_only_limit",
            note=("kalanı kapat" if is_last and n > 1 else
                  f"{POST_ONLY_RETRIES} deneme post-only, sonra market")))

    # beklenen maliyet: son dilim taker, diğerleri maker varsayımı
    maker_share = (n - 1) / n if n > 1 else 0.0
    est = (maker_share * maker_fee + (1 - maker_share) * taker_fee) * 1e4
    return ExecutionPlan(symbol, side, qty, notional, float(ref_price), out,
                         MAX_SLIPPAGE_BPS, round(est, 2),
                         reason=f"{n} dilim × {interval_sec}s")


def limit_price(side: str, best_bid: float, best_ask: float,
                aggressiveness: float = 0.0) -> float:
    """Post-only limit fiyatı.

    aggressiveness 0 = pasif (BUY→best_bid, SELL→best_ask), 1 = karşı tarafa
    geç (taker olur, post-only reddedilir). Varsayılan 0: maker ücreti alınır.

    UYARI: post-only emir, fiyat karşı tarafa geçtiği anda BORSA TARAFINDAN
    REDDEDİLİR. Bu bir hata değil, korumadır — dolmamış emir yeniden
    fiyatlanır (bkz. POST_ONLY_RETRIES)."""
    if side.upper() == "BUY":
        return float(best_bid + (best_ask - best_bid) * aggressiveness)
    return float(best_ask - (best_ask - best_bid) * aggressiveness)


def slippage_bps(fill_price: float, ref_price: float, side: str) -> float:
    """Gerçekleşen kayma (bps). Pozitif = aleyhte."""
    if ref_price <= 0:
        return 0.0
    d = (fill_price - ref_price) / ref_price
    return float((d if side.upper() == "BUY" else -d) * 1e4)


def funding_aware_delay(next_funding_sec: int, funding_rate: float,
                        side: str, window_sec: int = 900) -> Dict:
    """Funding uzlaşmasına yakın ve funding ALEYHTE ise girişi ertele.

    Perp'te funding 8 saatte bir ödenir. Long açarken funding pozitifse (long'lar
    öder) uzlaşmadan hemen önce girmek gereksiz maliyettir; uzlaşma geçtikten
    sonra girmek yıllık %1-3 kazandırır (küçük ama garantili)."""
    adverse = (funding_rate > 0 and side.upper() == "BUY") or \
              (funding_rate < 0 and side.upper() == "SELL")
    if adverse and 0 < next_funding_sec <= window_sec:
        return {"delay": True, "wait_sec": int(next_funding_sec) + 30,
                "reason": (f"funding {funding_rate*100:+.4f}% aleyhte ve uzlaşmaya "
                           f"{next_funding_sec}s kaldı → uzlaşma sonrası gir")}
    return {"delay": False, "wait_sec": 0, "reason": ""}
