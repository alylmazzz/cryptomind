"""
Fırsat motoru çekirdeği — şartname 29, 30, 34, 44, 53 ve 62.

VARSAYILAN ÇIKTI **NO_TRADE**'dir. Bir fırsatın yayımlanabilmesi için
kapıların HEPSİNİ geçmesi gerekir; biri bile düşerse gerekçesiyle reddedilir.

Şartnamenin mantrası burada kodlanmıştır:

    NO DATA      → NO TRADE
    NO EDGE      → NO TRADE
    NO LIQUIDITY → NO TRADE
    NO CONFIDENCE→ NO TRADE

⚠️ BU MODÜL EMİR GÖNDERMEZ. Yalnız değerlendirir ve sıralar. Yürütme
`risk/live_guard.py`'nin üç anahtarına bağlıdır ve şu an ÜÇÜ DE KAPALIDIR.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

from .costs import CostEstimate, estimate_costs, required_gross_move_pct


@dataclass
class Gates:
    """Yayımlama eşikleri. Şartname 30/53: bunlar elle 'iyi görünüyor' diye
    seçilmez — walk-forward ile kalibre edilir. Buradaki değerler BAŞLANGIÇ
    noktasıdır ve `calibrated=False` ile işaretlenir."""
    min_net_return_pct: float = 1.0
    min_expected_value_pct: float = 0.0
    min_execution_probability: float = 0.60
    min_liquidity_score: float = 0.60
    min_data_quality: float = 0.80
    min_confidence: float = 0.60
    max_risk_score: float = 70.0
    calibrated: bool = False          # walk-forward'dan geçmediyse False


@dataclass
class Opportunity:
    """Şartname 29/42: bütün stratejiler TEK formatta — böylece aynı sıralama
    motorundan geçebilirler."""
    id: str
    strategy: str
    symbol: str
    direction: str                     # LONG | SHORT
    horizon: str

    entry: float = 0.0
    target: float = 0.0
    stop: float = 0.0

    gross_return_pct: float = 0.0
    costs: Optional[Dict] = None
    net_return_pct: float = 0.0

    p_target_first: float = 0.0
    p_stop_first: float = 0.0
    p_timeout: float = 0.0
    expected_value_pct: float = 0.0

    execution_probability: float = 0.0
    liquidity_score: float = 0.0
    data_quality: float = 0.0
    confidence: float = 0.0
    risk_score: float = 100.0

    capital_required: float = 0.0
    max_capacity_usd: float = 0.0

    detected_at: float = field(default_factory=time.time)
    expires_at: float = 0.0

    action: str = "NO_TRADE"
    reject_reasons: List[str] = field(default_factory=list)
    score: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


def expected_value_pct(net_return_pct: float, stop_pct: float,
                       p_target: float, p_stop: float, p_timeout: float,
                       timeout_return_pct: float = 0.0) -> float:
    """Şartname 34. EV yalnız kazanç×olasılık değildir; kayıp ve zaman aşımı
    da girer. Olasılıklar 1'e toplanmıyorsa normalize edilir."""
    s = p_target + p_stop + p_timeout
    if s <= 0:
        return float("-inf")
    pt, ps, po = p_target / s, p_stop / s, p_timeout / s
    return float(pt * net_return_pct - ps * abs(stop_pct) + po * timeout_return_pct)


def evaluate(op: Opportunity, gates: Gates) -> Opportunity:
    """Kapılardan geçir. VARSAYILAN REDDİR."""
    r: List[str] = []

    if op.data_quality < gates.min_data_quality:
        r.append(f"VERI_KALITESI ({op.data_quality:.2f} < {gates.min_data_quality})")
    if op.liquidity_score < gates.min_liquidity_score:
        r.append(f"LIKIDITE ({op.liquidity_score:.2f} < {gates.min_liquidity_score})")
    if op.net_return_pct < gates.min_net_return_pct:
        r.append(f"NET_GETIRI (%{op.net_return_pct:.3f} < %{gates.min_net_return_pct})")
    if not math.isfinite(op.expected_value_pct) or \
            op.expected_value_pct <= gates.min_expected_value_pct:
        r.append(f"BEKLENEN_DEGER (%{op.expected_value_pct:.3f} ≤ "
                 f"%{gates.min_expected_value_pct})")
    if op.execution_probability < gates.min_execution_probability:
        r.append(f"GERCEKLESME_OLASILIGI ({op.execution_probability:.2f} < "
                 f"{gates.min_execution_probability})")
    if op.confidence < gates.min_confidence:
        r.append(f"GUVEN ({op.confidence:.2f} < {gates.min_confidence})")
    if op.risk_score > gates.max_risk_score:
        r.append(f"RISK ({op.risk_score:.0f} > {gates.max_risk_score})")
    if op.expires_at and op.expires_at < time.time():
        r.append("SURESI_DOLMUS")

    op.reject_reasons = r
    op.action = "NO_TRADE" if r else ("BUY" if op.direction == "LONG" else "SELL")

    if not r:
        # Şartname 41/27 — tek sayı değil, bileşenlerden puan
        op.score = float(round(
            25 * min(1.0, op.net_return_pct / 2.0)
            + 25 * min(1.0, max(0.0, op.expected_value_pct) / 1.0)
            + 20 * op.execution_probability
            + 15 * op.liquidity_score
            + 15 * op.confidence
            - 10 * (op.risk_score / 100.0), 2))
    else:
        op.score = 0.0

    if not gates.calibrated:
        op.notes.append(
            "Eşikler walk-forward ile KALİBRE EDİLMEDİ — başlangıç değerleri. "
            "Bu haliyle sıralama göstergedir, işlem gerekçesi değildir.")
    return op


def build_price_opportunity(symbol: str, direction: str, entry: float,
                            target_net_pct: float, stop_pct: float,
                            *, bid_depth: float, ask_depth: float,
                            spread_bps: float, notional: float,
                            p_target: float, p_stop: float, p_timeout: float,
                            horizon: str = "4h",
                            funding_rate_8h: float = 0.0,
                            holding_hours: float = 4.0,
                            data_quality: float = 1.0,
                            confidence: float = 0.0) -> Opportunity:
    """Fiyat-yönlü fırsatı uçtan uca kurar: maliyet → net hedef → EV → kapı."""
    c = estimate_costs(notional, bid_depth, ask_depth, spread_bps,
                       holding_hours=holding_hours,
                       funding_rate_8h=funding_rate_8h, direction=direction)
    gerekli_brut = required_gross_move_pct(target_net_pct, c)

    if direction == "LONG":
        target = entry * (1 + gerekli_brut / 100.0)
        stop = entry * (1 - stop_pct / 100.0)
    else:
        target = entry * (1 - gerekli_brut / 100.0)
        stop = entry * (1 + stop_pct / 100.0)

    ev = expected_value_pct(target_net_pct, stop_pct, p_target, p_stop, p_timeout)

    # Likidite skoru: emrin bant derinliğine oranı + spread
    derinlik = min(bid_depth, ask_depth)
    doluluk = notional / derinlik if derinlik > 0 else 1.0
    lik = max(0.0, min(1.0, (1.0 - doluluk) * (1.0 - min(1.0, spread_bps / 20.0))))

    op = Opportunity(
        id=f"{symbol}|{direction}|{horizon}|{int(time.time())}",
        strategy="price_first_passage", symbol=symbol,
        direction=direction, horizon=horizon,
        entry=float(entry), target=float(target), stop=float(stop),
        gross_return_pct=float(gerekli_brut),
        costs=c.to_dict(),
        net_return_pct=float(target_net_pct),
        p_target_first=float(p_target), p_stop_first=float(p_stop),
        p_timeout=float(p_timeout),
        expected_value_pct=float(ev),
        execution_probability=float(max(0.0, min(1.0, 1.0 - doluluk))),
        liquidity_score=float(lik),
        data_quality=float(data_quality),
        confidence=float(confidence),
        risk_score=float(min(100.0, 100.0 * (1.0 - lik) + 20.0 * doluluk)),
        capital_required=float(notional),
        max_capacity_usd=float(derinlik * 0.25),
    )
    if c.warnings:
        op.notes.extend(c.warnings)
    return op


def rank(ops: List[Opportunity]) -> List[Opportunity]:
    """Şartname 52: sıralama yüzde değişime göre DEĞİL, EV ve skora göre."""
    gecerli = [o for o in ops if o.action != "NO_TRADE"]
    return sorted(gecerli, key=lambda o: (-o.score, -o.expected_value_pct))
