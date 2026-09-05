"""Başabaş olasılık, gereken lift ve gerçek lift — şartname 8, 9, 101, 104.

BU DOSYA "MODEL İYİ Mİ?" SORUSUNU ÖLÇÜLEBİLİR YAPAR

Ham doğruluk göstermek yasaktır. %75 "hedef önce" oranı, kör taban zaten %73
ise HİÇBİR ŞEY demektir. Gösterilecek sayı:

    MODEL LIFT OVER BASELINE = p_model_selected − p_baseline

ve bu lift, işlem yapmayı haklı çıkaracak eşiği geçmelidir:

    RequiredLift = p_break_even − p_baseline

Eşikler ELLE UYDURULMAZ (şartname 101). "p > %70" gibi bir sabit yoktur;
başabaş noktası her hücrede kazanç/kayıp asimetrisinden ve maliyetten çözülür.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Optional

from .stats import Z95, wilson_ci


@dataclass
class Payoff:
    """Bir hücrenin NET getiri profili — hepsi yüzde, işaretli.

    net_win  : hedef vurulursa (+1,0 tanım gereği — hedef NET %1'e göre kuruldu)
    net_loss : stop vurulursa kaybedilen NET yüzde (POZİTİF sayı olarak)
    net_timeout : zaman aşımında gerçekleşen ORTALAMA net getiri.
                  ⚠️ Şartname 28: bu sıfır VARSAYILMAZ, ölçülür. Genelde
                  negatiftir çünkü maliyet ödenmiş, hareket olmamıştır.
    """
    net_win: float
    net_loss: float
    net_timeout: float


def expected_value(p_tp: float, p_sl: float, p_to: float, pay: Payoff) -> float:
    """Beklenen NET getiri (%). Olasılıklar 1'e toplanmıyorsa normalize edilir."""
    s = p_tp + p_sl + p_to
    if s <= 0:
        return float("-inf")
    return float((p_tp * pay.net_win - p_sl * abs(pay.net_loss)
                  + p_to * pay.net_timeout) / s)


def breakeven_probability(pay: Payoff, p_timeout: float = 0.0) -> Optional[float]:
    """EV = 0 olacak minimum hedef-önce olasılığı.

    p_tp·W − p_sl·L + p_to·T = 0 ,  p_sl = 1 − p_tp − p_to
      →  p_tp = ( L·(1 − p_to) − p_to·T ) / (W + L)

    Zaman aşımı yoksa şartnamedeki sadeleşmiş hâle iner: L / (W + L).
    """
    W, L, T = pay.net_win, abs(pay.net_loss), pay.net_timeout
    if W + L <= 0:
        return None
    p = (L * (1.0 - p_timeout) - p_timeout * T) / (W + L)
    if not math.isfinite(p):
        return None
    return float(min(1.0, max(0.0, p)))


@dataclass
class LiftResult:
    baseline: Optional[float]
    breakeven: Optional[float]
    required_lift: Optional[float]
    model_rate: Optional[float]
    actual_lift: Optional[float]
    actual_lift_lower95: Optional[float]
    edge: bool
    reason: str

    def to_dict(self) -> Dict:
        return asdict(self)


def evaluate_lift(baseline: Optional[float],
                  pay: Payoff,
                  p_timeout: float,
                  model_rate: Optional[float] = None,
                  n_model_eff: float = 0.0,
                  n_base_eff: float = 0.0) -> LiftResult:
    """Kenar var mı? Üç kapı birden geçilmeli.

    1. `actual_lift > required_lift`  (modelin kaldırdığı, gerekenden fazla)
    2. lift'in %95 ALT sınırı hâlâ gerekenin üstünde (şartname 9, 102)
    3. model oranı ölçülmüş olmalı — yoksa kenar İDDİA EDİLEMEZ

    Model yoksa `edge=False` ve sebep açıkça yazılır. Bu bir başarısızlık
    değil, dürüst durumdur: ölçülmemiş bir kenar yok sayılır.
    """
    be = breakeven_probability(pay, p_timeout)
    if baseline is None or be is None:
        return LiftResult(baseline, be, None, model_rate, None, None, False,
                          "TABAN_VEYA_BASABAS_HESAPLANAMADI")
    req = float(be - baseline)

    if model_rate is None:
        return LiftResult(baseline, be, req, None, None, None, False,
                          "MODEL_YOK — gerçek lift ölçülmedi, kenar iddia edilemez")

    act = float(model_rate - baseline)
    # İki oranın FARKI için standart hata; ikisi de etkin örnekleme göre
    ne_m = max(1.0, n_model_eff)
    ne_b = max(1.0, n_base_eff)
    se = math.sqrt(max(1e-12, model_rate * (1 - model_rate) / ne_m
                       + baseline * (1 - baseline) / ne_b))
    alt = act - Z95 * se

    if act <= req:
        return LiftResult(baseline, be, req, model_rate, act, alt, False,
                          f"LIFT_YETERSIZ ({act:+.4f} ≤ gereken {req:+.4f})")
    if alt <= req:
        return LiftResult(baseline, be, req, model_rate, act, alt, False,
                          f"LIFT_ALT_SINIRI_YETERSIZ ({alt:+.4f} ≤ {req:+.4f}) — "
                          f"nokta tahmin geçiyor ama örneklem bunu taşımıyor")
    return LiftResult(baseline, be, req, model_rate, act, alt, True,
                      "LIFT_GECERLI")


def net1_precision(basarili: int, yayimlanan: int) -> Optional[float]:
    """Şartname 104 — asıl KPI.

    Net1PercentPrecision = net %1 hedefi tutan sinyaller / YAYIMLANAN sinyaller.
    Payda yayımlanan sinyallerdir; başarısızları paydadan düşürmek yasaktır."""
    if yayimlanan <= 0:
        return None
    return float(basarili / yayimlanan)


def false_opportunity_rate(basarisiz: int, yayimlanan: int) -> Optional[float]:
    """Şartname 103 — yayımlanıp hedefi tutturamayanların oranı."""
    if yayimlanan <= 0:
        return None
    return float(basarisiz / yayimlanan)
