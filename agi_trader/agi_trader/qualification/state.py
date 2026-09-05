"""Nitelendirme durum makinesi — şartname 47, 56, 92, 100.

HER (parite × ufuk × yön) HÜCRESİ TAM BİR DURUMDA BULUNUR

    NO_DATA          yeterli veri yok
    RESEARCH_ONLY    veri var, doğrulama tamamlanmadı (validation_report yok)
    NO_EDGE          ölçüldü; maliyet sonrası EV pozitif değil
    UNVERIFIED       model var fakat OOS kenar kanıtlanmadı
    DEGRADED         kenar vardı; sürüklenme/kalibrasyon bozuldu
    QUALIFIED        OOS + kalibrasyon + EV + risk kapıları geçti
    HIGH_CONFIDENCE  QUALIFIED + canlı/gölge performansı da sınırlar içinde

`GUARANTEED` DİYE BİR DURUM YOKTUR VE EKLENEMEZ.
Bu, yorum değil kod kuralıdır: `QualificationState` bir sabit kümedir ve
`test_guaranteed_durumu_yoktur` bu kümede "GUARANTEE" geçen bir ad
bulunmadığını doğrular. Finansal piyasada gelecekteki bir fiyat hareketinin
garantisi yoktur; ürünün temel güvenilirlik şartı budur (şartname 116).

VARSAYILAN REDDİR. Bir hücre yalnız BÜTÜN kapıları geçerse yükselir.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional


class QualificationState:
    NO_DATA = "NO_DATA"
    RESEARCH_ONLY = "RESEARCH_ONLY"
    NO_EDGE = "NO_EDGE"
    UNVERIFIED = "UNVERIFIED"
    DEGRADED = "DEGRADED"
    QUALIFIED = "QUALIFIED"
    HIGH_CONFIDENCE = "HIGH_CONFIDENCE"

    ALL = (NO_DATA, RESEARCH_ONLY, NO_EDGE, UNVERIFIED, DEGRADED,
           QUALIFIED, HIGH_CONFIDENCE)
    TRADABLE = (QUALIFIED, HIGH_CONFIDENCE)

    # Şartname 92 — renk sistemi. "guaranteed" için yeşil rozet YOKTUR.
    COLOR = {
        NO_DATA: "gri", RESEARCH_ONLY: "gri", UNVERIFIED: "sari",
        NO_EDGE: "kirmizi", DEGRADED: "kirmizi",
        QUALIFIED: "yesil", HIGH_CONFIDENCE: "koyu_yesil",
    }


class RejectionCode:
    """Şartname 56 — 'neden işlem yok' için net kodlar. Serbest metin değil."""
    TARGET_PROBABILITY_LOW = "TARGET_PROBABILITY_LOW"
    CI_TOO_WIDE = "CI_TOO_WIDE"
    NEGATIVE_EV = "NEGATIVE_EV"
    NO_OOS_EDGE = "NO_OOS_EDGE"
    LOW_LIQUIDITY = "LOW_LIQUIDITY"
    INSUFFICIENT_L2 = "INSUFFICIENT_L2"
    DATA_STALE = "DATA_STALE"
    CALIBRATION_FAILED = "CALIBRATION_FAILED"
    MODEL_DRIFT = "MODEL_DRIFT"
    REGIME_UNSUPPORTED = "REGIME_UNSUPPORTED"
    FILL_PROBABILITY_LOW = "FILL_PROBABILITY_LOW"
    COST_TOO_HIGH = "COST_TOO_HIGH"
    CORRELATION_LIMIT = "CORRELATION_LIMIT"
    SAMPLE_TOO_SMALL = "SAMPLE_TOO_SMALL"
    NO_VALIDATION_REPORT = "NO_VALIDATION_REPORT"
    MISSING_MODEL = "MISSING_MODEL"
    SIGNAL_EXPIRED = "SIGNAL_EXPIRED"
    NO_QUALIFIED_HORIZON = "NO_QUALIFIED_HORIZON"
    MULTIPLE_TESTING = "MULTIPLE_TESTING"
    OVERFIT_RISK = "OVERFIT_RISK"
    REGIME_CONCENTRATED = "REGIME_CONCENTRATED"

    ALL = tuple(v for k, v in list(locals().items()) if k.isupper())

    TR = {
        TARGET_PROBABILITY_LOW: "hedef olasılığı yetersiz",
        CI_TOO_WIDE: "güven aralığı çok geniş",
        NEGATIVE_EV: "beklenen değer pozitif değil",
        NO_OOS_EDGE: "örneklem dışı kenar kanıtlanmadı",
        LOW_LIQUIDITY: "likidite yetersiz",
        INSUFFICIENT_L2: "L2 defter kapsamı yetersiz",
        DATA_STALE: "veri bayat",
        CALIBRATION_FAILED: "kalibrasyon bozuk",
        MODEL_DRIFT: "model sürüklenmesi",
        REGIME_UNSUPPORTED: "bu rejim için kalibrasyon yok",
        FILL_PROBABILITY_LOW: "dolum olasılığı düşük",
        COST_TOO_HIGH: "maliyet hedefi yiyor",
        CORRELATION_LIMIT: "korelasyon limiti",
        SAMPLE_TOO_SMALL: "örneklem çok küçük",
        NO_VALIDATION_REPORT: "doğrulama raporu yok",
        MISSING_MODEL: "model yok",
        SIGNAL_EXPIRED: "sinyalin süresi doldu",
        NO_QUALIFIED_HORIZON: "doğrulanmış ufuk yok",
        MULTIPLE_TESTING: "çoklu test düzeltmesini geçmiyor (DSR)",
        OVERFIT_RISK: "aşırı uyum olasılığı yüksek (PBO)",
        REGIME_CONCENTRATED: "kazanç tek döneme sıkışmış",
    }


@dataclass
class EvidenceGates:
    """Şartname 17 — minimum kanıt kapıları. Config'den yönetilir, başlangıçta
    MUHAFAZAKÂR. Bu sayılar 'iyi görünsün' diye değil, istatistiksel güç
    kaygısıyla seçildi: 200 etkin gözlemin altında %5'lik bir lift'i tespit
    etmek mümkün değildir."""
    min_effective_samples: float = 200.0
    min_tp_events: int = 30
    min_regime_samples: float = 100.0
    max_ci_width: float = 0.30
    max_calibration_error: float = 0.05      # ECE
    min_calibration_slope: float = 0.70
    max_calibration_slope: float = 1.40
    max_psi: float = 0.25
    min_data_quality: float = 0.80
    min_liquidity_score: float = 0.50
    min_shadow_samples: int = 50
    # Çoklu test kontrolü (şartname 41) — tek başına yeterli değil ama ZORUNLU
    min_dsr: float = 0.95
    max_pbo: float = 0.30
    # Alt dönem tutarlılığı (şartname 42): kazancın tek dönemde toplanmaması
    min_positive_subperiods: float = 0.50


@dataclass
class CellEvidence:
    """Bir hücrenin durum kararı için gereken HER ŞEY. Eksik alan `None`
    kalır ve `None` asla 'geçti' sayılmaz."""
    n_effective: float = 0.0
    n_tp_events: int = 0
    baseline: Optional[float] = None
    p_target_first: Optional[float] = None
    p_lower95: Optional[float] = None
    p_upper95: Optional[float] = None
    robust_ev: Optional[float] = None
    edge_proven: bool = False
    lift_reason: str = ""
    has_validation_report: bool = False
    has_model: bool = False
    calibration_ece: Optional[float] = None
    calibration_slope: Optional[float] = None
    psi: Optional[float] = None
    had_edge_before: bool = False
    shadow_n: int = 0
    shadow_within_bounds: Optional[bool] = None
    data_quality: Optional[float] = None
    liquidity_score: Optional[float] = None
    cost_model_valid: bool = False
    cost_model_measured: bool = False      # gerçek L2 VWAP mı?
    data_stale: bool = False
    regime_supported: bool = True
    dsr: Optional[float] = None
    pbo: Optional[float] = None
    positive_subperiod_frac: Optional[float] = None
    regime_concentrated: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class StateDecision:
    state: str
    rejection_reasons: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def tradable(self) -> bool:
        return self.state in QualificationState.TRADABLE and not self.rejection_reasons

    def to_dict(self) -> Dict:
        return {"status": self.state,
                "color": QualificationState.COLOR[self.state],
                "rejection_reasons": list(self.rejection_reasons),
                "rejection_reasons_tr": [RejectionCode.TR.get(c, c)
                                         for c in self.rejection_reasons],
                "notes": list(self.notes),
                "tradable": self.tradable}


def decide_state(ev: CellEvidence, gates: EvidenceGates = EvidenceGates()
                 ) -> StateDecision:
    """Kanıttan duruma. Sıra ÖNEMLİDİR: en temel eksiklik önce."""
    r: List[str] = []
    notlar: List[str] = []

    # 1 — veri var mı?
    if ev.n_effective < gates.min_effective_samples or \
            ev.n_tp_events < gates.min_tp_events:
        r.append(RejectionCode.SAMPLE_TOO_SMALL)
        return StateDecision(
            QualificationState.NO_DATA if ev.n_effective <= 0
            else QualificationState.RESEARCH_ONLY, r,
            [f"etkin örneklem {ev.n_effective:.0f} "
             f"(gereken {gates.min_effective_samples:.0f}), "
             f"hedef olayı {ev.n_tp_events} (gereken {gates.min_tp_events})"])

    if ev.data_stale:
        r.append(RejectionCode.DATA_STALE)
    if ev.data_quality is not None and ev.data_quality < gates.min_data_quality:
        # Şartnamede ayrı bir "veri kalitesi düşük" kodu yok; en yakın anlam
        # DATA_STALE'dir (kaynak sağlığı düştüğünde veri güvenilmez olur).
        if RejectionCode.DATA_STALE not in r:
            r.append(RejectionCode.DATA_STALE)
    if ev.liquidity_score is not None and \
            ev.liquidity_score < gates.min_liquidity_score:
        r.append(RejectionCode.LOW_LIQUIDITY)
    if not ev.cost_model_valid:
        r.append(RejectionCode.INSUFFICIENT_L2)
    if not ev.regime_supported:
        r.append(RejectionCode.REGIME_UNSUPPORTED)

    # 2 — güven aralığı taşınabilir mi?
    if ev.p_lower95 is not None and ev.p_upper95 is not None:
        if (ev.p_upper95 - ev.p_lower95) > gates.max_ci_width:
            r.append(RejectionCode.CI_TOO_WIDE)

    # 3 — model ve doğrulama var mı?
    if not ev.has_model:
        r.append(RejectionCode.MISSING_MODEL)
    if not ev.has_validation_report:
        r.append(RejectionCode.NO_VALIDATION_REPORT)

    # 4 — EV pozitif mi? (Bu ölçüldüyse ve negatifse durum NO_EDGE'dir.)
    ev_negatif = (ev.robust_ev is None) or (ev.robust_ev <= 0)
    if ev_negatif:
        r.append(RejectionCode.NEGATIVE_EV)

    # 5 — kenar kanıtlandı mı?
    if not ev.edge_proven:
        r.append(RejectionCode.NO_OOS_EDGE)

    # 6 — kalibrasyon
    kal_bozuk = False
    if ev.calibration_ece is not None and ev.calibration_ece > gates.max_calibration_error:
        kal_bozuk = True
    if ev.calibration_slope is not None and not (
            gates.min_calibration_slope <= ev.calibration_slope
            <= gates.max_calibration_slope):
        kal_bozuk = True
    if kal_bozuk:
        r.append(RejectionCode.CALIBRATION_FAILED)

    # 7 — sürüklenme
    surukleniyor = ev.psi is not None and ev.psi > gates.max_psi
    if surukleniyor:
        r.append(RejectionCode.MODEL_DRIFT)

    # 8 — ÇOKLU TEST KONTROLÜ (şartname 41). Bir hücrenin pozitif çıkması,
    # yüzlerce hücre denendiğinde şaşırtıcı DEĞİLDİR. DSR bunu denenen strateji
    # sayısına göre düzeltir; PBO backtest aşırı-uyum olasılığını ölçer.
    # İkisi de tek başına yeterli değildir ama ikisi de ZORUNLUDUR.
    if ev.dsr is None:
        r.append(RejectionCode.NO_OOS_EDGE)
    elif ev.dsr < gates.min_dsr:
        r.append(RejectionCode.MULTIPLE_TESTING)
    if ev.pbo is not None and ev.pbo > gates.max_pbo:
        r.append(RejectionCode.OVERFIT_RISK)

    # 9 — alt dönem tutarlılığı (şartname 42)
    if ev.regime_concentrated or (
            ev.positive_subperiod_frac is not None
            and ev.positive_subperiod_frac < gates.min_positive_subperiods):
        r.append(RejectionCode.REGIME_CONCENTRATED)

    # ── durum ataması ────────────────────────────────────────────────────
    if ev.had_edge_before and (surukleniyor or kal_bozuk):
        return StateDecision(QualificationState.DEGRADED, r,
                             ["daha önce kenar ölçülmüştü; kalibrasyon/"
                              "sürüklenme kapısı düştü → otomatik işlem kapalı"])
    if not ev.has_model or not ev.has_validation_report:
        return StateDecision(QualificationState.RESEARCH_ONLY, r,
                             ["taban oranları ölçüldü; model/doğrulama raporu "
                              "olmadan olasılık YAYIMLANMAZ"] + notlar)
    if ev_negatif:
        return StateDecision(QualificationState.NO_EDGE, r,
                             ["ölçüldü — maliyet sonrası beklenen değer "
                              "pozitif değil"] + notlar)
    if not ev.edge_proven:
        return StateDecision(QualificationState.UNVERIFIED, r, notlar)
    if r:
        return StateDecision(QualificationState.UNVERIFIED, r, notlar)

    if ev.shadow_n >= gates.min_shadow_samples and ev.shadow_within_bounds:
        # Şartname 17/23: TAHMİNİ maliyet modeli HIGH_CONFIDENCE üretemez.
        # Maliyet ölçülmediyse net hedefin kendisi tahminîdir.
        if not ev.cost_model_measured:
            return StateDecision(
                QualificationState.QUALIFIED, [],
                notlar + ["maliyet modeli ÖLÇÜLMEDİ (gerçek L2 VWAP yok) — "
                          "HIGH_CONFIDENCE verilemez"])
        return StateDecision(QualificationState.HIGH_CONFIDENCE, [], notlar)
    if ev.shadow_within_bounds is False:
        return StateDecision(QualificationState.DEGRADED,
                             [RejectionCode.CALIBRATION_FAILED],
                             ["gölge doğrulaması kalibrasyon sınırları dışında"])
    return StateDecision(QualificationState.QUALIFIED, [],
                         notlar + [f"gölge örneklemi {ev.shadow_n} < "
                                   f"{gates.min_shadow_samples} — HIGH_CONFIDENCE "
                                   f"için canlı doğrulama bekleniyor"])
