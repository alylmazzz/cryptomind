"""FAIL-FAST ŞEMA VE BİRİM DOĞRULAMASI (§IV, V, LXXXIII, P0-15).

NEDEN AYRI BİR DOSYA

Birim hataları bu projede iki kez sessizce yanlış sonuç üretti:

  1. `bid_depth` BAZ VARLIK cinsindeyken dolar sanıldı → BTC'de 10.000 $'lık
     emir "derinliği 34× aşıyor" göründü.
  2. DSR'a işlem-başı Sharpe verildi, oysa fonksiyon YILLIK bekliyordu →
     eşik bir mertebe yanlış kuruldu.

İkisi de "kod çalıştı, sayı çıktı, sayı yanlıştı" biçiminde. Testler bunları
sonradan yakaladı; bu modül **çalışma anında** yakalar ve HATA FIRLATIR —
sessizce devam etmez.

TASARIM KURALI: bir alan eksikse ya da birimi belirsizse `None` döndürüp
devam etmek YASAK. Eksik veri, yanlış veriden iyidir; yanlış veri sessizce
karar üretir.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Sequence

# Alan → (birim, geçerli aralık). `None` sınır = kontrol yok.
FIELD_UNITS: Dict[str, Dict] = {
    # olasılıklar
    "p_target_first": {"unit": "oran", "min": 0.0, "max": 1.0},
    "p_stop_first": {"unit": "oran", "min": 0.0, "max": 1.0},
    "p_timeout": {"unit": "oran", "min": 0.0, "max": 1.0},
    "p_target_lower95": {"unit": "oran", "min": 0.0, "max": 1.0},
    "p_target_upper95": {"unit": "oran", "min": 0.0, "max": 1.0},
    "baseline": {"unit": "oran", "min": 0.0, "max": 1.0},
    # yüzdeler
    "target_gross_pct": {"unit": "yüzde", "min": 0.0, "max": 100.0},
    "stop_pct": {"unit": "yüzde", "min": 0.0, "max": 100.0},
    "cost_pct": {"unit": "yüzde", "min": 0.0, "max": 100.0},
    "net_mean": {"unit": "yüzde", "min": -100.0, "max": 100.0},
    "robust_ev": {"unit": "yüzde", "min": -100.0, "max": 100.0},
    # dolar
    "max_capacity_usd": {"unit": "USD", "min": 0.0, "max": None},
    "bid_depth_usd": {"unit": "USD", "min": 0.0, "max": None},
    "ask_depth_usd": {"unit": "USD", "min": 0.0, "max": None},
    # baz puan
    "spread_bps": {"unit": "bps", "min": 0.0, "max": 100_000.0},
    # sayımlar
    "n_raw": {"unit": "adet", "min": 0.0, "max": None},
    "n_eff": {"unit": "adet", "min": 0.0, "max": None},
    "n_eff_used": {"unit": "adet", "min": 0.0, "max": None},
    # süreler
    "horizon_minutes": {"unit": "dakika", "min": 1.0, "max": None},
    "expected_half_life_sec": {"unit": "saniye", "min": 0.0, "max": None},
}

# Bir parite kartında BULUNMASI ZORUNLU alanlar (§88 şeması)
CARD_REQUIRED = (
    "symbol", "timestamp", "guaranteed", "best_horizon", "direction", "status",
    "market_price", "p_target_first", "p_target_first_lower95", "p_stop_first",
    "p_timeout", "baseline_target_rate", "required_probability_lift",
    "actual_probability_lift", "robust_expected_value", "cost_model",
    "rejection_reasons", "horizons",
)
HORIZON_REQUIRED = ("horizon", "horizon_minutes", "direction", "status")


class SchemaError(ValueError):
    """Şema/birim ihlali. YAKALANMAZ — çalışmayı durdurur."""


@dataclass
class Violation:
    where: str
    field: str
    problem: str
    value: Any = None

    def to_dict(self) -> Dict:
        return asdict(self)

    def __str__(self) -> str:
        return f"{self.where}.{self.field}: {self.problem} (={self.value!r})"


def _sayi(v: Any) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def check_units(d: Dict, where: str = "") -> List[Violation]:
    """Bilinen alanların birim ve aralık kontrolü."""
    out: List[Violation] = []
    for ad, kural in FIELD_UNITS.items():
        if ad not in d:
            continue
        ham = d[ad]
        if ham is None:
            continue
        f = _sayi(ham)
        if f is None:
            out.append(Violation(where, ad, f"sayı değil ({kural['unit']} bekleniyor)", ham))
            continue
        if kural["min"] is not None and f < kural["min"]:
            out.append(Violation(where, ad,
                                 f"{kural['unit']} alt sınırının altında "
                                 f"(min {kural['min']})", f))
        if kural["max"] is not None and f > kural["max"]:
            out.append(Violation(where, ad,
                                 f"{kural['unit']} üst sınırının üstünde "
                                 f"(max {kural['max']})", f))
    return out


def check_probabilities(d: Dict, where: str = "", tol: float = 1e-3
                        ) -> List[Violation]:
    """Üç sonuç olasılığı 1'e toplanmalı ve alt sınır nokta tahmini aşmamalı."""
    out: List[Violation] = []
    p = [_sayi(d.get(k)) for k in ("p_target_first", "p_stop_first", "p_timeout")]
    if all(x is not None for x in p):
        t = sum(p)
        if abs(t - 1.0) > tol:
            out.append(Violation(where, "p_*", f"olasılık toplamı 1 değil", round(t, 6)))
    alt = _sayi(d.get("p_target_lower95") or d.get("lower95"))
    nokta = _sayi(d.get("p_target_first"))
    ust = _sayi(d.get("p_target_upper95"))
    if alt is not None and nokta is not None and alt > nokta + 1e-9:
        out.append(Violation(where, "lower95", "alt sınır nokta tahmini AŞIYOR", alt))
    if ust is not None and nokta is not None and ust < nokta - 1e-9:
        out.append(Violation(where, "upper95", "üst sınır nokta tahminin ALTINDA", ust))
    return out


def check_card(card: Dict, strict: bool = True) -> List[Violation]:
    """Bir parite kartının tam doğrulaması.

    `guaranteed` alanı ŞEMA DEĞİŞMEZİDİR: yoksa ya da True ise bu bir şema
    ihlalidir, kozmetik bir eksik değil."""
    where = card.get("symbol") or "?"
    out: List[Violation] = []
    for ad in CARD_REQUIRED:
        if ad not in card:
            out.append(Violation(where, ad, "ZORUNLU alan eksik"))
    if card.get("guaranteed") is not False:
        out.append(Violation(where, "guaranteed",
                             "ŞEMA DEĞİŞMEZİ: her zaman False olmalı",
                             card.get("guaranteed")))
    out += check_units(card, where)
    out += check_probabilities(card, where)
    for h in card.get("horizons") or []:
        hw = f"{where}/{h.get('horizon')}/{h.get('direction')}"
        for ad in HORIZON_REQUIRED:
            if ad not in h:
                out.append(Violation(hw, ad, "ZORUNLU alan eksik"))
        out += check_units(h, hw)
        out += check_probabilities(h, hw)
        if h.get("tradable") and h.get("rejection_reasons"):
            out.append(Violation(hw, "tradable",
                                 "işlem yapılabilir ama red sebebi var"))
        if (h.get("status") in ("QUALIFIED", "HIGH_CONFIDENCE")
                and (_sayi(h.get("robust_ev")) or 0) <= 0):
            out.append(Violation(hw, "status",
                                 "QUALIFIED ama Robust EV ≤ 0", h.get("robust_ev")))
    if strict and out:
        raise SchemaError(f"{len(out)} şema ihlali: " +
                          " · ".join(str(v) for v in out[:5]))
    return out


def check_scan(scan: Dict, strict: bool = True) -> List[Violation]:
    """Tam tarama çıktısı — her kart ayrı doğrulanır."""
    out: List[Violation] = []
    for k in scan.get("cards") or []:
        out += check_card(k, strict=False)
    if strict and out:
        raise SchemaError(f"{len(out)} şema ihlali: " +
                          " · ".join(str(v) for v in out[:5]))
    return out


# ── VERİ KALİTESİ SKORU (§V) ───────────────────────────────────────────────

@dataclass
class DataQuality:
    score: float
    components: Dict[str, float]
    reasons: List[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


def data_quality(freshness_sec: Optional[float], max_fresh_sec: float,
                 completeness: Optional[float],
                 book_coverage: Optional[float],
                 schema_violations: int,
                 cost_model: Optional[str]) -> DataQuality:
    """Tek sayıya indirgenmiş veri kalitesi — ama bileşenler AYRI görünür.

    Tek bir skor göstermek, hangi boyutun bozuk olduğunu gizler; bu yüzden
    bileşenler her zaman birlikte döner. Ölçülemeyen bileşen **0** sayılır,
    "sorun yok" varsayılmaz."""
    def _norm(x: Optional[float], iyi: float, kotu: float) -> float:
        if x is None or not math.isfinite(x):
            return 0.0
        if iyi == kotu:
            return 1.0
        return max(0.0, min(1.0, (x - kotu) / (iyi - kotu)))

    b = {
        "freshness": _norm(None if freshness_sec is None
                           else max_fresh_sec - freshness_sec,
                           max_fresh_sec, 0.0),
        "completeness": _norm(completeness, 1.0, 0.5),
        "book_coverage": _norm(book_coverage, 1.0, 0.0),
        "schema": 1.0 if schema_violations == 0 else 0.0,
        "cost_model": (1.0 if cost_model == "MEASURED_L2_VWAP"
                       else 0.5 if cost_model == "ESTIMATED" else 0.0),
    }
    neden: List[str] = []
    if b["freshness"] < 0.5:
        neden.append(f"veri yaşı {freshness_sec}s (eşik {max_fresh_sec}s)")
    if b["schema"] == 0.0:
        neden.append(f"{schema_violations} şema ihlali")
    if b["cost_model"] < 1.0:
        neden.append(f"maliyet modeli {cost_model} — gerçek L2 değil")
    if b["book_coverage"] < 0.5:
        neden.append("defter kapsamı yetersiz")
    skor = 100.0 * sum(b.values()) / len(b)
    return DataQuality(round(skor, 1), {k: round(v, 3) for k, v in b.items()},
                       neden,
                       "Ölçülemeyen bileşen 0 sayılır; 'sorun yok' VARSAYILMAZ. "
                       "Tek skor bileşenlerden ayrı okunamaz.")
