"""YAKINSAMA MOTORU — "bu sayı kesinleşiyor mu?" sorusunun ölçülebilir cevabı.

NEDEN BU DOSYA VAR

Bir olasılık tahmini iki farklı sebeple güvenilmez olabilir:

  1. ÖRNEKLEM AZ  → sayı doğru yere gidiyor ama henüz varmadı (güven aralığı
     geniş). Bu, veri biriktikçe kendiliğinden düzelir.
  2. SAYI KAYIYOR → tahmin dönemden döneme, rejimden rejime savruluyor. Bu,
     veri biriktikçe DÜZELMEZ; çünkü ölçülen büyüklük sabit değildir.

İkisi çok farklı şeydir ve tek bir güven aralığı bunları ayırt etmez. Dar bir
aralık, ölçülen şeyin kararlı olduğunu KANITLAMAZ — yalnız o pencerede çok
gözlem olduğunu söyler.

BU MOTOR DÖRT ŞEYİ AYRI ÖLÇER

  A. Örneklem yeterliliği   — güven aralığı genişliği ve etkin örneklem
  B. Zamansal kararlılık    — dönem dönem tahminin sapması (yıl/çeyrek)
  C. Rejim kararlılığı      — düşük/normal/yüksek oynaklıkta aynı mı?
  D. Daralma hızı           — CI genişliği gerçekten ~1/√n ile mi daralıyor?

D maddesi kritiktir: örneklem dört katına çıktığında aralık yarıya inmiyorsa
gözlemler bağımsız değildir ve "daha çok veri toplayınca kesinleşir"
beklentisi YANLIŞTIR. Bu, örtüşen üçlü-bariyer etiketlerinde tam da olan şeydir.

VERDİKTLER
  CONVERGED        dört ölçüt de geçti; sayı bu veriyle kararlı sayılabilir
  CONVERGING       kayma yok ama örneklem yetersiz; veri biriktikçe düzelir
  REGIME_DEPENDENT ZAMANLA kararlı fakat REJİME koşullu; tek sayı temsil etmez
  UNSTABLE         ZAMANLA kayıyor; daha çok veri BUNU DÜZELTMEZ
  UNMEASURED       yakınsama ölçülemedi (yeterli alt bölme yok)

REJİM BAĞLILIĞI NEDEN AYRI BİR VERDİKT — ÖLÇÜLEREK BULUNDU
BTCUSDT 4h LONG kör tabanı: LOW_VOL %8,1 · NORMAL_VOL %14,7 · HIGH_VOL %27,4
· PANIC %48,4. Yayılım **40,3 puan**, yani altı kat. Bunun sebebi gizemli
değil, geometrik: hedef sabit yüzdedir (net %1), stop ise oynaklıkla ölçeklenir
(k·σ(H)). Oynaklık yükseldikçe hedef sigma cinsinden YAKLAŞIR.

Bu bir "ölçüm kararsızlığı" DEĞİLDİR — ölçüm doğru, büyüklük rejime bağlıdır.
İki durumu aynı etiketle işaretlemek, düzeltilebilir bir eksikle
düzeltilemez bir kaymayı karıştırır:

  • ZAMANLA kayma      → model bozuluyor; veri biriktirmek çözmez
  • REJİME koşulluluk  → beklenen davranış; hücreler ZATEN rejim bazında
                          ölçülüyor ve model rejimi özellik olarak kullanıyor

Bu yüzden rejim bağlılığı ayrı bir verdikttir ve kullanıcıya "bu satırı rejim
kırılımıyla oku" der.

⚠️ HİÇBİR VERDİKT "KESİN" DEMEZ. `CONVERGED` bile "bu ölçüm penceresinde
kararlı" demektir; gelecekteki bir rejim değişimini kapsamaz.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .stats import Z95, wilson_ci

CONVERGED = "CONVERGED"
CONVERGING = "CONVERGING"
REGIME_DEPENDENT = "REGIME_DEPENDENT"
UNSTABLE = "UNSTABLE"
UNMEASURED = "UNMEASURED"

# Eşikler — muhafazakâr; config'den ezilebilir.
MAX_CI_WIDTH = 0.10           # A: ±5 puan
MIN_EFF_SAMPLE = 400.0        # A
MAX_PERIOD_SPREAD = 0.15      # B: dönemler arası en büyük fark (15 puan)
MAX_REGIME_SPREAD = 0.20      # C: rejimler arası en büyük fark (20 puan)
MIN_SHRINK_RATIO = 0.55       # D: n×4 olunca aralık en az bu oranda daralmalı
                              #    (bağımsız gözlemde 0,50 beklenir)
MIN_GROUP_N = 60


@dataclass
class ConvergenceResult:
    verdict: str
    ci_width: Optional[float] = None
    n_effective: Optional[float] = None
    period_spread: Optional[float] = None
    period_estimates: List[Dict] = field(default_factory=list)
    regime_spread: Optional[float] = None
    regime_estimates: List[Dict] = field(default_factory=list)
    shrink_ratio: Optional[float] = None
    shrink_curve: List[Dict] = field(default_factory=list)
    checks: Dict[str, Optional[bool]] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


def _oran_ci(k: int, n: int, ess: Optional[float] = None
             ) -> Tuple[float, float, float]:
    """(oran, alt, üst) — nokta tahmini ham veriden, GENİŞLİK etkin örneklemden."""
    if n <= 0:
        return (float("nan"), 0.0, 1.0)
    p = k / n
    ne = int(round(max(1.0, ess if ess is not None else n)))
    lo, hi = wilson_ci(int(round(p * ne)), ne)
    return (p, lo, hi)


def shrink_curve(labels: np.ndarray, target: int, horizon_bars: int,
                 fractions: Sequence[float] = (0.125, 0.25, 0.5, 1.0)
                 ) -> Tuple[List[Dict], Optional[float]]:
    """D — aralık genişliği örneklemle nasıl daralıyor?

    Alt örneklemler zamanın BAŞINDAN alınır (rastgele değil): rastgele seçim
    örtüşen etiketleri karıştırıp sahte bir bağımsızlık yanılsaması yaratır.

    Dönen `shrink_ratio`: tam örneklemin genişliği ÷ dörtte birinin genişliği.
    Bağımsız gözlemde ~0,50'dir (4× veri → 2× dar aralık). 0,55'ten büyükse
    gözlemler bağımlıdır ve "veri biriktikçe kesinleşir" beklentisi zayıftır.
    """
    from .stats import effective_sample_size
    n = len(labels)
    if n < 200:
        return [], None
    egri: List[Dict] = []
    for f in fractions:
        m = int(n * f)
        if m < 50:
            continue
        alt = labels[:m]
        ess = effective_sample_size(alt.astype(float), horizon_bars)["used"]
        k = int((alt == target).sum())
        p, lo, hi = _oran_ci(k, m, ess)
        egri.append({"fraction": f, "n": m, "n_eff": round(ess, 1),
                     "p": round(float(p), 5), "ci_width": round(hi - lo, 5)})
    if len(egri) < 2:
        return egri, None
    ceyrek = next((r for r in egri if abs(r["fraction"] - 0.25) < 1e-9), egri[0])
    tam = egri[-1]
    if ceyrek["ci_width"] <= 0:
        return egri, None
    return egri, float(tam["ci_width"] / ceyrek["ci_width"])


def group_spread(gruplar: Dict[str, Tuple[int, int]],
                 min_n: int = MIN_GROUP_N) -> Tuple[Optional[float], List[Dict]]:
    """Alt gruplar arası tahmin farkı (en büyük − en küçük).

    `gruplar`: {ad: (başarı, toplam)}. Yeterli örneği olmayan grup ölçüme
    GİRMEZ ama listede `enough=False` ile görünür — sessizce düşmez."""
    out: List[Dict] = []
    oranlar: List[float] = []
    for ad, (k, n) in sorted(gruplar.items()):
        yeter = n >= min_n
        p = (k / n) if n else None
        out.append({"group": ad, "n": int(n),
                    "p": (round(float(p), 5) if p is not None else None),
                    "enough": bool(yeter)})
        if yeter and p is not None:
            oranlar.append(float(p))
    if len(oranlar) < 2:
        return None, out
    return float(max(oranlar) - min(oranlar)), out


def decide_verdict(checks: Dict[str, Optional[bool]]) -> str:
    """Kontrollerden verdikt — TEK KARAR NOKTASI.

    ⚠️ Bu fonksiyon var çünkü aynı karar bir zamanlar İKİ yerde veriliyordu
    (`assess` ve `research._convergence_for`). `REGIME_DEPENDENT` eklendiğinde
    biri güncellendi, diğeri unutuldu ve 472 hücre yanlış etiketlendi. Karar
    mantığı bundan sonra yalnız burada değişir.

    Sıra ÖNEMLİ: zamansal kayma en ağır teşhistir ve rejim bağlılığını ezer."""
    if checks.get("temporally_stable") is False:
        return UNSTABLE
    if checks.get("regime_stable") is False:
        return REGIME_DEPENDENT
    if (checks.get("temporally_stable") is None
            and checks.get("regime_stable") is None):
        return UNMEASURED
    if checks.get("sample_sufficient") and checks.get("shrinks_with_n") is not False:
        return CONVERGED
    return CONVERGING


def build_checks(ci_width: float, n_effective: float,
                 period_spread: Optional[float],
                 regime_spread: Optional[float],
                 shrink_ratio: Optional[float]) -> Dict[str, Optional[bool]]:
    """Ölçümlerden kontrol sözlüğü — eşikler TEK yerde uygulanır."""
    return {
        "sample_sufficient": bool(ci_width <= MAX_CI_WIDTH
                                  and n_effective >= MIN_EFF_SAMPLE),
        "temporally_stable": (None if period_spread is None
                              else bool(period_spread <= MAX_PERIOD_SPREAD)),
        "regime_stable": (None if regime_spread is None
                          else bool(regime_spread <= MAX_REGIME_SPREAD)),
        "shrinks_with_n": (None if shrink_ratio is None
                           else bool(shrink_ratio <= MIN_SHRINK_RATIO)),
    }


def explain(checks: Dict[str, Optional[bool]], ci_width: float,
            n_effective: float, period_spread: Optional[float],
            regime_spread: Optional[float],
            shrink_ratio: Optional[float]) -> List[str]:
    """Verdiktin GEREKÇESİ — bu da tek yerde üretilir."""
    neden: List[str] = []
    if not checks.get("sample_sufficient"):
        neden.append(f"güven aralığı {ci_width * 100:.1f} puan "
                     f"(eşik {MAX_CI_WIDTH * 100:.0f}) · etkin örneklem "
                     f"{n_effective:.0f} (eşik {MIN_EFF_SAMPLE:.0f})")
    if checks.get("temporally_stable") is False:
        neden.append(f"dönemler arası fark {period_spread * 100:.1f} puan — "
                     f"tahmin zamanla KAYIYOR, daha çok veri bunu düzeltmez")
    if checks.get("regime_stable") is False:
        neden.append(f"rejimler arası fark {regime_spread * 100:.1f} puan — "
                     f"ölçüm YANLIŞ değil, büyüklük rejime KOŞULLU; satırı "
                     f"rejim kırılımıyla okuyun")
    if checks.get("shrinks_with_n") is False:
        neden.append(f"örneklem 4× olunca aralık yalnız {shrink_ratio:.2f} "
                     f"oranında daraldı (bağımsızda 0,50) — gözlemler bağımlı")
    return neden


def assess(labels: np.ndarray, target: int, horizon_bars: int,
           n_effective: float,
           period_groups: Optional[Dict[str, Tuple[int, int]]] = None,
           regime_groups: Optional[Dict[str, Tuple[int, int]]] = None,
           ci_width: Optional[float] = None) -> ConvergenceResult:
    """Dört ölçütü uygula ve verdikt ver."""
    labels = np.asarray(labels)
    n = int(len(labels))
    k = int((labels == target).sum()) if n else 0
    if ci_width is None:
        _, lo, hi = _oran_ci(k, n, n_effective)
        ci_width = hi - lo

    egri, oran = shrink_curve(labels, target, horizon_bars)
    p_spread, p_list = group_spread(period_groups or {})
    r_spread, r_list = group_spread(regime_groups or {})

    kontrol = build_checks(ci_width, n_effective, p_spread, r_spread, oran)
    neden = explain(kontrol, ci_width, n_effective, p_spread, r_spread, oran)
    verdikt = decide_verdict(kontrol)

    return ConvergenceResult(
        verdict=verdikt, ci_width=round(float(ci_width), 5),
        n_effective=round(float(n_effective), 1),
        period_spread=(None if p_spread is None else round(p_spread, 5)),
        period_estimates=p_list,
        regime_spread=(None if r_spread is None else round(r_spread, 5)),
        regime_estimates=r_list,
        shrink_ratio=(None if oran is None else round(oran, 4)),
        shrink_curve=egri, checks=kontrol, reasons=neden,
        note=("CONVERGED 'kesin' demek DEĞİLDİR; bu ölçüm penceresinde kararlı "
              "demektir. Gelecekteki bir rejim değişimini kapsamaz."))


VERDICT_TR = {
    CONVERGED: "kararlı",
    CONVERGING: "yakınsıyor (örneklem yetersiz)",
    REGIME_DEPENDENT: "rejime koşullu",
    UNSTABLE: "kararsız — zamanla kayıyor",
    UNMEASURED: "yakınsama ölçülmedi",
}
VERDICT_COLOR = {
    CONVERGED: "yesil",
    CONVERGING: "sari",
    REGIME_DEPENDENT: "mavi",
    UNSTABLE: "kirmizi",
    UNMEASURED: "gri",
}
