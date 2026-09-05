"""
Harmonik formasyonlar — XABCD geometrisi, bacak oranları ve D bölgesi projeksiyonu.

`patterns.detect_harmonics` yalnız TAMAMLANMIŞ formasyonu bulur ve çizim için
yeterli bilgi vermez. Bu modül:

  1. TAMAMLANMIŞ XABCD  — beş nokta, dört bacak, her bacakta gerçekleşen/ideal
     Fibonacci oranı ve uyum skoru
  2. OLUŞMAKTA OLAN XABC — D noktası HENÜZ YOK; kuralın gerektirdiği D fiyatı
     iki bağımsız yoldan projekte edilir (XA geri çekilmesi ve BC uzantısı),
     kesişimleri **PRZ** (Potansiyel Dönüş Bölgesi) olur. Kullanıcının istediği
     "gelecekteki olası formasyon simülasyonu" budur.
  3. İŞLEM KARARI — giriş (D/PRZ), stop (D'nin ötesi), hedefler (AD bacağının
     %38,2 / %61,8 geri çekilmesi ve A seviyesi), R/R.

DÜRÜSTLÜK NOTU: Harmonik formasyonların örneklem dışı kâr ürettiğine dair
akademik kanıt yoktur; oranlar geleneksel Fibonacci kabulleridir. Bu modül
KARAR DESTEĞİ üretir — otomatik işlem açtırmaz. Projeksiyon bir tahmindir,
"fiyat buraya gelecek" demek değildir; "kural gerçekleşirse D burada olurdu"
demektir.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .patterns import find_pivots, HARMONIC_RULES

# Panelde buton olarak sunulacak formasyonlar (görünen ad + renk)
HARMONIC_META = {
    "Gartley":   {"key": "gartley",   "tr": "Gartley",           "color": "#00FF88"},
    "Butterfly": {"key": "butterfly", "tr": "Butterfly (Kelebek)", "color": "#FFB627"},
    "Bat":       {"key": "bat",       "tr": "Bat (Yarasa)",      "color": "#0099FF"},
    "Crab":      {"key": "crab",      "tr": "Crab (Yengeç)",     "color": "#FF3B5C"},
    "Cypher":    {"key": "cypher",    "tr": "Cypher",            "color": "#A855F7"},
    "Shark":     {"key": "shark",     "tr": "Shark (Köpekbalığı)", "color": "#22D3EE"},
}

# Kalite eşiği GERÇEK VERİ ile kalibre edildi. İlk denemede yalnız sentetik ders
# kitabı şekillerine bakıp 0,82 seçilmişti — o şekiller kurgu gereği kusursuz
# olduğu için eşik gerçek veride dedektörü neredeyse sıfırladı (540 pencerede
# 3 olay). Çift tepe kapılarında düşülen tuzağın aynısı.
#
# Doğru ölçüt: TAMAMLANMIŞ formasyonun gerçek veri oranı ile eşleştirilmiş
# oynaklıktaki rastgele yürüyüş oranı arasındaki FARK.
#   eşik   gerçek   rastgele   fark
#   0,55     %55       %33     +21
#   0,65     %43       %17     +27   ← SEÇİLEN (fark en yüksek)
#   0,75     %15        %7      +8
#   0,82      %5        %1      +4   (ikisi de sıfıra yaklaşıyor)
# Not: çift tepe/dip'te bu fark HİÇBİR ayarda pozitif çıkmamıştı; harmonikler
# bu sınavı geçiyor — şeklin gerçekten piyasaya özgü olduğunun göstergesi.
MIN_QUALITY_COMPLETE = 0.65
MIN_QUALITY_FORMING = 0.65


# ---------------------------------------------------------------------------
# ÖLÇÜLMÜŞ KANIT — panelde harmoniklerin yanında gösterilir.
# ---------------------------------------------------------------------------
# İki ayrı soru, iki ayrı cevap. Bu ayrım önemli çünkü harmonikler birinci
# sınavı GEÇİYOR, ikincisini geçemiyor:
#
# 1) ŞEKİL GERÇEK Mİ? — evet. Tamamlanmış harmonik, gerçek kripto verisinde
#    eşleştirilmiş oynaklıktaki rastgele yürüyüşten belirgin şekilde daha SIK
#    görülüyor (%43 vs %17, +27 puan). Çift tepe/dip bu sınavı HİÇBİR ayarda
#    geçememişti; harmonikler geçiyor.
#
# 2) YÖN BİLGİSİ VAR MI? — hayır. 20-bar ileri getiri olay çalışması
#    (kontrol = aynı paritenin rastgele anları, 540 gözlem):
#      LONG  (58 olay): %-0,92 · kontrol %-0,92 · t = -0,00  → fark TAM SIFIR
#      SHORT (84 olay): %+0,31 · kontrol %-0,92 · t = +1,14  → TERS yönde, anlamsız
#    Formasyon bazlı yön-düzeltilmiş getiriler de negatif:
#      bat %-0,51 · butterfly %-1,23 · shark %-2,05
#
# Sonuç: harmonik ÇİZİM olarak gerçek bir yapıdır ve gösterilmeye değer; ama
# "D'de al/sat" kararı ölçülmüş bir üstünlüğe dayanmaz. Panel bunu yazar.
HARMONIC_EVIDENCE = {
    "tested": True,
    "shape_real": True,          # 1. sınav geçildi
    "edge": "yok",               # 2. sınav geçilemedi
    "frequency": {"real_pct": 43, "random_pct": 17, "gap_pts": 27,
                  "note": ("Tamamlanmış harmonik gerçek veride rastgele "
                           "yürüyüşten %27 puan daha sık — şekil piyasaya özgü.")},
    "direction": {"fwd_bars": 20, "control_ret_pct": -0.92,
                  "long": {"n": 58, "ret_pct": -0.92, "t": -0.00},
                  "short": {"n": 84, "ret_pct": 0.31, "t": 1.14},
                  "per_pattern": {"bat": -0.51, "butterfly": -1.23, "shark": -2.05},
                  "note": ("Yön öngörüsü ÖLÇÜLDÜ ve bulunamadı: LONG'da fark tam "
                           "sıfır, SHORT'ta ters yönde ve anlamsız (t=1,14).")},
    "verdict": ("ŞEKİL GERÇEK — ama yön üstünlüğü ölçülemedi. Çizim ve seviyeler "
                "karar desteğidir; tek başına işlem gerekçesi değildir."),
}


# ===========================================================================
# Veri yapıları
# ===========================================================================
@dataclass
class HPoint:
    label: str          # "X" | "A" | "B" | "C" | "D"
    i: int
    price: float
    projected: bool = False


@dataclass
class HLeg:
    frm: str
    to: str
    x0: int
    y0: float
    x1: int
    y1: float
    ratio_name: str     # "AB/XA"
    ratio: float        # gerçekleşen
    ideal: str          # "0.786" veya "0.382–0.886"
    fit: float          # 0..1
    projected: bool = False


@dataclass
class HarmonicPattern:
    key: str
    name: str
    direction: str          # "LONG" | "SHORT"
    status: str             # "tamamlandı" | "oluşuyor"
    quality: float
    completion: float
    score: float
    color: str
    points: List[HPoint] = field(default_factory=list)
    legs: List[HLeg] = field(default_factory=list)
    prz: Optional[Dict] = None          # {lo, hi, i, mid, sources}
    entry: float = 0.0
    stop: float = 0.0
    targets: List[float] = field(default_factory=list)
    target_pcts: List[float] = field(default_factory=list)
    rr: float = 0.0
    entry_pct: float = 0.0              # girişin mevcut fiyata uzaklığı %
    clears_min_move: bool = False
    ratios: Dict[str, float] = field(default_factory=dict)
    action: str = ""                    # kullanıcıya net karar cümlesi
    note: str = ""
    # --- geçerlilik (detect_harmonics_rich içinde doldurulur) ---
    valid: bool = True                  # şu anki fiyattan HÂLÂ işlem açılabilir mi
    validity: str = "işlenebilir"       # işlenebilir | stop_ihlal | hedef_aşıldı
    validity_note: str = ""

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["points"] = [asdict(p) for p in self.points]
        d["legs"] = [asdict(l) for l in self.legs]
        return d


# ===========================================================================
# Oran yardımcıları
# ===========================================================================
def _rule_range(rule) -> Tuple[float, float]:
    if isinstance(rule, tuple):
        return float(rule[0]), float(rule[1])
    return float(rule), float(rule)


def _rule_text(rule) -> str:
    lo, hi = _rule_range(rule)
    return f"{lo:.3f}" if abs(hi - lo) < 1e-9 else f"{lo:.3f}–{hi:.3f}"


def _fit(val: float, rule) -> float:
    """Oranın kurala uyumu 0..1. Aralık içindeyse 1, dışında mesafeyle azalır."""
    lo, hi = _rule_range(rule)
    if lo <= val <= hi:
        return 1.0
    tol = max(0.08, (hi - lo) * 0.5)
    d = min(abs(val - lo), abs(val - hi))
    return float(max(0.0, 1.0 - d / tol))


def _ratio(a: float, b: float) -> float:
    return abs(a) / (abs(b) + 1e-12)


def _pct(a: float, b: float) -> float:
    return float((a - b) / (b + 1e-12) * 100.0)


# ===========================================================================
# İşlem planı (tamamlanmış veya projekte D için)
# ===========================================================================
def _trade_plan(D: float, A: float, direction: str, price: float,
                atr: float = 0.0) -> Dict:
    """Giriş D'de; hedefler AD bacağının Fibonacci geri çekilmeleri.

    Stop D'nin ÖTESİNDE — X seviyesi stop olarak kullanılmaz çünkü Butterfly
    (1.27) ve Crab (1.618) formasyonlarında D zaten X'in ötesindedir; X orada
    geçersizlik değil, geçilmiş bir seviyedir.

    STOP MESAFESİ = max(AD × %15, 1 × ATR).
    Eskiden yalnız `AD × %15` kullanılıyordu; bu, ödül/riski KURGU GEREĞİ sabit
    yapıyordu (0,382/0,15 = 2,55) ve R/R her formasyonda aynı çıkıyordu —
    ölçüldü: 69 kurulumun 69'unda R/R tam 2,55. Böyle bir sayı formasyon hakkında
    hiçbir şey söylemez. ATR tabanı, dar bacaklı bir kurulumun stop'unu piyasa
    gürültüsünün içine koymasını da engeller."""
    ad = A - D
    t1 = D + 0.382 * ad
    t2 = D + 0.618 * ad
    t3 = A
    stop_mesafe = max(abs(ad) * 0.15, float(atr) if atr and atr > 0 else 0.0)
    stop = D - math.copysign(stop_mesafe, ad)
    risk = abs(D - stop)
    reward = abs(t1 - D)
    return {"entry": float(D), "stop": float(stop),
            "targets": [float(t1), float(t2), float(t3)],
            "target_pcts": [round(_pct(t1, price), 2), round(_pct(t2, price), 2),
                            round(_pct(t3, price), 2)],
            "rr": round(float(reward / (risk + 1e-12)), 2),
            "stop_source": "ATR" if stop_mesafe > abs(ad) * 0.15 + 1e-12 else "AD%15"}


def _action_text(direction: str, status: str, entry: float, price: float,
                 t1_pct: float) -> str:
    """Kullanıcının net karar cümlesi."""
    side = "ALIŞ" if direction == "LONG" else "SATIŞ"
    dist = _pct(entry, price)
    if status == "oluşuyor":
        return (f"{side} kurulumu — fiyat PRZ bölgesine (%{dist:+.2f} uzakta) "
                f"gelirse {side.lower()} tetiklenir. Bölgeye gelmeden işlem YOK; "
                f"ilk hedef %{t1_pct:+.2f}.")
    if abs(dist) < 0.5:
        return (f"{side} bölgesi AKTİF — fiyat D noktasında (%{dist:+.2f}). "
                f"Stop D'nin ötesinde, ilk hedef %{t1_pct:+.2f}.")
    return (f"{side} noktası geçildi (%{dist:+.2f}) — formasyon tamamlandı, "
            f"giriş kaçırıldıysa kovalama; ilk hedef %{t1_pct:+.2f}.")


def _apply_h_validity(p: "HarmonicPattern", price: float) -> None:
    """Fiyat kurulumu geçersiz kıldı mı?

    Grafik formasyonlarında bulunan hatanın aynısı harmoniklerde de vardı:
    denetimde 69 kurulumun 10'unda fiyat STOP'u ihlal etmiş, 29'unda ilk hedefi
    geçmişti — hepsi hâlâ "işlem planı" diye sunuluyordu. Tamamlanmış bir
    formasyonda giriş D'dedir; fiyat stop'un ötesine geçtiyse kurulum ÖLMÜŞTÜR."""
    if not p.targets:
        return
    son_hedef = p.targets[-1]
    if p.direction == "LONG":
        if price <= p.stop:
            p.valid, p.validity = False, "stop_ihlal"
            p.validity_note = (f"GEÇERSİZ — fiyat ({price:.4g}) stop seviyesinin "
                               f"({p.stop:.4g}) altına indi.")
            return
        if price >= son_hedef:
            p.valid, p.validity = False, "hedef_aşıldı"
            p.validity_note = (f"TAMAMLANDI — fiyat ({price:.4g}) son hedefi "
                               f"({son_hedef:.4g}) geçti; kurulum bitti.")
            return
    elif p.direction == "SHORT":
        if price >= p.stop:
            p.valid, p.validity = False, "stop_ihlal"
            p.validity_note = (f"GEÇERSİZ — fiyat ({price:.4g}) stop seviyesinin "
                               f"({p.stop:.4g}) üstüne çıktı.")
            return
        if price <= son_hedef:
            p.valid, p.validity = False, "hedef_aşıldı"
            p.validity_note = (f"TAMAMLANDI — fiyat ({price:.4g}) son hedefi "
                               f"({son_hedef:.4g}) geçti; kurulum bitti.")
            return
    p.valid, p.validity, p.validity_note = True, "işlenebilir", ""


# ===========================================================================
# 1) TAMAMLANMIŞ XABCD
# ===========================================================================
def _complete_patterns(df: pd.DataFrame, pivots: List[Tuple[int, float, int]],
                       price: float, only: Optional[str],
                       atr: float = 0.0) -> List[HarmonicPattern]:
    out: List[HarmonicPattern] = []
    if len(pivots) < 5:
        return out

    for start in range(max(0, len(pivots) - 16), len(pivots) - 4):
        pts = pivots[start:start + 5]
        if len(pts) < 5:
            continue
        t = [p[2] for p in pts]
        if not all(t[k] != t[k + 1] for k in range(4)):
            continue

        (iX, X, _), (iA, A, _), (iB, B, _), (iC, Cc, _), (iD, D, tD) = pts
        XA, AB, BC, CD, AD = A - X, B - A, Cc - B, D - Cc, D - A
        if min(abs(XA), abs(AB), abs(BC)) < 1e-9:
            continue

        r = {"AB/XA": _ratio(AB, XA), "BC/AB": _ratio(BC, AB),
             "CD/BC": _ratio(CD, BC), "AD/XA": _ratio(AD, XA)}

        # BİR PENCERE = BİR HARMONİK. Eskiden her kural ayrı ayrı ekleniyordu ve
        # aynı beş nokta gartley+crab+cypher+shark diye DÖRT formasyon olarak
        # raporlanabiliyordu (ölçüldü: tek bir Gartley şeklinde 5 formasyon
        # döndü). Oranlar birbirine yakın olduğu için bu kaçınılmaz; doğru olan
        # EN İYİ UYAN kuralı seçmektir.
        aday = []
        for name, rule in HARMONIC_RULES.items():
            meta = HARMONIC_META.get(name)
            if not meta or (only and meta["key"] != only):
                continue
            fits = {
                "AB/XA": _fit(r["AB/XA"], rule["AB_XA"]),
                "BC/AB": _fit(r["BC/AB"], rule["BC_AB"]),
                "CD/BC": _fit(r["CD/BC"], rule["CD_BC"]),
                "AD/XA": _fit(r["AD/XA"], rule["AD_XA"]),
            }
            q = float(np.mean(list(fits.values())))
            if q < MIN_QUALITY_COMPLETE:
                continue
            aday.append((q, name, rule, meta, fits))
        if not aday:
            continue
        aday.sort(key=lambda t: -t[0])
        q, name, rule, meta, fits = aday[0]
        ikinci = aday[1][0] if len(aday) > 1 else 0.0
        for _ in (0,):                       # tek geçiş — girinti korunsun diye

            direction = "LONG" if tD == -1 else "SHORT"
            plan = _trade_plan(D, A, direction, price, atr)
            p = HarmonicPattern(
                key=meta["key"], name=meta["tr"], direction=direction,
                status="tamamlandı", quality=round(q, 3), completion=1.0,
                score=round(q * (0.7 + 0.3 * min(1.0, abs(plan["target_pcts"][0]) / 5)), 4),
                color=meta["color"],
                entry=plan["entry"], stop=plan["stop"], targets=plan["targets"],
                target_pcts=plan["target_pcts"], rr=plan["rr"],
                entry_pct=round(_pct(D, price), 2),
                clears_min_move=abs(plan["target_pcts"][0]) >= 1.0,
                ratios={k: round(v, 3) for k, v in r.items()},
                note=(f"{meta['tr']} — dört bacağın tamamı kural aralığında "
                      f"(uyum %{q*100:.0f}"
                      + (f"; en yakın rakip kural %{ikinci*100:.0f})." if ikinci > 0
                         else ").")),
            )
            p.action = _action_text(direction, "tamamlandı", D, price,
                                    plan["target_pcts"][0])
            p.points = [HPoint("X", iX, float(X)), HPoint("A", iA, float(A)),
                        HPoint("B", iB, float(B)), HPoint("C", iC, float(Cc)),
                        HPoint("D", iD, float(D))]
            # X→A BAZ bacaktır; oranlar ona GÖRE ölçülür, kendisi oran taşımaz.
            p.legs.append(HLeg("X", "A", int(iX), float(X), int(iA), float(A),
                               "XA (baz)", 1.0, "—", 1.0))
            for frm, to, x0, y0, x1, y1, rn in (
                    ("A", "B", iA, A, iB, B, "AB/XA"),
                    ("B", "C", iB, B, iC, Cc, "BC/AB"),
                    ("C", "D", iC, Cc, iD, D, "CD/BC")):
                rl = {"AB/XA": rule["AB_XA"], "BC/AB": rule["BC_AB"],
                      "CD/BC": rule["CD_BC"]}[rn]
                p.legs.append(HLeg(frm, to, int(x0), float(y0), int(x1), float(y1),
                                   rn, round(r[rn], 3), _rule_text(rl),
                                   round(fits[rn], 2)))
            # AD/XA bütünün kapanış oranı — ayrı bir bacak değil, künyede raporlanır
            p.ratios["AD/XA (kapanış)"] = round(r["AD/XA"], 3)
            out.append(p)
    return out


# ===========================================================================
# 2) OLUŞMAKTA OLAN XABC → D PROJEKSİYONU (PRZ)
# ===========================================================================
def _forming_patterns(df: pd.DataFrame, pivots: List[Tuple[int, float, int]],
                      price: float, only: Optional[str],
                      bars_ahead: int, atr: float = 0.0) -> List[HarmonicPattern]:
    """Son dört pivot XABC ise, kuralın gerektirdiği D'yi İKİ bağımsız yoldan
    projekte et ve kesişimi PRZ olarak ver.

      D₁ = A − (AD/XA) × XA      → XA bacağının geri çekilmesi
      D₂ = C − (CD/BC) × BC      → BC bacağının uzantısı

    İki projeksiyon örtüşüyorsa kurulum güçlüdür; örtüşmüyorsa zayıf sayılır."""
    out: List[HarmonicPattern] = []
    if len(pivots) < 4:
        return out

    n = len(df)
    last_i = n - 1
    for start in range(max(0, len(pivots) - 8), len(pivots) - 3):
        pts = pivots[start:start + 4]
        if len(pts) < 4:
            continue
        t = [p[2] for p in pts]
        if not all(t[k] != t[k + 1] for k in range(3)):
            continue
        (iX, X, _), (iA, A, _), (iB, B, _), (iC, Cc, tC) = pts
        # C yeterince yeni olmalı (formasyon hâlâ canlı)
        if last_i - iC > max(30, bars_ahead * 2):
            continue

        XA, AB, BC = A - X, B - A, Cc - B
        if min(abs(XA), abs(AB), abs(BC)) < 1e-9:
            continue
        r_ab = _ratio(AB, XA)
        r_bc = _ratio(BC, AB)

        for name, rule in HARMONIC_RULES.items():
            meta = HARMONIC_META.get(name)
            if not meta or (only and meta["key"] != only):
                continue
            f_ab = _fit(r_ab, rule["AB_XA"])
            f_bc = _fit(r_bc, rule["BC_AB"])
            q_sofar = (f_ab + f_bc) / 2
            if q_sofar < MIN_QUALITY_FORMING:
                continue

            ad_lo, ad_hi = _rule_range(rule["AD_XA"])
            cd_lo, cd_hi = _rule_range(rule["CD_BC"])
            d1 = sorted([A - ad_lo * XA, A - ad_hi * XA])
            d2 = sorted([Cc - cd_lo * BC, Cc - cd_hi * BC])

            lo = max(d1[0], d2[0])
            hi = min(d1[1], d2[1])
            overlap = lo <= hi
            if not overlap:
                # örtüşme yok → iki bölgenin birleşimini al, kaliteyi düşür
                lo, hi = min(d1[0], d2[0]), max(d1[1], d2[1])
            prz_mid = (lo + hi) / 2

            # D, C'nin ters yönünde olmalı (zikzak devam etsin)
            if (tC == 1 and prz_mid >= Cc) or (tC == -1 and prz_mid <= Cc):
                continue

            direction = "LONG" if tC == 1 else "SHORT"   # C tepe ise D dip → LONG
            q = q_sofar * (1.0 if overlap else 0.7)
            # Eşik, GÖSTERİLEN kaliteye uygulanmalı. Eskiden kapı yalnız ceza
            # ÖNCESİ `q_sofar`'a bakıyordu; iki D projeksiyonu örtüşmediğinde
            # ×0,7 cezası devreye girip panelde eşiğin ALTINDA kalite değerleri
            # görünüyordu (canlıda 0,47 ve 0,55 — "uyum eşiği %65" yazarken).
            if q < MIN_QUALITY_FORMING:
                continue
            plan = _trade_plan(prz_mid, A, direction, price, atr)
            width_pct = abs(_pct(hi, lo))

            p = HarmonicPattern(
                key=meta["key"], name=meta["tr"], direction=direction,
                status="oluşuyor", quality=round(q, 3), completion=0.75,
                score=round(q * 0.8 * (0.7 + 0.3 * min(1.0, abs(plan["target_pcts"][0]) / 5)), 4),
                color=meta["color"],
                entry=plan["entry"], stop=plan["stop"], targets=plan["targets"],
                target_pcts=plan["target_pcts"], rr=plan["rr"],
                entry_pct=round(_pct(prz_mid, price), 2),
                clears_min_move=abs(plan["target_pcts"][0]) >= 1.0,
                ratios={"AB/XA": round(r_ab, 3), "BC/AB": round(r_bc, 3)},
                note=(f"XABC tamam, D bekleniyor. PRZ iki bağımsız projeksiyonun "
                      f"{'KESİŞİMİ' if overlap else 'BİRLEŞİMİ (örtüşme yok → zayıf)'}; "
                      f"bölge genişliği %{width_pct:.2f}."),
            )
            p.action = _action_text(direction, "oluşuyor", prz_mid, price,
                                    plan["target_pcts"][0])
            d_i = min(last_i + bars_ahead, last_i + bars_ahead)
            p.points = [HPoint("X", iX, float(X)), HPoint("A", iA, float(A)),
                        HPoint("B", iB, float(B)), HPoint("C", iC, float(Cc)),
                        HPoint("D?", d_i, float(prz_mid), projected=True)]
            p.prz = {"lo": round(float(lo), 8), "hi": round(float(hi), 8),
                     "mid": round(float(prz_mid), 8), "i": int(d_i),
                     "i_from": int(iC), "overlap": bool(overlap),
                     "from_xa": [round(float(d1[0]), 8), round(float(d1[1]), 8)],
                     "from_bc": [round(float(d2[0]), 8), round(float(d2[1]), 8)]}
            legs = [("X", "A", iX, X, iA, A, "XA (baz)", None, 1.0, 1.0, False),
                    ("A", "B", iA, A, iB, B, "AB/XA", rule["AB_XA"], round(r_ab, 3), f_ab, False),
                    ("B", "C", iB, B, iC, Cc, "BC/AB", rule["BC_AB"], round(r_bc, 3), f_bc, False),
                    ("C", "D?", iC, Cc, d_i, prz_mid, "CD/BC", rule["CD_BC"], 0.0, 0.0, True)]
            for frm, to, x0, y0, x1, y1, rn, rl, rv, fv, proj in legs:
                p.legs.append(HLeg(frm, to, int(x0), float(y0), int(x1), float(y1),
                                   rn, rv, ("—" if rl is None else _rule_text(rl)),
                                   fv, projected=proj))
            out.append(p)
    return out


# ===========================================================================
# ANA GİRİŞ
# ===========================================================================
def detect_harmonics_rich(df: pd.DataFrame, pattern: Optional[str] = None,
                          top_n: int = 6, include_forming: bool = True,
                          bars_ahead: int = 12,
                          pivot_left: int = 3, pivot_right: int = 3) -> Dict:
    """Tüm harmonik formasyonlar — çizim geometrisi + işlem planıyla.

    pattern: yalnız belirli bir formasyon (ör. "butterfly"). None → hepsi.
    include_forming: tamamlanmamış XABC'lerin D projeksiyonunu da üret."""
    if df is None or len(df) < 40:
        return {"patterns": [], "available": {}, "reason": "yetersiz veri"}
    if not {"high", "low", "close"}.issubset(df.columns):
        return {"patterns": [], "available": {}, "reason": "eksik sütun"}

    sh, sl = find_pivots(df, pivot_left, pivot_right)
    pivots = sorted([(i, float(df["high"].iloc[i]), 1) for i in sh] +
                    [(i, float(df["low"].iloc[i]), -1) for i in sl])
    price = float(df["close"].iloc[-1])
    only = pattern.lower().strip() if pattern else None

    # ATR — stop mesafesinin tabanı; dar bacaklı kurulumun stop'u piyasa
    # gürültüsünün içine düşmesin diye.
    try:
        from .indicators import atr as _atr
        atr_v = float(_atr(df, 14).iloc[-1])
        if not np.isfinite(atr_v) or atr_v <= 0:
            atr_v = 0.0
    except Exception:
        atr_v = 0.0

    pats = _complete_patterns(df, pivots, price, only, atr_v)
    if include_forming:
        pats += _forming_patterns(df, pivots, price, only, bars_ahead, atr_v)

    # GEÇERLİLİK: fiyat stop'u ihlal etmiş ya da son hedefi geçmiş kurulumlar
    # işaretlenir. Yalnız TAMAMLANMIŞ formasyonlar için anlamlı — "oluşuyor"
    # durumunda giriş henüz gelmemiştir, geçersizlikten söz edilemez.
    for p in pats:
        if p.status == "tamamlandı":
            _apply_h_validity(p, price)
        # ≥%1 rozeti yalnız hâlâ işlenebilir kurulumlara
        p.clears_min_move = bool(p.valid and p.clears_min_move)

    # Aynı (formasyon, durum) için en iyisini tut
    best: Dict[str, HarmonicPattern] = {}
    for p in pats:
        k = f"{p.key}|{p.status}"
        if k not in best or p.score > best[k].score:
            best[k] = p
    ranked = sorted(best.values(), key=lambda x: -x.score)[:top_n]

    # Buton durumları: hangi formasyondan kaç tane var
    available = {m["key"]: {"tr": m["tr"], "color": m["color"],
                            "complete": 0, "forming": 0}
                 for m in HARMONIC_META.values()}
    for p in best.values():
        if p.key in available:
            available[p.key]["complete" if p.status == "tamamlandı" else "forming"] += 1

    return {"patterns": [p.to_dict() for p in ranked],
            "available": available,
            "price": price,
            "bars_ahead": bars_ahead,
            "evidence": HARMONIC_EVIDENCE,
            "min_quality": MIN_QUALITY_COMPLETE,
            "n_invalid": sum(1 for p in ranked if not p.valid)}
