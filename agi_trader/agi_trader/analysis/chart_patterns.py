"""
Çizilebilir grafik formasyonları (FAZ 5).

`patterns.py` formasyonu TESPİT eder ama yalnız birkaç fiyat noktası döndürür —
grafiğe çizilemez. Bu modül her formasyonu ÇİZİM GEOMETRİSİYLE birlikte üretir:
etiketli çapa noktaları (Sol Omuz / Baş / Sağ Omuz…), doğru parçaları, gölge
alanı, kırılım seviyesi, hedef ve iptal; ayrıca hedefin mevcut fiyattan
YÜZDE UZAKLIĞI.

Tespit edilenler
  Devam        : yükselen/alçalan/simetrik üçgen · boğa/ayı bayrağı · flama · dikdörtgen
  Dönüş        : yükselen/düşen kama · fincan-kulp · omuz-baş-omuz (+ters) · çift tepe/dip

DÜRÜSTLÜK NOTU: Grafik formasyonlarının örneklem dışı kâr ürettiğine dair güçlü
akademik kanıt YOKTUR. Bu modül formasyonları **tek başına işlem açtıran kapı**
olarak değil, (a) kullanıcıya görsel karar desteği ve (b) FAZ 4 meta-etiketleme
için ÖZELLİK olarak üretir. `runs/FAZ1-2_BULGULAR.md` disiplini geçerlidir:
kabul kapısından geçmeyen hiçbir sinyal otomatik işlem açtırmaz.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .patterns import find_pivots
from .indicators import true_range

# Kullanıcı hedefi: en az %1'lik bir hareket öngörülebilir olmalı.
MIN_MOVE_PCT = 1.0

# --- Arama genişliğini dengeleyen kabul eşikleri ---------------------------
# Bayrak/fincan dedektörlerinde arama uzayı (bitiş uzaklığı × pencere boyu)
# 9'dan ~200 kombinasyona çıkarıldı. Çoklu deneme, gürültüde de eşiği geçen bir
# pencere bulur: ölçüldü, bu eşikler olmadan bayrak rastgele yürüyüşün 25/25'inde,
# fincan 19/25'inde ateşliyordu. Eşikler o yüzden aramayla ORANTILI sıkılaştırıldı.
POLE_ATR_MULT = 2.2     # direk ≥ 2,2 × ATR × √bar (rastgele yürüyüşün ötesi)
CONS_MAX_FRAC = 0.55    # sıkışma aralığı ≤ direk boyunun %55'i
# CUP_MIN_R2 süpürülerek seçildi (tespit ↔ yanlış pozitif dengesi):
#   0,72 → tespit 15/15 ama YP 11/25 · 8/25
#   0,80 → YP 5/25 · 4/25
#   0,86 → YP 3/25 · 1/25, tespit kuyruk 0-8'de hâlâ 15/15   ← seçilen
#   0,90 → tespit kuyruk 8'de 1/15'e çöküyor
# Eşik bir tohumda seçilip AYRI bir tohumda (1234) doğrulandı.
CUP_MIN_R2 = 0.86       # U parabol uyumu (eskiden 0,55)
CUP_MIN_ATR_DEPTH = 3.0  # fincan derinliği ≥ 3 × ATR
CUP_MIN_HANDLE = 0.08   # kulp derinliği ≥ fincanın %8'i (kulpsuz fincan sayılmaz)

# --- Çift tepe/dip ayırt edicilik kapıları ---------------------------------
# İki benzer tepe ve aralarında bir çukur, rastgele seride DOĞAL OLARAK oluşur:
# ölçüldü, yalnız geometrik tanımla dedektör rastgele yürüyüşün %52-60'ında
# ateşliyordu (çift tepe 13-15/25, çift dip 8-12/25). Şekil "gerçek" ama
# bilgi taşımıyor. Ayırt edicilik ancak tanımı formasyonun ANLAMINA
# yaklaştırarak kazanılır — çift tepe bir DÖNÜŞ formasyonudur:
#   1) dönecek bir yükseliş olmalı (öncül trend),
#   2) boğaz anlamlı bir geri çekilme olmalı (yatay gürültü değil),
#   3) tepeler aralığın ZİRVESİNDE olmalı (orta yerde iki tümsek değil).
# Sabitler GERÇEK piyasa verisinde kalibre edildi (sentetik ders kitabı
# şekillerine göre ayarlamak yanılttı: adv=8 · Q=0,85 ile gerçek veride çift tepe
# oranı %41'den %0'a düştü — hiç ateşlemeyen dedektör de işe yaramaz).
# Gerçek veri ↔ rastgele yürüyüş oranları (90 kesit / 90 kesit, ölçüldü):
#   kapısız            gerçek %41  rastgele %40   ← ayırt edicilik YOK
#   yalnız derinlik    gerçek %16  rastgele %29
#   adv2 · Q0,50       gerçek %13  rastgele %27   ← SEÇİLEN (görünürlük korunur)
#   adv4 · Q0,60       gerçek  %4  rastgele %21
#   adv8 · Q0,85       gerçek  %0  rastgele %11   ← hiç ateşlemiyor, işe yaramaz
# HİÇBİR ayarda gerçek oran rastgeleyi geçmiyor. Kapılar bu yüzden ayırt edicilik
# için değil, KALİTE için tutuluyor: sığ bir dalgalanma boğaz sayılmasın diye.
DBL_MIN_DEPTH_ATR = 2.0   # boğaz derinliği ≥ 2,0 × ATR
DBL_MIN_DEPTH_PCT = 2.5   # ve ≥ tepe fiyatının %2,5'i
DBL_PRIOR_BARS = 40       # öncül trend penceresi
DBL_MIN_PRIOR_ADV = 2.0   # öncül yükseliş/düşüş ≥ %2 (dönecek bir hareket olmalı)
DBL_RANGE_Q = 0.50        # tepeler pencere yükseklerinin medyanının üstünde olmalı


# ---------------------------------------------------------------------------
# ÖLÇÜLMÜŞ KANIT — hangi formasyon ailesinin yön öngörüsü sınandı?
# ---------------------------------------------------------------------------
# Çift tepe/dip "rastgele seride de çıkıyor" diye işaretlenmişti; düzeltme
# denemesi sırasında ASIL sorunun sıklık değil BİLGİ olduğu ortaya çıktı.
#
# 1) Sıklık: eşleştirilmiş oynaklıkta rastgele yürüyüş, GERÇEK kripto verisinden
#    DAHA ÇOK çift tepe üretiyor (her kapı ayarında fark −%11 ile −%17).
#    Yani şeklin sık görülmesi piyasaya özgü bir olgu değil.
# 2) Yön: olay çalışması (20 bar ileri, aynı paritenin rastgele anları kontrol)
#    çift tepe → −%0,34 (kontrol −%0,92): düşüş öngörüyor ama fiyat DAHA AZ düştü
#    çift dip  → −%1,38 (kontrol −%0,92): yükseliş öngörüyor ama fiyat DAHA ÇOK düştü
#    İkisi de TERS yönde ve ikisi de anlamsız (|t| = 1,27 ve 0,81).
#
# Sonuç: kapılar formasyonun KALİTESİNİ yükseltir (ders kitabı ölçütleri), ama
# ölçülen bir yön üstünlüğü YOKTUR. Panel bunu açıkça yazar.
PATTERN_EVIDENCE = {
    "double_top": {
        "tested": True, "edge": "yok",
        "n_events": 219, "fwd_bars": 20,
        "event_ret_pct": -0.34, "control_ret_pct": -0.92, "t": 1.27,
        "note": ("Yön öngörüsü ÖLÇÜLDÜ ve bulunamadı: düşüş beklenirken fiyat "
                 "kontrol grubundan DAHA AZ düştü (−%0,34 vs −%0,92), üstelik "
                 "fark anlamsız (t=1,27). Eşleştirilmiş rastgele yürüyüş, gerçek "
                 "veriden daha çok çift tepe üretiyor."),
    },
    "double_bottom": {
        "tested": True, "edge": "yok",
        "n_events": 235, "fwd_bars": 20,
        "event_ret_pct": -1.38, "control_ret_pct": -0.92, "t": -0.81,
        "note": ("Yön öngörüsü ÖLÇÜLDÜ ve bulunamadı: yükseliş beklenirken fiyat "
                 "kontrol grubundan DAHA ÇOK düştü (−%1,38 vs −%0,92), fark "
                 "anlamsız (t=−0,81)."),
    },
}


# ===========================================================================
# Veri yapıları
# ===========================================================================
@dataclass
class PatternPoint:
    i: int              # bar indeksi
    price: float
    label: str          # "Sol Omuz", "Baş", "Tepe 1"…


@dataclass
class PatternLine:
    x0: int
    y0: float
    x1: int
    y1: float
    kind: str           # "direnç" | "destek" | "boyun" | "kanal" | "hedef"
    dashed: bool = False


@dataclass
class ChartPattern:
    key: str
    name: str
    family: str                     # "devam" | "dönüş"
    direction: str                  # "LONG" | "SHORT"
    status: str                     # "potansiyel" | "oluşuyor" | "tamamlandı" | "kırılım"
    completion: float               # 0..1
    quality: float                  # 0..1
    score: float                    # sıralama
    color: str
    start_i: int
    end_i: int
    breakout: float                 # kırılım seviyesi
    target: float
    stop: float
    target_pct: float               # mevcut fiyata göre % (yönlü)
    stop_pct: float
    rr: float
    clears_min_move: bool           # |hedef %| ≥ MIN_MOVE_PCT
    points: List[PatternPoint] = field(default_factory=list)
    lines: List[PatternLine] = field(default_factory=list)
    curve: List[Dict] = field(default_factory=list)      # fincan için {x,y}
    note: str = ""
    # --- geçerlilik (detect_chart_patterns içinde doldurulur) ---
    valid: bool = True              # şu anki fiyattan HÂLÂ işlem açılabilir mi
    validity: str = "işlenebilir"   # işlenebilir | stop_ihlal | hedef_aşıldı
    validity_note: str = ""

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["points"] = [asdict(p) for p in self.points]
        d["lines"] = [asdict(l) for l in self.lines]
        return d


# ===========================================================================
# Geometri yardımcıları
# ===========================================================================
def _fit_line(xs: List[int], ys: List[float]) -> Tuple[float, float, float]:
    """En küçük kareler doğrusu → (eğim, kesişim, R²)."""
    if len(xs) < 2:
        return 0.0, (ys[0] if ys else 0.0), 0.0
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / (ss_tot + 1e-12) if ss_tot > 1e-12 else 1.0
    return float(slope), float(intercept), float(max(0.0, min(1.0, r2)))


def _line_at(slope: float, intercept: float, x: int) -> float:
    return float(slope * x + intercept)


def _norm_slope(slope: float, price: float, bars: int) -> float:
    """Eğimi 'pencere boyunca % değişim' cinsine çevir — ölçekten bağımsız."""
    if price <= 0:
        return 0.0
    return float(slope * bars / price * 100.0)


def _boundary_fit(xs: List[int], ys: List[float], side: str, tol: float,
                  min_keep: int = 3) -> Tuple[float, float, float, List[int]]:
    """SINIR (zarf) doğrusu — trend çizgisine DOKUNAN pivotları seçerek uydurur.

    Neden gerekli: düz en-küçük-kareler bütün pivotlara uyar, oysa bir direnç
    çizgisi yalnız ona DEĞEN tepelerden geçer. Aradaki küçük salınım tepeleri
    (dirence değmeyenler) doğruyu aşağı çekip eğimi uydurur.

    Ölçüldü (yükselen üçgen, gürültülü sentetik):
        pivotlar 109,8 · 109,7 · 111,3 · 111,0 · 111,3  → eğim %+3,72/pencere
    Oysa gerçek direnç DÜZ. İlk iki pivot dirence değmiyor; ayıklanınca eğim
    ≈0 çıkıyor ve "yükselen üçgen" doğru sınıflanıyor.

    Yöntem — trend çizgisinin TANIMINI doğrudan uygular:
      "üstünde (üst sınırda) hiçbir pivot kalmayan, en çok DOKUNUŞU olan doğru".
    Her pivot çifti bir aday doğru verir (≤8 pivot → ≤28 aday). Bir aday, hiçbir
    pivot onu `tol`dan fazla aşmıyorsa geçerlidir; puanı `tol` içindeki pivot
    sayısıdır (dokunuş). En çok dokunuşlu aday seçilir, eşitlikte daha geniş
    x-aralığı yayanı. Sonra kesin eğim için dokunuşlardan en-küçük-kareler
    geçirilir.

    NEDEN yinelemeli budama DEĞİL: "uydur → en kötüyü at → yeniden uydur"
    denendi ve TAKILDI — en küçük kareler aykırı pivota EĞİLEREK uyuyor,
    artığı küçültüyor ve pivot bir daha ayıklanmıyor. (Ölçüldü: 109,8'lik
    dokunmayan pivot budanamadı.) Aşağıdaki yöntemde doğru, aykırıya uymak
    zorunda değil; sadece onu aşmaması yeterli.

    tol : "dokunma" toleransı, fiyat birimi (tipik olarak ~0,5 × ATR)
    side: "upper" (direnç) | "lower" (destek)
    Döner: (eğim, kesişim, R², kullanılan pivot indeksleri)
    """
    n = len(xs)
    if n < 2:
        m, b, r2 = _fit_line(xs, ys)
        return m, b, r2, list(xs)
    if n == 2 or tol <= 0:
        m, b, r2 = _fit_line(xs, ys)
        return m, b, r2, list(xs)

    up = (side == "upper")
    best = None                      # (dokunus, yayilim, m, b, dokunan_indeksler)
    for a in range(n):
        for c in range(a + 1, n):
            if xs[c] == xs[a]:
                continue
            m = (ys[c] - ys[a]) / (xs[c] - xs[a])
            b = ys[a] - m * xs[a]
            resid = [ys[k] - (m * xs[k] + b) for k in range(n)]
            # sınırı AŞAN pivot varsa bu doğru sınır olamaz
            asim = max(resid) if up else -min(resid)
            if asim > tol:
                continue
            touch = [k for k in range(n) if abs(resid[k]) <= tol]
            if len(touch) < 2:
                continue
            yayilim = xs[touch[-1]] - xs[touch[0]]
            cand = (len(touch), yayilim, m, b, touch)
            if best is None or cand[:2] > best[:2]:
                best = cand

    if best is None:                 # geçerli sınır bulunamadı → düz uydurma
        m, b, r2 = _fit_line(xs, ys)
        return m, b, r2, list(xs)

    touch = best[4]
    kept_x = [xs[k] for k in touch]
    kept_y = [ys[k] for k in touch]
    if len(kept_x) >= max(2, min_keep):
        m, b, r2 = _fit_line(kept_x, kept_y)
    else:
        m, b, r2 = best[2], best[3], 1.0
    return m, b, r2, kept_x


def _touch_tol(df: pd.DataFrame, price: float) -> float:
    """Trend çizgisine "dokunma" toleransı = tipik bar aralığının yarısı.

    ATR'den türetilir; sabit yüzde kullanmak sakin ve çalkantılı rejimlerde
    bozulurdu (aynı %0,2 birinde her pivotu dokunma sayar, diğerinde hiçbirini)."""
    try:
        t = float(true_range(df).tail(30).mean()) * 0.5
    except Exception:
        t = 0.0
    if not np.isfinite(t) or t <= 0:
        t = max(price, 1e-9) * 0.002
    return t


def _pct(a: float, b: float) -> float:
    """b'ye göre a'nın yüzde farkı."""
    return float((a - b) / (b + 1e-12) * 100.0)


def _mk(key, name, family, direction, status, completion, quality, color,
        start_i, end_i, breakout, target, stop, price, **kw) -> ChartPattern:
    tp = _pct(target, price)
    sp = _pct(stop, price)
    risk = abs(price - stop)
    reward = abs(target - price)
    rr = float(reward / (risk + 1e-12))
    # Skor: kalite × tamamlanma × durum ağırlığı × hareket büyüklüğü kazancı
    status_w = {"kırılım": 1.0, "tamamlandı": 0.9, "oluşuyor": 0.75, "potansiyel": 0.5}
    move_w = min(1.0, abs(tp) / 5.0)          # %5 ve üstü tam puan
    score = quality * (0.4 + 0.6 * completion) * status_w.get(status, 0.6) * (0.6 + 0.4 * move_w)
    return ChartPattern(
        key=key, name=name, family=family, direction=direction, status=status,
        completion=round(float(completion), 3), quality=round(float(quality), 3),
        score=round(float(score), 4), color=color,
        start_i=int(start_i), end_i=int(end_i),
        breakout=round(float(breakout), 8), target=round(float(target), 8),
        stop=round(float(stop), 8), target_pct=round(tp, 2), stop_pct=round(sp, 2),
        rr=round(rr, 2), clears_min_move=abs(tp) >= MIN_MOVE_PCT, **kw)


# ===========================================================================
# 1) ÜÇGENLER — yükselen / alçalan / simetrik
# ===========================================================================
def detect_triangles(df: pd.DataFrame, sh: List[int], sl: List[int],
                     lookback: int = 120) -> List[ChartPattern]:
    out: List[ChartPattern] = []
    n = len(df)
    if n < 40:
        return out
    lo_i = max(0, n - lookback)
    # Havuz 5 değil 8: zarf uydurması dokunmayanları zaten ayıklıyor, bu yüzden
    # geniş havuz artık gürültü değil SEÇENEK demek. Dar havuz (son 5) alçalan
    # üçgende gerçek trend pivotlarını kırpıyordu.
    hi_pool = [i for i in sh if i >= lo_i][-8:]
    lo_pool = [i for i in sl if i >= lo_i][-8:]
    if len(hi_pool) < 2 or len(lo_pool) < 2:
        return out

    highs = df["high"].values
    lows = df["low"].values
    price = float(df["close"].iloc[-1])
    last = n - 1

    tol = _touch_tol(df, price)
    sh_m, sh_b, sh_r2, hi_pts = _boundary_fit(
        hi_pool, [highs[i] for i in hi_pool], "upper", tol)
    sl_m, sl_b, sl_r2, lo_pts = _boundary_fit(
        lo_pool, [lows[i] for i in lo_pool], "lower", tol)
    if len(hi_pts) < 2 or len(lo_pts) < 2:
        return out

    span = max(10, last - min(hi_pts[0], lo_pts[0]))
    up_n = _norm_slope(sh_m, price, span)          # üst çizgi eğimi (% / pencere)
    dn_n = _norm_slope(sl_m, price, span)          # alt çizgi eğimi

    top_now = _line_at(sh_m, sh_b, last)
    bot_now = _line_at(sl_m, sl_b, last)
    height_start = abs(_line_at(sh_m, sh_b, min(hi_pts[0], lo_pts[0])) -
                       _line_at(sl_m, sl_b, min(hi_pts[0], lo_pts[0])))
    if height_start <= 0:
        return out

    # TEPE (apeks) KONTROLÜ — eskiden `top_now <= bot_now` ise formasyon
    # tümden reddediliyordu. Ama çizgilerin kesişmesi tam olarak KIRILIM ANIdır;
    # o anda körleşmek, üçgeni en işe yarar olduğu yerde kaybetmek demekti
    # (pivot düzeltmesinden sonra alçalan üçgende 4/20 kayıp bundandı).
    # Doğru ölçüt: apeks GEÇMİŞTE ÇOK GERİDE mi? Öyleyse formasyon bayattır.
    denom = sh_m - sl_m
    if abs(denom) > 1e-12:
        apex_x = (sl_b - sh_b) / denom
        if apex_x < last - 0.25 * span:
            return out                 # apeks çoktan geride: bayat üçgen
    elif top_now <= bot_now:
        return out                     # paralel ve ters sıralı: geometri bozuk

    # Apeks civarında çizgiler kesiştiğinde uç seviye artık güvenilmez; STOP
    # için çizgi yerine SON GERÇEK fiyat uç noktası kullanılır. Bu, kırılım
    # anında "LONG ama stop girişin üstünde" gibi geçersiz kurulumları önler.
    look_k = min(12, n)
    recent_low = float(lows[-look_k:].min())
    recent_high = float(highs[-look_k:].max())
    # LONG kurulumda stop kırılımın ALTINDA, SHORT'ta ÜSTÜNDE olmak zorunda;
    # apekste çizgiler ters sıraya geçebildiği için gerçek uç noktayla kelepçele.
    stop_long = min(bot_now, recent_low)
    if stop_long >= top_now:
        stop_long = top_now * 0.997
    stop_short = max(top_now, recent_high)
    if stop_short <= bot_now:
        stop_short = bot_now * 1.003

    quality = float(np.clip((sh_r2 + sl_r2) / 2, 0, 1)) * \
        min(1.0, (len(hi_pts) + len(lo_pts)) / 6.0)
    # daralma oranı = tamamlanma
    height_now = top_now - bot_now
    completion = float(np.clip(1.0 - height_now / (height_start + 1e-12), 0.05, 0.98))

    FLAT = 1.5      # |eğim| < %1,5/pencere → yatay say

    def add(key, name, direction, color, breakout, target, stop, note, extra_pts):
        st = "kırılım" if ((direction == "LONG" and price > breakout) or
                           (direction == "SHORT" and price < breakout)) else \
             ("tamamlandı" if completion > 0.75 else "oluşuyor")
        p = _mk(key, name, "devam", direction, st, completion, quality, color,
                min(hi_pts[0], lo_pts[0]), last, breakout, target, stop, price,
                note=note)
        p.lines = [
            PatternLine(hi_pts[0], _line_at(sh_m, sh_b, hi_pts[0]),
                        last, top_now, "direnç"),
            PatternLine(lo_pts[0], _line_at(sl_m, sl_b, lo_pts[0]),
                        last, bot_now, "destek"),
        ]
        p.points = extra_pts
        out.append(p)

    hp = [PatternPoint(i, float(highs[i]), f"T{k+1}") for k, i in enumerate(hi_pts)]
    lp = [PatternPoint(i, float(lows[i]), f"D{k+1}") for k, i in enumerate(lo_pts)]

    if abs(up_n) < FLAT and dn_n > FLAT:
        add("ascending_triangle", "Yükselen Üçgen", "LONG", "#00FF88",
            top_now, top_now + height_start, stop_long,
            "Yatay direnç + yükselen dipler: alıcı baskısı birikiyor.", hp + lp)
    elif abs(dn_n) < FLAT and up_n < -FLAT:
        add("descending_triangle", "Alçalan Üçgen", "SHORT", "#FF3B5C",
            bot_now, bot_now - height_start, stop_short,
            "Yatay destek + alçalan tepeler: satıcı baskısı birikiyor.", hp + lp)
    elif up_n < -FLAT and dn_n > FLAT:
        # simetrik: yön son eğilime göre
        trend_up = df["close"].iloc[-1] > df["close"].iloc[max(0, last - span)]
        if trend_up:
            add("symmetric_triangle", "Simetrik Üçgen", "LONG", "#0099FF",
                top_now, top_now + height_start, stop_long,
                "Daralan aralık; giriş trendi yukarı → yukarı kırılım beklenir.", hp + lp)
        else:
            add("symmetric_triangle", "Simetrik Üçgen", "SHORT", "#FFB627",
                bot_now, bot_now - height_start, stop_short,
                "Daralan aralık; giriş trendi aşağı → aşağı kırılım beklenir.", hp + lp)
    return out


# ===========================================================================
# 2) KAMALAR — yükselen (ayı) / düşen (boğa)
# ===========================================================================
def detect_wedges(df: pd.DataFrame, sh: List[int], sl: List[int],
                  lookback: int = 120) -> List[ChartPattern]:
    out: List[ChartPattern] = []
    n = len(df)
    if n < 40:
        return out
    lo_i = max(0, n - lookback)
    hi_pool = [i for i in sh if i >= lo_i][-8:]
    lo_pool = [i for i in sl if i >= lo_i][-8:]
    if len(hi_pool) < 2 or len(lo_pool) < 2:
        return out

    highs, lows = df["high"].values, df["low"].values
    price = float(df["close"].iloc[-1])
    last = n - 1
    # Üçgendeki ile aynı kusur: kamanın sınır çizgileri de yalnız DOKUNAN
    # pivotlardan geçmeli. Kama koşulları yön temelli olduğu için kirlenmeye
    # üçgenden daha dayanıklı, yine de daralma testi (h_now < h_start) bozulabilir.
    tol = _touch_tol(df, price)
    sh_m, sh_b, sh_r2, hi_pts = _boundary_fit(
        hi_pool, [highs[i] for i in hi_pool], "upper", tol)
    sl_m, sl_b, sl_r2, lo_pts = _boundary_fit(
        lo_pool, [lows[i] for i in lo_pool], "lower", tol)
    if len(hi_pts) < 2 or len(lo_pts) < 2:
        return out
    start = min(hi_pts[0], lo_pts[0])
    span = max(10, last - start)
    up_n = _norm_slope(sh_m, price, span)
    dn_n = _norm_slope(sl_m, price, span)

    h_start = _line_at(sh_m, sh_b, start) - _line_at(sl_m, sl_b, start)
    h_now = _line_at(sh_m, sh_b, last) - _line_at(sl_m, sl_b, last)
    if h_start <= 0 or h_now <= 0 or h_now >= h_start:
        return out                      # daralma yoksa kama değil

    quality = float(np.clip((sh_r2 + sl_r2) / 2, 0, 1)) * \
        min(1.0, (len(hi_pts) + len(lo_pts)) / 6.0)
    completion = float(np.clip(1.0 - h_now / h_start, 0.05, 0.98))
    MIN = 2.0

    def add(key, name, direction, color, breakout, target, stop, note):
        st = "kırılım" if ((direction == "LONG" and price > breakout) or
                           (direction == "SHORT" and price < breakout)) else \
             ("tamamlandı" if completion > 0.7 else "oluşuyor")
        p = _mk(key, name, "dönüş", direction, st, completion, quality, color,
                start, last, breakout, target, stop, price, note=note)
        p.lines = [
            PatternLine(hi_pts[0], _line_at(sh_m, sh_b, hi_pts[0]), last,
                        _line_at(sh_m, sh_b, last), "direnç"),
            PatternLine(lo_pts[0], _line_at(sl_m, sl_b, lo_pts[0]), last,
                        _line_at(sl_m, sl_b, last), "destek"),
        ]
        p.points = ([PatternPoint(i, float(highs[i]), f"T{k+1}") for k, i in enumerate(hi_pts)] +
                    [PatternPoint(i, float(lows[i]), f"D{k+1}") for k, i in enumerate(lo_pts)])
        out.append(p)

    if up_n > MIN and dn_n > MIN and dn_n > up_n:
        # yükselen kama → AYI (dipler tepelerden hızlı yükseliyor, momentum tükeniyor)
        bo = _line_at(sl_m, sl_b, last)
        add("rising_wedge", "Yükselen Kama", "SHORT", "#FF3B5C",
            bo, bo - h_start, _line_at(sh_m, sh_b, last),
            "Yükseliş içinde daralma: momentum tükeniyor, aşağı kırılım riski.")
    elif up_n < -MIN and dn_n < -MIN and up_n < dn_n:
        # düşen kama → BOĞA
        # NOT: yakınsama koşulu up_n < dn_n'dir (üst çizgi alttan HIZLI düşer, aralık
        # daralır). Burada eskiden `up_n > dn_n` yazıyordu — bu IRAKSAMA koşuludur ve
        # 255. satırdaki daralma kapısıyla asla birlikte sağlanamazdı; düşen kama
        # 20 sentetik denemede 0 kez tespit edilmişti. Yükselen kamada koşul zaten
        # doğruydu (dn_n > up_n), asimetri hatayı ele verdi.
        bo = _line_at(sh_m, sh_b, last)
        add("falling_wedge", "Düşen Kama", "LONG", "#00FF88",
            bo, bo + h_start, _line_at(sl_m, sl_b, last),
            "Düşüş içinde daralma: satış baskısı azalıyor, yukarı kırılım beklenir.")
    return out


# ===========================================================================
# 3) BAYRAK / FLAMA — sert hareket + karşı yönlü sıkışma
# ===========================================================================
def detect_flags(df: pd.DataFrame, lookback: int = 90) -> List[ChartPattern]:
    out: List[ChartPattern] = []
    n = len(df)
    if n < 45:
        return out
    close = df["close"].values
    highs, lows = df["high"].values, df["low"].values
    price = float(close[-1])
    last = n - 1
    atr_v = _touch_tol(df, price) * 2.0        # _touch_tol = 0,5×ATR

    # Direk: ≥ %3 tek yönlü hareket; ardından sıkışma.
    #
    # BİTİŞ UZAKLIĞI (`off`) — düzeltilen kusur: eskiden sıkışmanın SON BARA kadar
    # sürmesi şart koşuluyordu (`p1 = last - cons_len`). Kırılım gerçekleşince
    # sıkışma penceresi kırılım barlarını da içine alıyor, `cons_move` direğin
    # yönüne dönüyor ve "sıkışma ters yönde olmalı" kapısı formasyonu reddediyordu.
    # Ölçüldü: kuyruk 8 barda bu kapı 12 çekilişte 60 kez reddetti, tespit 0/12.
    # Kuyruk 14'te ise direk penceresi tümden bayrağın içine kayıyordu.
    # Artık formasyonun `off` bar önce BİTMİŞ olmasına izin veriliyor; geometri
    # yalnız formasyon penceresinde ölçülür, kırılım ise GÜNCEL fiyatla belirlenir.
    best: Dict[str, tuple] = {}
    for off in (0, 2, 4, 7, 11, 16, 22):
        pe = last - off
        if pe < 40:
            continue
        for pole_len in (8, 10, 14, 18, 24):
            for cons_len in (6, 8, 12, 18, 24):
                p0 = pe - cons_len - pole_len
                p1 = pe - cons_len
                if p0 < max(0, n - lookback):
                    continue
                pole_move = _pct(close[p1], close[p0])
                if abs(pole_move) < 3.0:
                    continue
                # DİREK ANORMAL Mİ? — sabit %3 eşiği tek başına YETMEZ; sakin bir
                # varlıkta %3 devasa, çalkantılıda sıradandır. Rastgele yürüyüş
                # `pole_len` barda tipik olarak ~ATR×√pole_len yol alır; gerçek
                # direk bunun belirgin katı olmalıdır.
                #
                # Bu kapı, arama uzayı 9'dan 175 kombinasyona çıkarıldığı için
                # ZORUNLU: ölçüldü, kapısız hâlde dedektör rastgele yürüyüşün
                # 25/25'inde bayrak "buluyordu" (çoklu deneme yanlılığı).
                pole_px = abs(float(close[p1]) - float(close[p0]))
                if pole_px < POLE_ATR_MULT * atr_v * math.sqrt(max(1, pole_len)):
                    continue
                seg_h, seg_l = highs[p1:pe + 1], lows[p1:pe + 1]
                seg_c = close[p1:pe + 1]
                if len(seg_c) < 5:
                    continue
                # SIKIŞMA DAR MI? — bayrak bir DURAKLAMADIR; gövdesi direğin
                # yanında küçük kalmalı. Bu olmadan herhangi bir yatay bölge
                # bayrak sayılıyor.
                cons_range = float(seg_h.max() - seg_l.min())
                if cons_range > pole_px * CONS_MAX_FRAC:
                    continue
                xs = list(range(p1, pe + 1))
                m_h, b_h, r2_h = _fit_line(xs, list(seg_h))
                m_l, b_l, r2_l = _fit_line(xs, list(seg_l))
                cons_move = _pct(seg_c[-1], seg_c[0])
                # sıkışma direğin karşı yönünde ve daha küçük olmalı
                if abs(cons_move) > abs(pole_move) * 0.6:
                    continue
                if pole_move > 0 and cons_move > 1.0:
                    continue
                if pole_move < 0 and cons_move < -1.0:
                    continue
                # kanal paralelliği (bayrak) veya daralma (flama)
                par = abs(_norm_slope(m_h - m_l, price, cons_len))
                is_pennant = par > 1.5
                quality = float(np.clip((r2_h + r2_l) / 2, 0, 1)) * \
                    min(1.0, abs(pole_move) / 8.0)
                if quality < 0.15:
                    continue

                pole_h = abs(close[p1] - close[p0])
                long_side = pole_move > 0
                bo = _line_at(m_h, b_h, pe) if long_side else _line_at(m_l, b_l, pe)
                target = bo + pole_h if long_side else bo - pole_h
                stop = _line_at(m_l, b_l, pe) if long_side else _line_at(m_h, b_h, pe)
                st = "kırılım" if ((long_side and price > bo) or
                                   (not long_side and price < bo)) else "oluşuyor"
                key = ("bull_pennant" if is_pennant else "bull_flag") if long_side else \
                      ("bear_pennant" if is_pennant else "bear_flag")
                # tazelik: aynı kalitede ise SON biten formasyon tercih edilir
                skor = quality - off * 0.004
                if key in best and best[key][0] >= skor:
                    continue
                best[key] = (skor, dict(
                    name=("Boğa Flaması" if is_pennant else "Boğa Bayrağı") if long_side
                         else ("Ayı Flaması" if is_pennant else "Ayı Bayrağı"),
                    long_side=long_side, st=st, cons_len=cons_len, quality=quality,
                    p0=p0, p1=p1, pe=pe, bo=bo, target=target, stop=stop,
                    pole_move=pole_move, pole_len=pole_len, is_pennant=is_pennant,
                    m_h=m_h, b_h=b_h, m_l=m_l, b_l=b_l, off=off))

    for key, (_, d) in best.items():
        p = _mk(key, d["name"], "devam", "LONG" if d["long_side"] else "SHORT", d["st"],
                min(0.95, d["cons_len"] / 20.0), d["quality"],
                "#0099FF" if d["long_side"] else "#FFB627",
                d["p0"], d["pe"], d["bo"], d["target"], d["stop"], price,
                note=(f"Direk %{d['pole_move']:+.1f} ({d['pole_len']} bar), ardından "
                      f"{d['cons_len']} barlık {'daralma' if d['is_pennant'] else 'kanal'}; "
                      f"hedef = kırılım + direk boyu."
                      + (f" Formasyon {d['off']} bar önce tamamlandı." if d["off"] else "")))
        p.lines = [
            PatternLine(d["p0"], float(close[d["p0"]]), d["p1"], float(close[d["p1"]]), "kanal"),
            PatternLine(d["p1"], _line_at(d["m_h"], d["b_h"], d["p1"]),
                        d["pe"], _line_at(d["m_h"], d["b_h"], d["pe"]), "direnç"),
            PatternLine(d["p1"], _line_at(d["m_l"], d["b_l"], d["p1"]),
                        d["pe"], _line_at(d["m_l"], d["b_l"], d["pe"]), "destek"),
        ]
        p.points = [PatternPoint(d["p0"], float(close[d["p0"]]), "Direk Başı"),
                    PatternPoint(d["p1"], float(close[d["p1"]]), "Direk Sonu")]
        out.append(p)
    return out


# ===========================================================================
# 4) FİNCAN VE KULP
# ===========================================================================
def detect_cup_handle(df: pd.DataFrame, lookback: int = 200) -> List[ChartPattern]:
    out: List[ChartPattern] = []
    n = len(df)
    if n < 60:
        return out
    close = df["close"].values
    highs, lows = df["high"].values, df["low"].values
    price = float(close[-1])
    last = n - 1
    start = max(0, n - lookback)
    atr_v = _touch_tol(df, price) * 2.0        # _touch_tol = 0,5×ATR

    # BİTİŞ UZAKLIĞI (`off`) — bayraktaki ile aynı düzeltme. Eskiden fincanın
    # SON BARA kadar sürmesi şart koşuluyordu; kırılım gerçekleşince kırılım
    # zirvesi `right_rim` oluyor, ±%6'lık kenar simetrisi kapısı düşüyor ve
    # parabol uyumu bozuluyordu. Ölçüldü: yalnız kuyruk=2'de çalışıyor,
    # kuyruk 0 / 4 / 8 / 14'te 0/12.
    # Ayrıca `cup_len` ızgarası 6 → 4 adıma incelildi: pencere gerçek fincan
    # başlangıcına denk gelmek zorunda olduğu için kaba ızgara şekli ıskalıyordu.
    best = None
    for off in (0, 2, 4, 7, 11, 16, 22):
        pe = last - off
        if pe < 50:
            continue
        for cup_len in range(30, min(160, pe - start + 1), 4):
            c0 = pe - cup_len
            if c0 < start:
                continue
            if len(close[c0:pe + 1]) < 20:
                continue
            bottom_i = int(np.argmin(lows[c0:pe + 1])) + c0
            bottom = float(lows[bottom_i])
            # KENARLAR dibin İKİ YANINDAKİ en yüksek noktalardır — eskiden sol
            # kenar `highs[c0:c0+5]` ile, yani pencerenin ilk 5 barından
            # ölçülüyordu. Pencere gerçek fincan başlangıcına tam oturmazsa kenar
            # fincanın yarısından okunuyor, ±%6'lık simetri kapısı sınırda kalıyor
            # ve tespit gürültüye bağlı hâle geliyordu (ölçüldü: 4/15 · 15/15 · 5/15
            # gibi kararsız sonuçlar). Bu tanım pencere hizasından bağımsızdır.
            if bottom_i <= c0 or bottom_i >= pe:
                continue
            left_rim = float(highs[c0:bottom_i].max())
            # dip ortada olmalı (U şekli, V değil)
            rel = (bottom_i - c0) / max(1, cup_len)
            if not (0.30 <= rel <= 0.70):
                continue
            depth = left_rim - bottom
            if depth <= 0:
                continue
            depth_pct = depth / left_rim * 100
            if not (8.0 <= depth_pct <= 55.0):
                continue
            # Derinlik varlığın kendi oynaklığına göre de ANLAMLI olmalı; yoksa
            # çalkantılı bir seride her salınım "fincan" sayılabiliyor.
            if depth < CUP_MIN_ATR_DEPTH * atr_v:
                continue
            right_seg = highs[bottom_i + 1:pe + 1]
            if len(right_seg) < 5:
                continue
            right_rim = float(right_seg.max())
            right_rim_i = int(np.argmax(right_seg)) + bottom_i + 1
            # sağ kenar sol kenara yakın olmalı (±%6)
            if abs(_pct(right_rim, left_rim)) > 6.0:
                continue
            # kulp: sağ kenardan sonra sığ geri çekilme
            handle = close[right_rim_i:pe + 1]
            handle_low = float(lows[right_rim_i:pe + 1].min()) if len(handle) else right_rim
            handle_depth = right_rim - handle_low
            if handle_depth > depth * 0.5:
                continue
            # KULP ZORUNLU: kulpsuz bir U yalnızca salınımdır. Bu kapı olmadan
            # rastgele seride her çukur "fincan-kulp" sayılıyordu.
            if handle_depth < depth * CUP_MIN_HANDLE:
                continue
            # U düzgünlüğü: parabol uyumu — YALNIZ FİNCAN bölümünde ölçülür.
            # Eskiden kulp ve kırılım barları da uydurmaya giriyordu; kulp
            # tanımı gereği U'dan sapar, dolayısıyla gerçek fincanlar bile
            # uyum eşiğini geçemiyordu (ölçüldü: kanonik denetimde 0/20).
            xs = np.arange(c0, right_rim_i + 1, dtype=float)
            ys = lows[c0:right_rim_i + 1].astype(float)
            if len(xs) < 20:
                continue
            try:
                coef = np.polyfit(xs, ys, 2)
                pred = np.polyval(coef, xs)
                ss_res = float(((ys - pred) ** 2).sum())
                ss_tot = float(((ys - ys.mean()) ** 2).sum())
                r2 = 1 - ss_res / (ss_tot + 1e-12)
            except Exception:
                continue
            if coef[0] <= 0 or r2 < CUP_MIN_R2:   # yukarı açık ve DÜZGÜN parabol şart
                continue
            # tazelik: aynı kalitede ise SON biten fincan tercih edilir
            q = float(np.clip(r2, 0, 1)) * min(1.0, cup_len / 60.0)
            skor = q - off * 0.004
            if best is None or skor > best[0]:
                best = (skor, c0, bottom_i, bottom, left_rim, right_rim, right_rim_i,
                        depth, handle_low, coef, pe, q, off)

    if not best:
        return out
    (_, c0, bottom_i, bottom, left_rim, right_rim, right_rim_i,
     depth, handle_low, coef, pe, q, off) = best
    bo = max(left_rim, right_rim)
    target = bo + depth
    stop = handle_low
    st = "kırılım" if price > bo else ("tamamlandı" if price > right_rim * 0.98 else "oluşuyor")
    p = _mk("cup_handle", "Fincan ve Kulp", "devam", "LONG", st,
            float(np.clip((pe - c0) / 120.0, 0.2, 0.98)), q, "#FFB627",
            c0, pe, bo, target, stop, price,
            note=(f"U tabanlı birikim (derinlik %{depth / left_rim * 100:.1f}) + sığ kulp; "
                  f"hedef = kenar + fincan derinliği."
                  + (f" Formasyon {off} bar önce tamamlandı." if off else "")))
    xs = np.arange(c0, pe + 1)
    p.curve = [{"x": int(x), "y": round(float(np.polyval(coef, x)), 8)}
               for x in xs[::max(1, len(xs) // 40)]]
    # Kenar çizgisi GÜNCEL bara kadar uzatılır (kırılım seviyesi görünür kalsın),
    # ama formasyonun kendisi `pe`'de biter.
    p.lines = [PatternLine(c0, left_rim, last, left_rim, "direnç", dashed=True)]
    p.points = [PatternPoint(c0, left_rim, "Sol Kenar"),
                PatternPoint(bottom_i, bottom, "Dip"),
                PatternPoint(right_rim_i, right_rim, "Sağ Kenar")]
    out.append(p)
    return out


# ===========================================================================
# 5) DİKDÖRTGEN / YATAY ARALIK
# ===========================================================================
def detect_rectangle(df: pd.DataFrame, sh: List[int], sl: List[int],
                     lookback: int = 120) -> List[ChartPattern]:
    out: List[ChartPattern] = []
    n = len(df)
    if n < 40:
        return out
    lo_i = max(0, n - lookback)
    hi_pool = [i for i in sh if i >= lo_i][-8:]
    lo_pool = [i for i in sl if i >= lo_i][-8:]
    if len(hi_pool) < 2 or len(lo_pool) < 2:
        return out
    highs, lows = df["high"].values, df["low"].values
    price = float(df["close"].iloc[-1])
    last = n - 1
    # Dikdörtgende kusur ORTALAMA ve SAPMA üzerinden işliyordu: aralığın ortasında
    # dönen bir salınım tepesi hem üst seviyeyi aşağı çekiyor hem sapmayı şişirip
    # "düz değil" dedirtiyordu. Sınıra DEĞEN pivotlar seçilerek ölçülür.
    tol = _touch_tol(df, price)
    _, _, _, hi_pts = _boundary_fit(hi_pool, [highs[i] for i in hi_pool], "upper", tol)
    _, _, _, lo_pts = _boundary_fit(lo_pool, [lows[i] for i in lo_pool], "lower", tol)
    # KAPSAMA ŞARTI — dikdörtgene özgü ve zorunlu. Yalnız "dokunanları seç" dersek
    # rastgele seride bile düz bir sınır uydurulabilir: ölçüldü, bu kapı olmadan
    # yanlış pozitif 0/25'ten 9/25'e fırladı. Dikdörtgenin TANIMI fiyatın iki
    # sınıra da TEKRAR TEKRAR gitmesidir; salınımların yarısı sınıra ulaşmıyorsa
    # bu bir aralık değil, başka bir şeydir.
    if (len(hi_pts) < max(3, (len(hi_pool) + 1) // 2) or
            len(lo_pts) < max(3, (len(lo_pool) + 1) // 2)):
        return out
    top = float(np.mean([highs[i] for i in hi_pts]))
    bot = float(np.mean([lows[i] for i in lo_pts]))
    if top <= bot:
        return out
    h = top - bot
    # düzlük: sapma yükseklikle kıyaslanınca küçük olmalı.
    # Artık DOKUNUŞLAR üzerinden ölçülüyor — doğru soru "aralığın sınırları düz mü",
    # "her salınım aynı yere mi gitti" değil.
    sd_t = float(np.std([highs[i] for i in hi_pts]))
    sd_b = float(np.std([lows[i] for i in lo_pts]))
    if (sd_t + sd_b) / 2 > h * 0.22:
        return out
    q = float(np.clip(1.0 - (sd_t + sd_b) / (2 * h * 0.22), 0, 1)) * \
        min(1.0, (len(hi_pts) + len(lo_pts)) / 6.0)
    long_side = abs(price - top) < abs(price - bot)
    bo = top if long_side else bot
    target = (top + h) if long_side else (bot - h)
    stop = bot if long_side else top
    st = "kırılım" if (price > top or price < bot) else "oluşuyor"
    start = min(hi_pts[0], lo_pts[0])
    p = _mk("rectangle", "Dikdörtgen (Yatay Aralık)", "devam",
            "LONG" if long_side else "SHORT", st,
            float(np.clip((last - start) / 80.0, 0.2, 0.95)), q, "#7C3AED",
            start, last, bo, target, stop, price,
            note="Yatay birikim; kırılım yönünde aralık yüksekliği kadar hedef.")
    p.lines = [PatternLine(start, top, last, top, "direnç"),
               PatternLine(start, bot, last, bot, "destek")]
    p.points = ([PatternPoint(i, float(highs[i]), "T") for i in hi_pts] +
                [PatternPoint(i, float(lows[i]), "D") for i in lo_pts])
    out.append(p)
    return out


# ===========================================================================
# 6) KANAL — yükselen / alçalan / yatay paralel kanal
# ===========================================================================
def detect_channel(df: pd.DataFrame, sh: List[int], sl: List[int],
                   lookback: int = 120) -> List[ChartPattern]:
    """Paralel trend kanalı: üst ve alt çizgi benzer eğimli.

    Üçgen/kama DARALIR, kanal PARALEL kalır — ayrı bir formasyondur ve
    piyasada en sık görülenlerdendir. İşlem mantığı: alt banttan al, üst banda
    sat (kanal içi) veya kırılımı takip et."""
    out: List[ChartPattern] = []
    n = len(df)
    if n < 40:
        return out
    lo_i = max(0, n - lookback)
    hi_pool = [i for i in sh if i >= lo_i][-8:]
    lo_pool = [i for i in sl if i >= lo_i][-8:]
    if len(hi_pool) < 2 or len(lo_pool) < 2:
        return out

    highs, lows = df["high"].values, df["low"].values
    price = float(df["close"].iloc[-1])
    last = n - 1
    # Kanalda kirlenme özellikle PARALELLİK testini bozar: kanala değmeyen bir
    # ara salınım tepesi üst çizgiyi içeri çeker, genişlik değişiyor görünür ve
    # `abs(h_now/h_start - 1) > 0.30` kapısı kanalı üçgen/kama sanıp reddeder.
    tol = _touch_tol(df, price)
    sh_m, sh_b, sh_r2, hi_pts = _boundary_fit(
        hi_pool, [highs[i] for i in hi_pool], "upper", tol)
    sl_m, sl_b, sl_r2, lo_pts = _boundary_fit(
        lo_pool, [lows[i] for i in lo_pool], "lower", tol)
    if len(hi_pts) < 2 or len(lo_pts) < 2:
        return out
    start = min(hi_pts[0], lo_pts[0])
    span = max(10, last - start)

    top_now = _line_at(sh_m, sh_b, last)
    bot_now = _line_at(sl_m, sl_b, last)
    h_start = _line_at(sh_m, sh_b, start) - _line_at(sl_m, sl_b, start)
    h_now = top_now - bot_now
    if h_now <= 0 or h_start <= 0:
        return out
    # PARALELLİK şartı: genişlik %30'dan fazla değişmemeli (değişirse üçgen/kama)
    if abs(h_now / h_start - 1.0) > 0.30:
        return out

    up_n = _norm_slope(sh_m, price, span)
    dn_n = _norm_slope(sl_m, price, span)
    if abs(up_n - dn_n) > 3.0:              # eğimler benzer olmalı
        return out

    slope = (up_n + dn_n) / 2
    quality = float(np.clip((sh_r2 + sl_r2) / 2, 0, 1)) * \
        min(1.0, (len(hi_pts) + len(lo_pts)) / 6.0)
    if quality < 0.2:
        return out

    # kanal içindeki konum: 0 = alt bant, 1 = üst bant
    pos = float(np.clip((price - bot_now) / (h_now + 1e-12), 0, 1))
    if slope > 1.5:
        key, name, color = "ascending_channel", "Yükselen Kanal", "#00FF88"
    elif slope < -1.5:
        key, name, color = "descending_channel", "Alçalan Kanal", "#FF3B5C"
    else:
        key, name, color = "horizontal_channel", "Yatay Kanal", "#0099FF"

    # Karar: alt banda yakınsa LONG (banda dönüş), üst banda yakınsa SHORT
    if pos <= 0.35:
        direction, target, stop = "LONG", top_now, bot_now - h_now * 0.3
    elif pos >= 0.65:
        direction, target, stop = "SHORT", bot_now, top_now + h_now * 0.3
    else:
        direction = "LONG" if slope > 0 else "SHORT"
        target = top_now if direction == "LONG" else bot_now
        stop = bot_now - h_now * 0.3 if direction == "LONG" else top_now + h_now * 0.3

    st = "kırılım" if (price > top_now or price < bot_now) else "oluşuyor"
    p = _mk(key, name, "devam", direction, st,
            float(np.clip(span / 100.0, 0.2, 0.95)), quality, color,
            start, last, top_now if direction == "LONG" else bot_now,
            target, stop, price,
            note=(f"Paralel kanal (eğim %{slope:+.1f}/pencere); fiyat bandın "
                  f"%{pos*100:.0f} seviyesinde. Bant içi işlem veya kırılım takibi."))
    p.lines = [
        PatternLine(start, _line_at(sh_m, sh_b, start), last, top_now, "direnç"),
        PatternLine(start, _line_at(sl_m, sl_b, start), last, bot_now, "destek"),
    ]
    p.points = ([PatternPoint(i, float(highs[i]), "T") for i in hi_pts[:3]] +
                [PatternPoint(i, float(lows[i]), "D") for i in lo_pts[:3]])
    out.append(p)
    return out


# ===========================================================================
# 7) OMUZ-BAŞ-OMUZ (+ ters) ve ÇİFT TEPE/DİP — çizilebilir sürüm
# ===========================================================================
def detect_hs_and_doubles(df: pd.DataFrame, sh: List[int], sl: List[int],
                          lookback: int = 160) -> List[ChartPattern]:
    out: List[ChartPattern] = []
    n = len(df)
    if n < 40:
        return out
    lo_i = max(0, n - lookback)
    H = [i for i in sh if i >= lo_i]
    L = [i for i in sl if i >= lo_i]
    highs, lows = df["high"].values, df["low"].values
    price = float(df["close"].iloc[-1])
    last = n - 1

    # ---- Omuz-Baş-Omuz (tepe) ----
    if len(H) >= 3:
        a, b, c = H[-3], H[-2], H[-1]
        ha, hb, hc = float(highs[a]), float(highs[b]), float(highs[c])
        if hb > ha and hb > hc and abs(_pct(hc, ha)) < 8.0:
            necks = [i for i in L if a < i < c]
            if len(necks) >= 2:
                n1, n2 = necks[0], necks[-1]
                neck = (float(lows[n1]) + float(lows[n2])) / 2
                height = hb - neck
                if height > 0:
                    q = float(np.clip(1 - abs(_pct(hc, ha)) / 8.0, 0, 1)) * \
                        min(1.0, height / (0.02 * price))
                    q = float(np.clip(q, 0, 1))
                    st = "kırılım" if price < neck else "tamamlandı"
                    p = _mk("head_shoulders", "Omuz-Baş-Omuz", "dönüş", "SHORT", st,
                            0.9 if price < neck else 0.75, q, "#A855F7",
                            a, last, neck, neck - height, hb, price,
                            note="Boyun çizgisi kırılırsa baş yüksekliği kadar düşüş hedeflenir.")
                    p.lines = [PatternLine(n1, neck, last, neck, "boyun", dashed=True),
                               PatternLine(a, ha, b, hb, "kanal"),
                               PatternLine(b, hb, c, hc, "kanal")]
                    p.points = [PatternPoint(a, ha, "Sol Omuz"),
                                PatternPoint(b, hb, "Baş"),
                                PatternPoint(c, hc, "Sağ Omuz"),
                                PatternPoint(n1, float(lows[n1]), "Boyun 1"),
                                PatternPoint(n2, float(lows[n2]), "Boyun 2")]
                    out.append(p)

    # ---- Ters Omuz-Baş-Omuz (dip) ----
    if len(L) >= 3:
        a, b, c = L[-3], L[-2], L[-1]
        la, lb, lc = float(lows[a]), float(lows[b]), float(lows[c])
        if lb < la and lb < lc and abs(_pct(lc, la)) < 8.0:
            necks = [i for i in H if a < i < c]
            if len(necks) >= 2:
                n1, n2 = necks[0], necks[-1]
                neck = (float(highs[n1]) + float(highs[n2])) / 2
                height = neck - lb
                if height > 0:
                    q = float(np.clip(1 - abs(_pct(lc, la)) / 8.0, 0, 1)) * \
                        min(1.0, height / (0.02 * price))
                    q = float(np.clip(q, 0, 1))
                    st = "kırılım" if price > neck else "tamamlandı"
                    p = _mk("inverse_head_shoulders", "Ters Omuz-Baş-Omuz", "dönüş",
                            "LONG", st, 0.9 if price > neck else 0.75, q, "#A855F7",
                            a, last, neck, neck + height, lb, price,
                            note="Boyun çizgisi yukarı kırılırsa baş derinliği kadar yükseliş hedeflenir.")
                    p.lines = [PatternLine(n1, neck, last, neck, "boyun", dashed=True),
                               PatternLine(a, la, b, lb, "kanal"),
                               PatternLine(b, lb, c, lc, "kanal")]
                    p.points = [PatternPoint(a, la, "Sol Omuz"),
                                PatternPoint(b, lb, "Baş"),
                                PatternPoint(c, lc, "Sağ Omuz"),
                                PatternPoint(n1, float(highs[n1]), "Boyun 1"),
                                PatternPoint(n2, float(highs[n2]), "Boyun 2")]
                    out.append(p)

    # ---- Çift Tepe / Çift Dip ----
    atr_v = _touch_tol(df, price) * 2.0        # _touch_tol = 0,5×ATR
    win_hi = highs[lo_i:last + 1]
    win_lo = lows[lo_i:last + 1]

    if len(H) >= 2:
        a, b = H[-2], H[-1]
        ha, hb = float(highs[a]), float(highs[b])
        if abs(_pct(hb, ha)) < 3.0 and (b - a) >= 5:
            mid = [i for i in L if a < i < b]
            if mid:
                # Boyun, tepeler arasındaki EN DÜŞÜK dip olmalı — eskiden
                # `lows[mid[0]]` ile ilk dip alınıyordu; birden fazla dip varsa
                # boyun yanlış yere konuyor ve derinlik olduğundan küçük çıkıyordu.
                neck_i = min(mid, key=lambda i: lows[i])
                neck = float(lows[neck_i])
                height = max(ha, hb) - neck
                peak = max(ha, hb)
                # (1) boğaz anlamlı bir geri çekilme mi?
                derin = (height >= DBL_MIN_DEPTH_ATR * atr_v and
                         height >= peak * DBL_MIN_DEPTH_PCT / 100.0)
                # (2) dönecek bir yükseliş var mıydı?
                p0 = max(0, a - DBL_PRIOR_BARS)
                taban = float(win_lo.min()) if a <= lo_i else float(lows[p0:a + 1].min())
                onculs = _pct(ha, taban)
                trendli = onculs >= DBL_MIN_PRIOR_ADV
                # (3) tepeler aralığın zirvesinde mi?
                zirvede = peak >= float(np.quantile(win_hi, DBL_RANGE_Q))
                if height > 0 and derin and trendli and zirvede:
                    q = float(np.clip(1 - abs(_pct(hb, ha)) / 3.0, 0, 1))
                    st = "kırılım" if price < neck else "tamamlandı"
                    p = _mk("double_top", "Çift Tepe", "dönüş", "SHORT", st,
                            0.85, q, "#FF3B5C", a, last, neck, neck - height,
                            max(ha, hb), price,
                            note=(f"İki eşit tepe; boyun altına kırılım düşüş sinyali. "
                                  f"Öncül yükseliş %{onculs:.0f}, boğaz derinliği "
                                  f"%{height / peak * 100:.1f}."))
                    p.lines = [PatternLine(neck_i, neck, last, neck, "boyun", dashed=True)]
                    p.points = [PatternPoint(a, ha, "Tepe 1"),
                                PatternPoint(b, hb, "Tepe 2"),
                                PatternPoint(neck_i, neck, "Boyun")]
                    out.append(p)

    if len(L) >= 2:
        a, b = L[-2], L[-1]
        la, lb = float(lows[a]), float(lows[b])
        if abs(_pct(lb, la)) < 3.0 and (b - a) >= 5:
            mid = [i for i in H if a < i < b]
            if mid:
                # boyun = dipler arasındaki EN YÜKSEK tepe (bkz. çift tepe notu)
                neck_i = max(mid, key=lambda i: highs[i])
                neck = float(highs[neck_i])
                dip = min(la, lb)
                height = neck - dip
                # aynı üç kapı, aynasal
                derin = (height >= DBL_MIN_DEPTH_ATR * atr_v and
                         height >= dip * DBL_MIN_DEPTH_PCT / 100.0)
                p0 = max(0, a - DBL_PRIOR_BARS)
                tavan = float(win_hi.max()) if a <= lo_i else float(highs[p0:a + 1].max())
                onculd = _pct(tavan, la)
                trendli = onculd >= DBL_MIN_PRIOR_ADV
                dipte = dip <= float(np.quantile(win_lo, 1.0 - DBL_RANGE_Q))
                if height > 0 and derin and trendli and dipte:
                    q = float(np.clip(1 - abs(_pct(lb, la)) / 3.0, 0, 1))
                    st = "kırılım" if price > neck else "tamamlandı"
                    p = _mk("double_bottom", "Çift Dip", "dönüş", "LONG", st,
                            0.85, q, "#00FF88", a, last, neck, neck + height,
                            dip, price,
                            note=(f"İki eşit dip; boyun üstüne kırılım yükseliş sinyali. "
                                  f"Öncül düşüş %{onculd:.0f}, boğaz yüksekliği "
                                  f"%{height / dip * 100:.1f}."))
                    p.lines = [PatternLine(neck_i, neck, last, neck, "boyun", dashed=True)]
                    p.points = [PatternPoint(a, la, "Dip 1"),
                                PatternPoint(b, lb, "Dip 2"),
                                PatternPoint(neck_i, neck, "Boyun")]
                    out.append(p)
    return out


# ===========================================================================
# ANA GİRİŞ
# ===========================================================================
def detect_chart_patterns(df: pd.DataFrame, top_n: int = 5,
                          min_move_pct: float = MIN_MOVE_PCT,
                          pivot_left: int = 3, pivot_right: int = 3) -> List[Dict]:
    """En yüksek skorlu `top_n` formasyonu ÇİZİM GEOMETRİSİYLE döndürür.

    min_move_pct: hedefi mevcut fiyattan bu yüzdeden yakın olan formasyonlar
    listelenir ama `clears_min_move=False` işaretlenir — kullanıcı %1'lik
    hareketi öngörüp kullanabilsin diye eşik açıkça raporlanır."""
    if df is None or len(df) < 40:
        return []
    need = {"open", "high", "low", "close"}
    if not need.issubset(df.columns):
        return []

    sh, sl = find_pivots(df, pivot_left, pivot_right)
    pats: List[ChartPattern] = []
    for fn in (lambda: detect_triangles(df, sh, sl),
               lambda: detect_wedges(df, sh, sl),
               lambda: detect_flags(df),
               lambda: detect_cup_handle(df),
               lambda: detect_rectangle(df, sh, sl),
               lambda: detect_channel(df, sh, sl),
               lambda: detect_hs_and_doubles(df, sh, sl)):
        try:
            pats.extend(fn())
        except Exception:
            continue

    # Aynı formasyon türünden en iyisini tut
    best: Dict[str, ChartPattern] = {}
    for p in pats:
        if p.key not in best or p.score > best[p.key].score:
            best[p.key] = p

    price = float(df["close"].iloc[-1])
    ranked = sorted(best.values(), key=lambda x: -x.score)[:top_n]
    out_d = []
    for p in ranked:
        _apply_validity(p, price)
        # ≥%1 rozeti YALNIZ hâlâ işlenebilir kurulumlara verilir
        p.clears_min_move = bool(p.valid and abs(p.target_pct) >= min_move_pct)
        d = p.to_dict()
        # Ölçülmüş kanıt varsa formasyona iliştir — kullanıcı, çizilen şeklin
        # sınanıp sınanmadığını ve sonucunu görmeden karar vermemeli.
        d["evidence"] = PATTERN_EVIDENCE.get(p.key)
        out_d.append(d)
    return out_d


def _apply_validity(p: ChartPattern, price: float) -> None:
    """Fiyat kurulumu geçersiz kıldı mı?

    Denetimde bulunan hata: fiyat stop'u ihlal etmiş ya da hedefi çoktan
    aşmış formasyonlar hâlâ "LONG/SHORT işlem" diye sunuluyordu. İkisi de
    kullanıcıyı yanıltır — birincisi anında zararla açtırır, ikincisi
    "hedef %-2" gibi yönüyle çelişen bir sayı gösterir."""
    # Hedef fiyata yapışıksa kurulum bitmiştir. Bu kapı olmadan rr, 2 basamağa
    # 0,00 yuvarlanıp "R/R ≤ 0" ihlali üretiyordu (denetimde 1 olay).
    if abs(p.target - price) / max(price, 1e-12) < 0.0005:
        p.valid, p.validity = False, "hedef_aşıldı"
        p.validity_note = (f"HEDEF ALINDI — fiyat ({price:.4g}) hedefe ({p.target:.4g}) "
                           f"yapıştı; kalan hareket %0,05'in altında.")
        return
    if p.direction == "LONG":
        if price <= p.stop:
            p.valid, p.validity = False, "stop_ihlal"
            p.validity_note = (f"GEÇERSİZ — fiyat ({price:.4g}) formasyonun iptal "
                               f"seviyesinin ({p.stop:.4g}) altına indi.")
            return
        if price >= p.target:
            p.valid, p.validity = False, "hedef_aşıldı"
            p.validity_note = (f"HEDEF ALINDI — fiyat ({price:.4g}) hedefi "
                               f"({p.target:.4g}) çoktan geçti; yeni giriş için geç.")
            return
    elif p.direction == "SHORT":
        if price >= p.stop:
            p.valid, p.validity = False, "stop_ihlal"
            p.validity_note = (f"GEÇERSİZ — fiyat ({price:.4g}) formasyonun iptal "
                               f"seviyesinin ({p.stop:.4g}) üstüne çıktı.")
            return
        if price <= p.target:
            p.valid, p.validity = False, "hedef_aşıldı"
            p.validity_note = (f"HEDEF ALINDI — fiyat ({price:.4g}) hedefi "
                               f"({p.target:.4g}) çoktan geçti; yeni giriş için geç.")
            return
    p.valid, p.validity = True, "işlenebilir"
    p.validity_note = ""


def trade_recommendation(patterns: List[Dict], price: float,
                         atr_pct: float = 0.0, tf: str = "4h") -> Dict:
    """"Grafiğe göre hangi yönde, yüzde kaçlık işlem daha optimal?"

    Yalnız HÂLÂ İŞLENEBİLİR formasyonlardan hesaplar. Çıktı: yön, hedef %,
    stop %, R/R, ulaşılabilirlik (ATR cinsinden kaç bar) ve gerekçe.

    ⚠️ Bu bir tavsiye DEĞİL, formasyonların geometrik çıkarımıdır. Formasyon
    ailesinin örneklem dışı yön öngörüsü bu projede ölçülmedi; ölçülen tek şey
    hedeflerin ARİTMETİK tutarlılığıdır. Boyutlama kullanıcıya aittir."""
    live = [p for p in patterns if p.get("valid", True)]
    # SINANMIŞ ve KANITSIZ aileler yön oyuna GİRMEZ. Ölçülmemiş bir formasyonu
    # göstermek ile ölçülüp çürütülmüş birine oy verdirmek aynı şey değildir:
    # çift tepe/dip'in yön öngörüsü olay çalışmasıyla sınandı ve bulunamadı
    # (bkz. PATTERN_EVIDENCE). Çizilmeye devam eder, karara katılmaz.
    kanitsiz = [p for p in live
                if (PATTERN_EVIDENCE.get(p.get("key", "")) or {}).get("edge") == "yok"]
    live = [p for p in live if p not in kanitsiz]
    if not live or price <= 0:
        if kanitsiz:
            return {"available": False,
                    "excluded_no_edge": [p["name"] for p in kanitsiz],
                    "reason": (f"{len(kanitsiz)} formasyon bulundu ama hepsinin yön "
                               f"öngörüsü ÖLÇÜLDÜ ve kanıtlanamadı; karara katılmıyorlar")}
        return {"available": False,
                "reason": ("işlenebilir formasyon yok — tespit edilenlerin hepsi "
                           "ya stop'u ihlal etmiş ya da hedefini aşmış")
                          if patterns else "formasyon tespit edilmedi"}

    lw = sum(p["score"] for p in live if p["direction"] == "LONG")
    sw = sum(p["score"] for p in live if p["direction"] == "SHORT")
    if lw <= 1e-9 and sw <= 1e-9:
        return {"available": False, "reason": "formasyon skorları sıfır"}

    side = "LONG" if lw > sw else "SHORT"
    same = [p for p in live if p["direction"] == side]
    opposite = [p for p in live if p["direction"] != side]
    w = sum(p["score"] for p in same)

    # Skor ağırlıklı hedef/stop — tek bir aykırı formasyon hedefi savurmasın
    tgt_pct = sum(p["target_pct"] * p["score"] for p in same) / w
    stp_pct = sum(p["stop_pct"] * p["score"] for p in same) / w
    rr = abs(tgt_pct) / (abs(stp_pct) + 1e-12)

    # ULAŞILABİLİRLİK: hedef, tipik bar hareketinin kaç katı?
    bars = (abs(tgt_pct) / atr_pct) if atr_pct > 0.01 else float("nan")
    if bars != bars:
        reach = "ATR ölçülemedi"
    elif bars <= 3:
        reach = f"~{bars:.1f} bar ({tf}) — yakın, normal dalgalanmayla ulaşılır"
    elif bars <= 10:
        reach = f"~{bars:.0f} bar ({tf}) — makul"
    else:
        reach = (f"~{bars:.0f} bar ({tf}) — UZAK; bu hedef tipik bar hareketinin "
                 f"{bars:.0f} katı, formasyon çok önce bozulabilir")

    # Güven: hemfikirlik × ortalama tamamlanma × kalite
    agree = w / (lw + sw)
    avg_comp = sum(p["completion"] * p["score"] for p in same) / w
    avg_q = sum(p["quality"] * p["score"] for p in same) / w
    conf = float(agree * (0.5 + 0.5 * avg_comp) * avg_q)

    if abs(tgt_pct) < MIN_MOVE_PCT:
        verdict = (f"BEKLE — hedef yalnız %{abs(tgt_pct):.2f}; %1'lik eşiğin altında, "
                   f"işlem ücreti ve kaymayı karşılamaz")
    elif rr < 1.0:
        verdict = (f"ZAYIF KURULUM — R/R {rr:.2f} (< 1). Riskin ödülden büyük; "
                   f"kazanma oranı %{100/(1+rr):.0f}'in üstünde olmalı ki başabaş gelsin")
    elif conf < 0.25:
        verdict = f"DÜŞÜK GÜVEN — formasyonlar zayıf/çelişkili (güven {conf:.2f})"
    else:
        verdict = (f"{side} yönünde %{abs(tgt_pct):.2f} hedef, %{abs(stp_pct):.2f} stop "
                   f"(R/R {rr:.2f})")

    return {
        "available": True,
        "direction": side,
        "target_pct": round(float(tgt_pct), 2),
        "stop_pct": round(float(stp_pct), 2),
        "rr": round(float(rr), 2),
        "confidence": round(conf, 3),
        "agreement": round(float(agree), 3),
        "n_supporting": len(same),
        "n_opposing": len(opposite),
        "n_invalid": len(patterns) - len(live),
        "reachability": reach,
        "bars_to_target": None if bars != bars else round(float(bars), 1),
        "atr_pct": round(float(atr_pct), 3),
        "verdict": verdict,
        "supporting": [{"name": p["name"], "target_pct": p["target_pct"],
                        "stop_pct": p["stop_pct"], "rr": p["rr"],
                        "score": p["score"], "status": p["status"]} for p in same],
        "breakeven_winrate": round(100.0 / (1.0 + rr), 1),
        # Öneriyi besleyen formasyonların kaçı SINANDI ve sonuç ne?
        "excluded_no_edge": [p["name"] for p in kanitsiz],
        "note": ("Hedef/stop formasyon geometrisinden gelir; yön öngörüsünün "
                 "örneklem dışı geçerliliği çoğu aile için ÖLÇÜLMEDİ. Başabaş "
                 "kazanma oranı sütunu, kurulumun matematiksel olarak ne kadar "
                 "isabet gerektirdiğini gösterir."),
    }


def pattern_consensus(patterns: List[Dict]) -> Dict:
    """Formasyonların ortak yönü — meta-etiketleme özelliği ve panel rozeti.

    Yalnız geçerli (valid) formasyonlar sayılır.
    Tek başına işlem açtırmaz (bkz. modül başlığındaki dürüstlük notu)."""
    patterns = [p for p in patterns if p.get("valid", True)]
    if not patterns:
        return {"bias": "NÖTR", "score": 0.0, "long": 0, "short": 0, "n": 0}
    lw = sum(p["score"] for p in patterns if p["direction"] == "LONG")
    sw = sum(p["score"] for p in patterns if p["direction"] == "SHORT")
    tot = lw + sw
    if tot <= 1e-9:
        return {"bias": "NÖTR", "score": 0.0, "long": 0, "short": 0, "n": len(patterns)}
    net = (lw - sw) / tot
    bias = "YUKARI" if net > 0.2 else ("AŞAĞI" if net < -0.2 else "NÖTR")
    return {"bias": bias, "score": round(float(net), 3),
            "long": sum(1 for p in patterns if p["direction"] == "LONG"),
            "short": sum(1 for p in patterns if p["direction"] == "SHORT"),
            "n": len(patterns),
            "actionable_1pct": sum(1 for p in patterns if p.get("clears_min_move"))}
