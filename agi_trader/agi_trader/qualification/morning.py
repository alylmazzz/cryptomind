"""SABAH FIRSAT MOTORU — 2. mesaj 24–33, 43–53.

GÖREV NE DEĞİL
"Sabah mutlaka bir coin seçmek" DEĞİLDİR. Görev, sabahın veriyle doğrulanmış
bölümünde bütün piyasayı ve bütün ufukları tarayıp gerçekten qualified olan
kombinasyonları tek haritada sunmaktır. Fırsat yoksa "NO QUALIFIED NET +%1
OPPORTUNITY" çıktısı sistemin DOĞRU çalıştığı anlamına gelir.

SABİT 08:00 YOK
Rapor saati elle seçilmez; 15 dakikalık slotlar ölçülür ve en iyisi öğrenilir.
Ama iki tuzak vardır ve ikisi de burada kapatılmıştır:

  1. AŞIRI UYUM — "geçen hafta 07:45 iyiydi" bir slot seçimi için yeterli
     değildir. Slot ancak 30 gün / 90 gün / tam OOS pencerelerinin HEPSİNDE
     tutarlıysa aday olur (`consistent` alanı).
  2. SLOT ZIPLAMASI — yeni slot, mevcut slotu yalnız İSTATİSTİKSEL OLARAK
     ANLAMLI biçimde geçiyorsa devralır (`MIN_IMPROVEMENT` + örneklem şartı).
     Aksi hâlde saat sabit kalır; günlük gürültüyle ileri geri sıçramaz.

READINESS BİR OLASILIK DEĞİLDİR
`MorningReadiness` operasyonel bir hazırlık skorudur: kaç qualified fırsat
var, defter ne kadar sağlıklı, sinyalin ömrü yeterli mi. "%87 kâr ihtimali"
diye okunamaz ve UI'da bu adla GÖSTERİLMEZ (2. mesaj 52).
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_TZ = "Europe/Istanbul"
WINDOW_START = (5, 0)          # 05:00 yerel
WINDOW_END = (11, 0)           # 11:00 yerel
SLOT_MINUTES = 15
SCAN_EVERY_MINUTES = 5

# Slot devri için asgari iyileşme (Net1PercentPrecision puanı) ve örneklem
MIN_IMPROVEMENT = 0.05
MIN_SLOT_SAMPLES = 30
WINDOWS = (30, 90, None)       # gün; None = tam OOS


def _tz(name: str):
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(name)
    except Exception:
        return dt.timezone.utc


def slots() -> List[str]:
    """05:00 … 10:45 arası 15 dakikalık slot etiketleri."""
    out: List[str] = []
    t = dt.datetime(2000, 1, 1, WINDOW_START[0], WINDOW_START[1])
    son = dt.datetime(2000, 1, 1, WINDOW_END[0], WINDOW_END[1])
    while t < son:
        out.append(t.strftime("%H:%M"))
        t += dt.timedelta(minutes=SLOT_MINUTES)
    return out


def slot_of(ts: dt.datetime, tz: str = DEFAULT_TZ) -> Optional[str]:
    """Bir zaman damgasının hangi sabah slotuna düştüğü (dışındaysa None)."""
    yerel = ts.astimezone(_tz(tz))
    dk = yerel.hour * 60 + yerel.minute
    bas = WINDOW_START[0] * 60 + WINDOW_START[1]
    son = WINDOW_END[0] * 60 + WINDOW_END[1]
    if not (bas <= dk < son):
        return None
    k = bas + ((dk - bas) // SLOT_MINUTES) * SLOT_MINUTES
    return f"{k // 60:02d}:{k % 60:02d}"


def in_window(ts: dt.datetime, tz: str = DEFAULT_TZ) -> bool:
    return slot_of(ts, tz) is not None


# ── slot öğrenimi (2. mesaj 25, 26, 44, 45) ────────────────────────────────

@dataclass
class SlotScore:
    slot: str
    n: int
    qualified_freq: Optional[float]
    net1_precision: Optional[float]
    robust_ev_mean: Optional[float]
    false_opportunity_rate: Optional[float]
    spread_bps_median: Optional[float]
    execution_probability: Optional[float]
    calibration_error: Optional[float]
    score: Optional[float]

    def to_dict(self) -> Dict:
        return asdict(self)


def _slot_metrics(kayitlar: Sequence[Dict], sonuclar: Dict[str, Dict],
                  slot: str, tz: str) -> SlotScore:
    n = tp = yay = 0
    ev_top = 0.0
    hata: List[float] = []
    spreadler: List[float] = []
    for p in kayitlar:
        try:
            t = dt.datetime.fromisoformat(p["timestamp"].replace("Z", "+00:00"))
        except Exception:
            continue
        if slot_of(t, tz) != slot:
            continue
        yay += 1
        if p.get("robust_ev") is not None:
            ev_top += float(p["robust_ev"])
        if p.get("spread_bps") is not None:
            spreadler.append(float(p["spread_bps"]))
        o = sonuclar.get(p["prediction_id"])
        if not o or o["outcome"] not in ("TP_FIRST", "SL_FIRST", "TIMEOUT"):
            continue
        n += 1
        if o["outcome"] == "TP_FIRST":
            tp += 1
        if o.get("prediction_error") is not None:
            hata.append(abs(float(o["prediction_error"])))
    if yay == 0:
        return SlotScore(slot, 0, None, None, None, None, None, None, None, None)
    prec = (tp / yay) if yay else None
    fo = ((yay - tp) / yay) if yay else None
    kal = float(np.mean(hata)) if hata else None
    ev = (ev_top / yay) if yay else None
    # Skor: hepsi ölçülmüş bileşenlerden; eksik bileşen 0 sayılır (varsayım yok)
    s = (0.45 * (prec or 0.0)
         + 0.30 * max(0.0, min(1.0, (ev or 0.0) / 0.5))
         + 0.15 * (1.0 - min(1.0, kal if kal is not None else 1.0))
         + 0.10 * min(1.0, yay / 20.0))
    return SlotScore(slot, n, qualified_freq=yay / max(1, len(kayitlar)),
                     net1_precision=prec, robust_ev_mean=ev,
                     false_opportunity_rate=fo,
                     spread_bps_median=(float(np.median(spreadler))
                                        if spreadler else None),
                     execution_probability=None,
                     calibration_error=kal, score=round(s, 4))


def learn_slots(led, tz: str = DEFAULT_TZ) -> Dict:
    """Her slotu 30g / 90g / tam OOS pencerelerinde AYRI ölç (2. mesaj 26).

    Rastgele bölme kullanılmaz; pencereler kronolojiktir. Bir slot ancak
    ÜÇ pencerede de üst yarıdaysa `consistent` sayılır — "son birkaç günde
    iyiydi" tek başına yeterli değildir.
    """
    tum = led.predictions()
    son = led.outcomes()
    simdi = dt.datetime.now(dt.timezone.utc)
    pencere_sonuc: Dict[str, List[Dict]] = {}
    for gun in WINDOWS:
        ad = "full" if gun is None else f"{gun}d"
        if gun is None:
            alt = tum
        else:
            kesim = simdi - dt.timedelta(days=gun)
            alt = []
            for p in tum:
                try:
                    if dt.datetime.fromisoformat(
                            p["timestamp"].replace("Z", "+00:00")) >= kesim:
                        alt.append(p)
                except Exception:
                    pass
        pencere_sonuc[ad] = [_slot_metrics(alt, son, s, tz).to_dict()
                             for s in slots()]

    # tutarlılık: üç pencerede de skor medyanın üstünde mi
    tutarli: Dict[str, bool] = {}
    for s in slots():
        bayrak = True
        for ad, satirlar in pencere_sonuc.items():
            skorlar = [r["score"] for r in satirlar if r["score"] is not None]
            bu = next((r["score"] for r in satirlar if r["slot"] == s), None)
            if not skorlar or bu is None or bu < float(np.median(skorlar)):
                bayrak = False
        tutarli[s] = bayrak

    yeterli = {s: all(
        next((r["n"] for r in pencere_sonuc[ad] if r["slot"] == s), 0)
        >= (MIN_SLOT_SAMPLES if ad == "full" else 5)
        for ad in pencere_sonuc) for s in slots()}

    adaylar = [s for s in slots() if tutarli[s] and yeterli[s]]
    en_iyi = None
    if adaylar:
        en_iyi = max(adaylar, key=lambda s: next(
            (r["score"] or 0.0) for r in pencere_sonuc["full"] if r["slot"] == s))
    return {"timezone": tz, "window": f"{WINDOW_START[0]:02d}:00–{WINDOW_END[0]:02d}:00",
            "slot_minutes": SLOT_MINUTES, "windows": pencere_sonuc,
            "consistent": tutarli, "sufficient_samples": yeterli,
            "best_slot": en_iyi,
            "note": ("Slot ancak 30g/90g/tam OOS pencerelerinin ÜÇÜNDE de üst "
                     "yarıda ve yeterli örneklemliyse aday olur.")}


def should_switch_slot(current: Optional[str], learned: Dict) -> Tuple[Optional[str], str]:
    """Slot devri — gürültüyle sıçramayı engelleyen kural (2. mesaj 45)."""
    aday = learned.get("best_slot")
    if aday is None:
        return current, "aday slot yok — mevcut saat korunuyor"
    if current is None:
        return aday, f"ilk slot ataması: {aday}"
    tam = {r["slot"]: r for r in learned["windows"]["full"]}
    a, m = tam.get(aday, {}), tam.get(current, {})
    if (a.get("n") or 0) < MIN_SLOT_SAMPLES:
        return current, (f"{aday} örneklemi yetersiz "
                         f"({a.get('n')} < {MIN_SLOT_SAMPLES}) — devir yok")
    fark = (a.get("score") or 0.0) - (m.get("score") or 0.0)
    if fark < MIN_IMPROVEMENT:
        return current, (f"{aday} iyileşmesi {fark:+.3f} < {MIN_IMPROVEMENT} — "
                         f"anlamlı değil, {current} korunuyor")
    return aday, f"{current} → {aday} (iyileşme {fark:+.3f})"


# ── canlı tetik (2. mesaj 27, 28, 29, 30) ──────────────────────────────────

@dataclass
class Readiness:
    """OPERASYONEL hazırlık skoru — kâr olasılığı DEĞİL."""
    score: float
    qualified_count: int
    best_robust_ev: Optional[float]
    best_lower95: Optional[float]
    data_quality: Optional[float]
    liquidity: Optional[float]
    signal_lifetime_sec: Optional[float]
    components: Dict = field(default_factory=dict)
    label: str = "MORNING OPPORTUNITY READINESS"
    disclaimer: str = ("Bu bir kâr olasılığı değildir; taramanın rapor "
                       "yayımlamaya hazır olup olmadığını ölçer.")

    def to_dict(self) -> Dict:
        return asdict(self)


READINESS_THRESHOLD = 0.60


def readiness(cards: Sequence[Dict], data_quality: Optional[float] = None,
              liquidity: Optional[float] = None) -> Readiness:
    kalifiye = [k for k in cards if k.get("best_horizon")]
    rev = [k.get("robust_expected_value") for k in kalifiye
           if k.get("robust_expected_value") is not None]
    lo = [k.get("p_target_first_lower95") for k in kalifiye
          if k.get("p_target_first_lower95") is not None]
    omur = [k.get("entry_valid_seconds") for k in kalifiye
            if k.get("entry_valid_seconds") is not None]
    bilesen = {
        "qualified": min(1.0, len(kalifiye) / 3.0),
        "robust_ev": min(1.0, max(0.0, (max(rev) if rev else 0.0) / 0.5)),
        "lower95": (max(lo) if lo else 0.0),
        "data_quality": (data_quality if data_quality is not None else 0.0),
        "liquidity": (liquidity if liquidity is not None else 0.0),
        "lifetime": min(1.0, (max(omur) if omur else 0.0) / 600.0),
    }
    agirlik = {"qualified": 0.30, "robust_ev": 0.25, "lower95": 0.20,
               "data_quality": 0.10, "liquidity": 0.10, "lifetime": 0.05}
    s = sum(bilesen[k] * agirlik[k] for k in bilesen)
    return Readiness(round(float(s), 4), len(kalifiye),
                     (max(rev) if rev else None), (max(lo) if lo else None),
                     data_quality, liquidity,
                     (max(omur) if omur else None),
                     {k: round(v, 4) for k, v in bilesen.items()})


def should_publish(cards: Sequence[Dict], rdy: Readiness, now_local: dt.datetime,
                   already_published: bool,
                   threshold: float = READINESS_THRESHOLD) -> Tuple[bool, str]:
    """Rapor şimdi yayımlanmalı mı? (2. mesaj 28, 29, 30)

    05:05'te çıkan ZAYIF bir fırsat sabah raporunu TÜKETMEZ; sistem ilk
    gördüğü fırsatı değil ilk GÜÇLÜ qualified fırsatı bekler. Pencere
    sonunda hâlâ yoksa "fırsat yok" raporu gönderilir — zorla seçim yapılmaz.
    """
    if already_published:
        return False, "bugünün raporu zaten yayımlandı"
    son = now_local.replace(hour=WINDOW_END[0], minute=WINDOW_END[1],
                            second=0, microsecond=0)
    kalifiye = [k for k in cards if k.get("best_horizon")]
    if now_local >= son:
        return True, ("pencere kapandı — " +
                      ("qualified fırsat var" if kalifiye
                       else "NO QUALIFIED NET +%1 OPPORTUNITY"))
    if not kalifiye:
        return False, "henüz qualified fırsat yok — bekleniyor"
    if rdy.score < threshold:
        return False, (f"hazırlık {rdy.score:.2f} < {threshold} — zayıf fırsat "
                       f"sabah raporunu tüketmez, güçlüsü bekleniyor")
    return True, f"qualified fırsat + hazırlık {rdy.score:.2f} ≥ {threshold}"


# ── rapor (2. mesaj 31, 32, 46, 47, 48, 49) ────────────────────────────────

def nearest_to_qualification(cards: Sequence[Dict], top: int = 5) -> List[Dict]:
    """Fırsat yokken 'neden bekliyoruz' sorusunun cevabı (2. mesaj 49).

    ⚠️ Bunlar İŞLEM ÖNERİSİ DEĞİLDİR ve öyle etiketlenmez; yalnız hangi
    kapının ne kadar farkla kaçırıldığını gösterir."""
    out: List[Dict] = []
    for k in cards:
        for h in k.get("horizons", []):
            if h.get("tradable"):
                continue
            eksik: List[str] = []
            if h.get("robust_ev") is not None and h["robust_ev"] <= 0:
                eksik.append(f"Robust EV {abs(h['robust_ev']):.3f} puan eksik")
            if (h.get("actual_lift") is not None
                    and h.get("required_lift") is not None
                    and h["actual_lift"] <= h["required_lift"]):
                eksik.append(f"lift {(h['required_lift']-h['actual_lift'])*100:.1f} "
                             f"puan eksik")
            if not eksik:
                eksik = [x for x in (h.get("rejection_reasons_tr") or [])][:2]
            if not eksik:
                continue
            out.append({
                "symbol": k["symbol"], "horizon": h["horizon"],
                "direction": h["direction"], "status": h["status"],
                "missing": eksik,
                "gap_score": abs(h.get("robust_ev") or 1.0),
                "not_a_trade": True,
            })
    return sorted(out, key=lambda r: r["gap_score"])[:top]


def morning_performance(led, tz: str = DEFAULT_TZ) -> Dict:
    """2. mesaj 43 — sabah algoritmasının KENDİ karnesi.

    Genel karneden AYRI tutulur: sabah penceresinde yayımlanan sinyaller
    farklı bir seçim sürecinden geçer (hazırlık eşiği + ilk-güçlü kuralı).
    Performans ayrıca TETİK SAATİ bazında kırılır ki hangi slotun gerçekten
    işe yaradığı ölçülebilsin — varsayılmasın."""
    son = led.outcomes()
    genel = {"published": 0, "tp_first": 0, "sl_first": 0, "timeout": 0,
             "not_traded": 0, "net_sum": 0.0, "ev_sum": 0.0}
    saatler: Dict[str, Dict] = {}
    for p in led.predictions():
        try:
            t = dt.datetime.fromisoformat(p["timestamp"].replace("Z", "+00:00"))
        except Exception:
            continue
        slot = slot_of(t, tz)
        if slot is None:
            continue                                # sabah penceresi dışında
        g = saatler.setdefault(slot, {
            "slot": slot, "published": 0, "tp_first": 0, "sl_first": 0,
            "timeout": 0, "not_traded": 0, "net_sum": 0.0, "ev_sum": 0.0})
        for d in (genel, g):
            d["published"] += 1
            if p.get("robust_ev") is not None:
                d["ev_sum"] += float(p["robust_ev"])
        o = son.get(p["prediction_id"])
        if not o:
            continue
        anahtar = {"TP_FIRST": "tp_first", "SL_FIRST": "sl_first",
                   "TIMEOUT": "timeout"}.get(o["outcome"], "not_traded")
        for d in (genel, g):
            d[anahtar] += 1
            if o.get("realized_net_pct") is not None:
                d["net_sum"] += float(o["realized_net_pct"])

    def _ozet(d: Dict) -> Dict:
        yay = d["published"]
        islem = d["tp_first"] + d["sl_first"] + d["timeout"]
        return {**{k: d[k] for k in ("published", "tp_first", "sl_first",
                                     "timeout", "not_traded")},
                "net1_precision": (d["tp_first"] / yay) if yay else None,
                "false_opportunity_rate": (((yay - d["tp_first"]) / yay)
                                           if yay else None),
                "avg_expected_ev": (d["ev_sum"] / yay) if yay else None,
                "realized_net_mean": (d["net_sum"] / islem) if islem else None}

    return {"timezone": tz, "window": f"{WINDOW_START[0]:02d}:00–"
                                      f"{WINDOW_END[0]:02d}:00",
            "overall": _ozet(genel),
            "by_trigger_hour": [{"slot": k, **_ozet(v)}
                                for k, v in sorted(saatler.items())],
            "note": ("Payda YAYIMLANAN sabah sinyalidir; başarısızlar "
                     "düşürülmez. Tetik saati kırılımı, slot seçiminin "
                     "gerekçesini görünür kılar.")}


def build_report(cards: Sequence[Dict], scanner: Dict, rdy: Readiness,
                 learned: Optional[Dict], trigger_time: Optional[str],
                 tz: str = DEFAULT_TZ, combos: Optional[int] = None) -> Dict:
    """Sabah raporu. Fırsat olsa da olmasa da AYNI şablonda gelir."""
    kalifiye = [k for k in cards if k.get("best_horizon")]
    yuksek = [k for k in kalifiye if k.get("status") == "HIGH_CONFIDENCE"]
    simdi = dt.datetime.now(_tz(tz))
    return {
        "title": "CRYPTOMIND — MORNING NET +%1 MAP",
        "date": simdi.strftime("%Y-%m-%d"),
        "generated": simdi.strftime("%H:%M"),
        "timezone": tz,
        "guarantee": "NONE",
        "guarantee_line": "GARANTİ: YOK — piyasa sonucu kesin değildir.",
        "morning_window": f"{WINDOW_START[0]:02d}:00–{WINDOW_END[0]:02d}:00",
        "best_historical_slot": (learned or {}).get("best_slot"),
        "live_trigger": trigger_time,
        "markets_scanned": scanner.get("markets_scanned"),
        "combinations": combos,
        "qualified": len(kalifiye),
        "high_confidence": len(yuksek),
        "readiness": rdy.to_dict(),
        "opportunities": [
            {"rank": i + 1, **{k: c.get(k) for k in (
                "symbol", "direction", "best_horizon",
                "earliest_qualified_horizon", "entry_low", "optimal_entry",
                "entry_high", "max_chase_price", "net_1pct_exit", "stop",
                "p_target_first", "p_target_first_lower95", "p_stop_first",
                "p_timeout", "expected_target_time_hours",
                "robust_expected_value", "fill_probability",
                "execution_probability", "cost_model", "max_capacity_usd",
                "status", "valid_until", "why_this_horizon")}}
            for i, c in enumerate(kalifiye[:5])],
        "empty_result": (None if kalifiye else
                         "NO QUALIFIED NET +%1 OPPORTUNITY"),
        "nearest_to_qualification": (nearest_to_qualification(cards)
                                     if not kalifiye else []),
        "closing_note": ("Fırsat yoksa bu sonuç sistemin doğru çalıştığı "
                         "anlamına gelir; düşük kaliteli alternatif üretilmez."),
    }
