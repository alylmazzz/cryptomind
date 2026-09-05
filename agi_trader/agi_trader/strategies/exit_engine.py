"""
ÇIKIŞ MOTORU — üç mod, tek hard stop, NET yarı-tepe kâr koruması, ATR/chandelier
trailing, zaman-stop (sleeve başına), edge-decay ve model çıkışı; peak-capture KPI.

Sıra (her fiyat güncellemesinde):
  1. HARD STOP        — hiçbir mod kaldıramaz; asgari tutmaya bakmaz
  2. asgari tutma     — 2+ için kapı (video: 0-15 dk zarar bölgesi)
  3. mod:
     FIXED_TARGET     — hedefte çık
     PARTIAL_AND_RUN  — kısmi kâr (koşucu yapar) → stop breakeven/yapısal; kalan: trailing + yarı-tepe
     DYNAMIC_PEAK     — sabit hedef yok; tepe takibi, yarı-tepe geri-verme, chandelier, devam olasılığı
  4. EDGE_DECAY       — komitenin güncel EV'si ≤ 0 ve devam kanıtı yoksa
  5. TIME_STOP        — sleeve/rejim başına süre

Yarı-tepe (half-peak giveback) NET üzerinden: net_gain = brüt − (giriş ücreti + tahmini
çıkış maliyeti). Silahlanma eşiği = max(maliyet × 3, 0,5 ATR, asgari MFE %0,30). Silahlandıktan
sonra net_gain ≤ tepe_net × retain (varsayılan 0,50; 0,35-0,70 sınırı) → PEAK_GIVEBACK_EXIT.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

FIXED_TARGET, PARTIAL_AND_RUN, DYNAMIC_PEAK = "FIXED_TARGET", "PARTIAL_AND_RUN", "DYNAMIC_PEAK"
MODES = (FIXED_TARGET, PARTIAL_AND_RUN, DYNAMIC_PEAK)


@dataclass
class ExitParams:
    retain_fraction: float = 0.50        # tepe NET kârın korunacak oranı (0,35–0,70)
    arm_cost_multiple: float = 2.0       # (3,0 idi) canlı: %0,2–0,35 net tepeler hiç silahlanmadı → zaman-stopunda eksiye döndü
    arm_min_mfe_pct: float = 0.20
    # KAZANCIN KAYBA DÖNMEMESİ: tepe net ≥ be_lock_cost_mult×maliyet olunca stop başabaş+maliyete çekilir (her modda)
    be_lock_cost_mult: float = 1.5
    # ZAMAN KİLİDİ: ufkun %60'ı geçince kârdaysa stop başabaş+maliyete (zaman-stopu eksiye düşürmesin)
    time_lock_frac: float = 0.6
    # ERKEN İPTAL: ufkun ilk %50'sinde MAE ≥ 0,6×stop ve tepe net ≤ 0 → kes (30 kazananın 0'ı bu kadar aleyhe gitti)
    early_abort_mae_frac: float = 0.6
    early_abort_window_frac: float = 0.5
    arm_atr_mult: float = 0.5
    chandelier_k: float = 2.5
    min_hold_sec: int = 900
    time_stop_sec: int = 3600
    edge_decay_enabled: bool = True
    model_exit_cont_prob: float = 0.35    # devam olasılığı bunun altına düşerse (DYNAMIC_PEAK)

    def validated(self) -> "ExitParams":
        self.retain_fraction = float(min(0.70, max(0.35, self.retain_fraction)))
        self.arm_cost_multiple = float(min(10.0, max(1.0, self.arm_cost_multiple)))
        self.arm_min_mfe_pct = float(min(3.0, max(0.05, self.arm_min_mfe_pct)))
        self.chandelier_k = float(min(6.0, max(1.0, self.chandelier_k)))
        self.be_lock_cost_mult = float(min(5.0, max(1.0, self.be_lock_cost_mult)))
        self.time_lock_frac = float(min(0.95, max(0.3, self.time_lock_frac)))
        self.early_abort_mae_frac = float(min(0.95, max(0.3, self.early_abort_mae_frac)))
        self.early_abort_window_frac = float(min(0.9, max(0.1, self.early_abort_window_frac)))
        return self


@dataclass
class PositionTrack:
    direction: str
    entry: float
    hard_stop: float
    target: Optional[float]
    opened_ts: float
    mode: str = PARTIAL_AND_RUN
    stop_pct: float = 1.0
    cost_pct_roundtrip: float = 0.2       # giriş + tahmini çıkış (NET için)
    atr_pct: float = 0.3
    highest_high: float = 0.0
    lowest_low: float = 0.0
    peak_gross_pct: float = 0.0
    peak_net_pct: float = 0.0
    armed: bool = False
    partial_done: bool = False
    trail_stop: Optional[float] = None
    be_locked: bool = False
    notes: list = field(default_factory=list)

    def sign(self) -> float:
        return 1.0 if self.direction == "LONG" else -1.0

    def gross_pct(self, price: float) -> float:
        return (price / self.entry - 1.0) * 100.0 * self.sign()

    def net_pct(self, price: float) -> float:
        return self.gross_pct(price) - self.cost_pct_roundtrip

    def mae_pct(self) -> float:
        """Girişten bu yana en kötü aleyhe hareket (%)."""
        if self.direction == "LONG":
            return max(0.0, (self.entry - (self.lowest_low or self.entry)) / self.entry * 100.0)
        return max(0.0, ((self.highest_high or self.entry) - self.entry) / self.entry * 100.0)

    def breakeven_plus(self) -> float:
        """Net sıfır çıkış fiyatı (maliyet dahil)."""
        return self.entry * (1.0 + self.sign() * self.cost_pct_roundtrip / 100.0)

    def giveback_level(self, p: ExitParams) -> Optional[float]:
        """Yarı-tepe çıkış fiyatı (UI: 'EXIT IF ≤ X'). Silahlanmadıysa None."""
        if not self.armed or self.peak_net_pct <= 0:
            return None
        keep_net = self.peak_net_pct * p.retain_fraction
        gross_needed = keep_net + self.cost_pct_roundtrip
        return self.entry * (1.0 + self.sign() * gross_needed / 100.0)


def arm_threshold_pct(track: PositionTrack, p: ExitParams) -> float:
    return max(track.cost_pct_roundtrip * p.arm_cost_multiple, p.arm_min_mfe_pct,
               p.arm_atr_mult * (track.atr_pct or 0.0))


def chandelier_stop(track: PositionTrack, p: ExitParams) -> Optional[float]:
    atr = track.atr_pct / 100.0 * track.entry
    if atr <= 0:
        return None
    if track.direction == "LONG":
        return track.highest_high - p.chandelier_k * atr if track.highest_high else None
    return track.lowest_low + p.chandelier_k * atr if track.lowest_low else None


def _stop_touched(level: float, bar_high: float, bar_low: float, s: float) -> bool:
    """Koruyucu seviye BAR İÇİNDE delindi mi?

    2026-09-05 ÖLÇÜMÜ: sert stop yalnız `price` (kapanış) ile karşılaştırılıyordu; oysa
    aynı fonksiyon `bar_low`/`bar_high`'ı zaten alıp MAE ve chandelier için kullanıyordu.
    Sonuç asimetriydi — ZARAR bar uçlarıyla ölçülüyor, STOP kapanışla kontrol ediliyordu:
    bar içinde stop delinip kapanış stopun üstünde biterse pozisyon YAŞAMAYA DEVAM ediyordu.
    Kanıt: 11 EARLY_ABORT işleminin hepsinde MAE %0,47–0,94 iken stop ~%0,52 idi; yani
    hepsi stop mesafesini aşmıştı ve ortalama −%0,77'de (STOP'un −%0,25'inin üç katı)
    kapandı. Risk modeli bu hâlde gerçeğin altını gösteriyordu — kaldıraç eklenecekse
    önce bu düzeltilmeliydi."""
    return (bar_low <= level) if s > 0 else (bar_high >= level)


def _stop_fill(level: float, bar_high: float, bar_low: float, s: float) -> float:
    """Stop dolum fiyatı: seviye bar aralığındaysa SEVİYE; bar tamamen seviyenin
    ötesine geçtiyse (boşluk) bardaki EN KÖTÜ fiyat. Boşlukta iyimser dolum
    varsaymak, düzeltilmek istenen yanlılığı geri getirirdi."""
    if s > 0:
        return float(level) if bar_high >= level else float(bar_low)
    return float(level) if bar_low <= level else float(bar_high)


def decide_exit(track: PositionTrack, price: float, bar_high: float, bar_low: float,
                p: ExitParams, now: float, cont_prob: Optional[float] = None,
                current_ev_pct: Optional[float] = None) -> Optional[Dict]:
    """Yan etki: tepe/highest_high/armed/trail_stop günceller. Çıkış yoksa None."""
    s = track.sign()
    # Bar ONCESI trailing seviyesi: bu bar icinde chandelier seviyeyi yukseltebilir ve
    # YENI seviyeyi AYNI barin dibiyle test etmek, bar icinde "once tepe, sonra dip"
    # sirasini varsaymak olurdu (gozlenemez). Bar-ici test yalniz onceden VAR OLAN
    # seviyeye uygulanir.
    prev_trail = track.trail_stop
    track.highest_high = max(track.highest_high or price, bar_high, price)
    track.lowest_low = min(track.lowest_low or price, bar_low, price) if track.lowest_low else min(bar_low, price)
    g = track.gross_pct(price)
    n = track.net_pct(price)
    if g > track.peak_gross_pct:
        track.peak_gross_pct = g
    if n > track.peak_net_pct:
        track.peak_net_pct = n
    age = now - track.opened_ts
    arm = arm_threshold_pct(track, p)
    if not track.armed and track.peak_net_pct >= arm:
        track.armed = True
        track.notes.append(f"tepe koruması silahlandı (net tepe %{track.peak_net_pct:.2f} ≥ eşik %{arm:.2f})")

    # 1) HARD STOP (kilitli stop dahil — başabaşa çekilmişse çıkış sebebi BE_LOCK)
    #    Bar İÇİ kontrol: kapanış değil, aleyhe uç. Dolum stop seviyesinde varsayılır.
    if _stop_touched(track.hard_stop, bar_high, bar_low, s):
        fill = _stop_fill(track.hard_stop, bar_high, bar_low, s)
        gf, nf = track.gross_pct(fill), track.net_pct(fill)
        return {"reason": ("BE_LOCK" if track.be_locked and nf >= -0.05 else "STOP"),
                "net_pct": nf, "gross_pct": gf, "age_sec": age, "exit_price": fill,
                "intrabar": bool((price - track.hard_stop) * s > 0)}
    # 1b) ERKEN İPTAL — kötü giriş: ufkun ilk yarısında stop'un %60'ına gitti, hiç kâra geçmedi
    if (age <= p.early_abort_window_frac * p.time_stop_sec and track.peak_net_pct <= 0.0
            and track.stop_pct > 0 and track.mae_pct() >= p.early_abort_mae_frac * track.stop_pct and n < 0):
        return {"reason": "EARLY_ABORT", "net_pct": n, "gross_pct": g, "age_sec": age, "mae_pct": track.mae_pct()}
    # 1c) BAŞABAŞ KİLİDİ — tepe net maliyetin ≥ be_lock_cost_mult katı olduysa stop başabaş+maliyete
    if not track.be_locked and track.peak_net_pct >= p.be_lock_cost_mult * track.cost_pct_roundtrip:
        be = track.breakeven_plus()
        if (be - track.hard_stop) * s > 0:
            track.hard_stop = be
        track.be_locked = True
        track.notes.append(f"başabaş kilidi: tepe net %{track.peak_net_pct:.2f} ≥ {p.be_lock_cost_mult}×maliyet → stop {be:.6g}")
    # 1d) ZAMAN KİLİDİ — ufkun %60'ı geçti, kârdayız, stop hâlâ başabaşın altında → başabaş+maliyet
    if not track.be_locked and age >= p.time_lock_frac * p.time_stop_sec and n >= 0.0:
        be = track.breakeven_plus()
        if (be - track.hard_stop) * s > 0:
            track.hard_stop = be
        track.be_locked = True
        track.notes.append(f"zaman kilidi: ufkun %{p.time_lock_frac*100:.0f}'i geçti, kârda → stop başabaş {be:.6g}")
    # trailing (chandelier) — silahlandıysa ve kârdaysa; hard stop'un daha iyi tarafında
    ch = chandelier_stop(track, p) if track.armed else None
    if ch is not None and (ch - track.hard_stop) * s > 0:
        track.trail_stop = ch if track.trail_stop is None else (max(track.trail_stop, ch) if s > 0 else min(track.trail_stop, ch))
    if age < p.min_hold_sec:
        return None
    # 3) mod
    if track.mode == FIXED_TARGET and track.target is not None and (price - track.target) * s >= 0:
        return {"reason": "TP", "net_pct": n, "gross_pct": g, "age_sec": age}
    if track.mode == PARTIAL_AND_RUN and track.target is not None and not track.partial_done \
            and (price - track.target) * s >= 0:
        return {"reason": "TP", "net_pct": n, "gross_pct": g, "age_sec": age}
    if track.armed:
        lvl = track.giveback_level(p)
        if lvl is not None and (price - lvl) * s <= 0:
            return {"reason": "GIVEBACK", "net_pct": n, "gross_pct": g, "age_sec": age,
                    "peak_net_pct": track.peak_net_pct, "level": lvl}
        if prev_trail is not None and _stop_touched(prev_trail, bar_high, bar_low, s):
            fill = _stop_fill(prev_trail, bar_high, bar_low, s)
            return {"reason": "TRAIL", "net_pct": track.net_pct(fill), "gross_pct": track.gross_pct(fill),
                    "age_sec": age, "level": prev_trail, "exit_price": fill,
                    "intrabar": bool((price - prev_trail) * s > 0)}
        if track.trail_stop is not None and (price - track.trail_stop) * s <= 0:
            return {"reason": "TRAIL", "net_pct": n, "gross_pct": g, "age_sec": age,
                    "level": track.trail_stop, "exit_price": price}
        if track.mode == DYNAMIC_PEAK and cont_prob is not None and cont_prob < p.model_exit_cont_prob:
            return {"reason": "MODEL_EXIT", "net_pct": n, "gross_pct": g, "age_sec": age,
                    "cont_prob": cont_prob}
    # 4) edge decay — kâr silahlanmadıysa ve güncel EV ≤ 0
    if p.edge_decay_enabled and current_ev_pct is not None and current_ev_pct <= 0 and not track.armed \
            and age >= 2 * p.min_hold_sec:
        return {"reason": "EDGE_DECAY", "net_pct": n, "gross_pct": g, "age_sec": age,
                "current_ev_pct": current_ev_pct}
    # 5) zaman
    if age >= p.time_stop_sec:
        return {"reason": "TIME_STOP", "net_pct": n, "gross_pct": g, "age_sec": age}
    return None


def continuation_probability(f: Dict, direction: str, remaining_frac: float, dist_target_atr: Optional[float] = None) -> Dict:
    """P(hareket devam eder) — SEZGİSEL, kalibre edilmemiş (kalibrasyon kaydı `p_cont` fişte tutulur, ölçülür).
    Girdiler: trend skoru, CVD, defter dengesi, EMA eğimi, RSI aşırılığı, kalan ufuk payı, hedefe ATR mesafesi."""
    import math as _m
    s = 1.0 if direction == "LONG" else -1.0
    x = 0.0; used = {}
    ts = f.get("trend_score")
    if ts is not None:
        x += (float(ts) - 0.5) * 1.6 * s; used["trend_score"] = ts
    cvd = f.get("cvd_ratio")
    if cvd is not None:
        x += 0.8 * float(cvd) * s; used["cvd"] = cvd
    obi = f.get("obi")
    if obi is not None:
        x += 0.8 * (float(obi) - 0.5) * 2.0 * s; used["obi"] = obi
    slope = f.get("ema_slope_pct")
    if slope is not None:
        x += 2.0 * max(-0.5, min(0.5, float(slope) * 10.0)) * s; used["ema_slope_pct"] = slope
    rsi = f.get("rsi")
    if rsi is not None and ((s > 0 and rsi > 75) or (s < 0 and rsi < 25)):
        x -= 0.4; used["rsi_extreme"] = rsi
    x -= 0.6 * (1.0 - max(0.0, min(1.0, remaining_frac)))
    if dist_target_atr is not None:
        x -= 0.1 * max(0.0, float(dist_target_atr) - 2.0)
    p = 1.0 / (1.0 + _m.exp(-x))
    return {"p": round(max(0.05, min(0.95, p)), 3), "inputs": used, "note": "sezgisel — kalibre edilmedi"}


def peak_capture_ratio(realized_net_pct: float, peak_net_pct: float) -> Optional[float]:
    """PCR = gerçekleşen net / mevcut olan azami net. Tepe yoksa None."""
    if peak_net_pct is None or peak_net_pct <= 0:
        return None
    return float(max(-1.0, min(1.0, realized_net_pct / peak_net_pct)))


def exit_quality(trade: Dict, post_exit_mfe_pct: Optional[float]) -> Dict:
    """Kapanış kalitesi: erken mi (çıkış sonrası MFE büyük), geç mi (PCR düşük)."""
    pcr = peak_capture_ratio(float(trade.get("net_pct_realized") or trade.get("pnl_pct") or 0.0),
                             float(trade.get("peak_net_pct") or 0.0))
    out = {"peak_capture": None if pcr is None else round(pcr, 3),
           "post_exit_mfe_pct": post_exit_mfe_pct}
    if post_exit_mfe_pct is not None and post_exit_mfe_pct >= 1.0:
        out["verdict"] = f"ERKEN ÇIKIŞ — çıkış sonrası +%{post_exit_mfe_pct:.2f} daha vardı"
    elif pcr is not None and pcr < 0.4:
        out["verdict"] = f"GEÇ ÇIKIŞ — tepenin yalnız %{pcr*100:.0f}'i alındı"
    else:
        out["verdict"] = "makul"
    return out
