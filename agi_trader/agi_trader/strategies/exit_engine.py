"""
ÇIKIŞ MOTORU v2 — kâr merdiveni (T1…T6), kâr KİLİDİ (ratchet), tek hard stop,
ATR/chandelier trailing, zaman-stop, edge-decay ve model çıkışı; peak-capture KPI.

────────────────────────────────────────────────────────────────────────────
NEDEN v2 — 2026-09-06 CANLI ÖLÇÜMÜ (200 işlem, MEXC kâğıt, 3,1 gün)
────────────────────────────────────────────────────────────────────────────
  brüt −1,48 $ · komisyon 5,47 $ · net −6,95 $ · PF 0,54
  ortalama TEPE net %0,316 · ortalama YAKALANAN %0,10  →  PCR = 0,103

  Yani: ortalama işlem GERÇEKTEN kâra geçiyordu; kârın %90'ı geri veriliyordu.
  İki yapısal kusur ölçüldü:

  (1) BAŞABAŞ KİLİDİ KÂRI SIFIRLIYORDU.  Eski kural: tepe net ≥ 1,5×maliyet
      olunca stop BAŞABAŞA çekiliyordu. Yarı-tepe koruması ise ancak
      tepe net ≥ max(2×maliyet, %0,20, 0,5×ATR) eşiğinde SİLAHLANIYORDU.
      Aradaki bantta (maliyet %0,2 için %0,30–%0,40) yalnız başabaş kilidi
      etkindi → kâr TAM olarak geri veriliyordu.
      Kanıt: BE_LOCK ile kapanan 38 işlem, ortalama tepe net %0,311,
             gerçekleşen %0,0115 → PCR 0,042.  Kaybedilen ≈ +1,33 $.
      v2: stop başabaşa değil, TEPENİN retain oranına kilitlenir (ratchet).
          Taban başabaş+maliyettir; asla aşağı inmez.

  (2) ASGARİ TUTMA SÜRESİ YALNIZ KÂRI ENGELLİYORDU.  `min_hold_sec` (900 sn)
      kapısı hard stop ve erken iptalin ARDINDAN, ama GIVEBACK/TRAIL'den ÖNCE
      duruyordu. Sonuç asimetri: ilk 15 dakikada ZARAR kapanabiliyor, KÂR
      korunamıyordu.
      Kanıt: 15 dakikadan kısa 52 işlem → net −4,01 $ (uzun olanlar −2,94 $).
             Bunların 30'u eşiğin üstüne (%0,20+) çıkıp tepesinin yarısının
             altında kapandı; tepede %50 korunsaydı +1,79 $.
      v2: KÂR KORUYUCU çıkışlar (GIVEBACK/TRAIL/merdiven) asgari tutmaya
          BAKMAZ. Takdire dayalı çıkışlar (sabit TP, EDGE_DECAY, TIME_STOP,
          MODEL_EXIT) eskisi gibi asgari tutmayı bekler.

  (3) TEK HEDEF → MERDİVEN.  Eski model tek `target` + tek `partial_tp`
      taşıyordu; 200 işlemin yalnız 7'sinde kısmi kâr alınabildi (kısmi kapı
      da asgari tutmanın arkasındaydı). v2 en çok 6 basamaklı bir KÂR
      MERDİVENİ taşır: her basamak parçalı satış yapar VE retain oranını
      yükseltir (kalan koşucu daha sıkı korunur).
      Basamaklar GELECEĞİN TEPELERİ DEĞİLDİR — R katları ve yapısal
      seviyelerdir; her basamak yalnız NET kârı maliyetin katı olduğunda
      ateşlenir ("borsa ücretinin üstünde oyna" kuralı koda geçirildi).

Sıra (her fiyat güncellemesinde):
  1. HARD STOP        — hiçbir mod kaldıramaz (bar İÇİ kontrol, dolum = seviye)
  1b ERKEN İPTAL      — kötü giriş
  1c KÂR KİLİDİ       — tepe net ≥ lock eşiği → stop = tepe×retain (ratchet)
  1d ZAMAN KİLİDİ     — ufkun %60'ı geçti, kârda → kilidi zorla
  2. KÂR MERDİVENİ    — T1…T6 kısmi çıkışlar (asgari tutmadan MUAF)
  3. KORUMA           — GIVEBACK / TRAIL (asgari tutmadan MUAF)
  4. TAKDİR           — sabit TP / MODEL_EXIT / EDGE_DECAY / TIME_STOP
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

FIXED_TARGET, PARTIAL_AND_RUN, DYNAMIC_PEAK = "FIXED_TARGET", "PARTIAL_AND_RUN", "DYNAMIC_PEAK"
MODES = (FIXED_TARGET, PARTIAL_AND_RUN, DYNAMIC_PEAK)

# Kâr merdiveni varsayılanları — R (stop mesafesi) katları ve her basamakta
# kapatılacak BAŞLANGIÇ miktarı payı. Toplam 1,00'in altındadır: son parça
# koşucu olarak kalır ve yalnız trailing/giveback ile kapanır.
DEFAULT_LADDER_R = (1.0, 1.8, 2.8, 4.0, 5.5, 7.5)
DEFAULT_LADDER_FRACS = (0.25, 0.20, 0.15, 0.15, 0.10, 0.10)


@dataclass
class ExitParams:
    retain_fraction: float = 0.50        # tepe NET kârın korunacak TABAN oranı (0,35–0,70)
    arm_cost_multiple: float = 2.0       # (3,0 idi) canlı: %0,2–0,35 net tepeler hiç silahlanmadı → zaman-stopunda eksiye döndü
    arm_min_mfe_pct: float = 0.20
    # KAZANCIN KAYBA DÖNMEMESİ: tepe net ≥ be_lock_cost_mult×maliyet olunca KÂR KİLİDİ devreye girer.
    # v2: stop artık BAŞABAŞA değil, tepenin retain oranına çekilir (taban = başabaş+maliyet).
    be_lock_cost_mult: float = 1.5
    # ZAMAN KİLİDİ: ufkun %60'ı geçince kârdaysa kilit zorlanır (zaman-stopu eksiye düşürmesin)
    time_lock_frac: float = 0.6
    # ERKEN İPTAL: ufkun ilk %50'sinde MAE ≥ 0,6×stop ve tepe net ≤ 0 → kes
    early_abort_mae_frac: float = 0.6
    early_abort_window_frac: float = 0.5
    arm_atr_mult: float = 0.5
    chandelier_k: float = 2.5
    min_hold_sec: int = 900
    time_stop_sec: int = 3600
    edge_decay_enabled: bool = True
    model_exit_cont_prob: float = 0.35    # devam olasılığı bunun altına düşerse (DYNAMIC_PEAK)
    # ─── v2 (hepsi geriye dönük güvenli varsayılanlarla) ────────────────────
    # AYRIM (ölçüldü): "kârı KORUMAK" ile "kâr ALMAK" aynı şey değildir.
    #   koruma  (GIVEBACK/TRAIL) → asgari tutmadan MUAF. 37.905 eşleştirilmiş yolda
    #           kâr kilidiyle birlikte +0,0088 puan/işlem (t = 8,84).
    #   kâr alma (merdiven/sabit TP) → asgari tutmayı BEKLER. Erken ölçekli çıkış
    #           kazananı kırpar, kaybedeni etkilemez: aynı ölçümde −0,0015 puan (t = −4,79).
    protect_before_min_hold: bool = True
    ladder_before_min_hold: bool = False
    # ─────────────────────────────────────────────────────────────────────────
    # KÂR MERDİVENİ VARSAYILAN OLARAK KAPALI — ÖLÇÜLDÜ, REDDEDİLDİ.
    #
    # İstenen davranış "birden çok tepeden kâr almak"tı ve kod yazıldı, test edildi,
    # 37.905 EŞLEŞTİRİLMİŞ yolda (40 parite × 14 gün × 1 dk bar) ölçüldü:
    #   merdiven         −0,00153 puan/işlem · t = −4,79 · %95 GA (−0,00215, −0,00090)
    #   kâr kilidi       +0,00881 puan/işlem · t = +8,84 · %95 GA (+0,00685, +0,01076)
    # Ölçekli çıkış KAZANANI kırpıyor, KAYBEDENİ etkilemiyor: ödeme oranı 0,711 → 0,355.
    # Kazanma oranındaki artış (%47,9 → %48,7) bunu karşılamıyor.
    #
    # Özellik SİLİNMEDİ: `ladder_enabled=True` ile açılır, testleri duruyor, tezgâh
    # (scripts/cm_exit_bench.py) yeniden ölçmek için hazır. Kanıt gelirse açılır.
    ladder_enabled: bool = False
    ladder_levels: int = 4                # kaç basamak kullanılacak (0–6; 0 = kapalı)
    ladder_r: List[float] = field(default_factory=lambda: list(DEFAULT_LADDER_R))
    ladder_fracs: List[float] = field(default_factory=lambda: list(DEFAULT_LADDER_FRACS))
    ladder_min_cost_mult: float = 2.0     # basamak NET kârı maliyetin bu katı değilse basamak YOK
    retain_step_per_level: float = 0.07   # her basamaktan sonra retain +0,07 (koşucu daha sıkı korunur)
    retain_max: float = 0.85
    lock_floor_breakeven: bool = True     # kilit tabanı başabaş+maliyetin altına inemez
    lock_mode: str = "profit"             # "profit" = tepe×retain kilitle (v2) · "breakeven" = v1 davranışı (A/B için)

    def validated(self) -> "ExitParams":
        self.retain_fraction = float(min(0.70, max(0.35, self.retain_fraction)))
        self.arm_cost_multiple = float(min(10.0, max(1.0, self.arm_cost_multiple)))
        self.arm_min_mfe_pct = float(min(3.0, max(0.05, self.arm_min_mfe_pct)))
        self.chandelier_k = float(min(6.0, max(1.0, self.chandelier_k)))
        self.be_lock_cost_mult = float(min(5.0, max(1.0, self.be_lock_cost_mult)))
        self.time_lock_frac = float(min(0.95, max(0.3, self.time_lock_frac)))
        self.early_abort_mae_frac = float(min(0.95, max(0.3, self.early_abort_mae_frac)))
        self.early_abort_window_frac = float(min(0.9, max(0.1, self.early_abort_window_frac)))
        self.ladder_levels = int(min(6, max(0, self.ladder_levels)))
        self.ladder_min_cost_mult = float(min(10.0, max(1.0, self.ladder_min_cost_mult)))
        self.retain_step_per_level = float(min(0.20, max(0.0, self.retain_step_per_level)))
        self.retain_max = float(min(0.95, max(self.retain_fraction, self.retain_max)))
        if self.lock_mode not in ("profit", "breakeven"):
            self.lock_mode = "profit"
        r = [float(x) for x in (self.ladder_r or DEFAULT_LADDER_R) if float(x) > 0]
        self.ladder_r = sorted(r)[:6] or list(DEFAULT_LADDER_R)
        f = [float(x) for x in (self.ladder_fracs or DEFAULT_LADDER_FRACS) if float(x) > 0]
        self.ladder_fracs = (f or list(DEFAULT_LADDER_FRACS))[:6]
        # merdiven toplamı 0,90'ı aşamaz — koşucu için pay kalmalı
        tot = sum(self.ladder_fracs[:max(1, self.ladder_levels)])
        if tot > 0.90:
            k = 0.90 / tot
            self.ladder_fracs = [round(x * k, 4) for x in self.ladder_fracs]
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
    # ─── v2 ─────────────────────────────────────────────────────────────────
    ladder: List[Dict] = field(default_factory=list)   # [{price, frac, source, net_pct, hit}]
    levels_hit: int = 0
    lock_price: Optional[float] = None                 # kâr kilidinin ürettiği stop seviyesi

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

    def retain_eff(self, p: "ExitParams") -> float:
        """Etkin koruma oranı — her merdiven basamağından sonra yükselir.
        Gerekçe: kısmi kâr alındıkça kalan koşucunun geri verilebilir payı azalmalı."""
        return float(min(p.retain_max, p.retain_fraction + p.retain_step_per_level * max(0, self.levels_hit)))

    def price_at_net(self, net_pct: float) -> float:
        """Verilen NET yüzdeyi veren çıkış fiyatı (maliyet eklenir)."""
        return self.entry * (1.0 + self.sign() * (net_pct + self.cost_pct_roundtrip) / 100.0)

    def giveback_level(self, p: "ExitParams") -> Optional[float]:
        """Yarı-tepe çıkış fiyatı (UI: 'EXIT IF ≤ X'). Silahlanmadıysa None."""
        if not self.armed or self.peak_net_pct <= 0:
            return None
        return self.price_at_net(self.peak_net_pct * self.retain_eff(p))

    def lock_level(self, p: "ExitParams") -> Optional[float]:
        """KÂR KİLİDİ seviyesi: tepe NET kârın retain oranı kilitlenir.
        Taban başabaş+maliyettir (v1 davranışı); tepe büyüdükçe yukarı YÜRÜR."""
        if self.peak_net_pct <= 0:
            return None
        s = self.sign()
        if getattr(p, "lock_mode", "profit") == "breakeven":
            return float(self.breakeven_plus())          # v1 davranışı (A/B karşılaştırması için)
        lvl = self.price_at_net(self.peak_net_pct * self.retain_eff(p))
        if p.lock_floor_breakeven:
            be = self.breakeven_plus()
            lvl = max(lvl, be) if s > 0 else min(lvl, be)
        return float(lvl)

    # ---------------------------------------------------------------- merdiven
    def build_ladder(self, p: "ExitParams", structure: Optional[List[float]] = None) -> List[Dict]:
        """T1…Tn kâr merdiveni.

        DÜRÜSTLÜK NOTU: bunlar "gelecekte oluşacak tepeler" DEĞİLDİR. Kimse altı
        tepeyi önceden bilemez. Bunlar, stop mesafesinin (R) katları ve —
        verilmişse — YAPISAL seviyelerden (swing high, VWAP, direnç) türeyen
        ÖLÇEKLİ ÇIKIŞ basamaklarıdır. Her basamak yalnız net kârı maliyetin
        `ladder_min_cost_mult` katını aştığında geçerlidir; aksi hâlde basamak
        atılır (komisyon için işlem yapılmaz)."""
        self.ladder = []
        if not p.ladder_enabled or p.ladder_levels <= 0 or self.stop_pct <= 0:
            return []
        s = self.sign()
        min_net = self.cost_pct_roundtrip * p.ladder_min_cost_mult
        prices: List[Dict] = []
        struct = [float(x) for x in (structure or []) if x and math.isfinite(float(x))]
        struct.sort(reverse=(s < 0))
        rs = list(p.ladder_r)[:p.ladder_levels]
        fr = list(p.ladder_fracs)[:p.ladder_levels]
        while len(fr) < len(rs):
            fr.append(fr[-1] if fr else 0.15)
        for i, r in enumerate(rs):
            net_at = r * self.stop_pct - self.cost_pct_roundtrip     # R katı BRÜT → NET
            if net_at < min_net:
                continue
            px = self.price_at_net(net_at)
            src = f"{r:g}R"
            # yapısal seviye bu basamağa yakınsa (±%25 R) onu KULLAN — gerçek direnç
            for st in struct:
                if abs(st - px) <= 0.25 * self.stop_pct / 100.0 * self.entry and (st - self.entry) * s > 0:
                    if self.net_pct(st) >= min_net:
                        px, src = float(st), f"{r:g}R≈yapı"
                    break
            prices.append({"price": float(px), "frac": float(fr[i]), "source": src,
                           "net_pct": round(self.net_pct(px), 4), "hit": False})
        # aynı fiyata düşen basamakları birleştir (yapısal eşleşme sonrası olabilir)
        out: List[Dict] = []
        for lv in prices:
            if out and abs(lv["price"] - out[-1]["price"]) <= 1e-12:
                out[-1]["frac"] = round(min(0.9, out[-1]["frac"] + lv["frac"]), 4)
                continue
            out.append(lv)
        self.ladder = out
        return out

    def next_level(self) -> Optional[Dict]:
        for lv in self.ladder:
            if not lv.get("hit"):
                return lv
        return None

    def ladder_state(self) -> List[Dict]:
        return [{"i": i + 1, "price": lv.get("price"), "frac": lv.get("frac"),
                 "source": lv.get("source"), "net_pct": lv.get("net_pct"), "hit": bool(lv.get("hit"))}
                for i, lv in enumerate(self.ladder)]


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


def _level_touched(level: float, bar_high: float, bar_low: float, s: float) -> bool:
    """Kâr seviyesi bar içinde görüldü mü (LONG: tepe ≥ seviye)."""
    return (bar_high >= level) if s > 0 else (bar_low <= level)


def decide_exit(track: PositionTrack, price: float, bar_high: float, bar_low: float,
                p: ExitParams, now: float, cont_prob: Optional[float] = None,
                current_ev_pct: Optional[float] = None) -> Optional[Dict]:
    """Yan etki: tepe/highest_high/armed/trail_stop/lock günceller. Çıkış yoksa None.

    Dönüş `partial: True` taşıyorsa POZİSYON KAPANMAZ — `fraction` kadar ölçekli
    çıkış yapılır ve pozisyon yaşamaya devam eder (kâr merdiveni)."""
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

    # 1) HARD STOP (kâr kilidi dahil) — bar İÇİ kontrol, dolum stop seviyesinde
    if _stop_touched(track.hard_stop, bar_high, bar_low, s):
        fill = _stop_fill(track.hard_stop, bar_high, bar_low, s)
        gf, nf = track.gross_pct(fill), track.net_pct(fill)
        # Sebep: kilit seviyesinden çıkıldıysa bu bir KÂR KORUMASIDIR (giveback);
        # başabaşa çekilmiş kilitse BE_LOCK; hiçbiri değilse gerçek STOP.
        locked_here = (track.lock_price is not None
                       and abs(track.hard_stop - float(track.lock_price)) <= 1e-12)
        if locked_here and nf > 0.0:
            reason = "GIVEBACK"
        elif track.be_locked and nf >= -0.05:
            reason = "BE_LOCK"
        else:
            reason = "STOP"
        return {"reason": reason, "net_pct": nf, "gross_pct": gf, "age_sec": age, "exit_price": fill,
                "peak_net_pct": track.peak_net_pct, "levels_hit": track.levels_hit,
                "intrabar": bool((price - track.hard_stop) * s > 0)}
    # 1b) ERKEN İPTAL — kötü giriş: ufkun ilk yarısında stop'un %60'ına gitti, hiç kâra geçmedi
    if (age <= p.early_abort_window_frac * p.time_stop_sec and track.peak_net_pct <= 0.0
            and track.stop_pct > 0 and track.mae_pct() >= p.early_abort_mae_frac * track.stop_pct and n < 0):
        return {"reason": "EARLY_ABORT", "net_pct": n, "gross_pct": g, "age_sec": age, "mae_pct": track.mae_pct()}
    # 1c) KÂR KİLİDİ (RATCHET) — tepe net ≥ eşik olduysa stop tepenin retain oranına yürür.
    #     v1'de bu stop BAŞABAŞA çekiliyor ve kâr TAM geri veriliyordu (38 işlem, PCR 0,042).
    if track.peak_net_pct > 0 and track.peak_net_pct >= p.be_lock_cost_mult * track.cost_pct_roundtrip:
        lock = track.lock_level(p)
        if lock is not None and (lock - track.hard_stop) * s > 0:
            if not track.be_locked:
                track.notes.append(
                    f"kâr kilidi: tepe net %{track.peak_net_pct:.2f} → stop {lock:.6g} "
                    f"(net %{track.peak_net_pct * track.retain_eff(p):.2f} kilitlendi)")
            track.hard_stop = float(lock)
            track.lock_price = float(lock)
            track.be_locked = True
    # 1d) ZAMAN KİLİDİ — ufkun %60'ı geçti, kârdayız, kilit yoksa en az başabaşa
    if not track.be_locked and age >= p.time_lock_frac * p.time_stop_sec and n >= 0.0:
        be = track.breakeven_plus()
        if (be - track.hard_stop) * s > 0:
            track.hard_stop = be
            track.lock_price = be
        track.be_locked = True
        track.notes.append(f"zaman kilidi: ufkun %{p.time_lock_frac*100:.0f}'i geçti, kârda → stop başabaş {be:.6g}")
    # trailing (chandelier) — silahlandıysa ve kârdaysa; hard stop'un daha iyi tarafında
    ch = chandelier_stop(track, p) if track.armed else None
    if ch is not None and (ch - track.hard_stop) * s > 0:
        track.trail_stop = ch if track.trail_stop is None else (max(track.trail_stop, ch) if s > 0 else min(track.trail_stop, ch))

    # 2) KÂR MERDİVENİ — kâr ALMAK olduğu için asgari tutmayı bekler (bkz. ExitParams notu)
    if p.ladder_enabled and track.ladder and (p.ladder_before_min_hold or age >= p.min_hold_sec):
        lv = track.next_level()
        if lv is not None and _level_touched(float(lv["price"]), bar_high, bar_low, s):
            lv["hit"] = True
            track.levels_hit += 1
            track.partial_done = True
            idx = track.levels_hit
            # basamak sonrası kilit yeniden hesaplanır (retain yükseldi)
            lock = track.lock_level(p)
            if lock is not None and (lock - track.hard_stop) * s > 0:
                track.hard_stop = float(lock)
                track.lock_price = float(lock)
                track.be_locked = True
            return {"reason": "LADDER_TP", "partial": True, "fraction": float(lv["frac"]),
                    "level": idx, "level_price": float(lv["price"]), "source": lv.get("source"),
                    "exit_price": float(lv["price"]), "net_pct": track.net_pct(float(lv["price"])),
                    "gross_pct": track.gross_pct(float(lv["price"])), "age_sec": age,
                    "retain_after": track.retain_eff(p), "peak_net_pct": track.peak_net_pct}

    # 3) KÂR KORUMA — GIVEBACK / TRAIL. asgari tutmaya bakmaz (protect_before_min_hold)
    protect_ok = p.protect_before_min_hold or age >= p.min_hold_sec
    if track.armed and protect_ok:
        lvl = track.giveback_level(p)
        if lvl is not None and (price - lvl) * s <= 0:
            return {"reason": "GIVEBACK", "net_pct": n, "gross_pct": g, "age_sec": age,
                    "peak_net_pct": track.peak_net_pct, "level": lvl, "levels_hit": track.levels_hit}
        if prev_trail is not None and _stop_touched(prev_trail, bar_high, bar_low, s):
            fill = _stop_fill(prev_trail, bar_high, bar_low, s)
            return {"reason": "TRAIL", "net_pct": track.net_pct(fill), "gross_pct": track.gross_pct(fill),
                    "age_sec": age, "level": prev_trail, "exit_price": fill,
                    "peak_net_pct": track.peak_net_pct, "levels_hit": track.levels_hit,
                    "intrabar": bool((price - prev_trail) * s > 0)}
        if track.trail_stop is not None and (price - track.trail_stop) * s <= 0:
            return {"reason": "TRAIL", "net_pct": n, "gross_pct": g, "age_sec": age,
                    "level": track.trail_stop, "exit_price": price, "peak_net_pct": track.peak_net_pct}

    # ---- buradan sonrası TAKDİRE dayalı çıkışlar: asgari tutma süresi geçerli ----
    if age < p.min_hold_sec:
        return None
    if track.mode == FIXED_TARGET and track.target is not None and (price - track.target) * s >= 0:
        return {"reason": "TP", "net_pct": n, "gross_pct": g, "age_sec": age}
    if track.mode == PARTIAL_AND_RUN and track.target is not None and not track.partial_done \
            and (price - track.target) * s >= 0:
        return {"reason": "TP", "net_pct": n, "gross_pct": g, "age_sec": age}
    if track.armed and track.mode == DYNAMIC_PEAK and cont_prob is not None and cont_prob < p.model_exit_cont_prob:
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
    Girdiler: trend skoru, CVD, defter dengesi, EMA eğimi, RSI aşırılığı, kalan ufuk payı, hedefe ATR mesafesi.

    KULLANIM SINIRI (v2): kalibre edilmediği için TEK BAŞINA çıkış üretmesine izin
    verilmez — `model_exit_cont_prob` kapısı yalnız DYNAMIC_PEAK modunda ve pozisyon
    SİLAHLANDIKTAN sonra (yani kâr zaten korunuyorken) çalışır."""
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
