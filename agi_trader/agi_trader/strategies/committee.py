"""
KOMİTE STRATEJİSİ — 12 rolün oyunu, yavaş bağlamı (CryptoMind 4h analizleri,
15 dk'da bir yenilenen önbellek) ve hızlı tetikleyiciyi (1 dk barlar, 30 sn
döngü) tek kararda birleştirir.

Akış (her parite, her döngü):
  1. Hızlı özellikler: z-skoru, RSI, σ, EMA20/50 mesafesi, 20-bar kırılım, son bar ivmesi
  2. Rejim rolü şablonu seçer: TREND → geri çekilme (pullback), RANGE/VOLATİL → ortalamaya dönüş
  3. Tetikleyici şart: şablona uygun hızlı sinyal yoksa "BEKLE" (oylama yapılmaz)
  4. Piyasa yapısı seviyeleri → yapısal stop/hedef; oynaklık stop'u ile bağdaştırılır
  5. Oylama: Σ w_r·c_r·s_r / Σ w_r·c_r  (w_r = taban × öğrenilmiş güvenilirlik)
  6. Veto rolleri (maliyet, risk, denetçi) her şeyi ezer
  7. İşlem fişi: yatırım, beklenen kâr, komisyon, kazanma olasılığı, EV, azami zarar, limitler

Bu modül emir göndermez; ağ erişimi yapmaz.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import roles as R
from . import video_scalp as VS
from . import contrib as _CB

STRATEGY_ID = "committee"
STRATEGY_NAME = "Komite Stratejisi (12 rol · CryptoMind verileriyle)"


@dataclass
class CommitteeParams(VS.ScalpParams):
    """Video parametreleri (stop/çıkış/tutma) + komite eşikleri."""
    theta: float = 0.25                  # |oy| bu eşiği geçmeli
    min_confidence: float = 0.35         # ağırlıklı ortalama güven eşiği
    rr_min: float = 1.2
    rr_max: float = 2.5
    partial_tp_r: float = 1.0            # 1R'de yarısını al, stop'u girişe çek
    partial_fraction: float = 0.5
    max_ctx_age_sec: int = 1800          # yavaş bağlam 30 dk'dan eskiyse bayat
    max_spread_bps: float = 15.0
    counter_trend_mult: float = 0.5
    p_maker_fill: float = 0.5            # ölçülene kadar muhafazakâr
    maker_wait_bars: int = 2             # limit emir kaç bar bekler
    chase_taker_ratio: float = 3.0       # maker dolmazsa: brüt/maliyet ≥ bu ise taker
    pullback_ema: int = 20
    pullback_band_atr: float = 0.5
    breakout_lookback: int = 20
    strict_qualification: bool = False
    # ek tetikleyiciler (daha sık ama daha küçük boyutlu giriş)
    enable_moderate: bool = True
    dip_z_moderate: float = 1.2
    rsi_max_moderate: float = 42.0
    enable_breakout: bool = True
    breakout_vol_ratio: float = 1.5
    enable_momentum: bool = True
    enable_catalyst: bool = True
    closed_bar_fallback: bool = True     # devam eden 1 dk barda titreşen sinyal yerine SON KAPANMIŞ bar sinyali de sayılır
    size_floor: float = 0.5              # yığılan çarpanlar (sleeve×inceleme×portföy×rol) bunun altına inmez (canlı: 30 $'lık pozisyonlar)
    stop_risk_size_warnings: int = 2     # giriş anında ≥2 uyaran (aşırı uzama/ters trend/hacim yok/geniş spread/haber) → boyut ×0,6
    stop_risk_veto_warnings: int = 3     # ≥3 uyaran → giriş yok (stop profili: stoplar bu uyaranlarla geliyor)
    rotation_margin_pct: float = 0.15    # rotasyon: EV_B − geçiş maliyeti − marj > kalan EV_A
    catalyst_min_score: float = 0.5
    # ── KANIT KAPILARI (2026-09-04, 85 canlı işlem: brüt −0,60 $ / komisyon 4,18 $ / net −4,78 $) ──
    # Kanıtlanmamış sleeve tam boyut alamaz: 25 $ tavan aynı 85 işlemde −4,78 → −0,92 $ (karşı-olgusal).
    probe_notional_usdt: float = 25.0
    # Aciliyet-0 sleeve'ler (defter momentumu/kırılım/katalizör) kanıtlanmadan taker giremez: taker girişler
    # 43 işlemde −6,02 $, maker girişler 42 işlemde +1,24 $; obi_momentum taker −2,48 $ / maker +0,52 $.
    taker_requires_proof: bool = True
    # Drawdown'da yarı boyut: son 20 işlemin net'i < 0 → boyut ×0,5 (aynı işlemlerde −4,78 → −2,10 $).
    derisk_trailing_n: int = 20
    derisk_mult: float = 0.5
    # Seans kapısı (4 saatlik UTC bloğu, son 14 gün): n ≥ 15 ve t ≤ −1,5 → ×0,5; t ≤ −2,5 → giriş yok.
    session_gate: bool = True
    session_min_n: int = 15
    session_t_half: float = -1.5
    session_t_block: float = -2.5
    # Fişte ULAŞILABİLİR hedef: sabit hedefe 85 işlemin 1'i ulaştı; EV ölçülmüş MFE ile de yazılır.
    achievable_target: bool = True

    def validated(self) -> "CommitteeParams":
        super().validated()
        self.probe_notional_usdt = float(min(1000.0, max(5.0, self.probe_notional_usdt)))
        self.derisk_trailing_n = int(min(200, max(5, self.derisk_trailing_n)))
        self.derisk_mult = float(min(1.0, max(0.1, self.derisk_mult)))
        self.session_min_n = int(min(500, max(5, self.session_min_n)))
        self.session_t_half = float(min(0.0, max(-5.0, self.session_t_half)))
        self.session_t_block = float(min(self.session_t_half, max(-9.0, self.session_t_block)))
        self.theta = float(min(0.9, max(0.05, self.theta)))
        self.min_confidence = float(min(0.9, max(0.1, self.min_confidence)))
        self.rr_min = float(min(3.0, max(0.8, self.rr_min)))
        self.rr_max = float(min(6.0, max(self.rr_min, self.rr_max)))
        self.partial_tp_r = float(min(2.0, max(0.3, self.partial_tp_r)))
        self.partial_fraction = float(min(0.8, max(0.0, self.partial_fraction)))
        self.max_ctx_age_sec = int(min(6 * 3600, max(300, self.max_ctx_age_sec)))
        self.max_spread_bps = float(min(100.0, max(1.0, self.max_spread_bps)))
        self.counter_trend_mult = float(min(1.0, max(0.0, self.counter_trend_mult)))
        self.p_maker_fill = float(min(1.0, max(0.0, self.p_maker_fill)))
        self.maker_wait_bars = int(min(20, max(0, self.maker_wait_bars)))
        self.chase_taker_ratio = float(min(20.0, max(1.0, self.chase_taker_ratio)))
        self.dip_z_moderate = float(min(self.dip_z, max(0.5, self.dip_z_moderate)))
        self.rsi_max_moderate = float(min(55.0, max(self.rsi_max, self.rsi_max_moderate)))
        self.breakout_vol_ratio = float(min(5.0, max(1.0, self.breakout_vol_ratio)))
        self.catalyst_min_score = float(min(1.0, max(0.2, self.catalyst_min_score)))
        return self

    @classmethod
    def from_dict(cls, d: Optional[Dict]) -> "CommitteeParams":
        p = cls()
        for k, v in (d or {}).items():
            if hasattr(p, k) and v is not None:
                cur = getattr(p, k)
                try:
                    setattr(p, k, bool(v) if isinstance(cur, bool) else type(cur)(v))
                except (TypeError, ValueError):
                    pass
        return p.validated()


@dataclass
class Verdict:
    symbol: str
    allowed: bool
    direction: Optional[str]
    score: float
    confidence: float
    template: str
    trigger: Optional[str]
    votes: List[Dict]
    vetoes: List[str]
    size_mult: float
    order_type: str
    plan: Optional[Dict]
    ticket: Optional[Dict]
    result: str
    notes: List[str] = field(default_factory=list)
    weights: Dict[str, float] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    fast: Dict = field(default_factory=dict)      # z, rsi, EMA uzaklığı… (panel "tetikleyiciye ne kadar yakın")

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["score"] = round(self.score, 3)
        d["confidence"] = round(self.confidence, 3)
        d["size_mult"] = round(self.size_mult, 3)
        d["weights"] = {k: round(v, 3) for k, v in self.weights.items()}
        for k in ("entry", "exit_mode", "competition", "valid_until", "regime", "silenced", "veto_review", "stop_risk", "atr_hint"):
            if hasattr(self, k):
                d[k] = getattr(self, k)
        return d


# ---------------------------------------------------------------- hızlı özellikler
def fast_features(df: pd.DataFrame, p: CommitteeParams) -> Dict:
    f = VS.compute_features(df, p)
    if not f.get("ok"):
        return f
    c = df["close"].astype(float)
    h = df["high"].astype(float) if "high" in df else c
    l = df["low"].astype(float) if "low" in df else c
    n = int(p.pullback_ema)
    ema = c.ewm(span=n, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1]) if len(tr) >= 14 else float("nan")
    price = float(c.iloc[-1])
    atr_pct = (atr / price * 100.0) if (price > 0 and math.isfinite(atr)) else float("nan")
    lb = int(p.breakout_lookback)
    hi_prev = float(h.iloc[-lb - 1:-1].max()) if len(h) > lb else float("nan")
    lo_prev = float(l.iloc[-lb - 1:-1].min()) if len(l) > lb else float("nan")
    f.update({
        "ema_fast": float(ema.iloc[-1]), "ema_slow": float(ema50.iloc[-1]),
        "dist_ema_pct": (price / float(ema.iloc[-1]) - 1.0) * 100.0,
        "ema_slope_pct": (float(ema.iloc[-1]) / float(ema.iloc[-4]) - 1.0) * 100.0 if len(ema) > 4 else 0.0,
        "trend_up": bool(float(ema.iloc[-1]) > float(ema50.iloc[-1])),
        "atr": atr, "atr_pct": atr_pct,
        "bar_up": bool(float(c.iloc[-1]) > float(c.iloc[-2])),
        "breakout_up": bool(math.isfinite(hi_prev) and price > hi_prev),
        "breakdown": bool(math.isfinite(lo_prev) and price < lo_prev),
        "swing_low": float(l.tail(lb).min()), "swing_high": float(h.tail(lb).max()),
    })
    # hacim oranı (son 3 bar / önceki 60 bar) ve EMA9×EMA21 yukarı kesişimi (son 3 bar)
    vol = df["volume"].astype(float) if "volume" in df else None
    f["vol_ratio"] = (float(vol.tail(3).mean() / max(1e-9, vol.tail(63).head(60).mean()))
                      if vol is not None and len(vol) >= 63 else None)
    e9 = c.ewm(span=9, adjust=False).mean()
    e21 = c.ewm(span=21, adjust=False).mean()
    f["ema_cross_up"] = bool(len(c) > 5 and float(e9.iloc[-1]) > float(e21.iloc[-1]) and
                             any(float(e9.iloc[-k]) <= float(e21.iloc[-k]) for k in range(2, 5)))
    return f


TRIGGER_PRIORITY = ("catalyst", "dip", "breakout", "pullback", "momentum", "dip_moderate", "top")
TRIGGER_TR = {"catalyst": "haber katalizörü", "dip": "dip", "dip_moderate": "ılımlı dip",
              "breakout": "kırılım", "pullback": "geri çekilme", "momentum": "momentum", "top": "tepe"}


def triggers(fast: Dict, template: str, p: CommitteeParams, allow_short: bool,
             news: Optional[Dict] = None) -> List[Dict]:
    """Ateşleyen BÜTÜN tetikleyiciler (öncelik sırasında). Her biri boyut çarpanı taşır:
    kesin dip ×1,0 · kırılım ×0,8 · geri çekilme ×1,0 · momentum ×0,7 · ılımlı dip ×0,6 ·
    haber katalizörü ×0,8 (yalnız hareketlilik DOĞRULANMIŞSA)."""
    out: List[Dict] = []
    if not fast.get("ok"):
        return out
    z, rsi = fast["z"], fast["rsi"]
    up = bool(fast.get("bar_up"))
    if p.enable_catalyst and news and news.get("confirmed") and not news.get("severe_risk") \
            and float(news.get("score") or 0.0) >= p.catalyst_min_score and up:
        out.append({"kind": "catalyst", "direction": "LONG", "size": 0.8,
                    "note": f"haber katalizörü doğrulandı (skor {news.get('score'):+.2f}, hacim ×{news.get('vol_ratio')})"})
    if template == "mean_reversion":
        if z <= -p.dip_z and rsi <= p.rsi_max and up:
            out.append({"kind": "dip", "direction": "LONG", "size": 1.0,
                        "note": f"dip + ilk yeşil bar (z={z:.2f}, RSI {rsi:.0f})"})
        elif p.enable_moderate and z <= -p.dip_z_moderate and rsi <= p.rsi_max_moderate and up:
            out.append({"kind": "dip_moderate", "direction": "LONG", "size": 0.6,
                        "note": f"ılımlı dip + yeşil bar (z={z:.2f}, RSI {rsi:.0f}) ×0,6"})
        if allow_short and z >= p.dip_z and rsi >= 100 - p.rsi_max and not up:
            out.append({"kind": "top", "direction": "SHORT", "size": 1.0,
                        "note": f"tepe + ilk kırmızı bar (z={z:.2f}, RSI {rsi:.0f})"})
    else:
        band = p.pullback_band_atr * (fast.get("atr_pct") or 0.3)
        d = fast.get("dist_ema_pct", 0.0)
        if fast.get("trend_up") and -band <= d <= band * 0.5 and up and rsi < 65:
            out.append({"kind": "pullback", "direction": "LONG", "size": 1.0,
                        "note": f"EMA{p.pullback_ema} geri çekilmesi (uzaklık %{d:.2f}, bant ±%{band:.2f}) + yeşil bar"})
        vr = fast.get("vol_ratio")
        if p.enable_breakout and fast.get("breakout_up") and fast.get("trend_up") and up \
                and vr is not None and vr >= p.breakout_vol_ratio and 55 <= rsi <= 72:
            out.append({"kind": "breakout", "direction": "LONG", "size": 0.8,
                        "note": f"{p.breakout_lookback}-bar kırılım + hacim ×{vr:.1f} (RSI {rsi:.0f})"})
        if p.enable_momentum and fast.get("ema_cross_up") and fast.get("trend_up") and up and 50 <= rsi <= 65:
            out.append({"kind": "momentum", "direction": "LONG", "size": 0.7,
                        "note": f"EMA9×EMA21 yukarı kesişim (RSI {rsi:.0f})"})
        if allow_short and not fast.get("trend_up") and -band * 0.5 <= d <= band and not up and rsi > 35:
            out.append({"kind": "pullback", "direction": "SHORT", "size": 1.0,
                        "note": f"EMA{p.pullback_ema} yükseliş düzeltmesi (uzaklık %{d:.2f}) + kırmızı bar"})
    out.sort(key=lambda t: TRIGGER_PRIORITY.index(t["kind"]) if t["kind"] in TRIGGER_PRIORITY else 99)
    return out


def trigger(fast: Dict, template: str, p: CommitteeParams, allow_short: bool,
            news: Optional[Dict] = None) -> Dict:
    """Öncelikli tek tetikleyici (ateşleyen yoksa BEKLE + neden). Bütün liste: triggers()."""
    if not fast.get("ok"):
        return {"kind": None, "direction": None, "note": fast.get("reason", "özellik yok")}
    fired = triggers(fast, template, p, allow_short, news)
    if fired:
        best = dict(fired[0])
        best["others"] = [t["kind"] for t in fired[1:]]
        return best
    return _wait_note(fast, template, p, allow_short)


def _wait_note(fast: Dict, template: str, p: CommitteeParams, allow_short: bool) -> Dict:
    z, rsi = fast["z"], fast["rsi"]
    if template == "mean_reversion":
        if z <= -p.dip_z and rsi <= p.rsi_max and fast.get("bar_up"):
            return {"kind": "dip", "direction": "LONG",
                    "note": f"dip + ilk yeşil bar (z={z:.2f}, RSI {rsi:.0f})"}
        if z <= -p.dip_z and rsi <= p.rsi_max:
            return {"kind": None, "direction": "LONG",
                    "note": f"dip var ama dönüş barı yok (z={z:.2f}) — bekle"}
        if allow_short and z >= p.dip_z and rsi >= 100 - p.rsi_max and not fast.get("bar_up"):
            return {"kind": "top", "direction": "SHORT",
                    "note": f"tepe + ilk kırmızı bar (z={z:.2f}, RSI {rsi:.0f})"}
        return {"kind": None, "direction": None, "note": f"ortalamaya dönüş koşulu yok (z={z:.2f}, RSI {rsi:.0f})"}
    # pullback: trend yukarı, fiyat EMA'ya yarım ATR içinde geri çekilmiş, bar yeşil
    band = p.pullback_band_atr * (fast.get("atr_pct") or 0.3)
    d = fast.get("dist_ema_pct", 0.0)
    if fast.get("trend_up") and -band <= d <= band * 0.5 and fast.get("bar_up") and rsi < 65:
        return {"kind": "pullback", "direction": "LONG",
                "note": f"EMA{p.pullback_ema} geri çekilmesi (uzaklık %{d:.2f}, bant ±%{band:.2f}) + yeşil bar"}
    if allow_short and not fast.get("trend_up") and -band * 0.5 <= d <= band and not fast.get("bar_up") and rsi > 35:
        return {"kind": "pullback", "direction": "SHORT",
                "note": f"EMA{p.pullback_ema} yükseliş düzeltmesi (uzaklık %{d:.2f}) + kırmızı bar"}
    return {"kind": None, "direction": None,
            "note": f"geri çekilme koşulu yok (EMA uzaklık %{d:.2f}, trend {'↑' if fast.get('trend_up') else '↓'})"}


# ---------------------------------------------------------------- plan
def build_plan(direction: str, price: float, fast: Dict, levels: Dict,
               p: CommitteeParams) -> Dict:
    """Yapısal + oynaklık stop; yapısal hedef rr_min..rr_max bandında."""
    s = 1.0 if direction == "LONG" else -1.0
    vol = VS.plan_trade(direction, price, fast.get("sigma_bar_pct"), p)
    vol_stop = vol["stop_pct"]
    atr_pct = fast.get("atr_pct") or 0.0
    struct_stop = None
    sup, res = levels.get("support"), levels.get("resistance")
    swing_lo, swing_hi = fast.get("swing_low"), fast.get("swing_high")
    if direction == "LONG":
        cands = [x for x in (sup, swing_lo) if x and x < price]
        if cands:
            lvl = max(cands)
            struct_stop = (price - lvl) / price * 100.0 + 0.2 * atr_pct
    else:
        cands = [x for x in (res, swing_hi) if x and x > price]
        if cands:
            lvl = min(cands)
            struct_stop = (lvl - price) / price * 100.0 + 0.2 * atr_pct
    if struct_stop is not None and 0.7 * vol_stop <= struct_stop <= 2.0 * vol_stop:
        stop_pct, stop_src = struct_stop, "yapısal (seviye − 0,2 ATR)"
    else:
        stop_pct, stop_src = vol_stop, "oynaklık (k·σ·√H)"
    stop_pct = float(np.clip(stop_pct, p.min_stop_pct, p.max_stop_pct))

    # hedef: yapısal direnç/destek mesafesi; yoksa rr × stop
    struct_tgt = None
    if direction == "LONG" and res and res > price:
        struct_tgt = (res - price) / price * 100.0
    elif direction == "SHORT" and sup and sup < price:
        struct_tgt = (price - sup) / price * 100.0
    note = ""
    invalid = None
    size_penalty = 1.0
    partial_near = None
    if struct_tgt is not None:
        if struct_tgt < p.rr_min * stop_pct:
            # Karşı-olgusal kanıt (canlı gölge: 14 hedef / 0 stop): yakın yapısal seviye VETO değil,
            # KISMİ kâr seviyesidir. Hedef rr'ye çekilir, boyut ×0,7, ilk direnç kısmi TP olur.
            target_pct = max(p.rr_min, p.rr) * stop_pct
            tgt_src = f"rr {max(p.rr_min, p.rr)}×stop (yapısal seviye %{struct_tgt:.2f} yakın → kısmi TP)"
            size_penalty = 0.7
            partial_near = float(price * (1.0 + s * struct_tgt / 100.0))
            note = f"yakın yapısal seviye %{struct_tgt:.2f} < {p.rr_min}×stop → kısmi TP oraya, boyut ×0,7"
        else:
            target_pct = min(struct_tgt, p.rr_max * stop_pct)
            tgt_src = "yapısal seviye" if struct_tgt <= p.rr_max * stop_pct else f"rr_max {p.rr_max}×stop"
    else:
        target_pct = max(p.rr_min, p.rr) * stop_pct
        tgt_src = f"rr {max(p.rr_min, p.rr)}×stop (seviye yok)"
    return {
        "direction": direction, "entry": float(price),
        "stop_pct": round(stop_pct, 4), "target_pct": round(target_pct, 4),
        "stop": float(price * (1.0 - s * stop_pct / 100.0)),
        "target": float(price * (1.0 + s * target_pct / 100.0)),
        "rr": round(target_pct / max(1e-9, stop_pct), 3),
        "stop_source": stop_src, "target_source": tgt_src,
        "partial_tp": (partial_near if partial_near is not None else float(price * (1.0 + s * p.partial_tp_r * stop_pct / 100.0))),
        "partial_tp_near": partial_near, "partial_fraction": p.partial_fraction, "size_penalty": size_penalty,
        "hold_bars": vol["hold_bars"], "invalid": invalid, "note": note,
    }


# ---------------------------------------------------------------- fiş
def achievable_target_pct(kind: str, plan_target_pct: float, atr_pct: Optional[float], hold_bars: Optional[float],
                          cost_pct: float, min_gross_to_cost: float, learned: Optional[Dict] = None) -> Dict:
    """ULAŞILABİLİR hedef (%): sleeve'in ÖLÇÜLMÜŞ tepe-brüt medyanı (n ≥ 8, `learned["mfe_by_sleeve"]`);
    yoksa sürüklenmesiz yürüyüşün beklenen azami lehte hareketi ≈ 0,8·σ·√H (σ = ATR%, H = ufuk barı).
    Alt sınır maliyet × asgari brüt/maliyet, üst sınır planın kendi hedefi. Sabit hedefe 85 canlı işlemin
    1'i ulaştı (kazanan MFE medyanı %0,55 / hedef medyanı %2,02): fişteki EV planın hedefiyle YANILTICI."""
    src = "formül 0,8·σ·√H"
    mfe = ((learned or {}).get("mfe_by_sleeve") or {}).get(kind)
    if isinstance(mfe, (int, float)) and mfe > 0:
        est, src = float(mfe), "ölçülmüş sleeve MFE medyanı"
    else:
        a = float(atr_pct or 0.0); h = float(hold_bars or 60.0)
        est = 0.8 * a * math.sqrt(max(1.0, h)) if a > 0 else float(plan_target_pct)
    lo = float(cost_pct) * float(min_gross_to_cost)
    est = float(min(float(plan_target_pct), max(lo, est)))
    return {"pct": round(est, 4), "source": src}


def trade_ticket(plan: Dict, notional: float, p_win: float, cost_pct_maker: float,
                 cost_pct_taker: float, order_type: str, p_maker_fill: float,
                 capital: float, daily_loss_left_pct: Optional[float],
                 exposure_room: float, achievable: Optional[Dict] = None) -> Dict:
    """Kullanıcının istediği bütün sayılar tek fişte — hiçbiri gizlenmez."""
    exp_cost = (p_maker_fill * cost_pct_maker + (1 - p_maker_fill) * cost_pct_taker
                if order_type == "maker" else cost_pct_taker)
    gross_win = notional * plan["target_pct"] / 100.0
    gross_loss = notional * plan["stop_pct"] / 100.0
    fee = notional * exp_cost / 100.0
    net_win = gross_win - fee
    net_loss = gross_loss + fee
    ev = p_win * net_win - (1.0 - p_win) * net_loss
    ach = None
    if achievable and achievable.get("pct") is not None:
        a_pct = float(achievable["pct"])
        ev_a = p_win * (notional * a_pct / 100.0 - fee) - (1.0 - p_win) * net_loss
        ach = {"achievable_target_pct": round(a_pct, 4), "achievable_source": achievable.get("source"),
               "ev_achievable_usdt": round(ev_a, 4),
               "ev_achievable_pct": round(ev_a / max(1e-9, notional) * 100.0, 4),
               "breakeven_win_rate_achievable": round(net_loss / max(1e-9, (notional * a_pct / 100.0 - fee) + net_loss), 3)}
    return {
        **(ach or {}),
        "investment_usdt": round(notional, 2),
        "capital_pct": round(notional / max(1e-9, capital) * 100.0, 2),
        "expected_profit_usdt": round(net_win, 4),
        "expected_profit_pct": round(plan["target_pct"] - exp_cost, 4),
        "max_loss_usdt": round(net_loss, 4),
        "max_loss_pct": round(plan["stop_pct"] + exp_cost, 4),
        "fee_usdt": round(fee, 4), "fee_pct_roundtrip": round(exp_cost, 4),
        "order_type": order_type, "p_maker_fill": p_maker_fill,
        "p_win": round(p_win, 3), "ev_usdt": round(ev, 4),
        "ev_pct": round(ev / max(1e-9, notional) * 100.0, 4),
        "breakeven_win_rate": round(net_loss / max(1e-9, net_win + net_loss), 3),
        "rr_net": round(net_win / max(1e-9, net_loss), 3),
        "daily_loss_left_pct": daily_loss_left_pct,
        "exposure_room_usdt": round(exposure_room, 2),
    }


def fast_summary(f: Dict) -> Dict:
    keys = ("z", "rsi", "bar_up", "trend_up", "dist_ema_pct", "ema_slope_pct", "atr_pct", "breakout_up", "breakdown",
            "vol_ratio", "ema_cross_up", "sigma_bar_pct", "bb_prev_pctile", "dist_vwap_pct", "rs_rank", "swept_low",
            "lower_wick_ratio", "range_ok", "range_pos", "move_4h_pct", "adx", "trend_score", "extended", "pullback_atr",
            "donchian_break", "bos_retest_up", "failed_breakdown", "failed_breakout", "obi", "microprice_dev_bps",
            "spread_bps", "regime_4h", "vwap", "ema_fast", "prior_swing_low", "prior_swing_high")
    out = {}
    for k in keys:
        v = f.get(k)
        if isinstance(v, float):
            out[k] = round(v, 4) if math.isfinite(v) else None
        elif isinstance(v, (bool, int, str)) or v is None:
            out[k] = v
    return out


# ---------------------------------------------------------------- komite
def _weight(role: str, learned: Dict) -> float:
    base = R.ROLE_BASE_WEIGHT.get(role, 0.5)
    rel = (learned.get("reliability") or {}).get(role)
    if rel is None:
        return base
    return base * (0.5 + float(rel))       # OnlineLearner ile aynı biçim


def evaluate(ctx: Dict, p: CommitteeParams, learned: Optional[Dict] = None) -> Verdict:
    """Komite kararı — çok-sleeve EV yarışması.

    ctx: symbol, price, df (1m), slow{signal, chart, ...}, qual_cell, book, fees{maker_bps,
    taker_bps, verified}, open_positions, max_open, exposure_room, capital, max_order,
    notional_fn, p_win, halted, paused_reason, daily_loss_left_pct, market_type, news,
    rs_rank, reliable_only, lifecycle, mode
    """
    from . import sleeves_fast as SF
    from . import entry_optimizer as EO
    learned = learned or {}
    sym = ctx["symbol"]
    price = float(ctx["price"])
    slow = ctx.get("slow") or {}
    signal = slow.get("signal")
    chart = slow.get("chart")
    fast = fast_features(ctx["df"], p)
    if fast.get("ok"):
        fast["price"] = price
        fast = SF.extra_features(ctx["df"], fast, ctx.get("rs_rank"), ctx.get("book"), slow)
    votes: List[R.RoleVote] = []
    notes: List[str] = []
    allow_short = bool(p.allow_short and ctx.get("market_type") != "spot")
    news = ctx.get("news")
    fs = fast_summary(fast) if fast.get("ok") else {}

    # 1) rejim → şablon + izinli sleeve'ler (zıt sistemler aynı anda oy kullanmaz)
    regime = (chart or {}).get("regime") or next(
        (l.get("detail") for l in (signal or {}).get("layer_breakdown") or []
         if l.get("layer") == "_regime"), None)
    regime_label = (regime or {}).get("label")
    vol_label = str((signal or {}).get("volatility") or "medium")
    rv = R.role_regime(regime, vol_label, "LONG", p.counter_trend_mult)
    template = getattr(rv, "template", "mean_reversion")
    allowed = SF.allowed_sleeves(regime_label)
    lifecycle = ctx.get("lifecycle")
    mode = ctx.get("mode", "paper")
    if lifecycle is not None:
        allowed = [k for k in allowed if lifecycle.can_trade(k, mode)]
    else:
        # FAIL-CLOSED (2026-09-06): yaşam döngüsü BAĞLANMAMIŞSA topluluk katkıları aday
        # kümesine GİREMEZ. Katkılar tanım gereği SHADOW'dur; kapı yokken onları "izinli"
        # saymak, dışarıdan gelen KANITSIZ kodun EV yarışmasına girip kanıtlanmış bir
        # sleeve'in yerine geçmesi demekti. Bu, katkı hattı eklendiğinde altı çekirdek
        # testi düşürerek ortaya çıktı — testler semptomdu, açık kapı asıl sorundu.
        allowed = [k for k in allowed if k not in _CB.CONTRIB]
    rel = (learned.get("reliability") or {})
    paused = set(learned.get("paused_sleeves") or [])
    if paused:
        allowed = [k for k in allowed if k not in paused]
    if ctx.get("reliable_only"):
        # SEÇİCİ modda yalnız ölçülmüş güvenilirliği ≥ 0,5 olan sleeve'ler (ölçülmemiş = kapalı)
        # ÖLÇÜLMÜŞ ve güvenilmez (<0,5) sleeve'ler kapanır; ölçülmemişler açık kalır (aksi hâlde kilit:
        # hiç işlem yok → hiç ölçüm yok → hiç sleeve yok). Boyutu portföy modu ×0,7 yapar.
        sleeve_rel = learned.get("sleeve_reliability") or {}
        allowed = [k for k in allowed if (k not in sleeve_rel) or sleeve_rel[k] >= 0.5]

    # 2) tetikleyiciler — mevcut (dip/kırılım/geri çekilme/momentum/katalizör) + yeni sleeve'ler
    all_trig = triggers(fast, template, p, allow_short, news) + \
        (triggers(fast, "pullback" if template == "mean_reversion" else "mean_reversion", p, allow_short, news) if fast.get("ok") else [])
    fired = [t for t in all_trig if t["kind"] in allowed]
    now_ts = float(ctx.get("now") or time.time())          # killzone etiketi (replay'de simülasyon saati)
    fired += SF.fire_sleeves(fast, allowed, news, p, allow_short, now_ts, ctx.get("df"))
    # Kapanmış-bar sinyali: 30 sn'lik örnekleme devam eden barı görür; "yeşil bar" koşulu titreşir ve
    # kurulum kaçar. Son KAPANMIŞ bar tetiklediyse ve fiyat kaçmadıysa (≤ 0,5 ATR) sinyal geçerli sayılır.
    closed_note = ""
    if p.closed_bar_fallback and len(ctx["df"]) > 40:
        try:
            dfc = ctx["df"].iloc[:-1]
            fc = fast_features(dfc, p)
            if fc.get("ok"):
                fc["price"] = float(dfc["close"].iloc[-1])
                fc = SF.extra_features(dfc, fc, ctx.get("rs_rank"), ctx.get("book"), slow)
                if abs(price / fc["price"] - 1.0) * 100.0 <= 0.5 * float(fc.get("atr_pct") or 0.3):
                    seen = {t["kind"] for t in fired}
                    extra = [t for t in triggers(fc, template, p, allow_short, news) +
                             triggers(fc, "pullback" if template == "mean_reversion" else "mean_reversion", p, allow_short, news) +
                             SF.fire_sleeves(fc, allowed, news, p, allow_short, now_ts, dfc)
                             if t["kind"] in allowed and t["kind"] not in seen]
                    if extra:
                        for t in extra:
                            t["note"] = (t.get("note") or "") + " (kapanmış bar sinyali)"
                        fired += extra
                        closed_note = "kapanmış bar sinyali: " + ", ".join(t["kind"] for t in extra)
        except Exception:
            pass
    # rejim seçicinin susturdukları — kaçırılan-fırsat motoru bunları da gölgeler
    silenced = []
    seen_k = {t["kind"] for t in fired}
    # "gölgede (yaşam döngüsü)" ile "rejimde kapalı" AYRI sebeplerdir; tek etiket kullanılırsa panel
    # ölçüm-amaçlı gölgeyi rejim kararı sanır (aynı hata sınıfı: "kurulmadı" ≠ "bozuldu").
    _shadow = set()
    if lifecycle is not None:
        _shadow = {k for k in SF.ALL_SLEEVES if not lifecycle.can_trade(k, mode)}
    for t in all_trig + SF.fire_sleeves(fast, [k for k in SF.ALL_SLEEVES if k not in allowed],
                                        news, p, allow_short, now_ts, ctx.get("df")):
        if t["kind"] not in allowed and t["kind"] not in seen_k and t["kind"] not in {x["kind"] for x in silenced}:
            silenced.append({"kind": t["kind"], "direction": t["direction"], "note": t.get("note"),
                             "gate": ("SLEEVE_DURAKLATILDI" if t["kind"] in paused
                                      else "YAŞAM_DÖNGÜSÜ" if t["kind"] in _shadow else "REJİM_SEÇİCİ")})
    silenced_plans = []
    for t in silenced:
        if t["direction"] == "SHORT" and not allow_short:
            continue
        try:
            pl_ = build_plan(t["direction"], price, fast, {}, p)
            silenced_plans.append({**t, "plan": {k: pl_.get(k) for k in ("entry", "stop", "target", "stop_pct", "target_pct", "rr")}})
        except Exception:
            continue
    # haber katalizörü yalnız Tier-1/2 kaynaklı haberle (Tier-3 sosyal tek başına tetiklemez)
    fired = [t for t in fired if not (t["kind"] == "catalyst" and news and news.get("tier12_items", 1) == 0)]
    if not fired:
        w = _wait_note(fast, template, p, allow_short) if fast.get("ok") else {"note": fast.get("reason", "özellik yok"), "direction": None}
        vv = Verdict(sym, False, w.get("direction"), 0.0, 0.0, template, None,
                     [rv.to_dict()], [], 0.0, "maker", None, None,
                     f"BEKLE — {w['note']}", [w["note"], f"izinli sleeve'ler: {', '.join(allowed)}"], fast=fs)
        if silenced_plans:
            vv.silenced = silenced_plans           # type: ignore[attr-defined]
            vv.notes.append("susturulan: " + ", ".join(f"{SF.SLEEVE_TR.get(t['kind'], t['kind'])} ({t.get('gate')})" for t in silenced_plans))
        vv.regime = regime_label                   # type: ignore[attr-defined]
        return vv
    direction = fired[0]["direction"]
    if direction == "SHORT" and not allow_short:
        return Verdict(sym, False, None, 0.0, 0.0, template, None, [rv.to_dict()], [],
                       0.0, "maker", None, None, "BEKLE — spot'ta SHORT yok", [], fast=fs)
    rv = R.role_regime(regime, vol_label, direction, p.counter_trend_mult)

    # 3) roller (yön için bir kez) — tetikleyen sleeve'in kendi kanaati de bir oydur
    ms = R.role_market_structure(price, fast.get("atr_pct") or 0.3, chart, signal)
    trg0 = fired[0]
    votes.append(R.RoleVote(role="sleeve_sinyali", score=(1.0 if direction == "LONG" else -1.0) * (0.5 + 0.5 * float(trg0.get("size", 1.0))),
                            confidence=0.6, notes=[str(trg0.get("note") or trg0["kind"])]))
    if closed_note:
        notes.append(closed_note)
    votes += [ms, R.role_formations(slow.get("patterns"), slow.get("harmonics"), slow.get("candles")),
              R.role_indicator_consensus(slow.get("indicators")), rv,
              R.role_qualification(ctx.get("qual_cell"), direction, p.strict_qualification),
              R.role_mover(slow.get("mover_pick")), R.role_macro(slow.get("events"), signal),
              R.role_news_social(news, slow.get("social"), signal), R.role_orchestrator(signal)]

    # 4) maliyet bileşenleri
    fees = ctx.get("fees") or {"maker_bps": 0.0, "taker_bps": 10.0}
    book = ctx.get("book") or {}
    spread = float(book.get("spread_bps") or 0.0)
    cost_taker = (2.0 * float(fees["taker_bps"]) + spread + 4.0) / 100.0
    cost_maker = (2.0 * float(fees["maker_bps"]) + 4.0) / 100.0
    p_win = float(ctx.get("p_win", 0.5))
    notional_fn = ctx.get("notional_fn")
    p_fill = float(learned.get("p_maker_fill", p.p_maker_fill))

    # 5) EV YARIŞMASI — her ateşleyen sleeve için plan + fiş; en yüksek EV kazanır
    levels = ms.levels or {}
    competition = []
    for t in fired:
        pl = build_plan(direction, price, fast, levels, p)
        if t.get("stop_hint") and (t["stop_hint"] - price) * (1 if direction == "LONG" else -1) < 0:
            sp = abs(price - t["stop_hint"]) / price * 100.0
            if p.min_stop_pct <= sp <= p.max_stop_pct:
                pl["stop_pct"], pl["stop"], pl["stop_source"] = round(sp, 4), float(t["stop_hint"]), f"{t['kind']} yapısal"
                pl["target_pct"] = round(max(pl["target_pct"], p.rr_min * sp), 4)
                pl["target"] = price * (1 + (1 if direction == "LONG" else -1) * pl["target_pct"] / 100.0)
        if t.get("target_hint") and (t["target_hint"] - price) * (1 if direction == "LONG" else -1) > 0:
            tp = abs(t["target_hint"] - price) / price * 100.0
            if tp >= p.rr_min * pl["stop_pct"]:
                pl["target_pct"], pl["target"], pl["target_source"] = round(tp, 4), float(t["target_hint"]), f"{t['kind']} hedefi"
                pl["invalid"] = None
            else:
                pl["invalid"] = f"{t['kind']}: hedef %{tp:.2f} < {p.rr_min}×stop"
        pl["rr"] = round(pl["target_pct"] / max(1e-9, pl["stop_pct"]), 3)
        notional = float(notional_fn(pl["stop_pct"])) if notional_fn else 50.0
        ach = (achievable_target_pct(t["kind"], pl["target_pct"], fast.get("atr_pct"),
                                     SF.SLEEVE_TIME_STOP_MIN.get(t["kind"], p.max_hold_sec // 60),
                                     p_fill * cost_maker + (1 - p_fill) * cost_taker, p.min_gross_to_cost, learned)
               if p.achievable_target else None)
        tk = trade_ticket(pl, notional * float(t.get("size", 1.0)), p_win, cost_maker, cost_taker,
                          "maker", p_fill, float(ctx.get("capital", 1000.0)),
                          ctx.get("daily_loss_left_pct"), float(ctx.get("exposure_room", 0.0)), achievable=ach)
        competition.append({"kind": t["kind"], "size": t.get("size", 1.0), "ev_pct": tk["ev_pct"],
                            "ev_achievable_pct": tk.get("ev_achievable_pct"), "achievable_target_pct": tk.get("achievable_target_pct"),
                            "rr": pl["rr"], "invalid": pl.get("invalid"), "note": t.get("note"),
                            "exit_mode": t.get("exit_mode") or SF.SLEEVE_EXIT_MODE.get(t["kind"], "PARTIAL_AND_RUN"),
                            "_plan": pl, "_notional": notional, "_ach": ach})
    valid = [c for c in competition if not c["invalid"]]
    pool = valid or competition
    pool.sort(key=lambda c: (-(c["ev_pct"] if c["ev_pct"] is not None else -99),
                             TRIGGER_PRIORITY.index(c["kind"]) if c["kind"] in TRIGGER_PRIORITY else 50))
    best = pool[0]
    trg = next(t for t in fired if t["kind"] == best["kind"])
    plan = best["_plan"]
    notional = best["_notional"]
    exit_mode = best["exit_mode"]
    achievable = best.get("_ach")
    if plan.get("invalid"):
        notes.append(plan["invalid"])
    for c in competition:
        c.pop("_plan", None); c.pop("_notional", None); c.pop("_ach", None)

    # 6) giriş optimizasyonu (bölge / optimal / max chase)
    entry = EO.optimize_entry(direction, price, book, {"vwap": fast.get("vwap"), "ema_fast": fast.get("ema_fast"),
                                                       "support": levels.get("support"), "resistance": levels.get("resistance")},
                              fast.get("atr_pct") or 0.3, plan["stop_pct"], plan["target_pct"],
                              float(fees["maker_bps"]), float(fees["taker_bps"]), p_win, p_fill, spread)
    order_type = entry.get("order_type") or "maker"
    # KANIT KAPISI — taker girişi kanıt ister: aciliyet-0 sleeve'ler (defter momentumu/kırılım/katalizör) kanıtlanmadan
    # (PROVEN) taker giremez; maker 1 bar bekler. Canlı: taker girişler −6,02 $ (43), maker +1,24 $ (42).
    sleeve_state = str((learned.get("sleeve_states") or {}).get(trg["kind"]) or "UNPROVEN")
    if order_type == "taker" and p.taker_requires_proof and sleeve_state != "PROVEN":
        order_type = "maker"
        notes.append(f"kanıt kapısı: {SF.SLEEVE_TR.get(trg['kind'], trg['kind'])} {sleeve_state} → taker yerine MAKER (1 bar)")

    # 7) veto rolleri
    ce = R.role_cost_execution(plan["target_pct"], cost_taker, cost_maker, spread,
                               float(book.get("bid_depth_usd") or 0.0), float(book.get("ask_depth_usd") or 0.0),
                               notional, p.min_gross_to_cost, p_fill, p.max_spread_bps,
                               book_ok=bool(book.get("ok", True)), book_stale=bool(book.get("stale", False)))
    rk = R.role_risk(sym, direction, ctx.get("open_positions") or {}, slow.get("corr"),
                     int(ctx.get("max_open", 3)), float(ctx.get("exposure_room", 0.0)),
                     notional, ctx.get("paused_reason"), p_win, plan["rr"], bool(ctx.get("halted")))
    au = R.role_auditor(direction, plan["entry"], plan["stop"], plan["target"], plan["target_pct"],
                        cost_maker if order_type == "maker" else cost_taker, notional,
                        float(ctx.get("max_order", notional)), slow.get("age_sec"), p.max_ctx_age_sec, p.rr_min)
    votes += [ce, rk, au]

    # 8) ağırlıklı oy
    num = den = 0.0
    weights: Dict[str, float] = {}
    conf_num = conf_den = 0.0
    for v in votes:
        if v.role in R.VETO_ROLES or not v.data_ok:
            continue
        w = _weight(v.role, learned)
        weights[v.role] = w
        num += w * v.confidence * v.score
        den += w * v.confidence
        conf_num += w * v.confidence
        conf_den += w
    score = num / den if den > 0 else 0.0
    confidence = conf_num / conf_den if conf_den > 0 else 0.0
    if not fees.get("verified", False):
        confidence *= 0.85                     # doğrulanmamış ücret → HIGH_CONFIDENCE yok
        notes.append("ücret doğrulanmadı (statik tablo) → güven ×0,85")
    dir_sign = 1.0 if direction == "LONG" else -1.0
    aligned = score * dir_sign

    vetoes = [v.veto for v in votes if v.veto]
    if plan.get("invalid"):
        vetoes.append("DENETÇİ: " + plan["invalid"])
    if entry.get("invalidated"):
        vetoes.append(f"MAX CHASE aşıldı ({entry.get('max_chase'):.6g}) — kovalama yok")
    if best.get("ev_pct") is not None and best["ev_pct"] <= 0:
        vetoes.append(f"NEGATİF NET EV (%{best['ev_pct']:.3f}) — NO TRADE")
    size_mult = float(trg.get("size", 1.0)) * float(plan.get("size_penalty") or 1.0)
    if plan.get("note"):
        notes.append(plan["note"])
    for v in votes:
        size_mult *= v.size_mult
    # STOP-RİSK SKORU: stop olanların profili (canlı) → giriş anındaki uyaranlar
    from ..learn.missed import WARN as _WARN, _safe as _wsafe
    _fw = {**fast, "news_severe": bool((news or {}).get("severe_risk")), "spread_bps": spread}
    stop_warnings = [tr_ for name, (pred, tr_) in _WARN.items() if _wsafe(pred, _fw, 1.0 if direction == "LONG" else -1.0)]
    stop_risk = {"n": len(stop_warnings), "warnings": stop_warnings}
    if len(stop_warnings) >= p.stop_risk_veto_warnings:
        vetoes.append(f"STOP RİSKİ: {len(stop_warnings)} uyaran ({', '.join(stop_warnings)})")
    elif len(stop_warnings) >= p.stop_risk_size_warnings:
        size_mult *= 0.6
        notes.append(f"stop riski: {len(stop_warnings)} uyaran ({', '.join(stop_warnings)}) → boyut ×0,6")
    size_mult = float(min(1.5, max(0.0, size_mult)))
    voted_ok = aligned >= p.theta and confidence >= p.min_confidence
    if not voted_ok and not vetoes:
        vetoes.append(f"OY {aligned:+.2f} < eşik {p.theta}" if aligned < p.theta
                      else f"GÜVEN {confidence:.2f} < {p.min_confidence}")
    # VETO İNCELEMESİ: formasyon varlıkları + indikatör al/sat sayımı + strateji uyumu + haber → yumuşak
    # vetolar (OY/GÜVEN) bütünsel kanıtla ×0,6 boyutla aşılabilir; sert vetolar asla.
    veto_review = None
    if vetoes:
        from . import veto_review as VR_
        veto_review = VR_.review(direction, vetoes, slow, fast, news, trg["kind"], allowed, template, regime_label, aligned)
        if veto_review["decision"] == "AÇ":
            notes.append("VETO İNCELEMESİ aştı: " + veto_review["summary_tr"])
            vetoes = []
            size_mult *= veto_review["size_mult"]
        else:
            notes.append("VETO İNCELEMESİ: " + veto_review["summary_tr"])
    allowed_trade = not vetoes and size_mult > 0.0
    if allowed_trade:
        size_mult = float(max(p.size_floor, size_mult))       # taban: haircut'lar çarpılır ama boyut sıfıra sürünmez

    ticket = trade_ticket(plan, notional * size_mult if allowed_trade else notional, p_win, cost_maker,
                          cost_taker, order_type, p_fill, float(ctx.get("capital", 1000.0)),
                          ctx.get("daily_loss_left_pct"), float(ctx.get("exposure_room", 0.0)), achievable=achievable)
    ticket["sleeve_state"] = sleeve_state
    if ticket.get("ev_achievable_pct") is not None and allowed_trade:
        if ticket["ev_achievable_pct"] <= 0:
            notes.append(f"ulaşılabilir hedef %{ticket['achievable_target_pct']:.2f} ({ticket.get('achievable_source')}) ile EV %{ticket['ev_achievable_pct']:+.3f} ≤ 0 "
                         f"→ yalnız kanıt boyutu (fişteki EV planın hedefine dayanır)")
    kind_tr = SF.SLEEVE_TR.get(trg["kind"], TRIGGER_TR.get(trg["kind"], trg["kind"]))
    result = (f"AÇ {direction} · {kind_tr} · EV %{best['ev_pct']:+.3f} · oy {aligned:+.2f} · güven {confidence:.2f} · ×{size_mult:.2f}"
              if allowed_trade else "VETO: " + "; ".join(vetoes))
    others = [c["kind"] for c in competition if c["kind"] != best["kind"]]
    if others:
        notes.append("yarışan sleeve'ler: " + ", ".join(f"{SF.SLEEVE_TR.get(k, k)}" for k in others))
    v = Verdict(sym, allowed_trade, direction, aligned, confidence, template, trg["kind"],
                [x.to_dict() for x in votes], vetoes, size_mult if allowed_trade else 0.0,
                order_type, plan, ticket, result, notes + [trg["note"]], weights, fast=fs)
    v.entry = entry                       # type: ignore[attr-defined]
    v.exit_mode = exit_mode               # type: ignore[attr-defined]
    v.competition = competition           # type: ignore[attr-defined]
    v.valid_until = time.time() + float(p.max_hold_sec) / 4.0   # type: ignore[attr-defined]
    v.regime = regime_label               # type: ignore[attr-defined]
    v.veto_review = veto_review           # type: ignore[attr-defined]
    v.stop_risk = stop_risk               # type: ignore[attr-defined]
    v.atr_hint = trg.get("atr_hint")      # type: ignore[attr-defined]  (swing: 4h ATR ile chandelier)
    if silenced_plans:
        v.silenced = silenced_plans       # type: ignore[attr-defined]
    return v


def describe() -> Dict:
    return {
        "id": STRATEGY_ID, "name": STRATEGY_NAME,
        "roles": [{"id": k, "title": v, "base_weight": R.ROLE_BASE_WEIGHT.get(k),
                   "veto": k in R.VETO_ROLES} for k, v in R.ROLE_TITLES.items()],
        "flow": ["hızlı özellikler (1 dk)", "rejim → şablon (geri çekilme / ortalamaya dönüş)",
                 "tetikleyici şart", "yapısal stop/hedef", "ağırlıklı oy (öğrenilmiş güvenilirlik)",
                 "maliyet/risk/denetçi vetosu", "işlem fişi", "maker-öncelikli emir",
                 "kapanışta post-mortem + ders"],
        "measured_caveats": [
            "Gösterge tablosu takip getirisi 4h'de negatif ölçüldü → taban ağırlık 0,30",
            "Harmonik ve mum formasyonlarının yön öngörüsü ölçüldü, bulunamadı → yön oyu yok",
            "Nitelendirme matrisinde sıfır QUALIFIED hücre → nitelenmemiş hücre boyut ×0,5-0,6",
            "Kazanma olasılığı öncülü 0,5; komitenin kendi paper kaydıyla Beta güncellenir",
        ],
    }


# ---------------------------------------------------------------- piyasa riski / nakit modu
def market_risk(slow_map: Dict[str, Dict], market_news: Optional[Dict] = None,
                breadth_caution: float = -0.3, breadth_cash: float = -0.5) -> Dict:
    """Portföy düzeyi risk-off kararı. Seviye 0 normal · 1 DİKKAT (yeni giriş yok) ·
    2 NAKİT MODU (açık pozisyonlar kapatılır). Girdi: ağır bağlamı olan paritelerin 4h
    konsensüsü (genişlik), BTC rejimi, haber tarayıcının piyasa risk puanı."""
    longs = shorts = n = 0
    btc_reg = None
    for sym, s in (slow_map or {}).items():
        sig = (s or {}).get("signal") or {}
        d = str(sig.get("direction") or "FLAT").upper()
        if d in ("LONG", "SHORT"):
            n += 1
            longs += d == "LONG"
            shorts += d == "SHORT"
        if sym.startswith("BTC"):
            btc_reg = ((s or {}).get("chart") or {}).get("regime") or {}
            btc_reg = btc_reg.get("label")
    breadth = (longs - shorts) / n if n else 0.0
    news_level = int((market_news or {}).get("level") or 0)
    reasons = []
    level = 0
    if n >= 3 and breadth <= breadth_caution:
        level = 1
        reasons.append(f"4h genişlik {breadth:+.2f} (LONG {longs} / SHORT {shorts})")
    if n >= 3 and breadth <= breadth_cash and btc_reg == "TREND AŞAĞI":
        level = 2
        reasons.append("BTC TREND AŞAĞI + genişlik çok negatif")
    if news_level >= 1:
        level = max(level, 1)
        reasons.append(f"haber risk-off puanı {(market_news or {}).get('risk_off_score')}")
    if news_level >= 2:
        level = 2
        reasons.append("haber: sistemik risk (borsa hack / düzenleyici darbe / depeg)")
    label = {0: "NORMAL", 1: "DİKKAT — yeni giriş yok", 2: "NAKİT MODU — pozisyonlar kapatılır"}[level]
    return {"level": level, "label": label, "reasons": reasons, "breadth": round(breadth, 3),
            "n_signals": n, "btc_regime": btc_reg, "news_level": news_level}
