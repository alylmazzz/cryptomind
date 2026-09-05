"""
Dip-Scalp — YouTube "Kripto Al-Sat botu yaptım, ve sonuç…" (Onur Nacos,
youtube.com/watch?v=xsszspuRYXs) videosundaki kurulumun CryptoMind'a uyarlanması.

Videoda anlatılan sistem (transkriptten):
  • düşen coini al, "azıcık" yükselince sat — ortalamaya dönüş (mean reversion)
  • TP/SL oranı 1 : 1,6  ("1.000 $ → 1.160'ta sat, 900'e düşerse de sat")
  • günlük zarar limiti  (5.000 $ sermayede 1.000 $ — "sabah en az 4.000 olsun")
  • tepe kârının yarısını geri verince çık ("%100 kârdan %50'ye düşünce 1.050'de çık")
  • 30 saniyede bir kontrol döngüsü
  • önce SANAL 5.000 $ ile self-test, gerçek para sonra
  • VPS'te 7/24

Videonun ÖLÇÜLMÜŞ sonucu (bu modülün asıl girdisi):
  • 10 işlem, +116 $ brüt … ama komisyon toplamı 323 $ → "Binance'i zengin ettik"
  • 0-15 dk'lık işlemler en çok zarar; 15-60 dk'lık işlemler en çok kâr
  • BNB ile komisyon %10 iner (3,30 → 3 $) ama "kurtarmıyor"

CryptoMind uyarlaması bu dersleri KURAL yapar:
  • KOMİSYON KAPISI: brüt hedef ≥ K × gidiş-dönüş maliyet; yoksa işlem YOK
  • asgari tutma 15 dk (stop hariç hiçbir çıkış daha erken tetiklenmez)
  • azami tutma 60 dk (zaman-stop) — 60 dk üstü videoda kazandırmamıştı
  • stop ölçeği ufka bağlı: k·σ_bar·√H (sabit yüzde değil — bkz. NET1 tuzağı #1)
  • BNB indirimi yalnız kullanıcı açarsa hesaba girer

Bu modül EMİR GÖNDERMEZ, ağa çıkmaz, RNG kullanmaz; saf fonksiyonlardır.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

VIDEO_URL = "https://www.youtube.com/watch?v=xsszspuRYXs"
STRATEGY_ID = "video_dip_scalp"
STRATEGY_NAME = "Dip-Scalp 1:1,6 (video kurulumu)"

# Binance spot taker %0,10; BNB ile %0,075. Vadeli %0,04/%0,05 — venue'ye göre ezilir.
BINANCE_SPOT_TAKER_BPS = 10.0
BNB_DISCOUNT = 0.25            # videoda "%10" dendi; Binance resmi oranı %25'tir —
                               # kullanıcı yüzdesini kendisi de girebilir


@dataclass
class ScalpParams:
    """Videodaki ayarlar + CryptoMind kapıları. Hepsi panelden değiştirilebilir."""
    rr: float = 1.6                     # hedef / stop oranı (video: 1x → 1,6x)
    dip_z: float = 1.5                  # z-skoru eşiği (|close − SMA| / σ)
    rsi_max: float = 35.0               # LONG için RSI tavanı (SHORT için 100 − bu)
    lookback: int = 20                  # SMA/σ penceresi (bar)
    stop_sigma_mult: float = 1.5        # stop = k · σ_bar · √H
    min_stop_pct: float = 0.35
    max_stop_pct: float = 4.0
    giveback: float = 0.5               # tepe kârın bu oranına düşünce çık (video: %50)
    giveback_activate_r: float = 0.5    # geri-verme kuralı en az 0,5R kâr görülünce silahlanır
    min_hold_sec: int = 15 * 60         # videoda 0-15 dk = zarar bölgesi
    max_hold_sec: int = 60 * 60         # videoda 15-60 dk = kâr bölgesi; sonrası zaman-stop
    loop_sec: int = 30                  # video: 30 sn'de bir
    fee_bps: float = BINANCE_SPOT_TAKER_BPS
    bnb_discount: bool = False
    bnb_discount_pct: float = BNB_DISCOUNT
    min_gross_to_cost: float = 2.0      # KOMİSYON KAPISI: brüt hedef ≥ 2× maliyet
    allow_short: bool = False           # spot'ta SHORT yok; vadelide kullanıcı açar
    bar_minutes: int = 1

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict]) -> "ScalpParams":
        p = cls()
        for k, v in (d or {}).items():
            if hasattr(p, k) and v is not None:
                cur = getattr(p, k)
                try:
                    setattr(p, k, type(cur)(v) if not isinstance(cur, bool) else bool(v))
                except (TypeError, ValueError):
                    pass
        return p.validated()

    def validated(self) -> "ScalpParams":
        """Anlamsız değerleri güvenli aralığa kelepçeler — sessiz saçmalık yerine."""
        self.rr = float(min(5.0, max(0.5, self.rr)))
        self.dip_z = float(min(4.0, max(0.5, self.dip_z)))
        self.rsi_max = float(min(50.0, max(10.0, self.rsi_max)))
        self.lookback = int(min(200, max(5, self.lookback)))
        self.stop_sigma_mult = float(min(5.0, max(0.5, self.stop_sigma_mult)))
        self.giveback = float(min(0.9, max(0.1, self.giveback)))
        self.giveback_activate_r = float(min(2.0, max(0.1, self.giveback_activate_r)))
        self.min_hold_sec = int(min(6 * 3600, max(0, self.min_hold_sec)))
        self.max_hold_sec = int(min(24 * 3600, max(self.min_hold_sec + 60, self.max_hold_sec)))
        self.loop_sec = int(min(600, max(10, self.loop_sec)))
        self.fee_bps = float(min(100.0, max(0.0, self.fee_bps)))
        self.bnb_discount_pct = float(min(0.9, max(0.0, self.bnb_discount_pct)))
        self.min_gross_to_cost = float(min(10.0, max(1.0, self.min_gross_to_cost)))
        self.bar_minutes = int(min(60, max(1, self.bar_minutes)))
        return self


# Videodaki BİREBİR kurulum — panelde "VİDEO AYARLARI" düğmesi bunu yükler.
VIDEO_PRESET = ScalpParams(rr=1.6, giveback=0.5, min_hold_sec=900, max_hold_sec=3600,
                           loop_sec=30, fee_bps=BINANCE_SPOT_TAKER_BPS,
                           bnb_discount=False).to_dict()
VIDEO_CAPITAL_USDT = 5000.0
VIDEO_DAILY_LOSS_PCT = 0.20     # 1.000 / 5.000 — bilinçli olarak VARSAYILAN DEĞİL (aşağıda)
DEFAULT_DAILY_LOSS_PCT = 0.05   # CryptoMind: %4-5. Videodaki %20 tek gecede sermayenin
                                # beşte birini feda eder; panelde ayrı seçenek olarak durur.


# ---------------------------------------------------------------- maliyet
def effective_fee_bps(p: ScalpParams) -> float:
    """Tek yön komisyon (bps). BNB indirimi yalnız açıksa uygulanır."""
    f = float(p.fee_bps)
    if p.bnb_discount:
        f *= (1.0 - float(p.bnb_discount_pct))
    return f


def roundtrip_cost_pct(p: ScalpParams, spread_bps: float = 0.0,
                       slippage_bps: float = 2.0) -> float:
    """Gidiş-dönüş maliyet (%): 2× komisyon + spread + 2× kayma.
    Videodaki "görmediğim komisyonlar" tam bu satırdır."""
    bps = 2.0 * effective_fee_bps(p) + max(0.0, spread_bps) + 2.0 * max(0.0, slippage_bps)
    return bps / 100.0


def fee_gate(target_gross_pct: float, cost_pct: float, p: ScalpParams) -> Dict:
    """KOMİSYON KAPISI. Videonun ölçülmüş dersi: brüt kâr maliyeti karşılamıyorsa
    bot 'kazanırken' hesap eriyor. Brüt hedef / maliyet ≥ K değilse işlem yok."""
    if cost_pct <= 0:
        return {"ok": True, "ratio": float("inf"), "note": "maliyet sıfır varsayıldı"}
    ratio = float(target_gross_pct) / float(cost_pct)
    ok = ratio >= p.min_gross_to_cost
    return {"ok": ok, "ratio": round(ratio, 2),
            "note": (f"brüt hedef %{target_gross_pct:.3f} = maliyetin {ratio:.2f} katı"
                     + ("" if ok else f" < {p.min_gross_to_cost} → İŞLEM YOK"))}


# ---------------------------------------------------------------- özellikler
def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    # Wilder yumuşatma
    au = up.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    ad = dn.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = au / ad.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.fillna(50.0)


def compute_features(df: pd.DataFrame, p: ScalpParams) -> Dict:
    """Son bar için: z-skoru, RSI, bar-başı σ (%), kapanış, SMA.
    Yalnız GEÇMİŞ ve MEVCUT barı kullanır; ileriye bakış yok."""
    if df is None or len(df) < max(p.lookback + 2, 16):
        return {"ok": False, "reason": f"yetersiz bar ({0 if df is None else len(df)})"}
    c = df["close"].astype(float)
    sma = c.rolling(p.lookback).mean()
    sd = c.rolling(p.lookback).std(ddof=0)
    last, m, s = float(c.iloc[-1]), float(sma.iloc[-1]), float(sd.iloc[-1])
    if not (math.isfinite(m) and math.isfinite(s)) or s <= 0 or last <= 0:
        return {"ok": False, "reason": "σ hesaplanamadı (düz seri?)"}
    z = (last - m) / s
    r = float(_rsi(c).iloc[-1])
    # bar-başı getiri σ'sı (%) — stop ölçeği için
    ret = np.log(c).diff().dropna().tail(max(p.lookback * 3, 30))
    sigma_bar_pct = float(ret.std(ddof=0) * 100.0) if len(ret) > 5 else float("nan")
    return {"ok": True, "close": last, "sma": m, "std": s, "z": float(z), "rsi": r,
            "sigma_bar_pct": sigma_bar_pct, "n_bars": int(len(df))}


# ---------------------------------------------------------------- sinyal
def signal(df: pd.DataFrame, p: ScalpParams, market_type: str = "spot") -> Dict:
    """Videodaki 'düşmüş coin' kuralı: fiyat SMA'nın dip_z·σ altında VE RSI aşırı
    satımda → LONG. Vadelide (ve izin verilmişse) simetrik SHORT.
    Çıktı her zaman gerekçe taşır; sinyal yoksa direction=None."""
    f = compute_features(df, p)
    if not f.get("ok"):
        return {"direction": None, "reasons": [f.get("reason", "özellik yok")], **f}
    reasons: List[str] = []
    direction: Optional[str] = None
    z, r = f["z"], f["rsi"]
    if z <= -p.dip_z and r <= p.rsi_max:
        direction = "LONG"
        reasons.append(f"dip: z={z:.2f} ≤ −{p.dip_z} ve RSI {r:.0f} ≤ {p.rsi_max:.0f}")
    elif (p.allow_short and market_type != "spot"
          and z >= p.dip_z and r >= 100.0 - p.rsi_max):
        direction = "SHORT"
        reasons.append(f"tepe: z={z:.2f} ≥ {p.dip_z} ve RSI {r:.0f} ≥ {100 - p.rsi_max:.0f}")
    else:
        reasons.append(f"koşul yok: z={z:.2f}, RSI {r:.0f}")
    return {"direction": direction, "reasons": reasons, **f}


def plan_trade(direction: str, entry: float, sigma_bar_pct: float,
               p: ScalpParams) -> Dict:
    """Stop = k·σ_bar·√H (H = azami tutma barı), hedef = rr × stop.
    Videodaki 1:1,6 oranı korunur; stop'un ÖLÇEĞİ ufka bağlıdır."""
    H = max(1, int(round(p.max_hold_sec / 60.0 / p.bar_minutes)))
    if not (isinstance(sigma_bar_pct, (int, float)) and math.isfinite(sigma_bar_pct)
            and sigma_bar_pct > 0):
        stop_pct = p.min_stop_pct
        note = "σ yok → asgari stop"
    else:
        stop_pct = float(np.clip(p.stop_sigma_mult * sigma_bar_pct * math.sqrt(H),
                                 p.min_stop_pct, p.max_stop_pct))
        note = f"stop = {p.stop_sigma_mult}·σ({sigma_bar_pct:.3f}%)·√{H}"
    target_pct = stop_pct * p.rr
    s = 1.0 if direction == "LONG" else -1.0
    return {
        "direction": direction, "entry": float(entry),
        "stop_pct": round(stop_pct, 4), "target_pct": round(target_pct, 4),
        "stop": float(entry * (1.0 - s * stop_pct / 100.0)),
        "target": float(entry * (1.0 + s * target_pct / 100.0)),
        "rr": p.rr, "hold_bars": H, "note": note,
    }


# ---------------------------------------------------------------- çıkış
@dataclass
class ExitState:
    """Açık pozisyonun çıkış kurallarına gereken asgari durumu."""
    direction: str
    entry: float
    stop: float
    target: float
    opened_ts: float
    peak_pnl_pct: float = 0.0            # görülen en yüksek gerçekleşmemiş kâr (%)
    stop_pct: float = 1.0                # R hesabı için

    def pnl_pct(self, price: float) -> float:
        s = 1.0 if self.direction == "LONG" else -1.0
        return (price / self.entry - 1.0) * 100.0 * s


def exit_decision(st: ExitState, price: float, p: ScalpParams,
                  now: Optional[float] = None) -> Optional[Dict]:
    """Sıra ÖNEMLİDİR ve videodaki mantığı izler:
      1. STOP        — her zaman, asgari tutmaya bakmaz (sermaye önce)
      2. TP          — hedef vuruldu
      3. GIVEBACK    — tepe kârın giveback oranına düştü (en az activate_r kâr görüldüyse)
      4. TIME_STOP   — azami tutma aşıldı
    2-4 asgari tutma süresinden ÖNCE tetiklenmez (video: 0-15 dk zarar bölgesi).
    Tepe kârı bu çağrıda GÜNCELLENİR (yan etki: st.peak_pnl_pct)."""
    now = time.time() if now is None else float(now)
    pnl = st.pnl_pct(price)
    st.peak_pnl_pct = max(st.peak_pnl_pct, pnl)
    age = now - st.opened_ts
    s = 1.0 if st.direction == "LONG" else -1.0

    if (price - st.stop) * s <= 0:
        return {"reason": "STOP", "pnl_pct": pnl, "age_sec": age}
    if age < p.min_hold_sec:
        return None
    if (price - st.target) * s >= 0:
        return {"reason": "TP", "pnl_pct": pnl, "age_sec": age}
    armed = st.peak_pnl_pct >= p.giveback_activate_r * max(1e-9, st.stop_pct)
    if armed and pnl <= st.peak_pnl_pct * p.giveback:
        return {"reason": "GIVEBACK", "pnl_pct": pnl, "age_sec": age,
                "peak_pnl_pct": st.peak_pnl_pct}
    if age >= p.max_hold_sec:
        return {"reason": "TIME_STOP", "pnl_pct": pnl, "age_sec": age}
    return None


# ---------------------------------------------------------------- panel
HOLD_BUCKETS = ((0, 15 * 60, "0-15 dk"), (15 * 60, 60 * 60, "15-60 dk"),
                (60 * 60, 10 ** 9, "60+ dk"))


def hold_bucket(hold_sec: float) -> str:
    for lo, hi, ad in HOLD_BUCKETS:
        if lo <= hold_sec < hi:
            return ad
    return HOLD_BUCKETS[-1][2]


def describe() -> Dict:
    """Panelde gösterilen strateji künyesi — ne yaptığı ve NEDEN böyle kurulduğu."""
    return {
        "id": STRATEGY_ID, "name": STRATEGY_NAME, "source": VIDEO_URL,
        "rules": [
            "Düşen pariteyi al (fiyat SMA'nın z·σ altında + RSI aşırı satım), küçük yükselişte sat",
            "Hedef/stop = 1 : 1,6 (video); stop ölçeği ufka bağlı k·σ·√H",
            "Tepe kârın %50'sini geri verince çık (video: '1.100'e çıkıp dönerse 1.050'de sat')",
            "Asgari tutma 15 dk, azami 60 dk — videoda ölçülen kâr bölgesi",
            "Günlük zarar limitine ulaşınca dur (video: 5.000 $'da 1.000 $)",
            "30 sn'de bir kontrol; önce sanal sermayeyle self-test, sonra gerçek",
        ],
        "measured_lessons": [
            "Videoda 10 işlem +116 $ ama komisyon 323 $ — kâr komisyona gitti",
            "0-15 dk işlemler zarar, 15-60 dk kâr bölgesi (videonun kendi verisi)",
            "BNB indirimi %25'e kadar iner ama komisyon problemini çözmez",
        ],
        "cryptomind_gates": [
            "KOMİSYON KAPISI: brüt hedef ≥ K× gidiş-dönüş maliyet, yoksa işlem yok",
            "Konsensüs motoru ters yönde güçlüyse VETO; aynı yöndeyse boyut artar",
            "Nitelendirme matrisi hücresi NİTELENMEMİŞSE boyut ×0,5 (katı modda VETO)",
            "Fırsat kapıları (net getiri, EV, likidite, veri kalitesi) geçilmeli",
            "HMM rejim çarpanı: trende karşı ×0,5, aşırı oynaklıkta ×0,6",
            "Sistem sağlığı RED/UNKNOWN ise hiçbir giriş yok",
        ],
        "video_preset": VIDEO_PRESET,
        "video_capital_usdt": VIDEO_CAPITAL_USDT,
        "video_daily_loss_pct": VIDEO_DAILY_LOSS_PCT,
        "default_daily_loss_pct": DEFAULT_DAILY_LOSS_PCT,
    }
