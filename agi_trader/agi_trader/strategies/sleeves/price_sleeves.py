"""
Fiyat-temelli sleeve'ler: trend, kesitsel momentum, kısa vadeli dönüş, vade yapısı.

Hepsi yalnız fiyat serisi kullanır — ek veri/API gerektirmez, bu yüzden
4,5 yıllık yerel arşivle TAM olarak doğrulanabilirler.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .base import Sleeve, vol_target_scale, cross_sectional_rank, zscore


# ===========================================================================
# 1) TREND — mevcut canlı çekirdek (referans/baseline)
# ===========================================================================
class TrendSleeve(Sleeve):
    """Trend200 + Mom20, varlık-bazında vol-hedefli.

    OOS-doğrulanmış (2022-2026, fit YOK): 17 varlık portföyünde Sharpe 1,37.
    PARAMETRELERİ OPTİMİZE ETME — literatür standardı oldukları için doğal
    olarak örneklem dışıdırlar; ayarlamak bu özelliği yok eder."""

    name = "trend"
    warmup = 230

    def __init__(self, sma: int = 200, mom: int = 20, vol_window: int = 30,
                 target_daily_vol: float = 0.025, **kw):
        super().__init__(sma=sma, mom=mom, vol_window=vol_window,
                         target_daily_vol=target_daily_vol, **kw)

    def positions(self, prices: pd.DataFrame, **ctx) -> pd.DataFrame:
        p = self.params
        above = prices > prices.rolling(p["sma"]).mean()
        momentum = prices.pct_change(p["mom"]) > 0
        raw = (above & momentum).astype(float)
        scale = vol_target_scale(prices, p["target_daily_vol"], p["vol_window"])
        n = max(1, prices.shape[1])
        return (raw * scale).fillna(0.0) / n


# ===========================================================================
# 2) KESİTSEL MOMENTUM — göreli güç
# ===========================================================================
class CrossSectionalSleeve(Sleeve):
    """Varlıkları geçmiş getiriye göre sırala, üstteki %top_pct'i tut.

    ⚠️ BU SLEEVE'DE BİR KEZ LOOK-AHEAD BUG'I YAŞANDI: seçim maskesi bugünün
    getirisini kullanıp bugünü seçiyordu → Sharpe 3,68 (sahte). Düzeltilince
    0,80'e çöktü. `Sleeve.returns` içindeki .shift(1) tek savunma değildir;
    sıralama da yalnız `lookback` penceresinin KAPANMIŞ verisiyle yapılır.

    Trend'den farkı: trend MUTLAK (fiyat > SMA), bu GÖRECELİdir (en iyiler).
    Yatay piyasada trend nakde geçer, bu yine de en güçlüyü tutar → korelasyon
    tam değildir."""

    name = "xsec_momentum"
    warmup = 120

    def __init__(self, lookback: int = 90, skip: int = 5, top_pct: float = 0.30,
                 vol_window: int = 30, target_daily_vol: float = 0.025,
                 min_assets: int = 4, neutral: bool = False, **kw):
        """neutral=True → dolar-nötr (üsttekiler long, alttakiler short).

        NEDEN ÖNEMLİ: sadece-long kesitsel momentum, trend sleeve'iyle ORTAK
        PİYASA BETASI taşır — ölçümde korelasyon +0,73 çıktı, yani yeni bir
        getiri akışı değil aynı bahsin tekrarı. Long-short kurgu bu ortak betayı
        götürür ve geriye saf GÖRECELİ güç kalır. Dağıtılabilirlik notu: short
        bacak perp/futures gerektirir, spot-only hesapta uygulanamaz."""
        super().__init__(lookback=lookback, skip=skip, top_pct=top_pct,
                         vol_window=vol_window, target_daily_vol=target_daily_vol,
                         min_assets=min_assets, neutral=neutral, **kw)
        if neutral:
            self.name = "xsec_neutral"

    def positions(self, prices: pd.DataFrame, **ctx) -> pd.DataFrame:
        p = self.params
        if prices.shape[1] < p["min_assets"]:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # "skip": son N günü atla (kısa vadeli dönüş etkisini momentumdan ayır —
        # Jegadeesh-Titman standardı). lookback penceresi t-skip'te kapanır.
        past = prices.shift(p["skip"])
        mom = past / past.shift(p["lookback"]) - 1.0

        rank = cross_sectional_rank(mom)
        scale = vol_target_scale(prices, p["target_daily_vol"], p["vol_window"])

        longs = (rank >= (1.0 - p["top_pct"])).astype(float)
        if not p["neutral"]:
            # yalnız MUTLAK olarak da pozitif momentumlular (ayı piyasası koruması)
            longs = longs.where(mom > 0, 0.0)
            w = (longs * scale).fillna(0.0)
            n_sel = longs.sum(axis=1).replace(0, np.nan)
            return w.div(n_sel, axis=0).fillna(0.0)

        shorts = (rank <= p["top_pct"]).astype(float)
        n_l = longs.sum(axis=1).replace(0, np.nan)
        n_s = shorts.sum(axis=1).replace(0, np.nan)
        w_l = (longs * scale).div(n_l, axis=0).fillna(0.0)
        w_s = (shorts * scale).div(n_s, axis=0).fillna(0.0)
        # her bacak brüt maruziyetin yarısı → net ≈ 0
        return (w_l - w_s) * 0.5


# ===========================================================================
# 3) KISA VADELİ DÖNÜŞ — trend ile negatif korelasyonlu
# ===========================================================================
class ShortReversalSleeve(Sleeve):
    """Yükseliş trendindeki varlıkta kısa vadeli aşırı satımı satın al.

    Mantık: uzun vadeli trend yukarıyken (fiyat > SMA200) 2-4 günlük sert
    düşüşler likidite kaynaklıdır ve genelde geri alınır. Trend sleeve'i
    düşüşte pozisyonu AZALTIRKEN bu sleeve ARTIRIR → yapısal olarak negatif
    korelasyon üretir (çeşitlendirme değeri buradan gelir).

    Yüksek devir hızı ⇒ maliyete duyarlıdır; maliyet modeli gerçekçi olmalı."""

    name = "short_reversal"
    warmup = 230

    def __init__(self, trend_sma: int = 200, lookback: int = 3,
                 entry_z: float = -1.0, hold: int = 3, z_window: int = 60,
                 vol_window: int = 30, target_daily_vol: float = 0.02, **kw):
        super().__init__(trend_sma=trend_sma, lookback=lookback, entry_z=entry_z,
                         hold=hold, z_window=z_window, vol_window=vol_window,
                         target_daily_vol=target_daily_vol, **kw)

    def positions(self, prices: pd.DataFrame, **ctx) -> pd.DataFrame:
        p = self.params
        uptrend = prices > prices.rolling(p["trend_sma"]).mean()
        short_ret = prices.pct_change(p["lookback"])
        z = zscore(short_ret, p["z_window"])

        entry = (uptrend & (z <= p["entry_z"])).astype(float)
        # `hold` gün boyunca pozisyonda kal (giriş sinyalini ileriye yay)
        held = entry.rolling(p["hold"], min_periods=1).max()
        held = held.where(uptrend, 0.0)                  # trend bozulursa çık

        scale = vol_target_scale(prices, p["target_daily_vol"], p["vol_window"])
        w = (held * scale).fillna(0.0)
        n_sel = held.sum(axis=1).replace(0, np.nan)
        return w.div(n_sel, axis=0).fillna(0.0)


# ===========================================================================
# 4) VADE YAPISI / TAŞIMA VEKİLİ — emtia ETF'lerinde roll getirisi
# ===========================================================================
class TermStructureSleeve(Sleeve):
    """Emtia ETF'lerinde contango/backwardation vekili.

    Gerçek vade eğrisi verisi ücretli olduğu için, spot-benzeri ve vadeli-tabanlı
    ETF çiftlerinin göreli performansı VEKİL olarak kullanılır (ör. altın için
    GLD spot'a yakın, DBC vadeli sepet). Oran yükseliyorsa backwardation
    (roll getirisi pozitif) → long.

    DÜRÜSTLÜK NOTU: bu bir vekildir, gerçek eğri değildir. Kabul kapısını
    geçemezse kullanılmaz — vekil olması onu meşru kılmaz."""

    name = "term_structure"
    warmup = 130

    def __init__(self, pairs: Optional[List[tuple]] = None, lookback: int = 60,
                 z_window: int = 120, entry_z: float = 0.5,
                 target_daily_vol: float = 0.02, vol_window: int = 30, **kw):
        super().__init__(pairs=pairs or [("DBC", "GLD"), ("USO", "UNG"),
                                         ("SLV", "GLD"), ("DBA", "CORN")],
                         lookback=lookback, z_window=z_window, entry_z=entry_z,
                         target_daily_vol=target_daily_vol, vol_window=vol_window, **kw)

    def positions(self, prices: pd.DataFrame, **ctx) -> pd.DataFrame:
        p = self.params
        out = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        usable = [(a, b) for a, b in p["pairs"]
                  if a in prices.columns and b in prices.columns]
        if not usable:
            return out

        scale = vol_target_scale(prices, p["target_daily_vol"], p["vol_window"])
        for a, b in usable:
            ratio = prices[a] / prices[b]
            slope = ratio.pct_change(p["lookback"])
            z = zscore(slope, p["z_window"])
            out[a] = out[a] + (z > p["entry_z"]).astype(float) * scale[a].fillna(0.0)
        n = max(1, len(usable))
        return (out / n).fillna(0.0)


# ===========================================================================
# Kayıt defteri
# ===========================================================================
PRICE_SLEEVES = {
    "trend": TrendSleeve,
    "xsec_momentum": CrossSectionalSleeve,
    "short_reversal": ShortReversalSleeve,
    "term_structure": TermStructureSleeve,
}
