"""
Sleeve (getiri akışı) temel arayüzü.

MOTİVASYON: Bugünkü kitapta 17 varlık var ama TEK BAHİS TÜRÜ (trend). 17 pozisyon
da aynı anda aynı yönde bozulabilir. Kanıtlanmış tek büyük Sharpe kazancı
çeşitlendirmeden geldi (kripto-only 1,07 → varlık-diversifiye 1,33). Aynı mantık
strateji boyutunda uygulanır: birbirinden BAĞIMSIZ getiri akışları eklemek.

Portföy Sharpe'ı ≈ SR_ortalama × √(N / (1 + (N−1)·ρ_ortalama))
  → 4 sleeve, tekil SR 0,9, ortalama korelasyon 0,25 ⇒ ≈ 1,9

Her sleeve şu sözleşmeye uyar:
  • `positions(prices)` → (tarih × varlık) hedef ağırlık matrisi, **t anında
    bilinen bilgiyle**. Look-ahead sorumluluğu sleeve'e aittir.
  • `returns(prices)`   → maliyet düşülmüş günlük portföy getirisi

KRİTİK KURAL: `positions` t günü kapanışıyla hesaplanır, getiriye
uygulanmadan önce `.shift(1)` ile bir gün ötelenir (base sınıfı yapar).
Bu ötelemenin unutulması bu projede Sharpe'ı 0,80'den 3,68'e çıkarmıştı.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Gerçekçi round-trip maliyet: taker %0.04 + volatiliteye bağlı kayma tabanı.
# FAZ 4 sonrası maker yürütmeyle düşürülecek (ölçüldü: Sharpe 1,05→1,08).
DEFAULT_COST = 0.0006


class Sleeve(ABC):
    """Tek bir bağımsız getiri akışı."""

    name: str = "sleeve"
    #: Bu sleeve'in çalışması için gereken asgari geçmiş bar sayısı
    warmup: int = 200

    def __init__(self, cost: float = DEFAULT_COST, **kwargs):
        self.cost = float(cost)
        self.params: Dict = dict(kwargs)

    # ------------------------------------------------------------------ arayüz
    @abstractmethod
    def positions(self, prices: pd.DataFrame, **ctx) -> pd.DataFrame:
        """(tarih × varlık) hedef ağırlık. Toplam brüt maruziyet ≤ 1.0 olmalı;
        ölçekleme allocator'ın işidir."""

    # ---------------------------------------------------------------- ortak iş
    def returns(self, prices: pd.DataFrame, **ctx) -> pd.Series:
        """Maliyet düşülmüş günlük portföy getirisi.

        `positions` çıktısı BURADA .shift(1) edilir — sleeve'ler bunu kendileri
        yapmamalı (çift öteleme sinyali bir gün geciktirir)."""
        pos = self.positions(prices, **ctx)
        if pos is None or pos.empty:
            return pd.Series(dtype=float)
        pos = pos.reindex(columns=prices.columns).fillna(0.0)
        rets = prices.pct_change().fillna(0.0)

        held = pos.shift(1).fillna(0.0)                 # ← look-ahead koruması
        turnover = (held - held.shift(1).fillna(0.0)).abs().sum(axis=1)
        gross = (held * rets).sum(axis=1)
        return (gross - turnover * self.cost).rename(self.name)

    def gross_exposure(self, prices: pd.DataFrame, **ctx) -> pd.Series:
        return self.positions(prices, **ctx).abs().sum(axis=1)

    def describe(self) -> Dict:
        return {"name": self.name, "cost": self.cost, "warmup": self.warmup,
                "params": self.params}


# ===========================================================================
# Ortak yardımcılar (sleeve'ler paylaşır)
# ===========================================================================
def vol_target_scale(prices: pd.DataFrame, target_daily_vol: float = 0.025,
                     window: int = 30, max_leverage: float = 1.0) -> pd.DataFrame:
    """Varlık bazında pozisyon ölçeği = hedef_vol / gerçekleşen_vol (tavanlı).

    Volatilite hedefleme, kanıtlanmış tek "bedava" Sharpe kazancıdır: oynak
    varlıkta küçük, sakin varlıkta büyük pozisyon → risk katkıları eşitlenir."""
    vol = prices.pct_change().rolling(window).std()
    return (target_daily_vol / (vol + 1e-9)).clip(0.0, max_leverage)


def zscore(df: "pd.DataFrame | pd.Series", window: int) -> "pd.DataFrame | pd.Series":
    m = df.rolling(window).mean()
    s = df.rolling(window).std()
    return (df - m) / (s + 1e-12)


def cross_sectional_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Satır bazında 0..1 yüzdelik sıra (NaN'lar korunur)."""
    return df.rank(axis=1, pct=True, na_option="keep")
