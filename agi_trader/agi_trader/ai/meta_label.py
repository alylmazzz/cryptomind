"""
Meta-etiketleme (FAZ 4) — birincil model YÖNÜ, ikincil model BÜYÜKLÜĞÜ belirler.

López de Prado, AFML bölüm 3. Fikir: yön tahmini zordur ve zaten bir modelimiz
var (trend sleeve'i). İkincil model yönü DEĞİŞTİRMEZ; yalnız "bu sinyal kârlı
olacak mı?" sorusunu cevaplar ve pozisyon büyüklüğünü ona göre ayarlar.

NEDEN BU YAPI:
  • Yön modelini bozmadan KESİNLİK (precision) artırılır — kötü sinyaller
    küçültülür, iyi olanlar büyütülür.
  • Formasyonlar, göstergeler ve fiyat-dışı sinyaller burada ÖZELLİK olur.
    Böylece "formasyon gördüm, işlem açıyorum" gibi kanıtsız bir kapı kurulmaz
    (bkz. analysis/chart_patterns.py dürüstlük notu).
  • Boyutlama kesirli Kelly ile: f = clip(2p−1, 0, 0.5)

BAĞIMLILIK: sklearn varsa gradient boosting, yoksa saf-numpy lojistik regresyon
(sunucuda sklearn kurulu değil — fallback ZORUNLU, sessiz çökme değil).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.light import LIGHT_MODE

try:
    if LIGHT_MODE:
        raise ImportError
    from sklearn.ensemble import GradientBoostingClassifier
    _HAS_SK = True
except Exception:
    _HAS_SK = False


# ===========================================================================
# Saf-numpy lojistik regresyon (sklearn yoksa)
# ===========================================================================
class _LogReg:
    """L2 düzenlileştirmeli lojistik regresyon (gradyan inişi, örnek ağırlıklı)."""

    def __init__(self, lr: float = 0.1, iters: int = 400, l2: float = 1e-3):
        self.lr, self.iters, self.l2 = lr, iters, l2
        self.w: Optional[np.ndarray] = None
        self.b = 0.0
        self.mu: Optional[np.ndarray] = None
        self.sd: Optional[np.ndarray] = None

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        w = w / (w.sum() + 1e-12) * len(w)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        self.w = np.zeros(Z.shape[1])
        self.b = 0.0
        for _ in range(self.iters):
            p = 1.0 / (1.0 + np.exp(-(Z @ self.w + self.b)))
            g = (p - y) * w
            self.w -= self.lr * (Z.T @ g / len(y) + self.l2 * self.w)
            self.b -= self.lr * float(g.mean())
        return self

    def predict_proba(self, X):
        Z = (np.asarray(X, float) - self.mu) / self.sd
        p = 1.0 / (1.0 + np.exp(-(Z @ self.w + self.b)))
        return np.column_stack([1 - p, p])


# ===========================================================================
# Özellik matrisi
# ===========================================================================
def build_features(df: pd.DataFrame,
                   pattern_scores: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Meta-model özellikleri — hepsi t anında BİLİNEN bilgiyle hesaplanır.

    pattern_scores: opsiyonel (tarih × formasyon-anahtarı) yönlü skor matrisi;
    `analysis/chart_patterns.detect_chart_patterns` çıktısından üretilir."""
    c = df["close"].astype(float)
    h, l = df["high"].astype(float), df["low"].astype(float)
    v = df["volume"].astype(float) if "volume" in df else pd.Series(1.0, index=df.index)

    r = c.pct_change()
    f = pd.DataFrame(index=df.index)
    # momentum / trend
    for k in (1, 3, 5, 10, 20):
        f[f"ret{k}"] = c.pct_change(k)
    f["sma_ratio_50"] = c / c.rolling(50).mean() - 1
    f["sma_ratio_200"] = c / c.rolling(200).mean() - 1
    # volatilite ve rejim
    f["vol20"] = r.rolling(20).std()
    f["vol_ratio"] = r.rolling(10).std() / (r.rolling(60).std() + 1e-12)
    # aralık / mum yapısı
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["atr_pct"] = tr.rolling(14).mean() / c
    f["close_pos"] = (c - l.rolling(14).min()) / (h.rolling(14).max() - l.rolling(14).min() + 1e-12)
    # hacim
    f["vol_z"] = (v - v.rolling(50).mean()) / (v.rolling(50).std() + 1e-12)
    # basit RSI
    up = r.clip(lower=0).rolling(14).mean()
    dn = (-r.clip(upper=0)).rolling(14).mean()
    f["rsi"] = 100 - 100 / (1 + up / (dn + 1e-12))
    # otokorelasyon (ortalamaya dönüş mü, trend mi)
    f["autocorr"] = r.rolling(50).apply(lambda x: pd.Series(x).autocorr(1), raw=False)

    if pattern_scores is not None and len(pattern_scores):
        ps = pattern_scores.reindex(df.index).fillna(0.0)
        for col in ps.columns:
            f[f"pat_{col}"] = ps[col]

    return f.replace([np.inf, -np.inf], np.nan)


# ===========================================================================
# Meta-etiketleyici
# ===========================================================================
@dataclass
class MetaResult:
    proba: np.ndarray
    size: np.ndarray
    model: str
    n_train: int
    features: List[str]


class MetaLabeler:
    """İkincil model: P(birincil sinyal kârlı olacak)."""

    def __init__(self, kelly_cap: float = 0.5, min_proba: float = 0.5):
        self.kelly_cap = float(kelly_cap)
        self.min_proba = float(min_proba)
        self.model = None
        self.features: List[str] = []
        self.name = "gbm" if _HAS_SK else "logreg"

    def fit(self, X: pd.DataFrame, y, sample_weight=None) -> "MetaLabeler":
        Xc = X.dropna()
        yv = pd.Series(np.asarray(y).ravel(), index=X.index).loc[Xc.index].values
        w = (pd.Series(np.asarray(sample_weight).ravel(), index=X.index).loc[Xc.index].values
             if sample_weight is not None else None)
        self.features = list(Xc.columns)
        if len(np.unique(yv)) < 2 or len(Xc) < 50:
            self.model = None                      # öğrenilecek sinyal yok
            return self
        if _HAS_SK:
            self.model = GradientBoostingClassifier(
                n_estimators=120, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=0)
            self.model.fit(Xc.values, yv, sample_weight=w)
        else:
            self.model = _LogReg().fit(Xc.values, yv, sample_weight=w)
        return self

    def predict(self, X: pd.DataFrame) -> MetaResult:
        n = len(X)
        if self.model is None:
            p = np.full(n, 0.5)
        else:
            Xv = X[self.features].fillna(0.0).values
            p = self.model.predict_proba(Xv)[:, 1]
        # kesirli Kelly: f = clip(2p−1, 0, cap)
        size = np.clip(2 * p - 1, 0.0, self.kelly_cap)
        size[p < self.min_proba] = 0.0
        return MetaResult(proba=p, size=size, model=self.name,
                          n_train=n, features=self.features)


def kelly_size(proba: float, cap: float = 0.5, min_proba: float = 0.5) -> float:
    """Tek nokta için kesirli Kelly büyüklüğü.

    TAM Kelly ASLA kullanılmaz: tahmin hatası varken tam Kelly iflas olasılığını
    hızla yükseltir. `cap` (varsayılan 0,5) yarım-Kelly tavanıdır."""
    if proba < min_proba:
        return 0.0
    return float(np.clip(2 * proba - 1, 0.0, cap))
