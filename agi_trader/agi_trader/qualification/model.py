"""Model katmanı — şartname 10, 18, 19, 36, 37, 39, 43, 84, 86, 113.

NE ÖĞRENİLİYOR
Üç rakip sonucun olasılığı: hedef önce / stop önce / zaman aşımı. Tek bir
"yukarı mı aşağı mı" sınıflandırıcısı bu soruya cevap VERMEZ; çünkü asıl soru
sıradır, yön değil (şartname 10).

MİMARİ (şartname 36, 37)
  GlobalCryptoModel   : bütün pariteler ortak yapıyı öğrenir (softmax
                        regresyon, L2 cezalı, deterministik)
  HorizonHead         : her ufuk için ayrı çıkış → tutarlı olasılık eğrisi
  PairCalibrationHead : parite başına Platt ölçekleme; az örnekli paritede
                        bağımsız model eğitmek yerine hiyerarşik düzeltme

NEDEN LİNEER
Bu ürünün canlı sunucusunda sklearn yok ve model ağırlıkları düz metin olarak
taşınabilmeli. Softmax regresyonun çıkarımı bir matris çarpımıdır — hangi
ortamda koşarsa koşsun aynı sayıyı verir. Doğrusal-olmayan sinyal olup
olmadığı AYRI bir meydan-okuyucu ile araştırılır ve sonucu raporlanır;
canlıya alınması ayrı bir iş kalemidir.

MODEL FALLBACK (şartname 113)
Model cevap veremiyorsa basit bir tabana düşüp "AL" ÜRETİLMEZ. `predict`
başarısız olursa `None` döner ve çağıran taraf NO_TRADE üretir.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

CLASSES = ("TP", "SL", "TIMEOUT")
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}


# ── softmax regresyon ──────────────────────────────────────────────────────

@dataclass
class SoftmaxModel:
    """L2 cezalı çok sınıflı lojistik regresyon. Ağırlıklar JSON'a sığar."""
    names: List[str]
    mean: List[float]
    scale: List[float]
    W: List[List[float]]              # (K, F+1) — son sütun kesişim
    l2: float
    n_train: int
    version: str = "1"

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict) -> "SoftmaxModel":
        return SoftmaxModel(**d)

    @property
    def feature_hash(self) -> str:
        return hashlib.sha256("|".join(self.names).encode()).hexdigest()[:16]

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != len(self.names):
            return None
        Z = (X - np.asarray(self.mean)) / np.asarray(self.scale)
        if not np.isfinite(Z).all():
            Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
        A = np.column_stack([Z, np.ones(len(Z))]) @ np.asarray(self.W).T
        A -= A.max(axis=1, keepdims=True)
        E = np.exp(A)
        return E / E.sum(axis=1, keepdims=True)


def fit_softmax(X: np.ndarray, y: np.ndarray, l2: float = 1.0,
                names: Optional[List[str]] = None,
                max_iter: int = 300) -> SoftmaxModel:
    """Deterministik eğitim — rastgelelik YOK, dolayısıyla tohum da yok."""
    from scipy.optimize import minimize

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    ort = X.mean(axis=0)
    olc = X.std(axis=0)
    olc[olc < 1e-12] = 1.0
    Z = np.column_stack([(X - ort) / olc, np.ones(len(X))])
    K, F = len(CLASSES), Z.shape[1]
    Y = np.zeros((len(y), K))
    Y[np.arange(len(y)), y] = 1.0

    def hedef(w):
        W = w.reshape(K, F)
        A = Z @ W.T
        A -= A.max(axis=1, keepdims=True)
        E = np.exp(A)
        S = E.sum(axis=1, keepdims=True)
        logp = A - np.log(S)
        nll = -float((Y * logp).sum()) / len(Z)
        P = E / S
        G = (P - Y).T @ Z / len(Z)
        ceza = np.array(W, copy=True)
        ceza[:, -1] = 0.0                      # kesişime ceza YOK
        return nll + 0.5 * l2 * float((ceza ** 2).sum()) / len(Z), \
            (G + l2 * ceza / len(Z)).ravel()

    w0 = np.zeros(K * F)
    r = minimize(hedef, w0, jac=True, method="L-BFGS-B",
                 options={"maxiter": max_iter, "ftol": 1e-12})
    return SoftmaxModel(names=list(names or [f"f{i}" for i in range(X.shape[1])]),
                        mean=ort.tolist(), scale=olc.tolist(),
                        W=r.x.reshape(K, F).tolist(), l2=float(l2),
                        n_train=int(len(X)))


# ── purged walk-forward (şartname 39) ──────────────────────────────────────

@dataclass
class Fold:
    train_start: int
    train_end: int
    test_start: int
    test_end: int


def purged_walk_forward(n: int, horizon_bars: int, n_folds: int = 5,
                        embargo_pct: float = 0.01,
                        min_train_frac: float = 0.35) -> List[Fold]:
    """Kronolojik, genişleyen pencere + purge + embargo.

    RASTGELE train/test bölmesi YASAKTIR (şartname 39). Etiketler geleceğe
    `horizon_bars` kadar uzandığı için eğitim penceresinin sonundan bu kadar
    bar SİLİNİR (purge); ayrıca test başlangıcından önce ek bir embargo
    bırakılır ki otokorelasyon eğitim setine sızmasın."""
    if n < 1000 or n_folds < 1:
        return []
    bas = int(n * min_train_frac)
    kalan = n - bas
    blok = kalan // n_folds
    if blok <= horizon_bars * 2:
        return []
    emb = max(1, int(n * embargo_pct))
    out: List[Fold] = []
    for k in range(n_folds):
        ts = bas + k * blok
        te = ts + blok if k < n_folds - 1 else n
        tr_end = ts - horizon_bars - emb        # purge + embargo
        if tr_end <= 100:
            continue
        out.append(Fold(0, tr_end, ts, te))
    return out


# ── kalibrasyon başlığı (şartname 36) ──────────────────────────────────────

@dataclass
class PlattHead:
    a: float = 1.0
    b: float = 0.0
    n: int = 0
    fitted: bool = False

    def apply(self, p: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return p
        q = np.clip(p, 1e-6, 1 - 1e-6)
        z = self.a * np.log(q / (1 - q)) + self.b
        return 1.0 / (1.0 + np.exp(-z))

    def to_dict(self) -> Dict:
        return asdict(self)


def fit_platt(p: np.ndarray, y: np.ndarray, min_n: int = 200) -> PlattHead:
    """Parite başlığı — OOS tahminler üzerinde tek boyutlu lojistik düzeltme.

    Az örnekli paritede EĞİTİLMEZ (fitted=False) ve global olasılık aynen
    kullanılır. "Az örnek → kendi modeli" hiyerarşik yaklaşımın tam tersidir."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(p) & np.isfinite(y)
    p, y = p[m], y[m]
    if len(p) < min_n or len(np.unique(y)) < 2:
        return PlattHead(1.0, 0.0, int(len(p)), False)
    from .stats import calibration_slope_intercept
    r = calibration_slope_intercept(p, y)
    if r["slope"] is None:
        return PlattHead(1.0, 0.0, int(len(p)), False)
    return PlattHead(float(r["slope"]), float(r["intercept"]), int(len(p)), True)


# ── değerlendirme ──────────────────────────────────────────────────────────

def decile_table(p: np.ndarray, y_tp: np.ndarray, n_bins: int = 10) -> List[Dict]:
    """OOS tahmin desiline göre gerçekleşen hedef-önce oranı.

    Modelin bilgi taşıyıp taşımadığının en dürüst tek tablosu budur: üst
    desil alt desilden anlamlı ölçüde yüksek değilse model bilgi taşımıyordur.
    Eşik ARAMASI yapılmaz (her eşik bir denemedir); desil sabit bir bölmedir."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y_tp, dtype=float)
    m = np.isfinite(p) & np.isfinite(y)
    p, y = p[m], y[m]
    if len(p) < n_bins * 20:
        return []
    kenar = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    kenar[0], kenar[-1] = -np.inf, np.inf
    out = []
    for i in range(n_bins):
        s = (p >= kenar[i]) & (p < kenar[i + 1])
        n = int(s.sum())
        out.append({"decile": i + 1, "n": n,
                    "p_mean": (float(p[s].mean()) if n else None),
                    "actual_tp": (float(y[s].mean()) if n else None)})
    return out


def top_decile_rate(dec: List[Dict]) -> Tuple[Optional[float], int]:
    if not dec:
        return None, 0
    ust = dec[-1]
    return ust["actual_tp"], ust["n"]
