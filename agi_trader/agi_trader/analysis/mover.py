"""
Günün "%1 hareket adayı" seçici — hangi parite bugün ≥%1 oynar?

ÖLÇÜLMÜŞ TEMEL (2022-2025, 5 parite, günlük):

  Hareket BÜYÜKLÜĞÜ (|getiri| ≥ %1)   AUC 0,56 – 0,62   → gerçek, mütevazı öngörü
  Hareket YÖNÜ      (yukarı/aşağı)     AUC 0,47 – 0,50   → SIFIR bilgi

Bu ayrım modülün tüm tasarımını belirler: sistem **hangi paritenin oynayacağını**
sıralar, **hangi yöne gideceğini SÖYLEMEZ**. Yön kararı ayrı bir tetikleyiciye
(formasyon kırılımı, seviye, trend kapısı) bırakılır.

TABAN ORANI UYARISI — en kritik nokta: pariteler zaten günlerin %57-81'inde %1
hareket ediyor. "BTC bugün %1 oynayacak" demek tek başına BİLGİ DEĞİLDİR. Bu
yüzden her tahminin yanında taban oranı ve **lift** (tahmin/taban) raporlanır;
lift ≈ 1 ise model hiçbir şey eklememiştir.

Gün-içi ARALIK (high-low) ölçütü kasıtlı olarak kullanılmaz: aralık ≥%1
günlerin %96,7-99,9'unda gerçekleşir — sorulmaya değmeyecek kadar kesindir.
Ölçüt kapanıştan kapanışa mutlak getiridir.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

MOVE_THRESHOLD = 0.01           # %1
FEATURES = [
    "vol20", "atr_pct", "prev_range", "range_expansion", "abs_mom5",
    "vol_of_vol", "dist_z", "gap_pct", "dow_weekend",
]
FEATURE_TR = {
    "vol20": "20 günlük gerçekleşen volatilite",
    "atr_pct": "ATR/fiyat (ortalama gerçek aralık)",
    "prev_range": "dünkü gün-içi aralık",
    "range_expansion": "aralık genişlemesi (5g/20g)",
    "abs_mom5": "5 günlük mutlak momentum",
    "vol_of_vol": "volatilitenin volatilitesi",
    "dist_z": "20 günlük ortalamadan uzaklık (z)",
    "gap_pct": "açılış boşluğu",
    "dow_weekend": "hafta sonu etkisi",
}


# ===========================================================================
# Özellikler — hepsi t-1'de bilinen bilgiyle
# ===========================================================================
def build_mover_features(df: pd.DataFrame) -> pd.DataFrame:
    """Günlük OHLCV → özellik matrisi. TÜM özellikler .shift(1) ile ötelenir:
    bugünü tahmin ederken bugünün verisi kullanılamaz."""
    c, h, l = df["close"].astype(float), df["high"].astype(float), df["low"].astype(float)
    o = df["open"].astype(float) if "open" in df else c.shift()
    r = c.pct_change()

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    rng = (h - l) / c.shift()

    f = pd.DataFrame(index=df.index)
    f["vol20"] = r.rolling(20).std()
    f["atr_pct"] = tr.rolling(14).mean() / c
    f["prev_range"] = rng
    f["range_expansion"] = rng.rolling(5).mean() / (rng.rolling(20).mean() + 1e-12)
    f["abs_mom5"] = r.rolling(5).sum().abs()
    f["vol_of_vol"] = r.rolling(20).std().rolling(20).std()
    f["dist_z"] = (c - c.rolling(20).mean()) / (c.rolling(20).std() + 1e-12)
    f["dist_z"] = f["dist_z"].abs()
    f["gap_pct"] = (o / c.shift() - 1).abs()
    f["dow_weekend"] = pd.Series(df.index.dayofweek, index=df.index).isin([5, 6]).astype(float)

    return f.shift(1).replace([np.inf, -np.inf], np.nan)


def move_labels(df: pd.DataFrame, threshold: float = MOVE_THRESHOLD) -> pd.Series:
    """Hedef: bugünün |kapanış-kapanış getirisi| ≥ eşik mi?"""
    return (df["close"].astype(float).pct_change().abs() >= threshold).astype(float)


# ===========================================================================
# Kalibre lojistik model (saf numpy — sunucuda sklearn yok)
# ===========================================================================
class MoverModel:
    """L2 lojistik regresyon + olasılık kalibrasyonu.

    Kalibrasyon şart: ham skor 'sıralama' verir ama '%73 olasılık' demek için
    çıktının gerçek frekansla eşleşmesi gerekir. Kalibre edilmemiş bir olasılık
    kullanıcıyı yanıltır."""

    def __init__(self, lr: float = 0.15, iters: int = 800, l2: float = 1e-3):
        self.lr, self.iters, self.l2 = lr, iters, l2
        self.w = self.b = self.mu = self.sd = None
        self.base_rate = 0.5
        self.features: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MoverModel":
        m = X.notna().all(axis=1) & y.notna()
        Xv = X[m].values.astype(float)
        yv = y[m].values.astype(float)
        if len(yv) < 100 or len(np.unique(yv)) < 2:
            self.base_rate = float(yv.mean()) if len(yv) else 0.5
            return self
        self.features = list(X.columns)
        self.base_rate = float(yv.mean())
        self.mu, self.sd = Xv.mean(0), Xv.std(0) + 1e-9
        Z = (Xv - self.mu) / self.sd
        self.w = np.zeros(Z.shape[1])
        self.b = math.log(self.base_rate / (1 - self.base_rate + 1e-12) + 1e-12)
        for _ in range(self.iters):
            p = 1.0 / (1.0 + np.exp(-(Z @ self.w + self.b)))
            g = p - yv
            self.w -= self.lr * (Z.T @ g / len(yv) + self.l2 * self.w)
            self.b -= self.lr * float(g.mean())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.w is None:
            return np.full(len(X), self.base_rate)
        Xv = X[self.features].values.astype(float)
        bad = ~np.isfinite(Xv).all(axis=1)
        Xv = np.nan_to_num(Xv, nan=0.0, posinf=0.0, neginf=0.0)
        Z = (Xv - self.mu) / self.sd
        p = 1.0 / (1.0 + np.exp(-(Z @ self.w + self.b)))
        p[bad] = self.base_rate
        return p

    def contributions(self, x: pd.Series) -> List[Dict]:
        """Bu tahmine hangi özellik ne kadar katkı yaptı (log-odds cinsinden)."""
        if self.w is None:
            return []
        out = []
        for k, name in enumerate(self.features):
            v = float(x.get(name, np.nan))
            if not np.isfinite(v):
                continue
            z = (v - self.mu[k]) / self.sd[k]
            out.append({"feature": name, "tr": FEATURE_TR.get(name, name),
                        "value": round(v, 6), "z": round(float(z), 2),
                        "contribution": round(float(z * self.w[k]), 4)})
        out.sort(key=lambda d: -abs(d["contribution"]))
        return out


# ===========================================================================
# Değerlendirme
# ===========================================================================
def auc_score(y, s) -> float:
    """ROC AUC (Mann-Whitney U). EŞİT SKORLAR ortalama sıra ile işlenir.

    Ties'ı yok saymak tehlikelidir: model sabit olasılık ürettiğinde (ör. taban
    orana düştüğünde) AUC 0,5 yerine keyfî bir değer çıkar ve işe yaramayan bir
    model 'mükemmel' görünebilir."""
    y = np.asarray(y, float); s = np.asarray(s, float)
    m = np.isfinite(y) & np.isfinite(s)
    y, s = y[m], s[m]
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks_sorted = np.arange(1, len(s) + 1, dtype=float)
    # eşit değer bloklarına ortalama sıra ata
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks_sorted[i:j + 1] = ranks_sorted[i:j + 1].mean()
        i = j + 1
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = ranks_sorted
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def brier_skill(y, p, base: float) -> float:
    """Brier beceri skoru: taban orana kıyasla iyileşme. ≤0 = model işe yaramıyor."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) == 0:
        return float("nan")
    bs = float(((p - y) ** 2).mean())
    bs_ref = float(((base - y) ** 2).mean())
    return float(1 - bs / (bs_ref + 1e-12))


def calibration_table(y, p, bins: int = 5) -> List[Dict]:
    """Tahmin edilen olasılık vs gerçekleşen frekans."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) < bins * 10:
        return []
    qs = np.quantile(p, np.linspace(0, 1, bins + 1))
    out = []
    for i in range(bins):
        sel = (p >= qs[i]) & (p <= qs[i + 1] if i == bins - 1 else p < qs[i + 1])
        if sel.sum() < 5:
            continue
        out.append({"bin": i + 1, "n": int(sel.sum()),
                    "tahmin": round(float(p[sel].mean()), 3),
                    "gerceklesen": round(float(y[sel].mean()), 3)})
    return out


# ===========================================================================
# Günlük seçim
# ===========================================================================
MIN_TRUSTED_AUC = 0.55          # bu AUC'nin altında modele güvenilmez


@dataclass
class MoverPick:
    symbol: str
    probability: float          # kalibre P(|hareket| ≥ %1)
    base_rate: float
    lift: float                 # tahmin / taban
    expected_move_pct: float    # beklenen |getiri| %
    expected_range_pct: float   # beklenen gün-içi aralık %
    rank: int
    model_trusted: bool = True  # bu paritede model örneklem dışı geçti mi?
    val_auc: Optional[float] = None
    evidence: List[Dict] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


def load_validation(output_dir: str = "runs") -> Dict[str, Dict]:
    """`research_mover.py` çıktısındaki parite-bazlı örneklem dışı skorlar.

    Canlı panel bunu okur: modelin ÖLÇÜLMÜŞ olarak çalışmadığı paritelerde
    olasılık gösterilmez, taban oranı gösterilir. Modelin nerede çalışmadığını
    gizlemek, çalıştığı yerdeki güveni de yok eder."""
    import json
    from pathlib import Path
    p = Path(output_dir)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[2] / p
    f = p / "mover_research.json"
    if not f.exists():
        return {}
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return {}
    # Sembol biçimi kaynaklara göre değişir: araştırma "AVAXUSDT", canlı ccxt
    # "AVAX/USDT". Normalize edilmezse eşleşme sessizce başarısız olur ve
    # doğrulamayı GEÇEMEYEN parite "geçti" sayılır — tam olarak gizlenmemesi
    # gereken şey gizlenir.
    return {_norm_symbol(r["symbol"]): r for r in d.get("rows", [])}


def _norm_symbol(s: str) -> str:
    """'AVAX/USDT:USDT' → 'AVAXUSDT'."""
    return str(s).upper().split(":")[0].replace("/", "").replace("-", "").strip()


def rank_movers(panel: Dict[str, pd.DataFrame],
                models: Optional[Dict[str, MoverModel]] = None,
                threshold: float = MOVE_THRESHOLD,
                validation: Optional[Dict[str, Dict]] = None) -> Dict:
    """Her parite için bugünün %1 hareket olasılığını hesapla ve sırala.

    panel      : {sembol: günlük OHLCV}
    models     : {sembol: eğitilmiş MoverModel}. Yoksa geçmişle eğitilir.
    validation : {sembol: {auc, brier_skill}} — modele güvenilip güvenilmeyeceği."""
    val = validation if validation is not None else load_validation()
    picks: List[MoverPick] = []
    for sym, df in panel.items():
        if df is None or len(df) < 250:
            continue
        X = build_mover_features(df)
        y = move_labels(df, threshold)
        model = (models or {}).get(sym)
        if model is None:
            model = MoverModel().fit(X.iloc[:-1], y.iloc[:-1])

        x_last = X.iloc[-1]
        p = float(model.predict_proba(X.iloc[[-1]])[0])
        base = float(model.base_rate)

        r = df["close"].pct_change().abs()
        rng = (df["high"] - df["low"]) / df["close"].shift()
        # beklenen büyüklük: son 20 günün medyanı, olasılıkla ölçeklenmez
        exp_move = float(r.tail(20).median() * 100)
        exp_range = float(rng.tail(20).median() * 100)

        v = val.get(_norm_symbol(sym)) or {}
        auc = v.get("auc")
        trusted = (auc is None) or (float(auc) >= MIN_TRUSTED_AUC)
        # güvenilmeyen paritede model çıktısı kullanılmaz, taban oranı gösterilir
        shown_p = p if trusted else base

        picks.append(MoverPick(
            symbol=sym, probability=round(shown_p, 4), base_rate=round(base, 4),
            lift=round(shown_p / (base + 1e-12), 3),
            expected_move_pct=round(exp_move, 2),
            expected_range_pct=round(exp_range, 2),
            rank=0, model_trusted=bool(trusted),
            val_auc=(round(float(auc), 3) if auc is not None else None),
            evidence=(model.contributions(x_last)[:4] if trusted else [])))

    picks.sort(key=lambda x: -x.probability)
    for i, p in enumerate(picks, 1):
        p.rank = i
        if not p.model_trusted:
            p.note = (f"Model bu paritede örneklem dışı geçemedi "
                      f"(AUC {p.val_auc}) — gösterilen değer TABAN ORANIDIR, tahmin değil.")
        elif p.lift >= 1.10:
            p.note = f"Model taban oranın %{(p.lift-1)*100:.0f} üzerinde — bugün olağandan hareketli."
        elif p.lift <= 0.90:
            p.note = f"Model taban oranın %{(1-p.lift)*100:.0f} altında — bugün olağandan sakin."
        else:
            p.note = "Model taban orandan anlamlı sapma görmüyor — bugün tipik bir gün."

    return {
        "threshold_pct": threshold * 100,
        "picks": [p.to_dict() for p in picks],
        "direction_warning": (
            "YÖN TAHMİN EDİLMEZ. Ölçüldü: yön için AUC 0,47-0,50 (sıfır bilgi), "
            "büyüklük için 0,56-0,62. Sistem hangi paritenin oynayacağını sıralar; "
            "hangi yöne gideceğini SÖYLEMEZ — yön kararı formasyon/seviye "
            "tetikleyicisine bırakılmalıdır."),
        "base_rate_warning": (
            "Pariteler zaten günlerin %57-81'inde ≥%1 hareket ediyor. Tek başına "
            "yüksek olasılık bilgi değildir; LIFT (tahmin/taban) sütununa bakın."),
    }
