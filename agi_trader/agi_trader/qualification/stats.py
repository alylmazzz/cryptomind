"""Belirsizlik ve kalibrasyon araçları — şartname 14, 15, 16, 18, 102, 112.

BU DOSYANIN VAR OLMA SEBEBİ
Bir olasılığı nokta tahmin olarak göstermek ("%82") kullanıcıyı yanıltır.
Örneklem küçükse %82'nin altında 40 da olabilir. Bu yüzden:

  • karar için NOKTA TAHMİN değil ALT GÜVEN SINIRI kullanılır (şartname 102),
  • örtüşen etiketlerde satır sayısı örneklem sayısı DEĞİLDİR (şartname 16),
  • "az örnek → yüksek güven" üretmek yasaktır (şartname 15).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Z95 = 1.959963984540054


def wilson_ci(k: int, n: int, z: float = Z95) -> Tuple[float, float]:
    """Wilson güven aralığı — küçük n ve uç oranlarda normal yaklaşımdan iyi.

    Normal yaklaşım p=0 ya da p=1'de sıfır genişlikte aralık üretir; bu,
    "3 gözlemde 3 başarı → %100 kesin" gibi saçma bir sonuç verir. Wilson
    bunu yapmaz."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1.0 + z * z / n
    merkez = (p + z * z / (2 * n)) / d
    yari = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, merkez - yari), min(1.0, merkez + yari))


def effective_sample_size(labels: np.ndarray, horizon_bars: int,
                          step_bars: int = 1) -> Dict[str, float]:
    """Etkin örneklem — şartname 16.

    ÜÇ SAYI birden döner çünkü üçü de farklı şey söyler:

    raw          : satır sayısı. Örtüşen 24h etiketlerinde 288 komşu örnek
                   NEREDEYSE AYNI geleceği paylaşır; bu sayı yanıltıcıdır.
    non_overlap  : ufuk uzunluğunda seyreltilmiş sayı (n·step/H). Muhafazakâr,
                   varsayımsız alt sınır. GÜVEN ARALIĞI BUNUNLA hesaplanır.
    autocorr     : etiket serisinin otokorelasyonundan türetilen ESS:
                   n / (1 + 2·Σ ρ_k). Gerçek bağımsızlık buna daha yakındır
                   ama bir model varsayımı taşır, bu yüzden karar sayısı değil.
    """
    n = int(len(labels))
    if n == 0:
        return {"raw": 0.0, "non_overlap": 0.0, "autocorr": 0.0, "used": 0.0}
    ortusme = max(1.0, horizon_bars / max(1, step_bars))
    non_ov = n / ortusme

    # Otokorelasyon ESS'i seyreltilmiş seride hesaplanır: 480 bin elemanlı
    # diziyi 576 gecikmede taramak hücre başına dakikalar sürerdi. Seyreltme
    # DETERMİNİSTİK (sabit adım) — tohumlu RNG bile kullanılmaz.
    adim = max(1, int(np.ceil(n / 80_000)))
    x = labels.astype(float)[::adim]
    nx = len(x)
    x = x - x.mean()
    var = float((x * x).mean())
    ess_ac = float(n)
    if var > 1e-12 and nx > 8:
        toplam = 0.0
        kmax = int(min(nx - 1, max(1, int(np.ceil(horizon_bars / adim)) * 2), 400))
        for k in range(1, kmax + 1):
            r = float((x[:-k] * x[k:]).mean() / var)
            if r <= 0:                      # ilk negatife inişte kes (Geyer)
                break
            toplam += r
        # ESS seyreltmeye göre DEĞİŞMEZ: nx/(1+2Σρ_seyreltilmiş) ile
        # n/(1+2Σρ_ham) aynı sayıya gider (üçgen örtüşmede ikisi de n/H).
        # Bu yüzden payda değil PAY da seyreltilmiş sayıdır — n kullanmak
        # ESS'i adım kadar şişirirdi.
        ess_ac = nx / (1.0 + 2.0 * toplam)
    # Karar sayısı MUHAFAZAKÂR olan: ikisinin küçüğü
    used = float(min(non_ov, ess_ac))
    return {"raw": float(n), "non_overlap": float(non_ov),
            "autocorr": float(ess_ac), "used": used}


def proportion_with_ci(k: int, n_raw: int, ess: float) -> Dict:
    """Oranın nokta tahmini ham veriden, GENİŞLİĞİ etkin örneklemden.

    Nokta tahmini bütün gözlemleri kullanır (daha verimli); güven aralığı
    etkin örneklemi kullanır (dürüst genişlik). İkisini karıştırmak, örtüşen
    pencerelerde aralığı sahte biçimde daraltır."""
    if n_raw <= 0:
        return {"p": None, "lower95": None, "upper95": None, "n_eff": 0.0}
    p = k / n_raw
    ne = max(1.0, float(ess))
    lo, hi = wilson_ci(int(round(p * ne)), int(round(ne)))
    return {"p": float(p), "lower95": float(lo), "upper95": float(hi),
            "n_eff": float(ne)}


def block_bootstrap_ci(labels: np.ndarray, target_value: int,
                       block: int, n_boot: int = 2000,
                       seed: int = 20260817) -> Dict:
    """Blok bootstrap — otokorelasyonu koruyarak oranın dağılımı.

    ⚠️ `seed` SABİT ve açık verilir. Python'un `hash()` fonksiyonu süreçler
    arası tuzlanır; ondan tohum türetmek sonucu tekrarlanamaz yapar. Bu hata
    bu projede bir kez yapıldı (şartname 84)."""
    n = len(labels)
    if n < block * 2:
        return {"lower95": None, "upper95": None, "n_boot": 0}
    rng = np.random.default_rng(seed)
    nblok = int(np.ceil(n / block))
    baslar = rng.integers(0, n - block, size=(n_boot, nblok))
    ok = (labels == target_value).astype(np.float64)
    idx = baslar[:, :, None] + np.arange(block)[None, None, :]
    ornek = ok[idx.reshape(n_boot, -1)[:, :n]]
    oranlar = ornek.mean(axis=1)
    return {"lower95": float(np.percentile(oranlar, 2.5)),
            "upper95": float(np.percentile(oranlar, 97.5)),
            "n_boot": int(n_boot)}


# ── kalibrasyon (şartname 18, 19, 105) ─────────────────────────────────────

def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    """Ortalama kare hata. 0 = mükemmel. Taban (her zaman ȳ) ile kıyaslanmalı."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(p) == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    y = np.asarray(y, dtype=float)
    if len(p) == 0:
        return float("nan")
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability_curve(p: np.ndarray, y: np.ndarray,
                      edges: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                                                0.6, 0.7, 0.8, 0.9, 1.0)
                      ) -> List[Dict]:
    """Güvenilirlik eğrisi: model '%80' dediğinde gerçekte kaç oldu?"""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    out: List[Dict] = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b if b < 1.0 else p <= b)
        n = int(m.sum())
        out.append({
            "bucket": f"{a:.0%}-{b:.0%}",
            "n": n,
            "predicted": (float(p[m].mean()) if n else None),
            "actual": (float(y[m].mean()) if n else None),
        })
    return out


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Beklenen kalibrasyon hatası — kova ağırlıklı |tahmin − gerçek|."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(p) == 0:
        return float("nan")
    kenar = np.linspace(0, 1, bins + 1)
    toplam = 0.0
    for a, b in zip(kenar[:-1], kenar[1:]):
        m = (p >= a) & (p < b if b < 1.0 else p <= b)
        if m.any():
            toplam += m.mean() * abs(float(p[m].mean()) - float(y[m].mean()))
    return float(toplam)


def calibration_slope_intercept(p: np.ndarray, y: np.ndarray,
                                eps: float = 1e-6) -> Dict[str, Optional[float]]:
    """logit(y) ~ a + b·logit(p). Mükemmel kalibrasyon: a=0, b=1.

    b < 1 → model AŞIRI kendine güveniyor (uçlara fazla gidiyor).
    b > 1 → fazla çekingen.
    """
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    y = np.asarray(y, dtype=float)
    if len(p) < 20 or len(np.unique(y)) < 2 or p.std() < 1e-9:
        return {"slope": None, "intercept": None, "n": int(len(p))}
    x = np.log(p / (1 - p))
    # Newton ile tek değişkenli lojistik regresyon
    b = np.array([0.0, 1.0])
    X = np.column_stack([np.ones_like(x), x])
    for _ in range(60):
        z = X @ b
        mu = 1.0 / (1.0 + np.exp(-z))
        W = mu * (1 - mu) + 1e-9
        g = X.T @ (y - mu)
        H = (X * W[:, None]).T @ X + 1e-9 * np.eye(2)
        try:
            adim = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return {"slope": None, "intercept": None, "n": int(len(p))}
        b = b + adim
        if np.max(np.abs(adim)) < 1e-9:
            break
    return {"intercept": float(b[0]), "slope": float(b[1]), "n": int(len(p))}


def psi(beklenen: np.ndarray, gozlenen: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index — şartname 76 sürüklenme göstergesi.
    < 0,10 kararlı · 0,10-0,25 dikkat · > 0,25 sürüklenme."""
    beklenen = np.asarray(beklenen, dtype=float)
    gozlenen = np.asarray(gozlenen, dtype=float)
    if len(beklenen) < bins or len(gozlenen) < bins:
        return float("nan")
    kenar = np.percentile(beklenen, np.linspace(0, 100, bins + 1))
    kenar[0], kenar[-1] = -np.inf, np.inf
    e = np.histogram(beklenen, kenar)[0] / len(beklenen)
    o = np.histogram(gozlenen, kenar)[0] / len(gozlenen)
    e = np.clip(e, 1e-6, None)
    o = np.clip(o, 1e-6, None)
    return float(np.sum((o - e) * np.log(o / e)))
