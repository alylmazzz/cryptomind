"""
Sleeve tahsisi — eşit risk katkısı (ERC) + portföy hedef-vol kaldıraç.

NEDEN ERC, eşit ağırlık değil: eşit ağırlıkta en oynak sleeve portföy riskinin
çoğunu tek başına üstlenir; "4 sleeve'im var" yanılsaması yaratır ama gerçekte
tek bahis kalırsınız. ERC her sleeve'in risk KATKISINI eşitler.

NEDEN portföy hedef-vol: getiri = Sharpe × vol kimliği gereği, risk sınırı
(kullanıcı kararı: MaxDD %15-20) doğrudan vol hedefine, o da kaldıraca çevrilir.
Kaldıraç bir "agresiflik ayarı" değil, risk sınırının matematiksel sonucudur.

Parametreler (plan tablosundan, keyfi değil):
  hedef yıllık vol     %18   → beklenen MaxDD ≈ 1,1 × vol ≈ %20
  maks kaldıraç        2,0×  → kuyruk riski kelepçesi
  sleeve risk bütçesi  ≤%40  → tek sleeve hâkim olamaz
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TARGET_ANNUAL_VOL = 0.18
MAX_LEVERAGE = 2.0
MAX_SLEEVE_RISK = 0.40
COV_WINDOW = 60
PERIODS_PER_YEAR = 365.0


# ===========================================================================
# ERC çekirdeği
# ===========================================================================
def erc_weights(cov: np.ndarray, max_weight: float = MAX_SLEEVE_RISK,
                iters: int = 200, tol: float = 1e-8) -> np.ndarray:
    """Eşit risk katkısı ağırlıkları (sabit nokta yinelemesi).

    RC_i = w_i · (Σw)_i ; hedef: tüm RC_i eşit.
    Başlangıç ters-volatilite (korelasyonlar eşitken ERC'nin tam çözümü)."""
    n = cov.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])

    vol = np.sqrt(np.maximum(np.diag(cov), 1e-18))
    w = (1.0 / vol)
    w = w / w.sum()

    for _ in range(iters):
        mrc = cov @ w                      # marjinal risk katkısı
        rc = w * mrc
        target = rc.mean()
        # RC'si hedefin altındaki sleeve'in ağırlığını artır
        w_new = w * (target / np.maximum(rc, 1e-18)) ** 0.5
        w_new = np.maximum(w_new, 0.0)
        s = w_new.sum()
        if s <= 0:
            break
        w_new /= s
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new

    # sleeve tavanı: aşanları kırp, farkı diğerlerine oransal dağıt
    for _ in range(10):
        over = w > max_weight
        if not over.any():
            break
        excess = (w[over] - max_weight).sum()
        w[over] = max_weight
        free = ~over
        if not free.any() or w[free].sum() <= 0:
            break
        w[free] += excess * w[free] / w[free].sum()
    return w / max(w.sum(), 1e-12)


def max_sharpe_weights(cov: np.ndarray, mu: np.ndarray,
                       max_weight: float = MAX_SLEEVE_RISK,
                       mu_shrink: float = 0.5, cov_shrink: float = 0.7
                       ) -> np.ndarray:
    """Beklenen-getiri farkındalı tahsis (kısıtlı maksimum-Sharpe).

    NEDEN ERC YETMEZ: ERC yalnız riski eşitler, beklenen getiriyi YOK SAYAR.
    Sharpe'ları eşit olan sleeve'lerde optimaldir; bizde değiller (trend 1,11 ·
    carry 0,80 · term 0,40 · xsec_neutral 0,20). Ölçümde ERC, düşük volatiliteli
    ama düşük Sharpe'lı carry'ye %41 ağırlık verip portföyü baseline'ın altına
    düşürdü.

    İki varlıkta teorik optimum Sharpe = √((S₁²+S₂²−2ρS₁S₂)/(1−ρ²)); trend+carry
    için bu 1,36 eder — ERC'nin bulduğu 0,95 değil.

    BÜZÜLTME (shrinkage) ZORUNLU: ham ortalama-varyans, tahmin hatasını
    kaldıraçlar ve klasik olarak örneklem dışında çöker. Burada
      • μ, kesitsel ortalamasına doğru büzülür (James-Stein ruhu)
      • Σ, köşegenine doğru büzülür (korelasyon tahmini gürültülüdür)
    ve negatif ağırlıklar kırpılır (sleeve'i ters çevirmek anlamsız)."""
    n = cov.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])

    mu_s = mu_shrink * mu + (1 - mu_shrink) * float(np.mean(mu))
    cov_s = cov_shrink * cov + (1 - cov_shrink) * np.diag(np.diag(cov))
    cov_s = cov_s + np.eye(n) * 1e-12

    try:
        raw = np.linalg.solve(cov_s, mu_s)
    except np.linalg.LinAlgError:
        raw = np.linalg.pinv(cov_s) @ mu_s

    raw = np.maximum(raw, 0.0)                    # sleeve'i short'lama
    if raw.sum() <= 1e-18:
        # tüm beklenen getiriler ≤0 → riske girme
        return np.zeros(n)
    w = raw / raw.sum()

    for _ in range(10):
        over = w > max_weight
        if not over.any():
            break
        excess = (w[over] - max_weight).sum()
        w[over] = max_weight
        free = ~over
        if not free.any() or w[free].sum() <= 0:
            break
        w[free] += excess * w[free] / w[free].sum()
    s = w.sum()
    return w / s if s > 0 else w


# ===========================================================================
# Zaman serisi tahsisi (backtest)
# ===========================================================================
def allocate(sleeve_returns: pd.DataFrame,
             target_annual_vol: float = TARGET_ANNUAL_VOL,
             max_leverage: float = MAX_LEVERAGE,
             max_sleeve_risk: float = MAX_SLEEVE_RISK,
             cov_window: int = COV_WINDOW,
             periods_per_year: float = PERIODS_PER_YEAR,
             method: str = "max_sharpe",
             mu_window: int = 252
             ) -> Tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Sleeve getirilerini birleştirir (tahsis + portföy hedef-vol kaldıraç).

    method: "max_sharpe" (varsayılan, büzültmeli) | "erc" (yalnız risk paritesi)
    mu_window: beklenen getiri tahmini penceresi (kovaryanstan uzun — ortalama
               tahmini varyans tahmininden çok daha gürültülüdür)

    Döndürür: (portföy getirisi, sleeve ağırlıkları, kaldıraç serisi)

    TÜM ağırlıklar t-1'e kadarki veriyle hesaplanır ve .shift(1) ile uygulanır —
    look-ahead bu katmanda da yasaktır."""
    R = sleeve_returns.fillna(0.0)
    cols = list(R.columns)
    n = len(cols)
    start = max(cov_window, mu_window if method == "max_sharpe" else cov_window)
    if n == 0 or len(R) < start + 5:
        empty = pd.Series(dtype=float)
        return empty, pd.DataFrame(columns=cols), empty

    W = pd.DataFrame(0.0, index=R.index, columns=cols)
    lev = pd.Series(0.0, index=R.index)
    target_daily_vol = target_annual_vol / np.sqrt(periods_per_year)

    for i in range(start, len(R)):
        window = R.iloc[i - cov_window:i]
        active = [c for c in cols if window[c].abs().sum() > 1e-12]
        if len(active) == 0:
            continue
        cov = np.atleast_2d(np.cov(window[active].values, rowvar=False, ddof=1))

        if method == "erc":
            w_act = erc_weights(cov, max_weight=max_sleeve_risk)
        else:
            mu = R[active].iloc[max(0, i - mu_window):i].mean().values
            w_act = max_sharpe_weights(cov, mu, max_weight=max_sleeve_risk)

        if w_act.sum() <= 1e-12:
            continue
        w = pd.Series(0.0, index=cols)
        w[active] = w_act
        W.iloc[i] = w.values

        port_vol = float(np.sqrt(max(w_act @ cov @ w_act, 1e-18)))
        lev.iloc[i] = float(np.clip(target_daily_vol / port_vol, 0.0, max_leverage))

    W_held = W.shift(1).fillna(0.0)
    lev_held = lev.shift(1).fillna(0.0)
    port = (W_held.mul(lev_held, axis=0) * R).sum(axis=1)
    return port.rename("portfolio"), W_held, lev_held


def current_allocation(sleeve_returns: pd.DataFrame,
                       target_annual_vol: float = TARGET_ANNUAL_VOL,
                       max_leverage: float = MAX_LEVERAGE,
                       max_sleeve_risk: float = MAX_SLEEVE_RISK,
                       cov_window: int = COV_WINDOW,
                       periods_per_year: float = PERIODS_PER_YEAR) -> Dict:
    """Canlı kullanım: bugünkü sleeve ağırlıkları + kaldıraç."""
    R = sleeve_returns.fillna(0.0).tail(cov_window)
    cols = list(R.columns)
    active = [c for c in cols if R[c].abs().sum() > 1e-12]
    if len(R) < 20 or not active:
        return {"weights": {c: 0.0 for c in cols}, "leverage": 0.0,
                "reason": "yetersiz geçmiş"}

    cov = np.atleast_2d(np.cov(R[active].values, rowvar=False, ddof=1))
    w_act = erc_weights(cov, max_weight=max_sleeve_risk)
    port_vol = float(np.sqrt(max(w_act @ cov @ w_act, 1e-18)))
    target_daily_vol = target_annual_vol / np.sqrt(periods_per_year)
    leverage = float(np.clip(target_daily_vol / port_vol, 0.0, max_leverage))

    weights = {c: 0.0 for c in cols}
    for c, w in zip(active, w_act):
        weights[c] = round(float(w), 4)

    rc = w_act * (cov @ w_act)
    rc = rc / max(rc.sum(), 1e-18)
    return {"weights": weights, "leverage": round(leverage, 3),
            "risk_contribution": {c: round(float(x), 4) for c, x in zip(active, rc)},
            "portfolio_vol_annual": round(port_vol * np.sqrt(periods_per_year), 4),
            "target_vol_annual": target_annual_vol}


def diversification_report(sleeve_returns: pd.DataFrame) -> Dict:
    """Sleeve'ler gerçekten bağımsız mı? Çeşitlendirmenin ÖLÇÜSÜ.

    Beklenen portföy Sharpe'ı ≈ SR_ort × √(N / (1 + (N−1)·ρ_ort))
    Bu formül planın 1,9 tahmininin kaynağıdır; burada gerçek veriyle ölçülür."""
    R = sleeve_returns.dropna(how="all").fillna(0.0)
    cols = [c for c in R.columns if R[c].abs().sum() > 1e-12]
    if len(cols) < 2:
        return {"n_sleeves": len(cols), "note": "karşılaştırma için ≥2 sleeve gerekli"}

    corr = R[cols].corr()
    off = corr.values[~np.eye(len(cols), dtype=bool)]
    rho = float(np.mean(off))
    sr = {c: float(R[c].mean() / (R[c].std() + 1e-12) * np.sqrt(PERIODS_PER_YEAR))
          for c in cols}
    sr_mean = float(np.mean(list(sr.values())))
    n = len(cols)
    expected = sr_mean * np.sqrt(n / (1 + (n - 1) * max(rho, -1 / (n - 1) + 1e-9)))
    return {
        "n_sleeves": n,
        "sleeve_sharpe": {k: round(v, 3) for k, v in sr.items()},
        "mean_sleeve_sharpe": round(sr_mean, 3),
        "mean_correlation": round(rho, 3),
        "max_pair_correlation": round(float(np.max(off)), 3),
        "expected_portfolio_sharpe": round(float(expected), 3),
        "correlation_matrix": corr.round(3).to_dict(),
    }
