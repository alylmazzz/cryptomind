"""
AI Ensemble Motoru (Derin Öğrenme / AI Araştırmacısı rolü).

Spec: tek model değil ENSEMBLE — Transformer/LSTM/TFT/GNN/RL/Bayesian/
GBM/RF/XGB/LightGBM/CatBoost/Autoencoder/HMM birlikte.

Gerçekçi, çalışan iskelet: bir sonraki N-bar yön olasılığını tahmin eden
ÇOK MODELLİ bir topluluk kurar. scikit-learn kuruluysa GradientBoosting +
LogisticRegression + RandomForest self-supervised olarak geçmiş barlar üzerinde
eğitilir (look-ahead bias'tan kaçınmak için yalnızca geçmiş pencere). sklearn
yoksa momentum / mean-reversion / trend-takip uzman modellerinin Bayesçi
ağırlıklı topluluğu kullanılır. Her iki yol da kalibre edilmiş bir olasılık ve
açıklama döndürür.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..core.models import LayerVote
from ..core.light import LIGHT_MODE
from ..analysis.indicators import rsi, macd, atr, ema, cci, mfi, adx, stoch_rsi, bollinger, williams_r, roc

try:
    from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                                  ExtraTreesClassifier, HistGradientBoostingClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

# HAFİF MOD (AGI_LIGHT_MODE=1) — paylaşımlı/az-bellekli sunucuda (CryptoMind'ın
# çalıştığı ARM 4GB VPS) ağır modeller HİÇ IMPORT EDİLMEZ. Bayrağı sonradan
# False yapmak yetmezdi: torch tek başına ~400 MB RSS tutuyor. Bu modda ensemble
# sklearn + uzman-model Bayes topluluğuna düşer (aynı arayüz, aynı çıktı şeması).
if LIGHT_MODE:
    _HAS_XGB = _HAS_LGBM = _HAS_TORCH = False
else:
    # Gradient boosting'in en iyileri (varsa otomatik devreye girer)
    try:
        from xgboost import XGBClassifier
        _HAS_XGB = True
    except Exception:
        _HAS_XGB = False

    try:
        from lightgbm import LGBMClassifier
        _HAS_LGBM = True
    except Exception:
        _HAS_LGBM = False

    try:
        import torch  # noqa: F401
        from .deep_models import DeepEnsemble
        _HAS_TORCH = True
    except Exception:
        _HAS_TORCH = False

# symbol+timeframe başına eğitilmiş derin modelleri bellekte tut
_DEEP_CACHE: dict = {}


_FEATURE_NAMES = ["ret1", "ret5", "ret10", "rsi", "stoch_rsi", "macd_hist", "ema_ratio",
                  "atr", "cci", "mfi", "adx", "willr", "bb_pctb", "roc12", "vol_z"]


def _features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    f = pd.DataFrame(index=df.index)
    f["ret1"] = c.pct_change()
    f["ret5"] = c.pct_change(5)
    f["ret10"] = c.pct_change(10)
    f["rsi"] = rsi(c, 14) / 100
    f["stoch_rsi"] = stoch_rsi(c) / 100
    line, sig, hist = macd(c)
    f["macd_hist"] = hist / (c + 1e-9)
    f["ema_ratio"] = ema(c, 20) / (ema(c, 50) + 1e-9) - 1
    f["atr"] = atr(df, 14) / (c + 1e-9)
    f["cci"] = cci(df) / 200
    f["mfi"] = mfi(df) / 100
    adx_v, _, _ = adx(df)
    f["adx"] = adx_v / 100
    f["willr"] = williams_r(df) / 100
    bb_u, bb_m, bb_l = bollinger(c)
    f["bb_pctb"] = (c - bb_l) / (bb_u - bb_l + 1e-9)
    f["roc12"] = roc(c, 12) / 100
    f["vol_z"] = (df["volume"] - df["volume"].rolling(20).mean()) / (df["volume"].rolling(20).std() + 1e-9)
    return f[_FEATURE_NAMES].replace([np.inf, -np.inf], 0).fillna(0)


def _ml_ensemble(df: pd.DataFrame, horizon: int = 3) -> Tuple[float, List[str], Dict]:
    """sklearn + XGBoost + LightGBM ile self-supervised yön tahmini (p_up).
    Ağaç-tabanlı modellerden ortalama özellik önemini de döndürür (açıklanabilirlik)."""
    f = _features(df)
    c = df["close"]
    target = (c.shift(-horizon) > c).astype(int)
    data = f.iloc[:-horizon]
    y = target.iloc[:-horizon]
    if len(data) < 80 or y.nunique() < 2:
        return 0.5, ["Yetersiz veri — nötr"], {}

    X = data.values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    x_now = scaler.transform(f.iloc[[-1]].values)

    models: Dict[str, object] = {
        "GradientBoosting": GradientBoostingClassifier(n_estimators=60, max_depth=3),
        "HistGB": HistGradientBoostingClassifier(max_iter=120, max_depth=4),
        "RandomForest": RandomForestClassifier(n_estimators=80, max_depth=5, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=80, max_depth=6, n_jobs=-1),
        "Logistic": LogisticRegression(max_iter=300),
    }
    if _HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=140, max_depth=3, learning_rate=0.08, subsample=0.85,
            colsample_bytree=0.85, eval_metric="logloss", verbosity=0, n_jobs=-1)
    if _HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=140, max_depth=4, learning_rate=0.08, subsample=0.85,
            colsample_bytree=0.85, verbose=-1, n_jobs=-1)

    probs, reasons = [], []
    importances = np.zeros(len(_FEATURE_NAMES))
    imp_count = 0
    for name, m in models.items():
        try:
            with warnings.catch_warnings():  # LightGBM/sklearn iyi-huylu feature-name uyarısını sustur
                warnings.simplefilter("ignore")
                m.fit(Xs, y.values)
                p = float(m.predict_proba(x_now)[0][1])
            probs.append(p)
            reasons.append(f"{name}: p(up)={p:.2f}")
            fi = getattr(m, "feature_importances_", None)
            if fi is not None and len(fi) == len(_FEATURE_NAMES):
                s = fi.sum()
                if s > 0:
                    importances += fi / s
                    imp_count += 1
        except Exception:
            continue
    if not probs:
        return 0.5, ["Model eğitimi başarısız — nötr"], {}

    feat_imp: Dict[str, float] = {}
    if imp_count:
        importances /= imp_count
        order = np.argsort(importances)[::-1][:4]
        feat_imp = {_FEATURE_NAMES[i]: round(float(importances[i]), 3) for i in order}
        reasons.append("En etkili özellikler: " + ", ".join(f"{k}={v}" for k, v in feat_imp.items()))
    return float(np.mean(probs)), reasons, feat_imp


def _heuristic_ensemble(df: pd.DataFrame) -> Tuple[float, List[str]]:
    """sklearn yoksa uzman-model topluluğu (momentum/mean-rev/trend)."""
    c = df["close"]
    reasons = []
    votes = []

    # Momentum uzmanı
    mom = c.pct_change(10).iloc[-1]
    p_mom = 1 / (1 + np.exp(-mom * 40))
    votes.append(p_mom); reasons.append(f"Momentum modeli: p(up)={p_mom:.2f}")

    # Trend-takip uzmanı (EMA dizilimi)
    trend = (ema(c, 20).iloc[-1] - ema(c, 50).iloc[-1]) / (c.iloc[-1] + 1e-9)
    p_trend = 1 / (1 + np.exp(-trend * 120))
    votes.append(p_trend); reasons.append(f"Trend modeli: p(up)={p_trend:.2f}")

    # Mean-reversion uzmanı (RSI)
    r = rsi(c, 14).iloc[-1]
    p_mr = 1 / (1 + np.exp((r - 50) / 12))  # düşük RSI -> yukarı beklentisi
    votes.append(p_mr); reasons.append(f"Mean-reversion modeli: p(up)={p_mr:.2f}")

    # Volatilite-ayarlı drift (HMM-benzeri rejim proxy)
    rets = c.pct_change().dropna()
    drift = rets.tail(20).mean() / (rets.tail(20).std() + 1e-9)
    p_reg = 1 / (1 + np.exp(-drift * 3))
    votes.append(p_reg); reasons.append(f"Rejim modeli: p(up)={p_reg:.2f}")

    return float(np.mean(votes)), reasons


def _deep_ensemble(df: pd.DataFrame, symbol: str, timeframe: str):
    """PyTorch Transformer + LSTM + RL toplulugu."""
    key = f"{symbol}_{timeframe}"
    de = _DEEP_CACHE.get(key)
    if de is None:
        de = DeepEnsemble(symbol, timeframe)
        de.fit_or_load(df)
        _DEEP_CACHE[key] = de
    meta = de.meta
    pred = de.predict(df)
    reasons = [
        f"Transformer p(up)={pred['models'].get('transformer', 0.5):.2f} | "
        f"LSTM p(up)={pred['models'].get('lstm', 0.5):.2f} | "
        f"RL aksiyon={pred['rl_action'].upper()} (p={pred['models'].get('rl_p_up', 0.5):.2f})",
    ]
    if meta.get("trained") or meta.get("from_cache"):
        reasons.append(f"Eğitim: {meta.get('n_samples','?')} örnek | "
                       f"Transformer val-acc {meta.get('transformer_val_acc','?')} | "
                       f"LSTM val-acc {meta.get('lstm_val_acc','?')} | "
                       f"RL ort. ödül {meta.get('rl_avg_reward','?')}"
                       + (" [cache]" if meta.get("from_cache") else ""))
    return pred["p_up"], reasons, pred


def ensemble_vote(df: pd.DataFrame, symbol: str = "?", timeframe: str = "?") -> LayerVote:
    backends: list = []   # (p_up, weight, label, reasons)

    if _HAS_TORCH:
        try:
            p_deep, r_deep, _ = _deep_ensemble(df, symbol, timeframe)
            backends.append((p_deep, 0.45, "deep(Transformer+LSTM+RL)", r_deep))
        except Exception as e:
            backends.append((0.5, 0.05, "deep(hata)", [f"Derin model hatası: {type(e).__name__}"]))

    # RL v2 — PPO ajanı (stable-baselines3) — sıralı pozisyon yönetimi öğrenir
    try:
        from .rl_agent import get_rl_agent, _HAS_RL
        if _HAS_RL and len(df) > 120:
            name = f"{symbol}_{timeframe}".replace("/", "_")
            ag = get_rl_agent(name, [df])
            if ag is not None:
                pr = ag.predict(df)
                if pr.get("available"):
                    backends.append((pr["p_up"], 0.25, "rl_ppo(PPO)",
                                     [f"PPO RL: aksiyon={pr['action'].upper()} p(up)={pr['p_up']} "
                                      f"(long {pr.get('p_long')}/short {pr.get('p_short')})"]))
    except Exception:
        pass

    feat_imp: Dict[str, float] = {}
    if _HAS_SKLEARN:
        p_ml, r_ml, feat_imp = _ml_ensemble(df)
        gbm_label = "sklearn+XGB+LGBM" if (_HAS_XGB or _HAS_LGBM) else "sklearn(GBM+RF+Logistic)"
        backends.append((p_ml, 0.35, gbm_label, r_ml))

    p_h, r_h = _heuristic_ensemble(df)
    backends.append((p_h, 0.2, "heuristic(mom+trend+mr+regime)", []))

    backends = [b for b in backends if b]
    wsum = sum(b[1] for b in backends) + 1e-9
    p_up = sum(b[0] * b[1] for b in backends) / wsum

    reasons: list = []
    labels = []
    for _, _, label, rs in backends:
        labels.append(label)
        reasons.extend(rs)
    reasons.insert(0, f"Aktif modeller: {', '.join(labels)} → birleşik p(up)={p_up:.2f}")

    score = 2 * p_up - 1
    confidence = float(min(0.95, 0.4 + abs(p_up - 0.5) * 1.6))

    return LayerVote(
        name="ai_ensemble",
        score=float(np.clip(score, -1, 1)),
        confidence=confidence,
        reasons=reasons[:9],
        detail={"p_up": p_up, "backends": labels, "feature_importance": feat_imp},
    )
