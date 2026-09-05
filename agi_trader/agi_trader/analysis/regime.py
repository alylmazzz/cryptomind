"""
Piyasa Rejimi Tespiti (HMM) + Dinamik Pozisyon Çarpanı.

Gizli Markov Modeli (hmmlearn) getiri serisine 3 gizli durum uydurur; her durum
ortalama getiri + varyansına göre yorumlanır:
  • Yüksek varyans            → VOLATİL  (risk-off, küçük pozisyon)
  • Belirgin pozitif drift     → TREND YUKARI
  • Belirgin negatif drift     → TREND AŞAĞI
  • Düşük drift + düşük varyans → RANGE / YATAY

Rejim, otonom motorun pozisyon BÜYÜKLÜĞÜNÜ ölçekler (dynamic sizing):
  trend (sinyalle uyumlu) = tam · range = 0.6 · volatil = 0.4 · trende-karşı = 0.5×

hmmlearn yoksa ADX + Bollinger genişliği + getiri vol ile sezgisel rejime düşer.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .indicators import adx, bollinger

try:
    from hmmlearn.hmm import GaussianHMM
    _HAS_HMM = True
except Exception:  # pragma: no cover
    _HAS_HMM = False

_BASE_MULT = {"TREND YUKARI": 1.0, "TREND AŞAĞI": 1.0, "RANGE / YATAY": 0.6, "VOLATİL": 0.4}
_EMOJI = {"TREND YUKARI": "📈", "TREND AŞAĞI": "📉", "RANGE / YATAY": "↔️", "VOLATİL": "🌪️"}


def _heuristic(df: pd.DataFrame) -> Dict:
    try:
        adx_v, pdi, mdi = adx(df)
        adx_last = float(adx_v.iloc[-1])
        bb_u, bb_m, bb_l = bollinger(df["close"])
        bbw = float((bb_u.iloc[-1] - bb_l.iloc[-1]) / (bb_m.iloc[-1] + 1e-12))
        rv = float(df["close"].pct_change().rolling(20).std().iloc[-1] or 0)
        up = float(pdi.iloc[-1]) > float(mdi.iloc[-1])
        if rv > 0.05 or bbw > 0.18:
            label = "VOLATİL"
        elif adx_last >= 25:
            label = "TREND YUKARI" if up else "TREND AŞAĞI"
        else:
            label = "RANGE / YATAY"
        return {"label": label, "method": "heuristic", "confidence": 0.5,
                "adx": round(adx_last, 1), "realized_vol": round(rv, 4)}
    except Exception:
        return {"label": "RANGE / YATAY", "method": "fallback", "confidence": 0.3,
                "adx": 0.0, "realized_vol": 0.0}


def detect_regime(df: pd.DataFrame) -> Dict:
    """Rejim sözlüğü: {label, emoji, method, confidence, multiplier(yön-bağımsız), ...}."""
    rets = df["close"].pct_change().dropna().values * 100.0
    base = None
    if _HAS_HMM and len(rets) >= 80:
        try:
            X = rets.reshape(-1, 1)
            model = GaussianHMM(n_components=3, covariance_type="diag",
                                n_iter=60, random_state=42)
            model.fit(X)
            states = model.predict(X)
            post = model.predict_proba(X)
            cur = int(states[-1])
            # her durumun ortalama + varyansı
            means = np.array([X[states == s].mean() if (states == s).any() else 0.0 for s in range(3)])
            vars = np.array([X[states == s].var() if (states == s).any() else 0.0 for s in range(3)])
            vol_state = int(np.argmax(vars))
            if cur == vol_state and vars[cur] > 1.5 * np.median(vars):
                label = "VOLATİL"
            elif means[cur] > 0.05 and abs(means[cur]) >= np.abs(means).mean():
                label = "TREND YUKARI"
            elif means[cur] < -0.05 and abs(means[cur]) >= np.abs(means).mean():
                label = "TREND AŞAĞI"
            else:
                label = "RANGE / YATAY"
            base = {"label": label, "method": "HMM", "confidence": round(float(post[-1, cur]), 2),
                    "state": cur, "state_mean": round(float(means[cur]), 3),
                    "state_var": round(float(vars[cur]), 3)}
        except Exception:
            base = None
    if base is None:
        base = _heuristic(df)

    base["emoji"] = _EMOJI.get(base["label"], "")
    base["multiplier"] = _BASE_MULT.get(base["label"], 0.6)
    return base


def position_multiplier(regime: Dict, direction: str, volatility: str = "medium") -> float:
    """Rejim + sinyal yönü + volatiliteye göre nihai pozisyon-boyut çarpanı (0..1.1)."""
    if not regime:
        return 1.0
    m = float(regime.get("multiplier", 0.6))
    label = regime.get("label", "")
    # trende-karşı işlem → yarıya indir
    if direction == "LONG" and label == "TREND AŞAĞI":
        m *= 0.5
    elif direction == "SHORT" and label == "TREND YUKARI":
        m *= 0.5
    # trendle-uyumlu → hafif bonus
    elif (direction == "LONG" and label == "TREND YUKARI") or \
         (direction == "SHORT" and label == "TREND AŞAĞI"):
        m = min(1.1, m * 1.1)
    # aşırı volatilite → ek kısıntı
    if volatility == "extreme":
        m *= 0.6
    elif volatility == "high":
        m *= 0.8
    return round(float(np.clip(m, 0.0, 1.1)), 3)
