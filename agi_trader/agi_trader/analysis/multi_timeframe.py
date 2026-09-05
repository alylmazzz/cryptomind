"""
Çok Zaman Dilimli (MTF) Confluence Motoru.

Spec: "Her timeframe bağımsız analiz edilmeli, sonra Multi Timeframe Analysis
olarak birleştirilmeli." Bu modül her TF'de teknik oyu hesaplar ve daha yüksek
zaman dilimlerine daha çok ağırlık vererek tek bir confluence skoru üretir.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from ..core.models import LayerVote
from .indicators import compute_all_indicators, technical_vote

# Zaman dilimi -> önem ağırlığı (yüksek TF daha güvenilir trend)
TF_WEIGHT = {
    "1m": 0.3, "3m": 0.35, "5m": 0.4, "15m": 0.6, "30m": 0.7,
    "1h": 0.9, "2h": 1.0, "4h": 1.3, "6h": 1.35, "8h": 1.4,
    "12h": 1.5, "1d": 1.7, "3d": 1.8, "1w": 2.0, "1M": 2.2, "1y": 2.4,
}


def multi_timeframe_vote(mtf_data: Dict[str, pd.DataFrame]) -> LayerVote:
    if not mtf_data:
        return LayerVote(name="multi_timeframe", score=0.0, confidence=0.0,
                         reasons=["MTF verisi yok"])

    weighted = 0.0
    wsum = 0.0
    reasons = []
    per_tf = {}
    for tf, df in mtf_data.items():
        try:
            ind = compute_all_indicators(df)
            v = technical_vote(df, ind)
        except Exception:
            continue
        w = TF_WEIGHT.get(tf, 1.0)
        weighted += v.score * v.confidence * w
        wsum += w * v.confidence
        per_tf[tf] = round(v.score, 2)
        arrow = "↑" if v.score > 0.1 else "↓" if v.score < -0.1 else "→"
        reasons.append(f"{arrow} {tf}: {v.bias.value} ({v.score:+.2f})")

    score = weighted / (wsum + 1e-12) if wsum else 0.0

    # Confluence: kaç TF aynı yönde
    aligned = sum(1 for s in per_tf.values() if (s > 0) == (score > 0) and abs(s) > 0.1)
    total = len([s for s in per_tf.values() if abs(s) > 0.1])
    confidence = min(1.0, 0.4 + 0.12 * aligned) if total else 0.3

    if total:
        reasons.insert(0, f"Confluence: {aligned}/{len(per_tf)} zaman dilimi hizalı")

    return LayerVote(
        name="multi_timeframe",
        score=float(np.clip(score, -1, 1)),
        confidence=float(confidence),
        reasons=reasons[:10],
        detail={"per_timeframe": per_tf},
    )
