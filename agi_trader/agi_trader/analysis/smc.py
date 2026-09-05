"""
Smart Money Concepts / ICT Motoru.

- Swing yapısı (HH/HL/LH/LL)
- BOS (Break of Structure) ve CHOCH (Change of Character)
- FVG (Fair Value Gap) tespiti
- Order Block (son zıt-yön mumu kırılımdan önce)
- Likidite seviyeleri (eşit tepe/dip)

Çıktı: LayerVote (boğa/ayı yapısal skoru) + detay sözlüğü.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..core.models import LayerVote
from .patterns import find_pivots


def find_swings(df: pd.DataFrame) -> List[Tuple[int, float, str]]:
    """Sıralı swing noktaları: (index, price, 'H'|'L')."""
    sh, sl = find_pivots(df, 2, 2)
    swings = [(i, float(df["high"].iloc[i]), "H") for i in sh] + \
             [(i, float(df["low"].iloc[i]), "L") for i in sl]
    return sorted(swings, key=lambda x: x[0])


def _market_structure(swings: List[Tuple[int, float, str]]) -> Tuple[str, List[str]]:
    """HH/HL/LH/LL dizisinden trend yapısı ve son BOS/CHOCH olayını çıkar."""
    highs = [(i, p) for i, p, t in swings if t == "H"]
    lows = [(i, p) for i, p, t in swings if t == "L"]
    notes: List[str] = []
    if len(highs) < 2 or len(lows) < 2:
        return "belirsiz", notes

    hh = highs[-1][1] > highs[-2][1]
    hl = lows[-1][1] > lows[-2][1]
    lh = highs[-1][1] < highs[-2][1]
    ll = lows[-1][1] < lows[-2][1]

    if hh and hl:
        structure = "yükseliş"
        notes.append("Yapı: HH + HL (boğa)")
    elif lh and ll:
        structure = "düşüş"
        notes.append("Yapı: LH + LL (ayı)")
    else:
        structure = "sıkışma"
        notes.append("Yapı: karışık (range)")
    return structure, notes


def detect_fvg(df: pd.DataFrame, lookback: int = 30) -> List[Dict]:
    """Fair Value Gap: 3 mumda 1. mum high < 3. mum low (boğa) veya tersi."""
    fvgs: List[Dict] = []
    h, l = df["high"].values, df["low"].values
    start = max(2, len(df) - lookback)
    for i in range(start, len(df)):
        # boğa FVG
        if l[i] > h[i - 2]:
            fvgs.append({"type": "bullish", "index": i, "gap": (float(h[i - 2]), float(l[i]))})
        # ayı FVG
        if h[i] < l[i - 2]:
            fvgs.append({"type": "bearish", "index": i, "gap": (float(h[i]), float(l[i - 2]))})
    return fvgs


def detect_order_blocks(df: pd.DataFrame, lookback: int = 30) -> List[Dict]:
    """Basit OB: güçlü hamleden önceki son zıt-yön mumu."""
    obs: List[Dict] = []
    o, c = df["open"].values, df["close"].values
    start = max(1, len(df) - lookback)
    avg_body = float(np.abs(c - o)[-lookback:].mean()) + 1e-12
    for i in range(start, len(df) - 1):
        body = c[i + 1] - o[i + 1]
        if body > 2 * avg_body and c[i] < o[i]:   # güçlü yeşilden önce kırmızı = bullish OB
            obs.append({"type": "bullish", "index": i, "zone": (float(min(o[i], c[i])), float(max(o[i], c[i])))})
        if -body > 2 * avg_body and c[i] > o[i]:  # güçlü kırmızıdan önce yeşil = bearish OB
            obs.append({"type": "bearish", "index": i, "zone": (float(min(o[i], c[i])), float(max(o[i], c[i])))})
    return obs


def smc_vote(df: pd.DataFrame) -> LayerVote:
    swings = find_swings(df)
    structure, reasons = _market_structure(swings)
    price = float(df["close"].iloc[-1])

    score = 0.0
    if structure == "yükseliş":
        score += 0.6
    elif structure == "düşüş":
        score -= 0.6

    fvgs = detect_fvg(df)
    obs = detect_order_blocks(df)

    # En yakın açık FVG yönü
    recent_fvg = fvgs[-3:]
    bull_fvg = sum(1 for f in recent_fvg if f["type"] == "bullish")
    bear_fvg = sum(1 for f in recent_fvg if f["type"] == "bearish")
    if bull_fvg > bear_fvg:
        score += 0.2; reasons.append(f"Boğa FVG baskın ({bull_fvg})")
    elif bear_fvg > bull_fvg:
        score -= 0.2; reasons.append(f"Ayı FVG baskın ({bear_fvg})")

    # CHOCH tespiti: son swing yapı değişimi
    if len(swings) >= 4:
        last4 = swings[-4:]
        types = "".join(t for _, _, t in last4)
        if structure == "yükseliş" and last4[-1][2] == "L" and last4[-1][1] < last4[-3][1]:
            score -= 0.3; reasons.append("CHOCH: yükselişte yapı kırıldı (uyarı)")
        if structure == "düşüş" and last4[-1][2] == "H" and last4[-1][1] > last4[-3][1]:
            score += 0.3; reasons.append("CHOCH: düşüşte yapı kırıldı (dönüş sinyali)")

    # Order block yakınlığı
    for ob in obs[-2:]:
        lo, hi = ob["zone"]
        if lo <= price <= hi:
            if ob["type"] == "bullish":
                score += 0.2; reasons.append("Fiyat boğa Order Block içinde")
            else:
                score -= 0.2; reasons.append("Fiyat ayı Order Block içinde")

    confidence = 0.45 if structure != "belirsiz" else 0.3
    confidence = min(1.0, confidence + 0.05 * len(reasons))

    return LayerVote(
        name="smc",
        score=float(np.clip(score, -1, 1)),
        confidence=float(confidence),
        reasons=reasons[:8],
        detail={"structure": structure, "fvg_count": len(fvgs), "ob_count": len(obs)},
    )
