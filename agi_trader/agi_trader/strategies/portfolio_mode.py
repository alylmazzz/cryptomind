"""
PORTFÖY MODU — RISK_ON · SELECTIVE · DEFENSIVE · CASH (tek RSI ile değil, çok kaynaklı).

Girdiler: evren genişliği (EMA20/EMA50 üstü %, 1 sa pozitif %), korelasyon şoku (1 dk
getiri ortalama çift korelasyonu), BTC/ETH rejimi, haber risk seviyesi, portföy drawdown'u,
model bozulması (sağlık), veri tazeliği. committee.market_risk'i genişletir.

Eylemler:
  RISK_ON    yeni giriş serbest, boyut ×1,0
  SELECTIVE  yeni giriş yalnız güvenilir sleeve'ler, boyut ×0,7
  DEFENSIVE  yeni giriş yok; kârdaki pozisyonların stop'u girişe (breakeven), boyut ×0,5
  CASH       bekleyen emirler iptal, açık pozisyonlar kapatılır (NAKİT MODU)
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

RISK_ON, SELECTIVE, DEFENSIVE, CASH = "RISK_ON", "SELECTIVE", "DEFENSIVE", "CASH"
LEVEL = {RISK_ON: 0, SELECTIVE: 1, DEFENSIVE: 2, CASH: 3}
LABEL_TR = {RISK_ON: "RİSK AÇIK", SELECTIVE: "SEÇİCİ — yalnız güvenilir sleeve'ler",
            DEFENSIVE: "SAVUNMA — yeni giriş yok, stoplar girişe", CASH: "NAKİT MODU — pozisyonlar kapatılır"}


def breadth(cross: Dict[str, Dict]) -> Dict:
    n = len(cross)
    if n == 0:
        return {"n": 0}
    a20 = sum(1 for f in cross.values() if f.get("above_ema20")) / n
    a50 = sum(1 for f in cross.values() if f.get("above_ema50")) / n
    pos1h = [f.get("ret_1h_pct") for f in cross.values() if f.get("ret_1h_pct") is not None]
    p1 = (sum(1 for x in pos1h if x > 0) / len(pos1h)) if pos1h else None
    hi = sum(1 for f in cross.values() if (f.get("dist_hi20_pct") or 1.0) <= 0.05) / n
    lo = sum(1 for f in cross.values() if (f.get("dist_lo20_pct") or 1.0) <= 0.05) / n
    return {"n": n, "pct_above_ema20": round(a20, 3), "pct_above_ema50": round(a50, 3),
            "pct_pos_1h": (None if p1 is None else round(p1, 3)),
            "new_highs_pct": round(hi, 3), "new_lows_pct": round(lo, 3)}


def correlation_shock(rets: Optional[pd.DataFrame]) -> Optional[float]:
    """1 dk getirilerin ortalama çift korelasyonu (0..1). ≥0,7 = tek beta işlemi."""
    if rets is None or rets.shape[1] < 3:
        return None
    c = rets.corr().values
    off = c[~np.eye(len(c), dtype=bool)]
    off = off[np.isfinite(off)]
    return float(np.mean(off)) if off.size else None


def decide_mode(br: Dict, corr: Optional[float], btc_regime: Optional[str], news_level: int,
                dd_pct: float, health_overall: Optional[str] = None, stale_share: float = 0.0,
                max_dd_pct: float = 15.0) -> Dict:
    reasons: List[str] = []
    score = 0
    n = br.get("n", 0)
    if n >= 10:
        a20, a50 = br.get("pct_above_ema20", 0.5), br.get("pct_above_ema50", 0.5)
        p1 = br.get("pct_pos_1h")
        if a20 <= 0.25 and a50 <= 0.35:
            score += 2; reasons.append(f"genişlik zayıf: EMA20 üstü %{a20*100:.0f}, EMA50 üstü %{a50*100:.0f}")
        elif a20 <= 0.4:
            score += 1; reasons.append(f"genişlik ılımlı zayıf: EMA20 üstü %{a20*100:.0f}")
        if p1 is not None and p1 <= 0.2:
            score += 1; reasons.append(f"1 sa pozitif yalnız %{p1*100:.0f}")
        if br.get("new_lows_pct", 0) >= 0.3:
            score += 1; reasons.append(f"20-bar yeni dip yapan %{br['new_lows_pct']*100:.0f}")
    if corr is not None and corr >= 0.7:
        score += 1; reasons.append(f"korelasyon şoku: ort. çift korelasyonu {corr:.2f}")
    if btc_regime == "TREND AŞAĞI":
        score += 1; reasons.append("BTC rejimi TREND AŞAĞI")
    if news_level >= 2:
        score += 3; reasons.append("haber: sistemik risk")
    elif news_level == 1:
        score += 1; reasons.append("haber: risk-off başlıkları")
    if dd_pct >= max_dd_pct * 0.66:
        score += 2; reasons.append(f"portföy drawdown %{dd_pct:.1f}")
    elif dd_pct >= max_dd_pct * 0.4:
        score += 1; reasons.append(f"drawdown %{dd_pct:.1f}")
    if health_overall in ("RED",):
        score += 3; reasons.append("sistem sağlığı RED")
    elif health_overall in ("DEGRADED", "UNKNOWN"):
        score += 1; reasons.append(f"sistem sağlığı {health_overall}")
    if stale_share >= 0.5:
        score += 2; reasons.append(f"veri bayat: paritelerin %{stale_share*100:.0f}'i")
    # RISK_ON ≤0 · SELECTIVE 1–2 · DEFENSIVE 3–4 · CASH ≥5 (ılımlı zayıf genişlik tek başına giriş kapatmaz;
    # canlı karşı-olgusal: SAVUNMA kapısı 3 kaçırılan / 0 kaçınılan; sistemik haber/RED yine anında CASH)
    mode = RISK_ON if score <= 0 else SELECTIVE if score <= 2 else DEFENSIVE if score <= 4 else CASH
    if news_level >= 2 or health_overall == "RED":
        mode = CASH
    return {"mode": mode, "level": LEVEL[mode], "label": LABEL_TR[mode], "score": score,
            "reasons": reasons, "breadth": br, "avg_corr": (None if corr is None else round(corr, 3)),
            "btc_regime": btc_regime, "news_level": news_level, "dd_pct": round(dd_pct, 2),
            "actions": {"new_entries": mode in (RISK_ON, SELECTIVE),
                        "size_mult": {RISK_ON: 1.0, SELECTIVE: 0.7, DEFENSIVE: 0.5, CASH: 0.0}[mode],
                        "tighten_stops": mode in (DEFENSIVE, CASH), "flatten": mode == CASH,
                        "reliable_sleeves_only": mode == SELECTIVE}}
