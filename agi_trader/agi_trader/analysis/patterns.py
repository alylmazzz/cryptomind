"""
Formasyon Tespit Motoru (Teknik Analiz & Formasyon Uzmanı rolü).

- Harmonik formasyonlar: Gartley, Bat, Butterfly, Crab, Cypher, Shark
  (Fibonacci oran tabanlı XABCD tespiti)
- Klasik formasyonlar: Double Top/Bottom, Head & Shoulders (+ ters)
- Swing high/low (pivot) tespiti — formasyonların ve maks/min analizinin temeli
- Her formasyon için: yön, tamamlanma %, kalite, hedef ve iptal seviyesi

Çıktı: List[PatternMatch] ve bir LayerVote (boğa/ayı skoru).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..core.models import Direction, LayerVote, PatternMatch


# ----------------------------------------------------------------------------
# Pivot (swing) tespiti
# ----------------------------------------------------------------------------
def find_pivots(df: pd.DataFrame, left: int = 3, right: int = 3) -> Tuple[List[int], List[int]]:
    """Swing high ve swing low indekslerini döndürür."""
    highs, lows = df["high"].values, df["low"].values
    n = len(df)
    sh, sl = [], []
    for i in range(left, n - right):
        window_h = highs[i - left:i + right + 1]
        window_l = lows[i - left:i + right + 1]
        if highs[i] == window_h.max() and (window_h.argmax() == left):
            sh.append(i)
        if lows[i] == window_l.min() and (window_l.argmin() == left):
            sl.append(i)
    return sh, sl


def _ratio(a: float, b: float) -> float:
    return abs(a) / (abs(b) + 1e-12)


def _within(value: float, target: float, tol: float) -> bool:
    return abs(value - target) <= tol


# ----------------------------------------------------------------------------
# Harmonik formasyonlar (Fibonacci oranları)
# ----------------------------------------------------------------------------
# Her formasyon için XABCD bacaklarının ideal Fibonacci oranları
HARMONIC_RULES = {
    "Gartley":   {"AB_XA": 0.618, "BC_AB": (0.382, 0.886), "CD_BC": (1.13, 1.618), "AD_XA": 0.786},
    "Bat":       {"AB_XA": (0.382, 0.5), "BC_AB": (0.382, 0.886), "CD_BC": (1.618, 2.618), "AD_XA": 0.886},
    "Butterfly": {"AB_XA": 0.786, "BC_AB": (0.382, 0.886), "CD_BC": (1.618, 2.618), "AD_XA": 1.27},
    "Crab":      {"AB_XA": (0.382, 0.618), "BC_AB": (0.382, 0.886), "CD_BC": (2.618, 3.618), "AD_XA": 1.618},
    "Cypher":    {"AB_XA": (0.382, 0.618), "BC_AB": (1.13, 1.414), "CD_BC": (0.786, 0.786), "AD_XA": 0.786},
    "Shark":     {"AB_XA": (0.446, 0.618), "BC_AB": (1.13, 1.618), "CD_BC": (0.886, 1.13), "AD_XA": 0.886},
}


def _check_ratio(val: float, rule) -> float:
    """Oranın kurala uyum skorunu döndürür (0..1)."""
    if isinstance(rule, tuple):
        lo, hi = rule
        if lo <= val <= hi:
            return 1.0
        # aralık dışı: yakınlığa göre azalan skor
        dist = min(abs(val - lo), abs(val - hi))
        return max(0.0, 1 - dist / 0.2)
    else:
        tol = 0.06
        return max(0.0, 1 - abs(val - rule) / tol) if _within(val, rule, 0.15) else 0.0


def detect_harmonics(df: pd.DataFrame) -> List[PatternMatch]:
    sh, sl = find_pivots(df, 3, 3)
    pivots = sorted([(i, df["high"].iloc[i], 1) for i in sh] +
                    [(i, df["low"].iloc[i], -1) for i in sl])
    matches: List[PatternMatch] = []
    if len(pivots) < 5:
        return matches

    # Son 5 ardışık pivot üzerinden XABCD ara
    for start in range(max(0, len(pivots) - 12), len(pivots) - 4):
        pts = pivots[start:start + 5]
        if len(pts) < 5:
            continue
        # zikzak olması için yön değişimleri kontrolü
        types = [p[2] for p in pts]
        if not (types[0] != types[1] and types[1] != types[2] and types[2] != types[3] and types[3] != types[4]):
            continue
        X, A, B, C, D = [p[1] for p in pts]
        iX, iA, iB, iC, idxD = [p[0] for p in pts]

        XA = A - X
        AB = B - A
        BC = C - B
        CD = D - C
        AD = D - A
        if abs(XA) < 1e-9 or abs(AB) < 1e-9 or abs(BC) < 1e-9:
            continue

        ab_xa = _ratio(AB, XA)
        bc_ab = _ratio(BC, AB)
        cd_bc = _ratio(CD, BC)
        ad_xa = _ratio(AD, XA)

        for name, rule in HARMONIC_RULES.items():
            s1 = _check_ratio(ab_xa, rule["AB_XA"])
            s2 = _check_ratio(bc_ab, rule["BC_AB"])
            s3 = _check_ratio(cd_bc, rule["CD_BC"])
            s4 = _check_ratio(ad_xa, rule["AD_XA"])
            quality = (s1 + s2 + s3 + s4) / 4
            if quality >= 0.6:
                # bullish harmonik: D bir dip (son pivot low) -> LONG
                direction = Direction.LONG if pts[4][2] == -1 else Direction.SHORT
                target = D + (0.618 * (A - D)) if direction == Direction.LONG else D - (0.618 * (D - A))
                invalidation = X
                matches.append(PatternMatch(
                    name=name, family="harmonic", direction=direction,
                    completion=1.0, quality=round(quality, 2), pivot_index=idxD,
                    points={"X": float(X), "A": float(A), "B": float(B), "C": float(C), "D": float(D)},
                    indices={"X": int(iX), "A": int(iA), "B": int(iB), "C": int(iC), "D": int(idxD)},
                    target=float(target), invalidation=float(invalidation),
                    ratios={"AB/XA": round(ab_xa, 3), "BC/AB": round(bc_ab, 3),
                            "CD/BC": round(cd_bc, 3), "AD/XA": round(ad_xa, 3)},
                    note=f"AB/XA={ab_xa:.2f} BC/AB={bc_ab:.2f} CD/BC={cd_bc:.2f} AD/XA={ad_xa:.2f}",
                ))
    # aynı index için en kaliteliyi tut
    best: Dict[int, PatternMatch] = {}
    for m in matches:
        if m.pivot_index not in best or m.quality > best[m.pivot_index].quality:
            best[m.pivot_index] = m
    return list(best.values())


# ----------------------------------------------------------------------------
# Klasik formasyonlar
# ----------------------------------------------------------------------------
def detect_classic(df: pd.DataFrame) -> List[PatternMatch]:
    sh, sl = find_pivots(df, 3, 3)
    out: List[PatternMatch] = []
    highs, lows = df["high"].values, df["low"].values
    n = len(df)

    # Double Top / Bottom (son iki belirgin swing)
    if len(sh) >= 2:
        i1, i2 = sh[-2], sh[-1]
        h1, h2 = highs[i1], highs[i2]
        match_pct = _ratio(h1 - h2, h1)
        if match_pct < 0.02 and (i2 - i1) > 4:
            ineck = int(i1 + lows[i1:i2 + 1].argmin())
            neckline = lows[ineck]
            out.append(PatternMatch(
                name="Double Top", family="classic", direction=Direction.SHORT,
                completion=0.8, quality=0.7, pivot_index=i2,
                points={"top1": float(h1), "top2": float(h2), "neckline": float(neckline)},
                indices={"top1": int(i1), "top2": int(i2), "neckline": ineck},
                target=float(neckline - (h2 - neckline)), invalidation=float(max(h1, h2) * 1.01),
                ratios={"tepe farkı %": round(match_pct * 100, 2)},
                note="Çift tepe — boyun çizgisi kırılırsa düşüş",
            ))
    if len(sl) >= 2:
        i1, i2 = sl[-2], sl[-1]
        l1, l2 = lows[i1], lows[i2]
        match_pct = _ratio(l1 - l2, l1)
        if match_pct < 0.02 and (i2 - i1) > 4:
            ineck = int(i1 + highs[i1:i2 + 1].argmax())
            neckline = highs[ineck]
            out.append(PatternMatch(
                name="Double Bottom", family="classic", direction=Direction.LONG,
                completion=0.8, quality=0.7, pivot_index=i2,
                points={"bottom1": float(l1), "bottom2": float(l2), "neckline": float(neckline)},
                indices={"bottom1": int(i1), "bottom2": int(i2), "neckline": ineck},
                target=float(neckline + (neckline - l2)), invalidation=float(min(l1, l2) * 0.99),
                ratios={"dip farkı %": round(match_pct * 100, 2)},
                note="Çift dip — boyun çizgisi kırılırsa yükseliş",
            ))

    # Head & Shoulders (3 tepe: orta en yüksek) ve tersi
    if len(sh) >= 3:
        a, b, cc = sh[-3], sh[-2], sh[-1]
        ha, hb, hc = highs[a], highs[b], highs[cc]
        shoulder_sym = _ratio(ha - hc, ha)
        if hb > ha and hb > hc and shoulder_sym < 0.04:
            ineck = int(a + lows[a:cc + 1].argmin())
            neckline = lows[ineck]
            out.append(PatternMatch(
                name="Head & Shoulders", family="classic", direction=Direction.SHORT,
                completion=0.75, quality=0.72, pivot_index=cc,
                points={"left": float(ha), "head": float(hb), "right": float(hc), "neckline": float(neckline)},
                indices={"left": int(a), "head": int(b), "right": int(cc), "neckline": ineck},
                target=float(neckline - (hb - neckline)), invalidation=float(hb * 1.01),
                ratios={"omuz simetri %": round(shoulder_sym * 100, 2)},
                note="Omuz-baş-omuz — düşüş formasyonu",
            ))
    if len(sl) >= 3:
        a, b, cc = sl[-3], sl[-2], sl[-1]
        la, lb, lc = lows[a], lows[b], lows[cc]
        shoulder_sym = _ratio(la - lc, la)
        if lb < la and lb < lc and shoulder_sym < 0.04:
            ineck = int(a + highs[a:cc + 1].argmax())
            neckline = highs[ineck]
            out.append(PatternMatch(
                name="Inverse Head & Shoulders", family="classic", direction=Direction.LONG,
                completion=0.75, quality=0.72, pivot_index=cc,
                points={"left": float(la), "head": float(lb), "right": float(lc), "neckline": float(neckline)},
                indices={"left": int(a), "head": int(b), "right": int(cc), "neckline": ineck},
                target=float(neckline + (neckline - lb)), invalidation=float(lb * 0.99),
                ratios={"omuz simetri %": round(shoulder_sym * 100, 2)},
                note="Ters OBO — yükseliş formasyonu",
            ))
    return out


# ----------------------------------------------------------------------------
# Maks/min nokta analizi (tüm önceki periyotları inceleyerek)
# ----------------------------------------------------------------------------
def extreme_analysis(df: pd.DataFrame) -> Dict[str, float]:
    """Geçmişin tamamına göre güncel fiyatın konumu + son swing maks/min."""
    c = float(df["close"].iloc[-1])
    hi = float(df["high"].max())
    lo = float(df["low"].min())
    rng = hi - lo + 1e-12
    return {
        "all_time_high": hi,
        "all_time_low": lo,
        "pct_from_high": (c - hi) / hi * 100,
        "pct_from_low": (c - lo) / lo * 100,
        "range_position": (c - lo) / rng,   # 0=dip, 1=tepe
        "recent_high_20": float(df["high"].iloc[-20:].max()),
        "recent_low_20": float(df["low"].iloc[-20:].min()),
    }


def find_confluence(patterns: List[PatternMatch], tol_pct: float = 0.6) -> List[Dict]:
    """Birden çok formasyon AYNI fiyat noktasını/bölgesini işaret ediyorsa grupla.
    Confluence (birleşim) = daha güçlü sinyal; özellikle aynı yöndeyse.

    Her formasyonun tamamlanma fiyatı (apex/pivot) baz alınır; tol_pct (%) içindeki
    formasyonlar tek bir 'birleşim bölgesi' sayılır. Döndürür: bölge listesi
    [{price, members:[isim...], directions, count, agree(yön birliği)}]."""
    if len(patterns) < 2:
        return []

    def apex_price(p: PatternMatch) -> float:
        for k in ("D", "top2", "bottom2", "right"):
            if k in p.points:
                return float(p.points[k])
        return float(next(iter(p.points.values()))) if p.points else 0.0

    items = [(apex_price(p), p) for p in patterns]
    items = [(pr, p) for pr, p in items if pr > 0]
    items.sort(key=lambda t: t[0])

    zones: List[Dict] = []
    used = [False] * len(items)
    for i in range(len(items)):
        if used[i]:
            continue
        base_price, base_p = items[i]
        members = [base_p]
        used[i] = True
        for j in range(i + 1, len(items)):
            if used[j]:
                continue
            pr, p = items[j]
            if abs(pr - base_price) / (base_price + 1e-12) * 100 <= tol_pct:
                members.append(p)
                used[j] = True
        if len(members) >= 2:
            prices = [apex_price(m) for m in members]
            dirs = [m.direction.value for m in members]
            longs = dirs.count("LONG")
            shorts = dirs.count("SHORT")
            agree = longs == 0 or shorts == 0   # hepsi aynı yön mü
            zones.append({
                "price": round(sum(prices) / len(prices), 6),
                "members": [m.name for m in members],
                "directions": dirs,
                "count": len(members),
                "agree": agree,
                "bias": "LONG" if longs > shorts else "SHORT" if shorts > longs else "MIXED",
            })
    return zones


def detect_patterns(df: pd.DataFrame) -> Tuple[List[PatternMatch], LayerVote]:
    patterns = detect_harmonics(df) + detect_classic(df)
    confluence = find_confluence(patterns)

    # Oy: en kaliteli/yakın formasyonlar
    score = 0.0
    reasons: List[str] = []
    if patterns:
        # son bara yakın olanlara ağırlık ver
        last_idx = len(df) - 1
        weighted = 0.0
        wsum = 0.0
        for p in patterns:
            recency = max(0.2, 1 - (last_idx - p.pivot_index) / max(last_idx, 1))
            w = p.quality * p.completion * recency
            dir_sign = 1 if p.direction == Direction.LONG else -1
            weighted += dir_sign * w
            wsum += w
            reasons.append(f"{'↑' if dir_sign>0 else '↓'} {p.name} ({p.family}, kalite {p.quality:.2f})")
        score = weighted / (wsum + 1e-12) if wsum else 0.0
        confidence = min(1.0, 0.45 + 0.1 * len(patterns))
        # Birleşim (confluence): aynı noktayı işaret eden formasyonlar güveni artırır
        for z in confluence:
            tag = "✓aynı yön" if z["agree"] else "⚠karışık yön"
            reasons.append(f"🎯 BİRLEŞİM: {z['count']} formasyon ~{z['price']:.4f} ({', '.join(z['members'])}) [{tag}]")
            if z["agree"]:
                confidence = min(1.0, confidence + 0.08 * z["count"])
                # aynı-yön birleşim skoru o yöne kuvvetlendirir
                score += (0.12 * z["count"]) * (1 if z["bias"] == "LONG" else -1)
    else:
        confidence = 0.3
        reasons.append("Belirgin formasyon yok")

    vote = LayerVote(
        name="pattern",
        score=float(np.clip(score, -1, 1)),
        confidence=float(confidence),
        reasons=reasons[:8],
        detail={"pattern_count": len(patterns), "confluence": confluence,
                "confluence_count": len(confluence)},
    )
    return patterns, vote
