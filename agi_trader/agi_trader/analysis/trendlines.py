"""
Trend Çizgisi + Kanal + Yatay Destek/Direnç katmanı.

"Gerçek grafik üzerine çizgiler çeksin" + "çekilen çizgiler ortak kararı
göstersin" isteğinin karşılığı. Pivot (swing) noktalarından:
  - DİRENÇ trend çizgisi (son tepe noktalarını birleştirir, eğimli)
  - DESTEK trend çizgisi (son dip noktalarını birleştirir, eğimli)
  - YATAY destek/direnç (kümelenen pivot fiyatları)
  - KANAL (paralel destek+direnç)

Ürettiği LayerVote, fiyatın bu çizgilere göre konumundan (kırılım / sıçrama /
reddedilme) yönlü bir skor çıkarır ve karar motorunun konsensüsüne katılır.
Çizilebilir çizgi geometrisi `build_lines()` ile chart.py'ye aktarılır.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.models import LayerVote
from .patterns import find_pivots
from .indicators import atr as atr_fn


def _fit_line(idxs: List[int], prices: List[float]) -> Optional[Tuple[float, float]]:
    """En küçük kareler ile doğru uydur → (eğim, kesişim). y = eğim*x + kesişim."""
    if len(idxs) < 2:
        return None
    x = np.asarray(idxs, dtype=float)
    y = np.asarray(prices, dtype=float)
    try:
        slope, intercept = np.polyfit(x, y, 1)
        return float(slope), float(intercept)
    except Exception:
        return None


def _line_at(line: Tuple[float, float], x: int) -> float:
    return line[0] * x + line[1]


def _touches(line: Tuple[float, float], idxs: List[int], prices: List[float],
             tol: float) -> int:
    """Çizgiye 'tol' (oransal) mesafede kaç pivot değiyor."""
    n = 0
    for i, p in zip(idxs, prices):
        proj = _line_at(line, i)
        if proj > 0 and abs(p - proj) / proj <= tol:
            n += 1
    return n


def detect_trendlines(df: pd.DataFrame, left: int = 3, right: int = 3,
                      max_pivots: int = 5) -> Dict:
    """Son swing noktalarından destek/direnç trend çizgileri + yatay seviyeler.
    Look-ahead yok: yalnızca df içindeki (geçmiş) barlar kullanılır."""
    n = len(df)
    if n < 30:
        return {"support": None, "resistance": None, "horizontals": [],
                "channel": False, "pivot_high_idx": [], "pivot_low_idx": []}

    highs_idx, lows_idx = find_pivots(df, left=left, right=right)
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    # Son birkaç pivotu al (en güncel yapı)
    h_idx = highs_idx[-max_pivots:] if highs_idx else []
    l_idx = lows_idx[-max_pivots:] if lows_idx else []
    res_line = _fit_line(h_idx, [high[i] for i in h_idx]) if len(h_idx) >= 2 else None
    sup_line = _fit_line(l_idx, [low[i] for i in l_idx]) if len(l_idx) >= 2 else None

    tol = 0.004  # %0.4 dokunma toleransı
    res_touch = _touches(res_line, h_idx, [high[i] for i in h_idx], tol) if res_line else 0
    sup_touch = _touches(sup_line, l_idx, [low[i] for i in l_idx], tol) if sup_line else 0

    # Yatay destek/direnç: tüm pivot fiyatlarını kümele (yakın fiyatları birleştir)
    piv_prices = [high[i] for i in highs_idx] + [low[i] for i in lows_idx]
    horizontals = _cluster_levels(piv_prices, close[-1], tol=0.006)

    # Kanal: destek ve direnç eğimleri benzerse (paralel) → kanal
    channel = False
    if res_line and sup_line:
        s1, s2 = res_line[0], sup_line[0]
        denom = (abs(s1) + abs(s2)) / 2 + 1e-9
        channel = abs(s1 - s2) / denom < 0.35

    return {
        "support": sup_line, "resistance": res_line,
        "support_touches": sup_touch, "resistance_touches": res_touch,
        "horizontals": horizontals, "channel": bool(channel),
        "pivot_high_idx": list(h_idx), "pivot_low_idx": list(l_idx),
    }


def _cluster_levels(prices: List[float], ref: float, tol: float = 0.006,
                    max_levels: int = 6) -> List[Dict]:
    """Yakın pivot fiyatlarını kümele → güçlü yatay seviyeler (dokunma sayısıyla)."""
    if not prices:
        return []
    pts = sorted(prices)
    clusters: List[List[float]] = [[pts[0]]]
    for p in pts[1:]:
        if abs(p - clusters[-1][-1]) / (clusters[-1][-1] + 1e-9) <= tol:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    levels = []
    for c in clusters:
        price = float(np.mean(c))
        levels.append({"price": round(price, 8), "touches": len(c),
                       "kind": "direnç" if price >= ref else "destek"})
    # En çok dokunulan + fiyata yakın olanlar öne
    levels.sort(key=lambda L: (-L["touches"], abs(L["price"] - ref) / (ref + 1e-9)))
    return levels[:max_levels]


def trendline_vote(df: pd.DataFrame) -> LayerVote:
    """Fiyatın trend çizgilerine göre konumundan yönlü oy üretir.
    Kırılım (breakout) > sıçrama (bounce) > reddedilme (rejection) hiyerarşisi."""
    tl = detect_trendlines(df)
    n = len(df)
    if n < 30:
        return LayerVote(name="trendline", score=0.0, confidence=0.0,
                         reasons=["Trend çizgisi için yetersiz veri"])
    price = float(df["close"].iloc[-1])
    last_x = n - 1
    a = float(atr_fn(df, 14).iloc[-1]) if n >= 15 else price * 0.01
    buf = max(0.0015, (a / (price + 1e-12)) * 0.5)  # kırılım tamponu

    res = tl["resistance"]
    sup = tl["support"]
    score = 0.0
    conf = 0.0
    reasons: List[str] = []

    res_at = _line_at(res, last_x) if res else None
    sup_at = _line_at(sup, last_x) if sup else None

    # 1) Kırılım / kırılış
    if res_at and price > res_at * (1 + buf):
        score += 0.6
        conf = max(conf, 0.7)
        reasons.append(f"Direnç trend çizgisi YUKARI kırıldı ({res_at:.4f}) — boğa")
    elif sup_at and price < sup_at * (1 - buf):
        score -= 0.6
        conf = max(conf, 0.7)
        reasons.append(f"Destek trend çizgisi AŞAĞI kırıldı ({sup_at:.4f}) — ayı")

    # 2) Sıçrama / reddedilme (çizgiye yakın + eğim yönü)
    if sup_at and abs(price - sup_at) / price <= buf * 1.5:
        slope_up = sup[0] >= 0
        score += 0.45 if slope_up else 0.25
        conf = max(conf, 0.6)
        reasons.append(f"Fiyat destek çizgisinde ({sup_at:.4f}) — sıçrama beklenir")
    if res_at and abs(price - res_at) / price <= buf * 1.5:
        slope_dn = res[0] <= 0
        score -= 0.45 if slope_dn else 0.25
        conf = max(conf, 0.6)
        reasons.append(f"Fiyat direnç çizgisinde ({res_at:.4f}) — reddedilme beklenir")

    # 3) Kanal içi konum (yön teyidi)
    if tl["channel"] and res_at and sup_at and res_at > sup_at:
        pos = (price - sup_at) / (res_at - sup_at + 1e-12)  # 0=dip,1=tepe
        if pos < 0.25:
            score += 0.2; reasons.append("Kanal dibinde — alım bölgesi")
        elif pos > 0.75:
            score -= 0.2; reasons.append("Kanal tepesinde — satım bölgesi")
        conf = max(conf, 0.5)

    # Dokunma sayısı güveni artırır (çok dokunulan çizgi = güçlü)
    touch_bonus = min(0.2, 0.05 * (tl["support_touches"] + tl["resistance_touches"]))
    conf = float(np.clip(conf + touch_bonus, 0.0, 0.9))
    score = float(np.clip(score, -1.0, 1.0))
    if not reasons:
        reasons.append("Fiyat trend çizgilerinin ortasında — net sinyal yok")

    return LayerVote(name="trendline", score=score, confidence=conf, reasons=reasons,
                     detail={"support_touches": tl["support_touches"],
                             "resistance_touches": tl["resistance_touches"],
                             "channel": tl["channel"],
                             "horizontals": tl["horizontals"][:3]})


def build_lines(df: pd.DataFrame, start: int = 0) -> Dict:
    """Chart.py için çizilebilir geometri: eğimli trend çizgileri (2 uç nokta) +
    yatay seviyeler. start = grafik penceresinin df içindeki başlangıç indeksi."""
    tl = detect_trendlines(df)
    n = len(df)
    out: Dict[str, object] = {"trendlines": [], "horizontals": tl["horizontals"]}
    last_x = n - 1

    def seg(line, idxs, kind):
        if not line or not idxs:
            return None
        x0 = max(min(idxs), start)
        x1 = last_x
        return {"kind": kind,
                "x0": int(x0 - start), "y0": round(_line_at(line, x0), 8),
                "x1": int(x1 - start), "y1": round(_line_at(line, x1), 8)}

    for line, idxs, kind in [(tl["resistance"], tl["pivot_high_idx"], "direnç"),
                             (tl["support"], tl["pivot_low_idx"], "destek")]:
        s = seg(line, idxs, kind)
        if s:
            out["trendlines"].append(s)
    out["channel"] = tl["channel"]
    return out
