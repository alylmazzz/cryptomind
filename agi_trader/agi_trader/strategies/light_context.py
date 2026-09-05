"""
HAFİF BAĞLAM — ağır 4h analiz hattına (orkestratör + formasyon + 300 gösterge) sığmayan
pariteler için komitenin yavaş bağlamını UCUZA üretir: rejim (HMM/heuristik), yatay
seviyeler (pivot kümeleme), 20/200-bar maks-min, ATR tabanlı beklenen aralık, 1 sa eğilim.

Neden ayrı katman: 15 paritelik ağır hat sunucuda RSS'i ~1,06 GB'ye çıkarıyor (tavan 1,4 GB);
40 pariteyi ağır hatta almak tavanı aşardı. Hafif bağlam parite başına ~500 bar tutar,
15 dk'da bir yenilenir ve yalnız piyasa yapısı + rejim rollerini besler. Diğer roller
"veri yok" der — komite dürüstçe daha düşük güvenle karar verir.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

CACHE_TTL = 900


def _pivots(h: np.ndarray, l: np.ndarray, win: int = 5):
    hi_idx, lo_idx = [], []
    n = len(h)
    for i in range(win, n - win):
        if h[i] == h[i - win:i + win + 1].max():
            hi_idx.append(i)
        if l[i] == l[i - win:i + win + 1].min():
            lo_idx.append(i)
    return hi_idx, lo_idx


def horizontals(df: pd.DataFrame, price: float, tol_pct: float = 0.5, max_levels: int = 6) -> List[Dict]:
    """Pivot tepe/dipleri %tol içinde kümele; dokunuş sayısıyla sırala."""
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    hi, lo = _pivots(h, l)
    pts = [float(h[i]) for i in hi] + [float(l[i]) for i in lo]
    pts.sort()
    clusters: List[List[float]] = []
    for p in pts:
        if clusters and abs(p - clusters[-1][-1]) / max(1e-12, clusters[-1][-1]) * 100.0 <= tol_pct:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    lv = [{"price": round(float(np.mean(c)), 8), "touches": len(c),
           "kind": "direnç" if float(np.mean(c)) >= price else "destek"} for c in clusters if len(c) >= 2]
    lv.sort(key=lambda x: -x["touches"])
    return lv[:max_levels]


def build_light_context(symbol: str, df4: pd.DataFrame, df1: Optional[pd.DataFrame] = None,
                        events: Optional[List[Dict]] = None, now: Optional[float] = None) -> Dict:
    now = time.time() if now is None else now
    c = df4["close"].astype(float)
    price = float(c.iloc[-1])
    h = df4["high"].astype(float); l = df4["low"].astype(float)
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1]) if len(tr) >= 14 else float("nan")
    try:
        from ..analysis.regime import detect_regime
        regime = detect_regime(df4)
    except Exception:
        regime = {"label": "RANGE / YATAY", "multiplier": 0.6, "confidence": 0.3, "method": "yok"}
    win = df4.tail(200)
    lo_all, hi_all = float(win["low"].min()), float(win["high"].max())
    rp = (price - lo_all) / max(1e-12, hi_all - lo_all)
    bias_1h = None
    swing = None
    if df1 is not None and len(df1) >= 60:
        c1 = df1["close"].astype(float)
        e20s, e50s = c1.ewm(span=20, adjust=False).mean(), c1.ewm(span=50, adjust=False).mean()
        e20, e50 = float(e20s.iloc[-1]), float(e50s.iloc[-1])
        bias_1h = "LONG" if e20 > e50 else "SHORT"
        e20_4, e50_4 = float(c.ewm(span=20, adjust=False).mean().iloc[-1]), float(c.ewm(span=50, adjust=False).mean().iloc[-1])
        close_1h = float(c1.iloc[-1]); prev_1h = float(c1.iloc[-2])
        # 1 sa geri çekilme: kapanış EMA20(1h) çevresinde (±%0,3) ve EMA50 üstünde, son 1 sa barı yeşil
        pullback_1h = bool(abs(close_1h / e20 - 1.0) <= 0.003 and close_1h >= e50 and close_1h > prev_1h)
        swing = {"ema20_1h": e20, "ema50_1h": e50, "close_1h": close_1h, "pullback_1h": pullback_1h,
                 "trend_4h_up": bool(e20_4 > e50_4 and price > e50_4), "atr_pct_4h": (round(atr / price * 100.0, 4) if math.isfinite(atr) else None),
                 "swing_low_4h": float(df4["low"].tail(10).min()), "ret_7d_pct": (round(float(price / float(c.iloc[-43]) - 1.0) * 100.0, 3) if len(c) > 43 else None),
                 "regime": (regime or {}).get("label")}
    return {
        "symbol": symbol, "tier": "light", "tf": "4h",
        "signal": None, "patterns": None, "harmonics": None, "indicators": None, "candles": None,
        "mover_pick": None, "corr": None, "social": None,
        "chart": {
            "levels": {"expected_high": (price + atr) if math.isfinite(atr) else None,
                       "expected_low": (price - atr) if math.isfinite(atr) else None},
            "trendlines": {"horizontals": horizontals(df4, price), "channel": False},
            "extremes": {"range_position": round(float(rp), 3),
                         "recent_high_20": float(df4["high"].tail(20).max()),
                         "recent_low_20": float(df4["low"].tail(20).min()),
                         "all_time_high": hi_all, "all_time_low": lo_all},
            "regime": regime, "smc": {"swings": []},
        },
        "light": {"atr_pct_4h": round(atr / price * 100.0, 4) if math.isfinite(atr) else None,
                  "bias_1h": bias_1h, "bars_4h": int(len(df4))},
        "swing": swing,
        "events": events, "age_sec": 0.0, "built_ts": now,
    }


class LightContextCache:
    """Parite başına hafif bağlam; TTL dolunca broker'dan 4h/1h çekip yeniden kurar."""

    def __init__(self, broker, events_fn=None, ttl: int = CACHE_TTL):
        self.broker = broker
        self.events_fn = events_fn
        self.ttl = ttl
        self._c: Dict[str, Dict] = {}

    def get(self, symbol: str, now: Optional[float] = None) -> Optional[Dict]:
        now = time.time() if now is None else now
        hit = self._c.get(symbol)
        if hit and now - hit["built_ts"] < self.ttl:
            out = dict(hit)
            out["age_sec"] = now - hit["built_ts"]
            return out
        try:
            df4 = self.broker.fetch_ohlcv(symbol, "4h", limit=300)
            df1 = self.broker.fetch_ohlcv(symbol, "1h", limit=120)
        except Exception:
            return hit
        ev = None
        if self.events_fn:
            try:
                ev = self.events_fn()
            except Exception:
                ev = None
        ctx = build_light_context(symbol, df4, df1, ev, now)
        self._c[symbol] = ctx
        return dict(ctx)
