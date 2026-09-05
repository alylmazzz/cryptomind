"""
Grafik veri üreticisi (Görselleştirme rolü).

Seçili parite + zaman dilimi için, grafikte ÇİZİLEBİLİR formasyon geometrisini
üretir: her formasyonun nokta(lar)ı (bar indeksi + fiyat), birleştirme çizgisi,
yön oku konumu, sayısal oranları (Fib vb.), hedef ve iptal seviyeleri; ayrıca
SMC bölgeleri (FVG / Order Block) ve sinyal seviyeleri (giriş/stop/TP/tahmin).

Hafiftir: yalnız OHLCV + formasyon/SMC tespiti çalıştırır (derin model YOK), bu
yüzden grafik anında gelir. Giriş/stop/TP gibi karar seviyeleri, varsa /api/analyze
önbelleğinden geçirilir (signal arg). Yoksa grafik yine tüm formasyonları gösterir.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

import numpy as np

from ..analysis.patterns import detect_harmonics, detect_classic, extreme_analysis, find_confluence
from ..analysis.smc import detect_fvg, detect_order_blocks, find_swings
from ..analysis import indicators as ind
from ..core.models import Direction, PatternMatch


_FIB_LBL = {"fib_0": "0.0", "fib_236": "0.236", "fib_382": "0.382", "fib_500": "0.5",
            "fib_618": "0.618", "fib_786": "0.786", "fib_1000": "1.0"}


def _ser(series, start: int):
    """Pandas serisini pencereye dilimle, NaN -> None (JSON güvenli liste)."""
    vals = series.iloc[start:].tolist()
    return [None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), 8)
            for v in vals]


def _overlays(df: pd.DataFrame, start: int) -> Dict:
    """Grafikte çizilecek indikatör serileri (mevcut indicators.py'den türetilir).
    Hepsi mum penceresiyle hizalı dizilerdir; yatay seviyeler tek değerdir."""
    c = df["close"]
    bb_u, bb_m, bb_l = ind.bollinger(c)
    tenkan, kijun, span_a, span_b = ind.ichimoku(df)
    st, st_dir = ind.supertrend(df)
    ov = {
        "bb": {"upper": _ser(bb_u, start), "mid": _ser(bb_m, start), "lower": _ser(bb_l, start)},
        "ichimoku": {"tenkan": _ser(tenkan, start), "kijun": _ser(kijun, start),
                     "span_a": _ser(span_a, start), "span_b": _ser(span_b, start)},
        "vwap": _ser(ind.vwap(df), start),
        "supertrend": {"value": _ser(st, start), "dir": _ser(st_dir, start)},
        "ema": {"20": _ser(ind.ema(c, 20), start), "50": _ser(ind.ema(c, 50), start),
                "200": _ser(ind.ema(c, 200), start)},
        # yatay seviyeler (tek değer): Fibonacci + Pivot
        "fib": [{"label": _FIB_LBL.get(k, k), "price": round(float(v), 6)}
                for k, v in ind.fib_levels(df).items()],
        "pivots": {k: round(float(v), 6) for k, v in ind.pivot_points(df).items()},
    }
    return ov


def _volume_profile(win: pd.DataFrame, bins: int = 24) -> Dict:
    """VPVR: pencere fiyat aralığını dilimleyip hacmi fiyata göre dağıtır.
    POC (Point of Control) = en yüksek hacimli fiyat dilimi."""
    lo = float(win["low"].min())
    hi = float(win["high"].max())
    if hi <= lo:
        return {"bins": [], "poc": None, "max_vol": 0.0}
    edges = np.linspace(lo, hi, bins + 1)
    tp = ((win["high"] + win["low"] + win["close"]) / 3).values
    vol = win["volume"].values
    hist = np.zeros(bins)
    idx = np.clip(np.digitize(tp, edges) - 1, 0, bins - 1)
    for k, v in zip(idx, vol):
        hist[k] += v
    max_vol = float(hist.max()) if hist.size else 0.0
    poc_i = int(hist.argmax()) if hist.size else 0
    out_bins = [{"lo": round(float(edges[i]), 6), "hi": round(float(edges[i + 1]), 6),
                 "vol": round(float(hist[i]), 4)} for i in range(bins)]
    poc = round(float((edges[poc_i] + edges[poc_i + 1]) / 2), 6) if max_vol else None
    return {"bins": out_bins, "poc": poc, "max_vol": max_vol}


def _regime(df: pd.DataFrame) -> Dict:
    """Piyasa rejimi: HMM tabanlı (analysis.regime) + ADX/BB tooltip alanları."""
    try:
        from ..analysis.regime import detect_regime
        r = detect_regime(df)
        adx_v, pdi, mdi = ind.adx(df)
        bb_u, bb_m, bb_l = ind.bollinger(df["close"])
        bbw = float((bb_u.iloc[-1] - bb_l.iloc[-1]) / (bb_m.iloc[-1] + 1e-12))
        rv = float(df["close"].pct_change().rolling(20).std().iloc[-1] or 0)
        up = float(pdi.iloc[-1]) > float(mdi.iloc[-1])
        return {"label": r["label"], "emoji": r["emoji"], "method": r.get("method"),
                "confidence": r.get("confidence"), "multiplier": r.get("multiplier"),
                "adx": round(float(adx_v.iloc[-1]), 1), "bb_width": round(bbw, 3),
                "realized_vol": round(rv, 4), "direction": "up" if up else "down"}
    except Exception:
        return {"label": "—", "emoji": "", "adx": 0, "bb_width": 0, "realized_vol": 0}


def _clamp_idx(i: int, start: int, n: int) -> int:
    """Mutlak df indeksini pencereye taşı (start offset) ve sınırla."""
    return max(0, min(n - 1, i - start))


def _pattern_payload(p: PatternMatch, start: int, n: int) -> Dict:
    """Tek formasyonu çizilebilir yapıya dönüştür."""
    # Noktaları (indeks + fiyat) sıralı çıkar — polyline için indeks sırasıyla
    pts = []
    for name, price in p.points.items():
        idx = p.indices.get(name)
        if idx is None:
            continue
        pts.append({"label": name, "x": _clamp_idx(int(idx), start, n), "y": float(price)})
    pts.sort(key=lambda d: d["x"])

    # Yön oku: tamamlanma noktası (pivot_index) — neckline değil, son nokta
    apex = next((q for q in pts if q["label"] in ("D", "top2", "bottom2", "right")), pts[-1] if pts else None)

    # Etiket: ad + kalite + sayısal oranlar (küçük yazı)
    ratio_txt = " ".join(f"{k}={v}" for k, v in p.ratios.items())
    return {
        "name": p.name,
        "family": p.family,
        "direction": p.direction.value,
        "quality": round(float(p.quality), 2),
        "completion": round(float(p.completion), 2),
        "points": pts,
        "apex": apex,
        "target": float(p.target) if p.target is not None else None,
        "invalidation": float(p.invalidation) if p.invalidation is not None else None,
        "ratios": p.ratios,
        "ratio_text": ratio_txt,
        "note": p.note,
        "label": f"{p.name} · k{p.quality:.2f}",
    }


def build_chart(orch, symbol: str, tf: str, bars: int = 200,
                signal: Optional[Dict] = None) -> Dict:
    df, source = orch.data.fetch_ohlcv_with_meta(symbol, tf)
    if df is None or len(df) < 20:
        return {"symbol": symbol, "timeframe": tf, "error": "yeterli veri yok", "source": source}

    n_full = len(df)
    start = max(0, n_full - bars)
    win = df.iloc[start:]
    n = len(win)

    # --- mumlar ---
    idx = win.index
    candles = [{
        "t": (idx[i].isoformat() if hasattr(idx[i], "isoformat") else str(idx[i])),
        "o": float(win["open"].iloc[i]), "h": float(win["high"].iloc[i]),
        "l": float(win["low"].iloc[i]), "c": float(win["close"].iloc[i]),
        "v": float(win["volume"].iloc[i]),
    } for i in range(n)]

    # --- formasyonlar (tam df üzerinde tespit, sonra pencereye taşı) ---
    patterns = detect_harmonics(df) + detect_classic(df)
    pat_payload = [_pattern_payload(p, start, n) for p in patterns]
    # birleşim (confluence): aynı noktayı işaret eden formasyonları işaretle
    confluence = find_confluence(patterns)
    for pp in pat_payload:
        apex_y = pp["apex"]["y"] if pp.get("apex") else None
        pp["confluence"] = False
        if apex_y is not None:
            for z in confluence:
                if abs(apex_y - z["price"]) / (z["price"] + 1e-12) * 100 <= 0.8:
                    pp["confluence"] = True
                    break

    # --- SMC bölgeleri ---
    fvgs = []
    for f in detect_fvg(df):
        if f["index"] < start:
            continue
        lo, hi = f["gap"]
        fvgs.append({"type": f["type"], "x": _clamp_idx(f["index"], start, n),
                     "from": min(lo, hi), "to": max(lo, hi),
                     "size_pct": round(abs(hi - lo) / (min(lo, hi) + 1e-12) * 100, 2)})
    obs = []
    for o in detect_order_blocks(df):
        if o["index"] < start:
            continue
        lo, hi = o["zone"]
        obs.append({"type": o["type"], "x": _clamp_idx(o["index"], start, n),
                    "from": min(lo, hi), "to": max(lo, hi)})

    swings = [{"x": _clamp_idx(i, start, n), "y": price, "kind": kind}
              for i, price, kind in find_swings(df) if i >= start]

    # --- trend çizgileri + yatay destek/direnç (gerçek grafiğe çizilir) ---
    try:
        from ..analysis.trendlines import build_lines
        trendlines = build_lines(df, start)
    except Exception:
        trendlines = {"trendlines": [], "horizontals": [], "channel": False}

    # --- seviyeler (varsa analiz sinyalinden) ---
    levels = {}
    if signal:
        fc = signal.get("forecast") or {}
        levels = {
            "entry": signal.get("entry"),
            "stop_loss": signal.get("stop_loss"),
            "take_profits": signal.get("take_profits") or [],
            "direction": signal.get("direction"),
            "confidence": signal.get("confidence"),
            "expected_high": fc.get("expected_high"),
            "expected_low": fc.get("expected_low"),
            "band95": fc.get("band95"),
        }

    ex = extreme_analysis(df)
    return {
        "symbol": symbol, "timeframe": tf,
        "source": source,
        "bars": n, "candles": candles,
        "patterns": pat_payload,
        "confluence": confluence,
        "overlays": _overlays(df, start),
        "volume_profile": _volume_profile(win),
        "regime": _regime(df),
        "smc": {"fvg": fvgs, "order_blocks": obs, "swings": swings},
        "trendlines": trendlines,
        "levels": levels,
        "extremes": {
            "all_time_high": ex["all_time_high"], "all_time_low": ex["all_time_low"],
            "range_position": round(ex["range_position"], 3),
            "recent_high_20": ex["recent_high_20"], "recent_low_20": ex["recent_low_20"],
        },
        "counts": {
            "patterns": len(pat_payload), "fvg": len(fvgs),
            "order_blocks": len(obs), "swings": len(swings),
        },
    }
