"""
STRATEJİ AİLESİ (sleeve'ler) + REJİM SEÇİCİ + EV YARIŞMASI.

Her sleeve farklı bir kenar hipotezidir; hepsi aynı anda oy kullanmaz. Rejim seçici
o piyasada anlamlı olanları açar, kalanı sessiz kalır (zıt sistemler birbirini
sıfırlamasın). Ateşleyen sleeve'ler arasında son karar "en çok oy" değil, en yüksek
DOĞRULANMIŞ net EV'dir (committee.compete).

Mevcut tetikleyiciler (committee.triggers): dip, dip_moderate, pullback, breakout,
momentum, catalyst. Bu modül ekler:
  squeeze_breakout   Bollinger bant genişliği yüzdelik ≤ 20 sonrası hacimli kırılım
  sweep_reversal     önceki swing low altına süpürme + hızlı geri alım (fitil + hacim)
  range_edge         net yatay range; yalnız alt kenarda (ortada işlem yok)
  vwap_reversion     yataydaysa VWAP'tan ≥ 1 ATR uzaklaşma sonrası dönüş barı
  vwap_continuation  trendde VWAP'a geri çekilme + devam barı
  rs_momentum        evren içinde göreli güç üst %20 + kırılım/kesişim
  news_overreaction  ayı haberi hızla fiyatlandı, devam başarısız → dönüş (küçük boyut)

Her aday: {kind, direction, size, exit_mode, note, target_hint?, stop_hint?}.
Lifecycle: bütün sleeve'ler PAPER aşamasındadır (lifecycle.py); canlı için kanıt gerekir.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

EXIT_FIXED, EXIT_PARTIAL_RUN, EXIT_DYNAMIC_PEAK = "FIXED_TARGET", "PARTIAL_AND_RUN", "DYNAMIC_PEAK"

from . import sleeves_video as SV

SLEEVE_TR = {
    **SV.SLEEVE_TR_VIDEO,
    "squeeze_breakout": "sıkışma-kırılım", "sweep_reversal": "likidite süpürme dönüşü",
    "range_edge": "range kenarı", "vwap_reversion": "VWAP dönüşü", "vwap_continuation": "VWAP devamı",
    "rs_momentum": "göreli güç momentumu", "news_overreaction": "haber aşırı-tepki dönüşü",
    "dip": "dip", "dip_moderate": "ılımlı dip", "pullback": "geri çekilme", "breakout": "kırılım",
    "momentum": "momentum", "catalyst": "haber katalizörü", "top": "tepe",
    "adaptive_trend": "uyarlanır trend (çok-ufuk + geri çekilme)", "donchian_breakout": "Donchian 55 kırılımı",
    "bos_retest": "yapı kırılımı + yeniden test", "failed_breakdown": "başarısız kırılma dönüşü (LONG)",
    "failed_breakout": "başarısız kırılım dönüşü (SHORT)", "obi_momentum": "defter dengesi / mikro-fiyat momentumu",
    "swing_trend": "swing trend (4h trend + 1h geri çekilme, 1–3 gün)",
}
SLEEVE_EXIT_MODE = {
    **SV.SLEEVE_EXIT_MODE_VIDEO,
    "dip": EXIT_PARTIAL_RUN, "dip_moderate": EXIT_FIXED, "pullback": EXIT_PARTIAL_RUN,
    "breakout": EXIT_DYNAMIC_PEAK, "momentum": EXIT_DYNAMIC_PEAK, "catalyst": EXIT_DYNAMIC_PEAK,
    "squeeze_breakout": EXIT_DYNAMIC_PEAK, "sweep_reversal": EXIT_PARTIAL_RUN, "range_edge": EXIT_FIXED,
    "vwap_reversion": EXIT_FIXED, "vwap_continuation": EXIT_PARTIAL_RUN, "rs_momentum": EXIT_DYNAMIC_PEAK,
    "news_overreaction": EXIT_FIXED, "top": EXIT_FIXED,
    "adaptive_trend": EXIT_DYNAMIC_PEAK, "donchian_breakout": EXIT_DYNAMIC_PEAK, "bos_retest": EXIT_PARTIAL_RUN,
    "failed_breakdown": EXIT_PARTIAL_RUN, "failed_breakout": EXIT_FIXED, "obi_momentum": EXIT_FIXED,
    "swing_trend": EXIT_DYNAMIC_PEAK,
}
# sleeve başına zaman-stop (dk) — scalp ile kırılım aynı süreyi kullanmaz
SLEEVE_TIME_STOP_MIN = {
    **SV.SLEEVE_TIME_STOP_MIN_VIDEO,
    "dip": 90, "dip_moderate": 90, "pullback": 120, "breakout": 180, "momentum": 120,      # kazanan medyanı 45 dk → ufuk 90
    "catalyst": 180, "squeeze_breakout": 180, "sweep_reversal": 90, "range_edge": 120,
    "vwap_reversion": 60, "vwap_continuation": 120, "rs_momentum": 180, "news_overreaction": 60,
    "adaptive_trend": 240, "donchian_breakout": 240, "bos_retest": 180, "failed_breakdown": 90,
    "failed_breakout": 90, "obi_momentum": 45, "swing_trend": 4320,     # swing: 3 gün
}

# Aciliyet: maker kaç bar bekleyebilir? 0 = anında taker (kırılım/katalizör kaçar), 2 = sabırlı (dip/geri çekilme)
SLEEVE_URGENCY = {
    **SV.SLEEVE_URGENCY_VIDEO,
    "catalyst": 0, "breakout": 0, "squeeze_breakout": 0, "donchian_breakout": 0, "momentum": 0, "rs_momentum": 0,
    "obi_momentum": 0, "sweep_reversal": 1, "failed_breakdown": 1, "bos_retest": 1, "vwap_continuation": 1,
    "adaptive_trend": 1, "dip": 2, "dip_moderate": 2, "pullback": 2, "range_edge": 2, "vwap_reversion": 2,
    "news_overreaction": 2, "failed_breakout": 1, "top": 2, "swing_trend": 2,
}


def cvd_from_trades(trades: list, now_ms: Optional[float] = None, window_sec: int = 300) -> Dict:
    """Son işlemlerden kümülatif hacim deltası: (alıcı taker − satıcı taker) / toplam, USD ağırlıklı.
    'burst' = son 60 sn'deki taker hacmi, pencere ortalamasının kaç katı."""
    if not trades:
        return {"cvd_ratio": None, "cvd_n": 0}
    now_ms = now_ms or max(float(t.get("timestamp") or 0) for t in trades)
    lo = now_ms - window_sec * 1000
    buy = sell = 0.0; n = 0; last60 = 0.0
    for t in trades:
        ts = float(t.get("timestamp") or 0)
        if ts < lo:
            continue
        usd = float(t.get("cost") or (float(t.get("price") or 0) * float(t.get("amount") or 0)))
        if usd <= 0:
            continue
        n += 1
        if str(t.get("side", "")).lower() == "buy":
            buy += usd
        else:
            sell += usd
        if ts >= now_ms - 60_000:
            last60 += usd
    tot = buy + sell
    if tot <= 0 or n < 5:
        return {"cvd_ratio": None, "cvd_n": n}
    per_min = tot / max(1.0, window_sec / 60.0)
    return {"cvd_ratio": round((buy - sell) / tot, 3), "cvd_n": n, "cvd_buy_usd": round(buy, 2), "cvd_sell_usd": round(sell, 2),
            "cvd_burst": round(last60 / per_min, 2) if per_min > 0 else None}


# Rejim → açık sleeve'ler (zıt sistemler aynı anda oy kullanmasın)
REGIME_SLEEVES = {
    "TREND YUKARI": ["pullback", "breakout", "momentum", "squeeze_breakout", "vwap_continuation",
                     "rs_momentum", "catalyst", "adaptive_trend", "donchian_breakout", "bos_retest", "obi_momentum",
                     "dip", "dip_moderate", "sweep_reversal", "swing_trend"]       # 4h yukarı trendde 1m derin dip = trend-içi alım (video kurulumu)
                    + SV.REGIME_SLEEVES_VIDEO["TREND YUKARI"],
    "RANGE / YATAY": ["dip", "dip_moderate", "range_edge", "vwap_reversion", "sweep_reversal",
                      "catalyst", "news_overreaction", "failed_breakdown", "failed_breakout", "obi_momentum"]
                     + SV.REGIME_SLEEVES_VIDEO["RANGE / YATAY"],
    "VOLATİL": ["dip", "sweep_reversal", "catalyst", "failed_breakdown"] + SV.REGIME_SLEEVES_VIDEO["VOLATİL"],
    "TREND AŞAĞI": ["sweep_reversal", "catalyst", "dip", "failed_breakdown", "failed_breakout"]
                   + SV.REGIME_SLEEVES_VIDEO["TREND AŞAĞI"],
}


ALL_SLEEVES = sorted({k for v in REGIME_SLEEVES.values() for k in v} | {"failed_breakout", "news_overreaction"}
                     | set(SV.ALL_VIDEO_SLEEVES))


def allowed_sleeves(regime_label: Optional[str]) -> List[str]:
    return list(REGIME_SLEEVES.get(regime_label or "RANGE / YATAY", REGIME_SLEEVES["RANGE / YATAY"]))


# ---------------------------------------------------------------- ek hızlı özellikler
def _adx(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> Optional[float]:
    if len(c) < 2 * n + 2:
        return None
    up = h.diff(); dn = -l.diff()
    plus = ((up > dn) & (up > 0)) * up
    minus = ((dn > up) & (dn > 0)) * dn
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / n, adjust=False).mean()
    pdi = 100.0 * plus.ewm(alpha=1.0 / n, adjust=False).mean() / atr.replace(0, np.nan)
    mdi = 100.0 * minus.ewm(alpha=1.0 / n, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = dx.ewm(alpha=1.0 / n, adjust=False).mean()
    v = float(adx.iloc[-1])
    return v if math.isfinite(v) else None


def extra_features(df: pd.DataFrame, f: Dict, rs_rank: Optional[float] = None,
                   book: Optional[Dict] = None, slow: Optional[Dict] = None) -> Dict:
    """committee.fast_features çıktısını genişletir (aynı df, ek hesap yok: tek geçiş).
    book → OBI/mikro-fiyat; slow → 4h rejim etiketi (çok-ufuk trend skoru)."""
    c = df["close"].astype(float)
    h = df["high"].astype(float) if "high" in df else c
    l = df["low"].astype(float) if "low" in df else c
    v = df["volume"].astype(float) if "volume" in df else None
    price = float(c.iloc[-1])
    n = len(c)
    # Bollinger bant genişliği (%) ve 120-bar yüzdeliği
    sma = c.rolling(20).mean(); sd = c.rolling(20).std(ddof=0)
    bw = (4.0 * sd / sma * 100.0)
    bw_now = float(bw.iloc[-1]) if math.isfinite(float(bw.iloc[-1])) else None
    hist = bw.dropna().tail(120)
    bw_pctile = float((hist < bw_now).mean() * 100.0) if (bw_now is not None and len(hist) >= 40) else None
    # önceki 5 barın bant genişliği yüzdeliği (sıkışma "öncesi" olmalı)
    bw_prev = bw.dropna().tail(126).head(120)
    bw_prev_pctile = (float((bw_prev < float(bw.iloc[-6])).mean() * 100.0)
                      if len(bw_prev) >= 40 and n > 6 and math.isfinite(float(bw.iloc[-6])) else None)
    # VWAP (mevcut tampon ~ son 150 bar)
    if v is not None and float(v.sum()) > 0:
        tp = (h + l + c) / 3.0
        vwap = float((tp * v).sum() / v.sum())
    else:
        vwap = float(c.mean())
    # önceki swing low/high (son 3 bar hariç, 20 bar)
    prior = slice(max(0, n - 23), n - 3)
    prior_low = float(l.iloc[prior].min()) if n > 25 else float("nan")
    prior_high = float(h.iloc[prior].max()) if n > 25 else float("nan")
    last_low, last_high, last_close = float(l.iloc[-1]), float(h.iloc[-1]), price
    swept_low = bool(math.isfinite(prior_low) and last_low < prior_low and last_close > prior_low)
    swept_high = bool(math.isfinite(prior_high) and last_high > prior_high and last_close < prior_high)
    wick_ratio = None
    rng = last_high - last_low
    if rng > 0:
        o = float(df["open"].iloc[-1]) if "open" in df else float(c.iloc[-2])
        body_low = min(o, last_close)
        wick_ratio = (body_low - last_low) / rng           # alt fitil payı
    # range tespiti (60 bar): genişlik ≥ 3 ATR, EMA20 eğimi düz
    atr_pct = f.get("atr_pct") or 0.0
    hi60, lo60 = float(h.tail(60).max()), float(l.tail(60).min())
    width_pct = (hi60 - lo60) / price * 100.0
    slope = abs(float(f.get("ema_slope_pct") or 0.0))
    range_ok = bool(width_pct >= 3.0 * atr_pct and slope < 0.05 and not f.get("breakout_up") and not f.get("breakdown"))
    range_pos = (price - lo60) / max(1e-12, hi60 - lo60)
    # 4 saatlik hareket (240 bar yoksa mevcut tamponun tümü)
    back = min(240, n - 1)
    move_4h_pct = float(price / float(c.iloc[-1 - back]) - 1.0) * 100.0 if back > 10 else 0.0
    # --- çok-ufuk trend skoru (A1/A8/A15), Donchian 55 (A7/C26), yapı kırılımı + retest (B22/C39),
    #     başarısız kırılım/kırılma (D41/D42), defter dengesi / mikro-fiyat (I103/I106) ---
    adx = _adx(h, l, c) if n >= 40 else None
    reg_label = ((slow or {}).get("chart") or {}).get("regime", {}).get("label") if slow else None
    comps = [bool(f.get("trend_up")), float(f.get("ema_slope_pct") or 0.0) > 0.0, price > vwap]
    if adx is not None:
        comps.append(adx >= 20.0)
    if rs_rank is not None:
        comps.append(rs_rank >= 0.6)
    if reg_label:
        comps.append(reg_label == "TREND YUKARI")
    trend_score = sum(1.0 for x in comps if x) / len(comps)
    extended = bool((f.get("dist_ema_pct") or 0.0) > 2.0 * atr_pct) if atr_pct else False
    pullback_atr = (-(f.get("dist_ema_pct") or 0.0) / atr_pct) if atr_pct else 0.0
    hi55_prev = float(h.iloc[-56:-1].max()) if n > 56 else float("nan")
    donchian_break = bool(math.isfinite(hi55_prev) and price > hi55_prev)
    older_high = float(h.iloc[max(0, n - 43):n - 13].max()) if n > 45 else float("nan")
    broke = bool(math.isfinite(older_high) and float(c.iloc[n - 13:n - 3].max()) > older_high) if n > 45 else False
    bos_retest_up = bool(broke and abs(price - older_high) / price * 100.0 <= 0.3 * atr_pct and price >= older_high * 0.999)
    lo20_ref = float(l.iloc[max(0, n - 27):n - 7].min()) if n > 30 else float("nan")
    hi20_ref = float(h.iloc[max(0, n - 27):n - 7].max()) if n > 30 else float("nan")
    fb_low = float(l.iloc[n - 7:n - 1].min()) if n > 30 else float("nan")
    fb_high = float(h.iloc[n - 7:n - 1].max()) if n > 30 else float("nan")
    broke_down = bool(math.isfinite(lo20_ref) and float(c.iloc[n - 7:n - 1].min()) < lo20_ref) if n > 30 else False
    broke_up = bool(math.isfinite(hi20_ref) and float(c.iloc[n - 7:n - 1].max()) > hi20_ref) if n > 30 else False
    vr_ = f.get("vol_ratio")
    failed_breakdown = bool(broke_down and price > lo20_ref and bool(f.get("bar_up")))
    failed_breakout = bool(broke_up and price < hi20_ref and not f.get("bar_up") and (vr_ is None or vr_ < 1.2))
    obi = microprice_dev_bps = None
    b = book or {}
    bd, ad = float(b.get("bid_depth_usd") or 0.0), float(b.get("ask_depth_usd") or 0.0)
    if bd + ad > 0:
        obi = bd / (bd + ad)
        if b.get("bid") and b.get("ask"):
            bid, ask = float(b["bid"]), float(b["ask"])
            mid = (bid + ask) / 2.0
            micro = (bid * ad + ask * bd) / (bd + ad)
            microprice_dev_bps = (micro - mid) / mid * 1e4 if mid > 0 else None
    f.update({
        "adx": adx, "trend_score": round(trend_score, 3), "extended": extended, "pullback_atr": round(pullback_atr, 3),
        "donchian_hi55": hi55_prev, "donchian_break": donchian_break, "bos_level": older_high, "bos_retest_up": bos_retest_up,
        "failed_breakdown": failed_breakdown, "failed_breakout": failed_breakout, "fb_low": fb_low, "fb_high": fb_high,
        "obi": obi, "microprice_dev_bps": microprice_dev_bps, "spread_bps": b.get("spread_bps"),
        "cvd_ratio": b.get("cvd_ratio"), "cvd_burst": b.get("cvd_burst"),
        "swing": (slow or {}).get("swing"),
        "regime_4h": reg_label,
        "bb_width_pct": bw_now, "bb_width_pctile": bw_pctile, "bb_prev_pctile": bw_prev_pctile,
        "vwap": vwap, "dist_vwap_pct": (price / vwap - 1.0) * 100.0 if vwap else None,
        "prior_swing_low": prior_low, "prior_swing_high": prior_high,
        "swept_low": swept_low, "swept_high": swept_high, "lower_wick_ratio": wick_ratio,
        "range_ok": range_ok, "range_low": lo60, "range_high": hi60, "range_pos": range_pos,
        "range_width_pct": width_pct, "move_4h_pct": move_4h_pct, "rs_rank": rs_rank,
    })
    f = SV.video_features(df, f)          # video kaynaklı kurulumların özellikleri (aynı df, ek çekim yok)
    return f


def relative_strength_ranks(cross: Dict[str, Dict]) -> Dict[str, float]:
    """Evren içinde 1 sa getiri × hacim oranıyla sıralama → 0..1 (1 = en güçlü)."""
    rows = [(s, (f.get("ret_1h_pct") or 0.0) * (1.0 + 0.2 * min(3.0, (f.get("vol_ratio") or 1.0))))
            for s, f in cross.items() if f.get("ret_1h_pct") is not None]
    if len(rows) < 3:
        return {}
    rows.sort(key=lambda x: x[1])
    n = len(rows)
    return {s: (i / (n - 1)) for i, (s, _) in enumerate(rows)}


# ---------------------------------------------------------------- sleeve tetikleyicileri
def fire_sleeves(f: Dict, allowed: List[str], news: Optional[Dict], p, allow_short: bool = False,
                 now_ts: Optional[float] = None) -> List[Dict]:
    """Ek sleeve'ler (committee.triggers'ın yanında). Yalnız `allowed` listesindekiler."""
    out: List[Dict] = []
    if not f.get("ok"):
        return out
    out += SV.fire_video_sleeves(f, allowed, p, allow_short, now_ts)     # video kaynaklı kurulumlar
    up = bool(f.get("bar_up"))
    rsi = float(f.get("rsi") or 50.0)
    atr = float(f.get("atr_pct") or 0.3)
    vr = f.get("vol_ratio")

    if "squeeze_breakout" in allowed and f.get("breakout_up") and up and \
            f.get("bb_prev_pctile") is not None and f["bb_prev_pctile"] <= 20 and vr is not None and vr >= 1.3:
        out.append({"kind": "squeeze_breakout", "direction": "LONG", "size": 0.8,
                    "exit_mode": EXIT_DYNAMIC_PEAK,
                    "note": f"sıkışma (bant genişliği %{f['bb_prev_pctile']:.0f} yüzdelik) sonrası hacimli kırılım ×{vr:.1f}"})
    if "sweep_reversal" in allowed and f.get("swept_low") and up and \
            (f.get("lower_wick_ratio") or 0.0) >= 0.4 and (vr is None or vr >= 1.2):
        out.append({"kind": "sweep_reversal", "direction": "LONG", "size": 0.8, "exit_mode": EXIT_PARTIAL_RUN,
                    "stop_hint": float(f["prior_swing_low"]) * (1.0 - 0.3 * atr / 100.0),
                    "note": f"swing low ({f['prior_swing_low']:.6g}) süpürüldü, geri alındı; alt fitil %{(f.get('lower_wick_ratio') or 0)*100:.0f}"})
    if "range_edge" in allowed and f.get("range_ok") and f.get("range_pos") is not None and \
            f["range_pos"] <= 0.15 and up and rsi < 50:
        mid = (f["range_low"] + f["range_high"]) / 2.0
        out.append({"kind": "range_edge", "direction": "LONG", "size": 0.7, "exit_mode": EXIT_FIXED,
                    "target_hint": mid, "stop_hint": f["range_low"] * (1.0 - 0.5 * atr / 100.0),
                    "note": f"range alt kenarı (konum %{f['range_pos']*100:.0f}, genişlik %{f['range_width_pct']:.2f})"})
    dv = f.get("dist_vwap_pct")
    if "vwap_reversion" in allowed and dv is not None and dv <= -1.0 * atr and up and rsi < 45:
        out.append({"kind": "vwap_reversion", "direction": "LONG", "size": 0.7, "exit_mode": EXIT_FIXED,
                    "target_hint": f["vwap"],
                    "note": f"VWAP'ın %{abs(dv):.2f} altında (≥1 ATR) + dönüş barı"})
    if "vwap_continuation" in allowed and dv is not None and f.get("trend_up") and \
            -0.3 * atr <= dv <= 0.3 * atr and up and 45 <= rsi < 65:
        out.append({"kind": "vwap_continuation", "direction": "LONG", "size": 0.9, "exit_mode": EXIT_PARTIAL_RUN,
                    "note": f"trendde VWAP'a geri çekilme (uzaklık %{dv:+.2f}) + devam barı"})
    rs = f.get("rs_rank")
    if "rs_momentum" in allowed and rs is not None and rs >= 0.8 and up and \
            (f.get("ema_cross_up") or f.get("breakout_up")) and rsi < 75:
        out.append({"kind": "rs_momentum", "direction": "LONG", "size": 0.7, "exit_mode": EXIT_DYNAMIC_PEAK,
                    "note": f"göreli güç sıralaması %{rs*100:.0f} + kırılım/kesişim"})
    ts_ = float(f.get("trend_score") or 0.0)
    pb = float(f.get("pullback_atr") or 0.0)
    obi = f.get("obi")
    if "adaptive_trend" in allowed and ts_ >= 0.7 and not f.get("extended") and 0.5 <= pb <= 1.5 and up and \
            (vr is None or vr >= 0.9) and (obi is None or obi >= 0.45) and rsi < 70 and \
            (f.get("cvd_ratio") is None or f["cvd_ratio"] > -0.2):
        psl = f.get("prior_swing_low")
        out.append({"kind": "adaptive_trend", "direction": "LONG", "size": 1.0, "exit_mode": EXIT_DYNAMIC_PEAK,
                    "stop_hint": (float(psl) * (1.0 - 0.3 * atr / 100.0) if psl is not None and math.isfinite(float(psl)) else None),
                    "note": f"trend skoru {ts_:.2f} (ADX {f.get('adx') if f.get('adx') is None else round(f['adx'])}) · EMA'ya {pb:.1f} ATR geri çekilme · devam barı"})
    if "donchian_breakout" in allowed and f.get("donchian_break") and up and vr is not None and vr >= 1.3 and \
            f.get("trend_up") and rsi < 80:
        out.append({"kind": "donchian_breakout", "direction": "LONG", "size": 0.8, "exit_mode": EXIT_DYNAMIC_PEAK,
                    "stop_hint": float(f.get("ema_fast") or 0.0) or None,
                    "note": f"55-bar Donchian kırılımı ({f.get('donchian_hi55'):.6g}) hacim ×{vr:.1f}"})
    if "bos_retest" in allowed and f.get("bos_retest_up") and up:
        lvl = float(f.get("bos_level"))
        out.append({"kind": "bos_retest", "direction": "LONG", "size": 0.9, "exit_mode": EXIT_PARTIAL_RUN,
                    "stop_hint": lvl * (1.0 - 1.0 * atr / 100.0),
                    "note": f"yapı kırılımı ({lvl:.6g}) yeniden test edildi, tutuyor"})
    if "failed_breakdown" in allowed and f.get("failed_breakdown") and up and (vr is None or vr >= 1.0):
        psh = f.get("prior_swing_high")
        out.append({"kind": "failed_breakdown", "direction": "LONG", "size": 0.8, "exit_mode": EXIT_PARTIAL_RUN,
                    "stop_hint": float(f["fb_low"]) * (1.0 - 0.3 * atr / 100.0),
                    "target_hint": (float(psh) if psh is not None and math.isfinite(float(psh)) and float(psh) > f.get("price", 0) * 1.005 else None),
                    "note": f"20-bar düşük kırıldı ({f.get('fb_low'):.6g}) ama geri alındı — başarısız kırılma"})
    if "failed_breakout" in allowed and allow_short and f.get("failed_breakout"):
        out.append({"kind": "failed_breakout", "direction": "SHORT", "size": 0.7, "exit_mode": EXIT_FIXED,
                    "stop_hint": float(f["fb_high"]) * (1.0 + 0.3 * atr / 100.0),
                    "note": "20-bar yüksek kırıldı ama hacim yok, fiyat range'e döndü — başarısız kırılım"})
    cvd = f.get("cvd_ratio")
    flow_ok = (obi is not None and obi >= 0.65 and (f.get("microprice_dev_bps") or 0.0) >= 1.0) or \
              (cvd is not None and cvd >= 0.25 and obi is not None and obi >= 0.55)
    if "obi_momentum" in allowed and flow_ok and up and (cvd is None or cvd > -0.1) and \
            (f.get("trend_up") or f.get("ema_cross_up")) and (f.get("spread_bps") is None or f["spread_bps"] <= 8.0):
        out.append({"kind": "obi_momentum", "direction": "LONG", "size": 0.6, "exit_mode": EXIT_FIXED,
                    "note": f"defter dengesi %{(obi or 0)*100:.0f} alıcı · mikro-fiyat +{(f.get('microprice_dev_bps') or 0):.1f} bps"
                            + (f" · CVD {cvd:+.2f}" if cvd is not None else "") + " · kısa ufuk"})
    sw = f.get("swing") or {}
    if "swing_trend" in allowed and sw.get("trend_4h_up") and sw.get("pullback_1h") and (sw.get("regime") == "TREND YUKARI") and rsi < 70 \
            and sw.get("atr_pct_4h") and sw.get("swing_low_4h"):
        a4 = float(sw["atr_pct_4h"])
        out.append({"kind": "swing_trend", "direction": "LONG", "size": 0.6, "exit_mode": EXIT_DYNAMIC_PEAK,
                    "stop_hint": float(sw["swing_low_4h"]) * (1.0 - 0.2 * a4 / 100.0), "atr_hint": a4,
                    "note": f"swing: 4h trend yukarı + 1h EMA20 geri çekilmesi (7g %{sw.get('ret_7d_pct')}) · 1–3 gün tutum"})
    if "news_overreaction" in allowed and news and news.get("data_ok") and \
            float(news.get("score") or 0.0) <= -0.5 and not news.get("severe_risk") and \
            (f.get("move_4h_pct") or 0.0) <= -2.0 * atr and up and (vr is None or vr < 1.0):
        out.append({"kind": "news_overreaction", "direction": "LONG", "size": 0.5, "exit_mode": EXIT_FIXED,
                    "note": f"ayı haberi fiyatlandı (4 sa %{f.get('move_4h_pct'):.2f}), hacim sönüyor, dönüş barı"})
    return out
