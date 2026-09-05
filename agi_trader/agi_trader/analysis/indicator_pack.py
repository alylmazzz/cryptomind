"""
Gösterge tablosu genişletmesi — 129 → ~300.

`indicator_board.build_board` çekirdek 129 göstergeyi üretir; bu modül geri
kalanını ekler. Ayrı dosya olmasının sebebi salt uzunluk değil: buradakiler
çekirdekten FARKLI BİLGİ SINIFLARI (oynaklık tahmincileri, istatistiksel rejim
ölçütleri, çoklu zaman dilimi hizası) ve ayrı test edilebilmeleri gerekir.

⚠️ SAYININ KENDİSİ BİR KALİTE ÖLÇÜSÜ DEĞİLDİR. 300 gösterge, 300 bağımsız kanıt
demek değil; çoğu aynı fiyat serisinin aynı özelliğini ölçer. `indicator_board`
aile indirgemesi bu yüzden vardır ve asıl bakılması gereken sayı odur.
Ölçüldü: bu konsensüsü takip etmek örneklem dışında para KAYBETTİRİYOR
(bkz. `BOARD_EVIDENCE`). Tablo kapsam gösterir, sinyal üretmez.

Dolgu yasağı: her gösterge ya farklı bir HESAP ya farklı bir PENCERE ya da farklı
bir ZAMAN DİLİMİ olmalı. "Aynı şeyi iki adla yazmak" sayıyı büyütür, bilgiyi değil.
"""
from __future__ import annotations

import math
from typing import Callable, Dict

import numpy as np
import pandas as pd

from . import indicators as I

AL, SAT, NOTR = "AL", "SAT", "NÖTR"


def _f(x) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def _sig(buy: bool, sell: bool) -> str:
    if buy and not sell:
        return AL
    if sell and not buy:
        return SAT
    return NOTR


def _thr(v: float, lo: float, hi: float, invert: bool = False) -> str:
    if not math.isfinite(v):
        return NOTR
    return _sig(v < lo, v > hi) if invert else _sig(v > hi, v < lo)


def _last(s) -> float:
    try:
        return _f(pd.Series(s).iloc[-1])
    except Exception:
        return float("nan")


def _slope_r2(y: np.ndarray):
    """Doğrusal regresyon eğimi ve R² — trend gücünün ölçekten bağımsız ölçüsü."""
    n = len(y)
    if n < 3 or not np.isfinite(y).all():
        return float("nan"), float("nan")
    x = np.arange(n, dtype=float)
    m, b = np.polyfit(x, y, 1)
    pred = m * x + b
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float(m), float(1 - ss_res / (ss_tot + 1e-12))


# ===========================================================================
def extend_board(df: pd.DataFrame, add: Callable) -> None:
    """`add(name, kategori, deger, sinyal, gerekce, aile)` ile ~175 gösterge ekler."""
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    o = df["open"].astype(float)
    v = df["volume"].astype(float) if "volume" in df else pd.Series(1.0, index=df.index)
    px = float(c.iloc[-1])
    n = len(c)
    cv, hv, lv, ov, vv = c.values, h.values, l.values, o.values, v.values
    ret = c.pct_change()
    atr14 = _last(I.atr(df, 14))
    atr_u = atr14 if atr14 > 0 else px * 0.01

    # ─────────────── A) ÖZEL / UYARLANIR ORTALAMALAR (~20) ───────────────
    def _lsma(s: pd.Series, k: int) -> float:
        y = s.tail(k).to_numpy(dtype=float)
        m, _ = _slope_r2(y)
        if not math.isfinite(m):
            return float("nan")
        x = np.arange(len(y), dtype=float)
        b = y.mean() - m * x.mean()
        return float(m * (len(y) - 1) + b)

    for k in (10, 20, 50, 100):
        val = _lsma(c, k)
        add(f"LSMA {k}", "adaptive_ma", val, _sig(px > val, px < val),
            f"fiyat {'>' if px > val else '<'} doğrusal regresyon ortalaması", "lsma")

    def _alma(s: pd.Series, k=20, offset=0.85, sigma=6.0) -> float:
        y = s.tail(k).to_numpy(dtype=float)
        if len(y) < k:
            return float("nan")
        m_ = offset * (k - 1); sd = k / sigma
        w = np.exp(-((np.arange(k) - m_) ** 2) / (2 * sd * sd))
        return float((y * w).sum() / w.sum())

    for k in (9, 21, 50):
        val = _alma(c, k)
        add(f"ALMA {k}", "adaptive_ma", val, _sig(px > val, px < val),
            f"Arnaud Legoux ortalaması", "alma")

    def _trima(s, k):
        return _last(I.sma(I.sma(s, max(2, k // 2)), max(2, k // 2 + 1)))

    for k in (14, 30):
        val = _trima(c, k)
        add(f"TRIMA {k}", "adaptive_ma", val, _sig(px > val, px < val),
            "üçgen ağırlıklı ortalama", "trima")

    for k in (14, 50):
        val = _last(c.ewm(alpha=1.0 / k, adjust=False).mean())     # Wilder / SMMA
        add(f"SMMA {k}", "adaptive_ma", val, _sig(px > val, px < val),
            "Wilder yumuşatması", "smma")

    for k in (10, 20, 50):
        val = _last((c * v).rolling(k).sum() / (v.rolling(k).sum() + 1e-12))
        add(f"VWMA {k}", "adaptive_ma", val, _sig(px > val, px < val),
            "hacim ağırlıklı ortalama", "vwma")

    # sinüs ağırlıklı
    k = 20
    w = np.sin(np.pi * (np.arange(1, k + 1)) / (k + 1))
    val = float((c.tail(k).to_numpy() * w).sum() / w.sum()) if n >= k else float("nan")
    add("Sinüs ağırlıklı MA 20", "adaptive_ma", val, _sig(px > val, px < val),
        "uçları bastıran ağırlıklandırma", "swma")

    # TSF — zaman serisi tahmini (LSMA'nın bir bar ilerisi)
    for k in (14, 30):
        y = c.tail(k).to_numpy(dtype=float)
        m, _ = _slope_r2(y)
        val = _lsma(c, k) + (m if math.isfinite(m) else 0.0)
        add(f"TSF {k}", "adaptive_ma", val, _sig(px > val, px < val),
            "regresyonun bir bar ileri tahmini", "tsf")

    # tipik / medyan / ağırlıklı kapanış ortalamaları
    tp = (h + l + c) / 3.0
    mp = (h + l) / 2.0
    wc = (h + l + 2 * c) / 4.0
    for ad, ser, fam in (("Tipik fiyat MA 20", tp, "price_ma"),
                         ("Medyan fiyat MA 20", mp, "price_ma"),
                         ("Ağırlıklı kapanış MA 20", wc, "price_ma")):
        val = _last(I.sma(ser, 20))
        add(ad, "adaptive_ma", val, _sig(px > val, px < val), "fiyat türevi ortalama", fam)

    # ─────────────── B) ÇOKLU PERİYOT OSİLATÖRLER (~44) ───────────────
    for k in (5, 9, 25, 50):
        r = _last(I.rsi(c, k))
        add(f"RSI {k}", "momentum", r, _thr(r, 45, 55), f"RSI {r:.1f}", "rsi")
    for k in (5, 9, 21):
        kk, dd = I.stoch(df, k)
        add(f"Stochastic {k}", "momentum", _last(kk), _thr(_last(kk), 20, 80, invert=True),
            f"%K({k})", "stoch")
    for k in (7, 50):
        val = _last(I.cci(df, k))
        add(f"CCI {k}", "momentum", val, _thr(val, -100, 100), f"CCI {val:.0f}", "cci")
    for k in (7, 28):
        val = _last(I.williams_r(df, k))
        add(f"Williams %R {k}", "momentum", val, _thr(val, -80, -20, invert=True),
            f"{val:.1f}", "wr")
    for k in (5, 20, 60, 120):
        val = _last(I.roc(c, k))
        add(f"ROC {k}", "momentum", val, _sig(val > 0, val < 0),
            f"%{val:+.2f} ({k} bar)", "roc")
    for k in (5, 20, 60):
        val = _last(I.momentum(c, k))
        add(f"Momentum {k}", "momentum", val, _sig(val > 0, val < 0), f"{k} bar", "mom")
    for k in (7, 28):
        val = _last(I.mfi(df, k))
        add(f"MFI {k}", "volume", val, _thr(val, 20, 80, invert=True), f"{val:.1f}", "mfi")
    for k in (7, 28):
        val = _last(I.cmo(c, k))
        add(f"CMO {k}", "momentum", val, _thr(val, -50, 50), f"{val:.1f}", "cmo")
    for k in (9, 21):
        val = _last(I.trix(c, k))
        add(f"TRIX {k}", "momentum", val, _sig(val > 0, val < 0), "sıfır çizgisi", "trix")
    for k in (7, 28):
        a_, p_, m_ = I.adx(df, k)
        add(f"ADX {k}", "trend_strength", _last(a_),
            NOTR if _last(a_) < 20 else _sig(_last(p_) > _last(m_), _last(m_) > _last(p_)),
            f"ADX {_last(a_):.1f}", "adx")
    for k in (14, 50):
        au, ad_ = I.aroon(df, k)
        d_ = _last(au) - _last(ad_)
        add(f"Aroon {k}", "trend_strength", d_, _sig(d_ > 0, d_ < 0), "yukarı/aşağı", "aroon")
    for k in (7, 28):
        vp, vm = I.vortex(df, k)
        d_ = _last(vp) - _last(vm)
        add(f"Vortex {k}", "trend_strength", d_, _sig(d_ > 0, d_ < 0), "VI+ / VI−", "vortex")
    for fast, slow, sgn in ((5, 13, 4), (19, 39, 9)):
        ml, ms, mh = I.macd(c, fast, slow, sgn)
        add(f"MACD {fast}/{slow}", "momentum", _last(ml) - _last(ms),
            _sig(_last(ml) > _last(ms), _last(ml) < _last(ms)), "sinyal kesişimi",
            "macd")
    for k in (14, 30):
        st, sd = I.supertrend(df, k, 3.0)
        add(f"SuperTrend {k}", "trend_strength", _last(st), _sig(_last(sd) > 0, _last(sd) < 0),
            "yön", "supertrend")
    # SMI — Stochastic Momentum Index
    for k in (14, 25):
        hh = h.rolling(k).max(); ll = l.rolling(k).min()
        mid = (hh + ll) / 2
        num = (c - mid).ewm(span=3).mean().ewm(span=3).mean()
        den = ((hh - ll) / 2).ewm(span=3).mean().ewm(span=3).mean()
        val = _last(100 * num / (den + 1e-12))
        add(f"SMI {k}", "momentum", val, _thr(val, -40, 40), f"{val:.1f}", "smi")
    # Laguerre RSI (kısa gecikmeli)
    try:
        g = 0.5
        L0 = L1 = L2 = L3 = 0.0
        for x in cv[-120:]:
            p0 = L0; p1 = L1; p2 = L2
            L0 = (1 - g) * x + g * p0
            L1 = -g * L0 + p0 + g * p1
            L2 = -g * L1 + p1 + g * p2
            L3 = -g * L2 + p2 + g * L3
        cu = max(L0 - L1, 0) + max(L1 - L2, 0) + max(L2 - L3, 0)
        cd = max(L1 - L0, 0) + max(L2 - L1, 0) + max(L3 - L2, 0)
        val = float(cu / (cu + cd + 1e-12))
        add("Laguerre RSI", "momentum", val, _thr(val, 0.2, 0.8, invert=True),
            f"{val:.2f}", "laguerre")
    except Exception:
        pass

    # ─────────────── C) OYNAKLIK TAHMİNCİLERİ (~18) ───────────────
    lhc = np.log(hv / np.maximum(lv, 1e-12))
    lco = np.log(cv / np.maximum(ov, 1e-12))
    lho = np.log(hv / np.maximum(ov, 1e-12))
    llo = np.log(lv / np.maximum(ov, 1e-12))

    def _tail_mean(a, k):
        a = a[-k:]
        a = a[np.isfinite(a)]
        return float(a.mean()) if a.size else float("nan")

    park = math.sqrt(max(_tail_mean(lhc ** 2, 20), 0) / (4 * math.log(2))) * 100
    add("Parkinson oynaklık", "volatility", park, NOTR,
        f"%{park:.2f} (yüksek-düşük tabanlı)", "vol_est")
    gk = math.sqrt(max(0.5 * _tail_mean(lhc ** 2, 20) -
                       (2 * math.log(2) - 1) * _tail_mean(lco ** 2, 20), 0)) * 100
    add("Garman-Klass oynaklık", "volatility", gk, NOTR, f"%{gk:.2f}", "vol_est")
    rs = math.sqrt(max(_tail_mean(lho * (lho - lco) + llo * (llo - lco), 20), 0)) * 100
    add("Rogers-Satchell oynaklık", "volatility", rs, NOTR,
        f"%{rs:.2f} (sürüklenmeye dayanıklı)", "vol_est")
    for k in (10, 20, 60):
        hv_ = _f(ret.tail(k).std() * 100)
        add(f"Tarihsel oynaklık {k}", "volatility", hv_, NOTR, f"%{hv_:.2f}", "hv")
    dd_ = _f(ret[ret < 0].tail(40).std() * 100)
    add("Aşağı yönlü sapma", "volatility", dd_, NOTR, f"%{dd_:.2f}", "downside")
    up_ = _f(ret[ret > 0].tail(40).std() * 100)
    add("Yukarı/aşağı sapma oranı", "volatility", up_ / (dd_ + 1e-12),
        _sig(up_ > dd_ * 1.15, dd_ > up_ * 1.15),
        "yukarı hareketler daha oynak" if up_ > dd_ else "aşağı hareketler daha oynak",
        "vol_skew")
    for k in (7, 50):
        a_ = _last(I.atr(df, k)) / px * 100
        add(f"ATR% {k}", "volatility", a_, NOTR, f"%{a_:.2f}", "atr")
    sk = _f(ret.tail(60).skew())
    add("Getiri çarpıklığı", "volatility", sk, _sig(sk > 0.3, sk < -0.3),
        f"{sk:+.2f} — {'sağa çarpık' if sk > 0 else 'sola çarpık'}", "moments")
    ku = _f(ret.tail(60).kurt())
    add("Getiri basıklığı", "volatility", ku, NOTR,
        f"{ku:+.2f} — {'kalın kuyruk' if ku > 1 else 'normale yakın'}", "moments")
    for nb, kb in ((20, 1.5), (20, 2.5), (50, 2.0)):
        bu, bm, bl = I.bollinger(c, nb, kb)
        pb = (px - _last(bl)) / (_last(bu) - _last(bl) + 1e-12)
        add(f"Bollinger %B {nb}/{kb}", "volatility", pb, _thr(pb, 0.0, 1.0, invert=True),
            f"%B {pb:.2f}", "bb")
    bw = ((I.bollinger(c)[0] - I.bollinger(c)[2]) / (I.bollinger(c)[1] + 1e-12))
    pct = _f((bw.tail(120).rank(pct=True)).iloc[-1]) * 100
    add("Bollinger genişlik yüzdeliği", "volatility", pct, NOTR,
        f"%{pct:.0f} — {'aşırı sıkışma' if pct < 15 else 'aşırı genişleme' if pct > 85 else 'normal'}",
        "bb_pct")
    for k in (20, 50):
        du, dm, dl = I.donchian(df, k)
        pos = (px - _last(dl)) / (_last(du) - _last(dl) + 1e-12)
        add(f"Donchian konumu {k}", "volatility", pos, _thr(pos, 0.25, 0.75),
            f"kanalın %{pos*100:.0f}'i", "donch")

    # ─────────────── D) TREND / REGRESYON (~20) ───────────────
    for k in (10, 20, 50, 100):
        m, r2 = _slope_r2(cv[-k:])
        nm = m * k / px * 100 if px else float("nan")
        add(f"Regresyon eğimi {k}", "trend_strength", nm, _sig(nm > 0.5, nm < -0.5),
            f"%{nm:+.1f}/pencere", "reg")
        add(f"Regresyon R² {k}", "trend_strength", r2, NOTR,
            f"{r2:.2f} — {'düzgün trend' if r2 > 0.7 else 'dağınık'}", f"regr2_{k}")
    for k in (20, 60):
        y = cv[-k:]
        x = np.arange(k, dtype=float)
        cr = _f(np.corrcoef(x, y)[0, 1]) if np.isfinite(y).all() else float("nan")
        add(f"Zaman korelasyonu {k}", "trend_strength", cr, _sig(cr > 0.5, cr < -0.5),
            f"{cr:+.2f}", "tcorr")
    for k in (14, 40):
        net = abs(cv[-1] - cv[-k]) if n > k else float("nan")
        yol = float(np.abs(np.diff(cv[-k:])).sum())
        er = net / (yol + 1e-12)
        add(f"Verimlilik oranı {k}", "trend_strength", er, NOTR,
            f"{er:.2f} — {'trendli' if er > 0.4 else 'çalkantılı'}", "er")
    for k in (14, 40):
        rng = float(hv[-k:].max() - lv[-k:].min())
        rwi_u = (cv[-1] - lv[-k:].min()) / (atr_u * math.sqrt(k) + 1e-12)
        rwi_d = (hv[-k:].max() - cv[-1]) / (atr_u * math.sqrt(k) + 1e-12)
        add(f"Random Walk Index {k}", "trend_strength", rwi_u - rwi_d,
            _sig(rwi_u > rwi_d * 1.2, rwi_d > rwi_u * 1.2),
            "rastgele yürüyüşün ötesi", "rwi")
    for k in (30, 60):
        sma_ = I.sma(c, k)
        tii = _f(((c > sma_).tail(k).mean()) * 100)
        add(f"Trend Yoğunluğu {k}", "trend_strength", tii, _thr(tii, 40, 60),
            f"barların %{tii:.0f}'i ortalamanın üstünde", "tii")
    ck_hi = _f(h.rolling(10).max().shift(1).iloc[-1]) - 2 * atr_u
    ck_lo = _f(l.rolling(10).min().shift(1).iloc[-1]) + 2 * atr_u
    add("Chande Kroll stop", "trend_strength", ck_hi, _sig(px > ck_hi, px < ck_lo),
        "uzun/kısa stop bandına göre", "chande_kroll")

    # ─────────────── E) HACİM (~20) ───────────────
    for k in (10, 30):
        vo = _f((I.sma(v, 5).iloc[-1] / (I.sma(v, k).iloc[-1] + 1e-12) - 1) * 100)
        add(f"Hacim osilatörü 5/{k}", "volume", vo, _sig(vo > 10, vo < -10),
            f"%{vo:+.0f}", "volosc")
    for k in (5, 20):
        vr = _f(v.pct_change(k).iloc[-1] * 100)
        add(f"Hacim ROC {k}", "volume", vr, _sig(vr > 0, vr < 0), f"%{vr:+.0f}", "volroc")
    for k in (10, 50):
        obv_ = I.obv(df)
        sl = _f(obv_.iloc[-1] - obv_.iloc[-k]) if n > k else float("nan")
        add(f"OBV eğimi {k}", "volume", sl, _sig(sl > 0, sl < 0),
            "birikim" if sl > 0 else "dağıtım", "obv")
    # VZO — Volume Zone Oscillator
    for k in (14, 28):
        vp_ = (np.sign(c.diff()) * v).ewm(span=k).mean()
        vt_ = v.ewm(span=k).mean()
        vz = _f(100 * vp_.iloc[-1] / (vt_.iloc[-1] + 1e-12))
        add(f"VZO {k}", "volume", vz, _thr(vz, -15, 15), f"{vz:+.1f}", "vzo")
    # hacim ağırlıklı RSI
    vw = (c.diff() * v)
    up_v = vw.clip(lower=0).rolling(14).sum()
    dn_v = (-vw.clip(upper=0)).rolling(14).sum()
    vrsi = _f(100 - 100 / (1 + up_v.iloc[-1] / (dn_v.iloc[-1] + 1e-12)))
    add("Hacim ağırlıklı RSI", "volume", vrsi, _thr(vrsi, 40, 60), f"{vrsi:.1f}", "vrsi")
    nvi, pvi = I.nvi_pvi(df)
    for ad, ser, fam in (("PVI", pvi, "pvi"),):
        d_ = _f(ser.iloc[-1] - I.sma(ser, 50).iloc[-1])
        add(ad, "volume", d_, _sig(d_ > 0, d_ < 0), "50 ortalamasına göre", fam)
    for k in (10, 40):
        upv = float(vv[-k:][np.diff(np.concatenate([[cv[-k-1]], cv[-k:]])) > 0].sum())
        dnv = float(vv[-k:][np.diff(np.concatenate([[cv[-k-1]], cv[-k:]])) < 0].sum())
        add(f"Yukarı/aşağı hacim {k}", "volume", upv - dnv, _sig(upv > dnv, dnv > upv),
            "alıcı hacmi baskın" if upv > dnv else "satıcı hacmi baskın", "udv")
    vpct = _f(v.tail(120).rank(pct=True).iloc[-1]) * 100
    add("Hacim yüzdeliği", "volume", vpct, NOTR,
        f"%{vpct:.0f} — {'olağandışı yüksek' if vpct > 90 else 'düşük' if vpct < 20 else 'normal'}",
        "vol_pct")
    ti = _f((v.tail(20).sum() / (np.abs(np.diff(cv[-21:])).sum() + 1e-12)))
    add("İşlem yoğunluğu", "volume", ti, NOTR, "birim harekete düşen hacim", "trade_int")
    for k in (10, 40):
        clv = ((c - l) - (h - c)) / ((h - l) + 1e-12)
        adl = (clv * v).cumsum()
        sl = _f(adl.iloc[-1] - adl.iloc[-k]) if n > k else float("nan")
        add(f"A/D eğimi {k}", "volume", sl, _sig(sl > 0, sl < 0),
            "birikim" if sl > 0 else "dağıtım", "adl")

    # ─────────────── F) İSTATİSTİKSEL / REJİM (~20) ───────────────
    def _hurst(y, max_lag=20):
        y = np.asarray(y, dtype=float)
        if len(y) < max_lag * 3 or not np.isfinite(y).all():
            return float("nan")
        lags = range(2, max_lag)
        tau = [np.sqrt(np.std(y[lag:] - y[:-lag])) for lag in lags]
        tau = np.asarray(tau)
        if not np.isfinite(tau).all() or (tau <= 0).any():
            return float("nan")
        m, _ = np.polyfit(np.log(list(lags)), np.log(tau), 1)
        return float(m * 2.0)

    hu = _hurst(cv[-160:])
    add("Hurst üsteli", "trend_strength", hu, _sig(hu > 0.55, hu < 0.45),
        f"{hu:.2f} — {'trend kalıcı' if hu > 0.55 else 'ortalamaya dönüş' if hu < 0.45 else 'rastgele'}",
        "hurst")
    fd = 2.0 - hu if math.isfinite(hu) else float("nan")
    add("Fraktal boyut", "trend_strength", fd, NOTR,
        f"{fd:.2f} — {'düzgün' if fd < 1.45 else 'pürüzlü'}", "hurst")
    r = ret.dropna().to_numpy()
    for lag in (1, 5, 10):
        if len(r) > lag + 40:
            ac = _f(np.corrcoef(r[:-lag], r[lag:])[0, 1])
        else:
            ac = float("nan")
        add(f"Otokorelasyon gecikme {lag}", "trend_strength", ac,
            _sig(ac > 0.1, ac < -0.1),
            f"{ac:+.3f} — {'momentum' if ac > 0.1 else 'dönüş' if ac < -0.1 else 'yok'}",
            "acf")
    for q in (2, 5, 10):
        if len(r) > q * 30:
            v1 = float(np.var(r[-q * 30:], ddof=1))
            agg = np.add.reduceat(r[-q * 30:], np.arange(0, q * 30, q))
            vq = float(np.var(agg, ddof=1)) / q
            vr = vq / (v1 + 1e-18)
        else:
            vr = float("nan")
        add(f"Varyans oranı q={q}", "trend_strength", vr, _sig(vr > 1.15, vr < 0.85),
            f"{vr:.2f} — {'trendli' if vr > 1.15 else 'ortalamaya dönen' if vr < 0.85 else 'rastgele'}",
            "vratio")
    # Ornstein-Uhlenbeck yarı-ömür
    try:
        y = pd.Series(cv[-120:])
        dy = y.diff().dropna()
        yl = y.shift(1).dropna()
        bcoef = float(np.polyfit(yl.values, dy.values, 1)[0])
        hl = -math.log(2) / bcoef if bcoef < 0 else float("nan")
        add("OU yarı-ömür", "trend_strength", hl, NOTR,
            f"{hl:.0f} bar" if math.isfinite(hl) else "ortalamaya dönmüyor", "ou")
    except Exception:
        pass
    # Shannon entropisi (getiri işaret dizisi)
    try:
        s3 = np.sign(r[-120:])
        vals, cnt = np.unique(s3, return_counts=True)
        p_ = cnt / cnt.sum()
        ent = float(-(p_ * np.log2(p_ + 1e-12)).sum())
        add("İşaret entropisi", "trend_strength", ent, NOTR,
            f"{ent:.2f} bit — {'öngörülemez' if ent > 1.5 else 'yapılı'}", "entropy")
    except Exception:
        pass
    # ardışık seri uzunlukları
    sgn = np.sign(np.diff(cv[-120:]))
    runs = 1
    for i in range(len(sgn) - 1, 0, -1):
        if sgn[i] == sgn[i - 1]:
            runs += 1
        else:
            break
    add("Güncel seri uzunluğu", "structure", runs * (sgn[-1] if len(sgn) else 0),
        _sig(runs >= 4 and sgn[-1] < 0, runs >= 4 and sgn[-1] > 0),
        f"{runs} bar {'yükseliş' if len(sgn) and sgn[-1] > 0 else 'düşüş'}", "runs")
    for k in (20, 60, 120):
        pr = _f(c.tail(k).rank(pct=True).iloc[-1]) * 100
        add(f"Fiyat yüzdeliği {k}", "structure", pr, _thr(pr, 25, 75),
            f"son {k} barın %{pr:.0f}'inde", "pctile")

    # ─────────────── G) PİVOT / SEVİYE (~14) ───────────────
    ph, pl, pc = float(hv[-2]), float(lv[-2]), float(cv[-2])
    piv = (ph + pl + pc) / 3.0
    rng = ph - pl
    for ad, lvl in (("Fibonacci R1", piv + 0.382 * rng), ("Fibonacci R2", piv + 0.618 * rng),
                    ("Fibonacci R3", piv + 1.000 * rng), ("Fibonacci S1", piv - 0.382 * rng),
                    ("Fibonacci S2", piv - 0.618 * rng), ("Fibonacci S3", piv - 1.000 * rng)):
        add(ad, "structure", lvl, _sig(px > lvl, px < lvl),
            f"fiyat {'üstünde' if px > lvl else 'altında'}", "fib_pivot")
    for ad, lvl in (("Camarilla R3", pc + rng * 1.1 / 4), ("Camarilla R4", pc + rng * 1.1 / 2),
                    ("Camarilla S3", pc - rng * 1.1 / 4), ("Camarilla S4", pc - rng * 1.1 / 2)):
        add(ad, "structure", lvl, _sig(px > lvl, px < lvl), "Camarilla seviyesi", "camarilla")
    wood = (ph + pl + 2 * pc) / 4.0
    add("Woodie pivot", "structure", wood, _sig(px > wood, px < wood),
        "kapanış ağırlıklı pivot", "woodie")
    for k in (52, 104):
        if n > k:
            hi_, lo_ = float(hv[-k:].max()), float(lv[-k:].min())
            pos = (px - lo_) / (hi_ - lo_ + 1e-12)
            add(f"{k} bar aralık konumu", "structure", pos, _thr(pos, 0.25, 0.75),
                f"aralığın %{pos*100:.0f}'i", "range")

    # ─────────────── H) MUM / YAPI (~18) ───────────────
    body = float(cv[-1] - ov[-1])
    rng1 = float(hv[-1] - lv[-1]) + 1e-12
    ust = float(hv[-1] - max(cv[-1], ov[-1]))
    alt = float(min(cv[-1], ov[-1]) - lv[-1])
    add("Üst fitil oranı", "structure", ust / rng1, _sig(ust / rng1 < 0.15, ust / rng1 > 0.5),
        f"{ust/rng1:.2f} — {'satış baskısı' if ust/rng1 > 0.5 else 'yok'}", "wick")
    add("Alt fitil oranı", "structure", alt / rng1, _sig(alt / rng1 > 0.5, alt / rng1 < 0.15),
        f"{alt/rng1:.2f} — {'alım desteği' if alt/rng1 > 0.5 else 'yok'}", "wick")
    add("Gövde/aralık", "structure", body / rng1, _sig(body / rng1 > 0.6, body / rng1 < -0.6),
        f"{body/rng1:+.2f} — {'güçlü mum' if abs(body/rng1) > 0.6 else 'kararsız'}", "candle_ext")
    ic = bool(hv[-1] <= hv[-2] and lv[-1] >= lv[-2])
    oc = bool(hv[-1] >= hv[-2] and lv[-1] <= lv[-2])
    add("İç/dış bar", "structure", 1.0 if ic else (-1.0 if oc else 0.0), NOTR,
        "iç bar (sıkışma)" if ic else "dış bar (genişleme)" if oc else "normal", "in_out")
    gap = _f((ov[-1] / cv[-2] - 1) * 100)
    add("Açılış boşluğu", "structure", gap, _sig(gap > 0.5, gap < -0.5), f"%{gap:+.2f}", "gap")
    for k in (10, 30):
        up_n = int((np.diff(cv[-k - 1:]) > 0).sum())
        add(f"Yükselen bar oranı {k}", "structure", up_n / k * 100,
            _thr(up_n / k * 100, 40, 60), f"%{up_n/k*100:.0f}", "upbars")
    for k in (20, 50, 200):
        ma = _last(I.sma(c, k))
        dist = (px - ma) / atr_u
        add(f"SMA{k}'e ATR uzaklığı", "trend_ma", dist, _thr(dist, -0.5, 0.5),
            f"{dist:+.1f} ATR — {'aşırı gerilmiş' if abs(dist) > 3 else 'normal'}",
            "ma_dist")
    for k in (10, 30):
        rr = _f((h.rolling(k).max().iloc[-1] - l.rolling(k).min().iloc[-1]) / px * 100)
        add(f"{k} bar aralık genişliği", "volatility", rr, NOTR, f"%{rr:.1f}", "rngw")
    hh = bool(hv[-1] >= hv[-20:].max()); ll_ = bool(lv[-1] <= lv[-20:].min())
    add("20 bar yeni zirve/dip", "structure", 1.0 if hh else (-1.0 if ll_ else 0.0),
        _sig(hh, ll_), "yeni zirve" if hh else "yeni dip" if ll_ else "aralık içi", "newhl")

    # ─────────────── H2) EK KLASİKLER (~16) ───────────────
    try:
        tk, kj, sa, sb = I.ichimoku(df)
        chik = float(cv[-1]) - float(cv[-27]) if n > 27 else float("nan")
        add("Chikou span", "trend_strength", chik, _sig(chik > 0, chik < 0),
            "gecikmeli çizgi 26 bar önceki fiyatın üstünde" if chik > 0 else "altında",
            "ichimoku_ext")
        tw = _last(sa) - _last(sb)
        prev_tw = _f(pd.Series(sa).iloc[-6] - pd.Series(sb).iloc[-6]) if n > 6 else float("nan")
        add("Kumo bükülmesi", "trend_strength", tw,
            _sig(tw > 0 and prev_tw <= 0, tw < 0 and prev_tw >= 0),
            "bulut yön değiştirdi" if (tw > 0) != (prev_tw > 0) else "bulut yönü sabit",
            "ichimoku_ext")
    except Exception:
        pass
    # Elder Impulse: EMA13 yönü + MACD histogram yönü aynı mı?
    e13 = I.ema(c, 13)
    _, _, mh2 = I.macd(c)
    e_up = _f(e13.iloc[-1]) > _f(e13.iloc[-2])
    h_up = _f(mh2.iloc[-1]) > _f(mh2.iloc[-2])
    add("Elder Impulse", "momentum", 1.0 if (e_up and h_up) else (-1.0 if (not e_up and not h_up) else 0.0),
        _sig(e_up and h_up, (not e_up) and (not h_up)),
        "ortalama ve momentum aynı yönde" if e_up == h_up else "çelişkili", "elder_impulse")
    for kn, kk_ in ((10, 1.5), (30, 2.5)):
        ku_, km_, kl_ = I.keltner(df, kn, kk_)
        add(f"Keltner {kn}/{kk_}", "volatility", px - _last(km_),
            _sig(px > _last(ku_), px < _last(kl_)),
            "üst bandın üstünde" if px > _last(ku_) else
            "alt bandın altında" if px < _last(kl_) else "kanal içi", "keltner")
    # TTM Squeeze — Bollinger, Keltner'ın İÇİNE girdiyse sıkışma var
    bu_, bm_, bl2 = I.bollinger(c, 20, 2.0)
    ku2, km2, kl2 = I.keltner(df, 20, 1.5)
    sq = bool(_last(bu_) < _last(ku2) and _last(bl2) > _last(kl2))
    add("TTM Squeeze", "volatility", 1.0 if sq else 0.0, NOTR,
        "SIKIŞMA — kırılım yaklaşıyor, yön belirsiz" if sq else "sıkışma yok", "ttm")
    for kc in (14, 40):
        try:
            from .indicators_ext import _choppiness
            ch = _f(_choppiness(df, kc))
        except Exception:
            ch = float("nan")
        add(f"Choppiness {kc}", "volatility", ch, NOTR,
            f"{ch:.0f} — {'yatay' if ch > 61 else 'trendli' if ch < 38 else 'ara'}", "choppiness")
    for ku_n in (14, 50):
        try:
            from .indicators_ext import _ulcer
            ul = _f(_ulcer(c, ku_n))
        except Exception:
            ul = float("nan")
        add(f"Ulcer {ku_n}", "volatility", ul, NOTR, f"{ul:.2f}", "ulcer")
    kl_line = I.chaikin_osc(df)
    add("Chaikin Osc sinyali", "volume", _last(kl_line) - _f(I.sma(kl_line, 10).iloc[-1]),
        _sig(_last(kl_line) > _f(I.sma(kl_line, 10).iloc[-1]),
             _last(kl_line) < _f(I.sma(kl_line, 10).iloc[-1])),
        "10 ortalamasına göre", "chaikin_sig")
    # fiyat-hacim doğrulaması
    pc_ = _f(c.pct_change(10).iloc[-1])
    vc_ = _f(v.rolling(10).mean().pct_change(10).iloc[-1])
    add("Fiyat-hacim doğrulaması", "volume", pc_ * vc_,
        _sig(pc_ > 0 and vc_ > 0, pc_ < 0 and vc_ > 0),
        "yükseliş hacimle destekli" if pc_ > 0 and vc_ > 0 else
        "düşüş hacimle destekli" if pc_ < 0 and vc_ > 0 else "hacim onaylamıyor",
        "pv_confirm")
    add("Mass Index 25", "volatility", _last(I.mass_index(df, 9, 25)), NOTR,
        "dönüş uyarısı eşiği 27", "mass")
    add("Coppock 22", "momentum", _last(I.wma(I.roc(c, 22) + I.roc(c, 17), 12)),
        _sig(_last(I.wma(I.roc(c, 22) + I.roc(c, 17), 12)) > 0,
             _last(I.wma(I.roc(c, 22) + I.roc(c, 17), 12)) < 0),
        "uzun vadeli dip göstergesi", "coppock")

    # ─────────────── I) ÇOKLU ZAMAN DİLİMİ (~16) ───────────────
    # Barları toplayarak üst zaman dilimi üret — aynı gösterge, FARKLI ölçek:
    # bu gerçekten yeni bilgidir, periyot değiştirmekten farklıdır.
    for kat, mult in (("orta", 4), ("üst", 12)):
        try:
            g = np.arange(n) // mult
            agg = pd.DataFrame({
                "open": o.groupby(g).first(), "high": h.groupby(g).max(),
                "low": l.groupby(g).min(), "close": c.groupby(g).last(),
                "volume": v.groupby(g).sum()})
            if len(agg) < 60:
                continue
            ac = agg["close"]
            apx = float(ac.iloc[-1])
            for k in (20, 50):
                m_ = _last(I.ema(ac, k))
                add(f"{kat.upper()} TF EMA {k}", "trend_ma", m_, _sig(apx > m_, apx < m_),
                    f"{mult}× birleştirilmiş barda", "mtf_ma")
            rr = _last(I.rsi(ac, 14))
            add(f"{kat.upper()} TF RSI 14", "momentum", rr, _thr(rr, 45, 55),
                f"{mult}× barda RSI {rr:.1f}", "mtf_rsi")
            ml, ms, mh = I.macd(ac)
            add(f"{kat.upper()} TF MACD", "momentum", _last(ml) - _last(ms),
                _sig(_last(ml) > _last(ms), _last(ml) < _last(ms)),
                "üst dilim sinyal kesişimi", "mtf_macd")
            a_, p_, m2 = I.adx(agg, 14)
            add(f"{kat.upper()} TF ADX", "trend_strength", _last(a_),
                NOTR if _last(a_) < 20 else _sig(_last(p_) > _last(m2), _last(m2) > _last(p_)),
                f"üst dilim trend gücü", "mtf_adx")
            sl, r2 = _slope_r2(ac.tail(20).to_numpy(dtype=float))
            add(f"{kat.upper()} TF eğim", "trend_strength", sl / apx * 100 * 20,
                _sig(sl > 0, sl < 0), "üst dilim regresyon eğimi", "mtf_slope")
            st_, sd_ = I.supertrend(agg, 10, 3.0)
            add(f"{kat.upper()} TF SuperTrend", "trend_strength", _last(st_),
                _sig(_last(sd_) > 0, _last(sd_) < 0), "üst dilim yön", "mtf_st")
            bu, bm, bl_ = I.bollinger(ac)
            pb = (apx - _last(bl_)) / (_last(bu) - _last(bl_) + 1e-12)
            add(f"{kat.upper()} TF Bollinger %B", "volatility", pb,
                _thr(pb, 0.0, 1.0, invert=True), f"%B {pb:.2f}", "mtf_bb")
            add(f"{kat.upper()} TF konum", "structure",
                _f(ac.tail(50).rank(pct=True).iloc[-1]) * 100,
                _thr(_f(ac.tail(50).rank(pct=True).iloc[-1]) * 100, 25, 75),
                "üst dilim aralık konumu", "mtf_pos")
        except Exception:
            continue
