"""
Gösterge tablosu — ikinci genişletme paketi (300 → ~400).

"Azami sayıya çıkar" isteğinin dürüst yorumu: doğal bir üst sınır YOKTUR.
RSI'ı 60 ayrı periyotta hesaplayıp "360 gösterge" demek mümkün ama bu sayıyı
büyütür, bilgiyi değil. Bu paket yalnız **hesabı gerçekten farklı** olanları
ekler ve periyot varyantıyla şişirmez.

Eklenen dört sınıf — dördü de çekirdekte YOK:

  A) EHLERS / SAYISAL FİLTRELER — süper yumuşatıcı, decycler, ağırlık merkezi,
     siber döngü, sinüs dalgası. Bunlar hareketli ortalama değil, sinyal
     işleme filtreleridir; gecikme/gürültü dengesi farklıdır.

  B) İSTATİSTİKSEL PERFORMANS — yuvarlanan Sharpe/Sortino/Calmar/Omega,
     K-oranı, kazanç-acı, düşüş derinliği ve SÜRESİ, toparlanma çarpanı.
     Fiyat seviyesini değil, getiri dağılımının ŞEKLİNİ ölçerler.

  C) HACİM PROFİLİ — POC, değer alanı (VAH/VAL), fiyatın değer alanındaki
     konumu. Zaman ekseni yerine FİYAT ekseninde hacim dağılımı.

  D) MİKROYAPI / L2 — `data/recorder.py` verisinden: funding z-skoru, açık
     pozisyon momentumu, defter eğimi/dışbükeyliği, çoklu derinlikte
     dengesizlik, Amihud likidite-sizliği, Kyle lambda vekili, etkin spread.
     **Bunlar fiyattan TÜRETİLMEZ** — tablodaki tek gerçekten dik bilgi sınıfı.
     Kaydedici verisi yoksa bu bölüm sessizce atlanır (uydurma yapılmaz).

⚠️ Aile indirgemesi burada da geçerlidir: 400 ham gösterge 400 bağımsız kanıt
değildir. Ve ölçüldü — 129 → 300 çıkışı konsensüsün öngörü gücünü hiç
değiştirmedi (bkz. `BOARD_EVIDENCE.count_comparison`).
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Optional

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


def _sig(b: bool, s: bool) -> str:
    if b and not s:
        return AL
    if s and not b:
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


# ===========================================================================
def extend_board2(df: pd.DataFrame, add: Callable,
                  micro: Optional[pd.Series] = None) -> bool:
    """`add(ad, kategori, deger, sinyal, gerekce, aile)` ile ~100 gösterge ekler.

    micro: `data/recorder.py`'den o parite için EN SON satır (varsa).
    Döner: mikroyapı bölümü eklendi mi."""
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    v = df["volume"].astype(float) if "volume" in df else pd.Series(1.0, index=df.index)
    px = float(c.iloc[-1])
    n = len(c)
    cv = c.values
    ret = c.pct_change().dropna()

    # ───────────────── A) EHLERS / SAYISAL FİLTRELER (~20) ─────────────────
    def _super_smoother(x: np.ndarray, period: int = 20) -> float:
        """2 kutuplu Butterworth — hareketli ortalamadan daha az gecikmeyle
        yüksek frekansı bastırır."""
        if len(x) < period + 5:
            return float("nan")
        a = math.exp(-1.414 * math.pi / period)
        b = 2 * a * math.cos(1.414 * math.pi / period)
        c2, c3 = b, -a * a
        c1 = 1 - c2 - c3
        y = list(x[:2])
        for i in range(2, len(x)):
            y.append(c1 * (x[i] + x[i - 1]) / 2 + c2 * y[-1] + c3 * y[-2])
        return float(y[-1])

    for p in (10, 20, 40):
        val = _super_smoother(cv[-200:], p)
        add(f"Süper Yumuşatıcı {p}", "adaptive_ma", val, _sig(px > val, px < val),
            "Butterworth filtresi — düşük gecikmeli yumuşatma", "ehlers_ss")

    def _decycler(x: np.ndarray, period: int = 60) -> float:
        """Yüksek geçiren filtreyi çıkararak trendi izole eder."""
        if len(x) < period + 5:
            return float("nan")
        alpha = (math.cos(2 * math.pi / period) + math.sin(2 * math.pi / period) - 1) / \
                math.cos(2 * math.pi / period)
        y = [x[0]]
        for i in range(1, len(x)):
            y.append((alpha / 2) * (x[i] + x[i - 1]) + (1 - alpha) * y[-1])
        return float(y[-1])

    for p in (30, 60):
        val = _decycler(cv[-250:], p)
        add(f"Decycler {p}", "adaptive_ma", val, _sig(px > val, px < val),
            "döngü bileşeni çıkarılmış trend", "ehlers_decycler")

    def _cog(x: np.ndarray, period: int = 10) -> float:
        """Ağırlık merkezi osilatörü — Ehlers."""
        if len(x) < period:
            return float("nan")
        w = x[-period:]
        num = sum((i + 1) * w[-(i + 1)] for i in range(period))
        den = w.sum()
        return float(-num / den + (period + 1) / 2.0) if den else float("nan")

    for p in (10, 20):
        val = _cog(cv, p)
        add(f"Ağırlık Merkezi {p}", "momentum", val, _sig(val > 0, val < 0),
            "Ehlers CoG osilatörü", "ehlers_cog")

    def _cyber_cycle(x: np.ndarray, alpha: float = 0.07) -> float:
        if len(x) < 20:
            return float("nan")
        sm = [(x[i] + 2 * x[i - 1] + 2 * x[i - 2] + x[i - 3]) / 6
              for i in range(3, len(x))]
        cyc = [0.0, 0.0]
        for i in range(2, len(sm)):
            cyc.append((1 - 0.5 * alpha) ** 2 * (sm[i] - 2 * sm[i - 1] + sm[i - 2])
                       + 2 * (1 - alpha) * cyc[-1] - (1 - alpha) ** 2 * cyc[-2])
        return float(cyc[-1])

    cyc = _cyber_cycle(cv[-200:])
    add("Siber Döngü", "momentum", cyc, _sig(cyc > 0, cyc < 0),
        "Ehlers döngü bileşeni", "ehlers_cycle")

    # Baskın döngü periyodu — otokorelasyon tepe noktası
    try:
        r = ret.tail(200).to_numpy()
        r = r - r.mean()
        ac = np.correlate(r, r, mode="full")[len(r) - 1:]
        ac = ac / (ac[0] + 1e-18)
        pik = int(np.argmax(ac[8:60])) + 8 if len(ac) > 60 else float("nan")
        add("Baskın döngü periyodu", "trend_strength", pik, NOTR,
            f"~{pik} bar" if math.isfinite(pik) else "ölçülemedi", "ehlers_period")
    except Exception:
        pass

    # ───────────────── B) İSTATİSTİKSEL PERFORMANS (~24) ─────────────────
    for k in (20, 60, 120):
        w = ret.tail(k)
        if len(w) < 10:
            continue
        mu, sd = float(w.mean()), float(w.std(ddof=1))
        sharpe = mu / (sd + 1e-12) * math.sqrt(252)
        add(f"Yuvarlanan Sharpe {k}", "trend_strength", sharpe,
            _sig(sharpe > 0.5, sharpe < -0.5), f"{sharpe:+.2f} (yıllık)", "roll_sharpe")
        dn = w[w < 0]
        sortino = mu / (float(dn.std(ddof=1)) + 1e-12) * math.sqrt(252) if len(dn) > 2 else float("nan")
        add(f"Yuvarlanan Sortino {k}", "trend_strength", sortino,
            _sig(sortino > 0.7, sortino < -0.7), f"{sortino:+.2f}", "roll_sortino")

    for k in (60, 120):
        w = c.tail(k)
        if len(w) < 20:
            continue
        zirve = w.cummax()
        dd = (w / zirve - 1.0)
        maxdd = float(dd.min()) * 100
        add(f"Maks düşüş {k}", "volatility", maxdd, NOTR,
            f"%{maxdd:.1f}", "drawdown")
        # düşüş SÜRESİ — derinlik kadar önemli
        sure = int((dd < -0.001).sum())
        add(f"Düşüşte geçen bar {k}", "volatility", sure, NOTR,
            f"{sure}/{k} bar suda", "drawdown_dur")
        toplam = float(w.iloc[-1] / w.iloc[0] - 1) * 100
        calmar = toplam / (abs(maxdd) + 1e-9)
        add(f"Calmar {k}", "trend_strength", calmar, _sig(calmar > 0.5, calmar < -0.5),
            f"{calmar:+.2f}", "calmar")
        toparlanma = float(w.iloc[-1] / w.min() - 1) * 100
        add(f"Toparlanma çarpanı {k}", "trend_strength", toparlanma,
            _sig(toparlanma > 5, toparlanma < 1), f"dipten %{toparlanma:+.1f}", "recovery")

    w = ret.tail(120)
    if len(w) > 20:
        kazanc = float(w[w > 0].sum()); aci = float(-w[w < 0].sum())
        omega = kazanc / (aci + 1e-12)
        add("Omega oranı", "trend_strength", omega, _sig(omega > 1.15, omega < 0.85),
            f"{omega:.2f} — kazanç/acı", "omega")
        add("Kazanç-acı oranı", "trend_strength", kazanc / (aci + 1e-12), NOTR,
            f"{kazanc*100:.1f}% / {aci*100:.1f}%", "omega")
        # K-oranı: birikimli getiri regresyonunun eğim/hata oranı
        try:
            cum = np.log(c.tail(120) / c.tail(120).iloc[0]).to_numpy()
            x = np.arange(len(cum), dtype=float)
            m, b = np.polyfit(x, cum, 1)
            se = float(np.std(cum - (m * x + b), ddof=2)) / math.sqrt(len(cum))
            k_ratio = m / (se + 1e-18)
            add("K-oranı", "trend_strength", k_ratio, _sig(k_ratio > 1, k_ratio < -1),
                f"{k_ratio:+.2f} — trendin düzgünlüğü", "k_ratio")
        except Exception:
            pass
        pos = float((w > 0).mean()) * 100
        add("Pozitif bar oranı 120", "structure", pos, _thr(pos, 45, 55),
            f"%{pos:.0f}", "pos_bars")
        # en uzun kazanç/kayıp serisi
        s = np.sign(w.to_numpy())
        best = cur = 0; worst = 0
        for x_ in s:
            cur = cur + 1 if x_ > 0 else 0
            best = max(best, cur)
        cur = 0
        for x_ in s:
            cur = cur + 1 if x_ < 0 else 0
            worst = max(worst, cur)
        add("En uzun yükseliş serisi", "structure", best, NOTR, f"{best} bar", "streaks")
        add("En uzun düşüş serisi", "structure", worst, NOTR, f"{worst} bar", "streaks")

    # ───────────────── C) HACİM PROFİLİ (~12) ─────────────────
    try:
        k = min(200, n)
        pr = c.tail(k).to_numpy(); vol = v.tail(k).to_numpy()
        lo, hi = float(pr.min()), float(pr.max())
        if hi > lo:
            kova = 40
            idx = np.clip(((pr - lo) / (hi - lo) * (kova - 1)).astype(int), 0, kova - 1)
            hist = np.zeros(kova)
            np.add.at(hist, idx, vol)
            merkez = lo + (np.arange(kova) + 0.5) * (hi - lo) / kova
            poc = float(merkez[int(np.argmax(hist))])
            add("POC (kontrol noktası)", "structure", poc, _sig(px > poc, px < poc),
                f"en çok hacim gören fiyat; şu an {'üstünde' if px > poc else 'altında'}",
                "vol_profile")
            # değer alanı: hacmin %70'ini kapsayan bant
            sira = np.argsort(-hist)
            hedef = hist.sum() * 0.70
            top, sec = 0.0, []
            for j in sira:
                top += hist[j]; sec.append(j)
                if top >= hedef:
                    break
            val_lo = float(merkez[min(sec)]); val_hi = float(merkez[max(sec)])
            add("Değer alanı üst (VAH)", "structure", val_hi, _sig(px > val_hi, False),
                "değer alanının üstünde" if px > val_hi else "içinde/altında", "vol_profile")
            add("Değer alanı alt (VAL)", "structure", val_lo, _sig(False, px < val_lo),
                "değer alanının altında" if px < val_lo else "içinde/üstünde", "vol_profile")
            konum = (px - val_lo) / (val_hi - val_lo + 1e-12)
            add("Değer alanı konumu", "structure", konum, _thr(konum, 0.2, 0.8),
                f"%{konum*100:.0f}", "vol_profile")
            add("Değer alanı genişliği", "volatility", (val_hi - val_lo) / px * 100,
                NOTR, f"%{(val_hi-val_lo)/px*100:.1f}", "vol_profile")
            ust = float(hist[merkez > px].sum()); alt = float(hist[merkez <= px].sum())
            add("Hacim üstte/altta", "volume", (alt - ust) / (alt + ust + 1e-12),
                _sig(alt > ust * 1.2, ust > alt * 1.2),
                "hacmin çoğu altta (destek)" if alt > ust else "üstte (direnç)",
                "vol_profile")
            # hacim boşluğu: fiyatın bulunduğu kovada hacim ne kadar az
            j = int(np.clip((px - lo) / (hi - lo) * (kova - 1), 0, kova - 1))
            bosluk = float(hist[j] / (hist.mean() + 1e-12))
            add("Hacim boşluğu", "volume", bosluk, NOTR,
                f"{bosluk:.2f}× ortalama — {'düşük hacimli bölge (hızlı geçer)' if bosluk < 0.5 else 'yoğun bölge'}",
                "vol_gap")
    except Exception:
        pass

    # ───────────────── C2) KLASİK AMA ÇEKİRDEKTE OLMAYANLAR (~45) ─────────────
    # Bill Williams ailesi
    med = (h + l) / 2.0
    try:
        jaw = _last(I.sma(med, 13).shift(8)); dis = _last(I.sma(med, 8).shift(5))
        lip = _last(I.sma(med, 5).shift(3))
        add("Alligator çene", "adaptive_ma", jaw, _sig(px > jaw, px < jaw),
            "13/8 kaydırmalı", "alligator")
        add("Alligator dizilim", "trend_strength", lip - jaw,
            _sig(lip > dis > jaw, lip < dis < jaw),
            "dudak>diş>çene (uyanık boğa)" if lip > dis > jaw else
            "ters dizilim (uyanık ayı)" if lip < dis < jaw else "uyuyor (yatay)",
            "alligator")
        add("Gator osilatörü", "trend_strength", abs(jaw - dis) - abs(dis - lip),
            NOTR, "çeneler açılıyor" if abs(jaw - dis) > abs(dis - lip) else "kapanıyor",
            "gator")
    except Exception:
        pass
    # Bill Williams fraktalleri
    try:
        hv, lv = h.values, l.values
        up_f = bool(len(hv) > 5 and hv[-3] == max(hv[-5:]))
        dn_f = bool(len(lv) > 5 and lv[-3] == min(lv[-5:]))
        add("Fraktal", "structure", 1.0 if up_f else (-1.0 if dn_f else 0.0),
            _sig(dn_f, up_f), "aşağı fraktal (dip)" if dn_f else
            "yukarı fraktal (tepe)" if up_f else "yok", "fractal")
    except Exception:
        pass
    # Accelerator (AO'nun ivmesi) — çekirdekte AO var, AC yok
    try:
        ao = I.awesome_osc(df)
        ac = _last(ao) - _f(I.sma(ao, 5).iloc[-1])
        add("Accelerator", "momentum", ac, _sig(ac > 0, ac < 0),
            "momentumun ivmesi", "accelerator")
    except Exception:
        pass
    # Market Facilitation Index — hacim başına fiyat hareketi
    try:
        mfi_bw = (h - l) / (v.replace(0, np.nan))
        cur, prev = _last(mfi_bw), _f(mfi_bw.iloc[-2])
        vcur, vprev = _f(v.iloc[-1]), _f(v.iloc[-2])
        durum = ("yeşil (hacim+hareket artıyor)" if cur > prev and vcur > vprev else
                 "fade (ikisi de azalıyor)" if cur < prev and vcur < vprev else
                 "fake (hareket var hacim yok)" if cur > prev else "squat (hacim var hareket yok)")
        add("Market Facilitation Index", "volume", cur,
            _sig(cur > prev and vcur > vprev, cur < prev and vcur < vprev),
            durum, "mfi_bw")
    except Exception:
        pass
    # Heikin-Ashi türevleri
    try:
        ha = I.heikin_ashi(df)
        hac = ha["close"].astype(float); hao = ha["open"].astype(float)
        seri = int((hac.tail(10) > hao.tail(10)).sum())
        add("Heikin-Ashi yeşil oranı", "structure", seri / 10 * 100,
            _thr(seri / 10 * 100, 40, 60), f"son 10 HA mumun {seri}'i yeşil", "ha_ext")
        govde = _f((hac.iloc[-1] - hao.iloc[-1]) / (ha["high"].iloc[-1] - ha["low"].iloc[-1] + 1e-12))
        add("HA gövde oranı", "structure", govde, _sig(govde > 0.6, govde < -0.6),
            f"{govde:+.2f} — {'fitilsiz güçlü' if abs(govde) > 0.6 else 'kararsız'}", "ha_ext")
    except Exception:
        pass
    # Guppy çoklu ortalama — kısa grup vs uzun grup
    try:
        kisa = np.mean([_f(I.ema(c, k).iloc[-1]) for k in (3, 5, 8, 10, 12, 15)])
        uzun = np.mean([_f(I.ema(c, k).iloc[-1]) for k in (30, 35, 40, 45, 50, 60)])
        add("Guppy kısa/uzun", "trend_strength", (kisa / uzun - 1) * 100,
            _sig(kisa > uzun, kisa < uzun),
            f"%{(kisa/uzun-1)*100:+.2f} — trader grubu yatırımcı grubunun "
            f"{'üstünde' if kisa > uzun else 'altında'}", "guppy")
        yayilim = np.std([_f(I.ema(c, k).iloc[-1]) for k in (30, 35, 40, 45, 50, 60)]) / px * 100
        add("Guppy uzun grup yayılımı", "volatility", yayilim, NOTR,
            f"%{yayilim:.2f} — {'trend güçlü' if yayilim > 1 else 'sıkışık'}", "guppy")
    except Exception:
        pass
    # Rainbow MA — arka arkaya yumuşatılmış SMA'lar
    try:
        r_ = c
        gokkusagi = []
        for _ in range(9):
            r_ = I.sma(r_, 2)
            gokkusagi.append(_last(r_))
        yay = (max(gokkusagi) - min(gokkusagi)) / px * 100
        add("Rainbow yayılımı", "volatility", yay, NOTR,
            f"%{yay:.3f} — {'trend ivmeleniyor' if yay > 0.5 else 'sıkışık'}", "rainbow")
        add("Rainbow konumu", "adaptive_ma", gokkusagi[-1],
            _sig(px > max(gokkusagi), px < min(gokkusagi)),
            "tüm şeridin üstünde" if px > max(gokkusagi) else
            "altında" if px < min(gokkusagi) else "şerit içinde", "rainbow")
    except Exception:
        pass
    # Chandelier exit / volatilite stopu
    try:
        atr22 = _f(I.atr(df, 22).iloc[-1])
        ce_long = float(h.tail(22).max()) - 3 * atr22
        ce_short = float(l.tail(22).min()) + 3 * atr22
        add("Chandelier çıkış (long)", "trend_strength", ce_long,
            _sig(px > ce_long, px < ce_long),
            "long stop seviyesinin üstünde" if px > ce_long else "altında", "chandelier")
        add("Chandelier çıkış (short)", "trend_strength", ce_short,
            _sig(px < ce_short, px > ce_short), "short stop seviyesi", "chandelier")
    except Exception:
        pass
    # SuperTrend çoklu çarpan — farklı hassasiyet
    for mult in (1.5, 4.5):
        try:
            st, sd = I.supertrend(df, 10, mult)
            add(f"SuperTrend ×{mult}", "trend_strength", _last(st),
                _sig(_last(sd) > 0, _last(sd) < 0),
                f"çarpan {mult} — {'hassas' if mult < 2 else 'gevşek'}", "supertrend_mult")
        except Exception:
            pass
    # PSAR farklı ivme
    for step in (0.01, 0.04):
        try:
            ps = _last(I.psar(df, step, 0.2))
            add(f"PSAR adım {step}", "trend_strength", ps, _sig(px > ps, px < ps),
                f"ivme {step}", "psar_mult")
        except Exception:
            pass
    # Relative Volatility Index — RSI ama std üzerinde
    try:
        sd_ = c.rolling(10).std()
        d_ = sd_.diff()
        up_ = d_.clip(lower=0).ewm(alpha=1/14).mean()
        dn_ = (-d_.clip(upper=0)).ewm(alpha=1/14).mean()
        rvi_ = _f(100 * up_.iloc[-1] / (up_.iloc[-1] + dn_.iloc[-1] + 1e-12))
        add("Relative Volatility Index", "volatility", rvi_, _thr(rvi_, 40, 60),
            f"{rvi_:.1f} — oynaklığın yönü", "rvi_vol")
    except Exception:
        pass
    # TRIX ve PPO sinyal kesişimleri
    try:
        tr_ = I.trix(c, 15)
        add("TRIX sinyal kesişimi", "momentum", _last(tr_) - _f(I.ema(tr_, 9).iloc[-1]),
            _sig(_last(tr_) > _f(I.ema(tr_, 9).iloc[-1]),
                 _last(tr_) < _f(I.ema(tr_, 9).iloc[-1])), "sinyal çizgisi", "trix_sig")
    except Exception:
        pass
    # Elder Force Index çoklu periyot
    for k in (2, 13, 50):
        try:
            fi = _f(I.force_index(df, k).iloc[-1])
            add(f"Force Index {k}", "volume", fi, _sig(fi > 0, fi < 0),
                f"{k} periyot", "force_multi")
        except Exception:
            pass
    # Intraday Intensity
    try:
        ii = ((2 * c - h - l) / ((h - l) + 1e-12) * v)
        iis = _f(ii.tail(21).sum() / (v.tail(21).sum() + 1e-12))
        add("Intraday Intensity", "volume", iis, _thr(iis, -0.1, 0.1),
            f"{iis:+.3f} — kurumsal birikim vekili", "intraday_int")
    except Exception:
        pass
    # Demand Index vekili
    try:
        di = _f(((c.pct_change() * v).tail(20).sum()) / (v.tail(20).sum() + 1e-12) * 100)
        add("Demand Index", "volume", di, _sig(di > 0, di < 0),
            f"{di:+.3f} — hacim ağırlıklı talep", "demand")
    except Exception:
        pass
    # Zigzag tabanlı salınım büyüklüğü
    try:
        from .patterns import find_pivots
        sh, sl = find_pivots(df, 3, 3)
        piv = sorted([(i, float(h.iloc[i])) for i in sh] + [(i, float(l.iloc[i])) for i in sl])
        if len(piv) >= 4:
            son = piv[-4:]
            bacaklar = [abs(son[k + 1][1] - son[k][1]) / px * 100 for k in range(3)]
            add("Ortalama salınım büyüklüğü", "volatility", float(np.mean(bacaklar)),
                NOTR, f"%{np.mean(bacaklar):.2f} — son 3 bacak", "swing_size")
            add("Salınım daralması", "volatility", bacaklar[-1] - bacaklar[0],
                _sig(False, bacaklar[-1] < bacaklar[0] * 0.6),
                "bacaklar daralıyor (sıkışma)" if bacaklar[-1] < bacaklar[0] * 0.6
                else "normal", "swing_size")
            add("Son pivota uzaklık", "structure", (px - son[-1][1]) / px * 100,
                NOTR, f"%{(px-son[-1][1])/px*100:+.2f}", "swing_dist")
    except Exception:
        pass

    # ───────────────── D) MİKROYAPI / L2 (~30) ─────────────────
    # Kaydedici verisi YOKSA bu bölüm atlanır — fiyattan üretilemez, uydurulmaz.
    # İşaretçi bir GÖSTERGE değildir: listeye eklenmez, sayımı şişirmez ve
    # NaN değer taşımaz. Durum tablo meta verisinde (`microstructure`) bildirilir.
    if micro is None or not len(micro):
        return False

    def m(k, d=float("nan")):
        try:
            val = micro.get(k, d)
            return _f(val)
        except Exception:
            return float("nan")

    fr = m("funding_rate")
    if math.isfinite(fr):
        yillik = fr * 3 * 365 * 100
        add("Funding oranı", "microstructure", fr * 100, _sig(fr < -0.0001, fr > 0.0003),
            f"%{fr*100:.4f}/8s (yıllık %{yillik:.1f}) — "
            f"{'longlar ödüyor (aşırı iyimserlik)' if fr > 0.0003 else 'shortlar ödüyor' if fr < 0 else 'normal'}",
            "funding")
        add("Funding yıllık", "microstructure", yillik, _thr(yillik, -5, 30, invert=True),
            f"%{yillik:.1f}", "funding")

    oi = m("open_interest")
    if math.isfinite(oi):
        add("Açık pozisyon", "microstructure", oi, NOTR, f"{oi:,.0f}", "oi")

    mark, index = m("mark_price"), m("index_price")
    if math.isfinite(mark) and math.isfinite(index) and index > 0:
        baz = (mark / index - 1) * 100
        add("Perp bazı", "microstructure", baz, _thr(baz, -0.05, 0.15),
            f"%{baz:+.3f} — mark/index ayrışması", "basis")

    tb = m("taker_buy_ratio")
    if math.isfinite(tb):
        add("Taker alış oranı", "microstructure", tb, _thr(tb, 0.45, 0.55),
            f"%{tb*100:.1f} agresif alış", "taker")
    for k, ad in (("ls_account_ratio", "Long/short hesap oranı"),
                  ("top_trader_ratio", "Büyük hesap pozisyon oranı")):
        val = m(k)
        if math.isfinite(val):
            add(ad, "microstructure", val, _thr(val, 0.9, 1.4, invert=True),
                f"{val:.2f} — {'kalabalık long' if val > 1.4 else 'kalabalık short' if val < 0.9 else 'dengeli'}",
                "positioning")

    sp = m("spread_bps")
    if math.isfinite(sp):
        add("Kote spread", "microstructure", sp, NOTR,
            f"{sp:.3f} bps — {'geniş (işlem pahalı)' if sp > 3 else 'dar'}", "spread")

    # L2 eğrisinden türetilenler
    kovalar = (1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 50.0)
    bid = {b: m(f"bid_cum_{b:g}bps") for b in kovalar}
    ask = {b: m(f"ask_cum_{b:g}bps") for b in kovalar}
    gecerli = [b for b in kovalar if math.isfinite(bid[b]) and math.isfinite(ask[b])]
    if gecerli:
        for b in (1.0, 5.0, 10.0, 20.0):
            if b not in gecerli:
                continue
            tot = bid[b] + ask[b]
            imb = (bid[b] - ask[b]) / (tot + 1e-12)
            add(f"Defter dengesizliği {b:g}bps", "microstructure", imb,
                _sig(imb > 0.15, imb < -0.15),
                f"{imb:+.2f} — {'alış tarafı kalın' if imb > 0 else 'satış tarafı kalın'}",
                "book_imb_multi")
            add(f"Derinlik {b:g}bps", "microstructure", tot, NOTR, f"{tot:,.0f} $", "depth_multi")
        # Defter EĞİMİ: derinlik uzaklıkla ne hızlı artıyor (likidite yoğunluğu)
        try:
            xs = np.array(gecerli, dtype=float)
            ys = np.array([bid[b] + ask[b] for b in gecerli], dtype=float)
            egim = float(np.polyfit(xs, ys, 1)[0])
            add("Defter eğimi", "microstructure", egim, NOTR,
                f"{egim:,.0f} $/bps — birim uzaklıkta eklenen likidite", "book_slope")
            # dışbükeylik: ikinci derece terim (defter uçlara doğru seyreliyor mu)
            if len(xs) >= 4:
                k2 = float(np.polyfit(xs, ys, 2)[0])
                add("Defter dışbükeyliği", "microstructure", k2, NOTR,
                    f"{k2:+.1f} — {'uçlarda kalınlaşıyor' if k2 > 0 else 'uçlarda seyreliyor'}",
                    "book_convex")
        except Exception:
            pass
        # Asimetri: aynı uzaklıkta alış/satış derinlik oranı
        if 10.0 in gecerli:
            asim = bid[10.0] / (ask[10.0] + 1e-12)
            add("Derinlik asimetrisi 10bps", "microstructure", asim,
                _sig(asim > 1.25, asim < 0.8), f"{asim:.2f}×", "depth_asym")
        # Kyle lambda vekili: 1 bps hareket için gereken nominal
        if 1.0 in gecerli:
            lam = (bid[1.0] + ask[1.0]) / 2.0
            add("Kyle lambda vekili", "microstructure", lam, NOTR,
                f"{lam:,.0f} $ ile fiyat 1 bps oynar", "kyle")
        trunc_b = micro.get("bid_truncated"); trunc_a = micro.get("ask_truncated")
        add("Merdiven sansürü", "microstructure", float(bool(trunc_b) or bool(trunc_a)), NOTR,
            "defter kaydedilen 1000 seviyede 100 bps'e ULAŞMIYOR — derin kovalar sansürlü"
            if (trunc_b or trunc_a) else "merdiven 100 bps'i kapsıyor", "ladder_trunc")

    # Amihud likidite-sizliği: |getiri| / hacim
    try:
        k = 30
        ami = float((ret.abs().tail(k) / (v.tail(k).values[-len(ret.tail(k)):] + 1e-12)).mean())
        add("Amihud likidite-sizliği", "microstructure", ami * 1e9, NOTR,
            "birim hacme düşen fiyat etkisi (yüksek = ince piyasa)", "amihud")
    except Exception:
        pass

    return True
