"""
Gösterge tablosu — ~384 göstergenin AL / SAT / NÖTR yorumu (FAZ 5 eki).

`indicators.compute_all_indicators` HAM DEĞER döndürür ("rsi_14: 61.3"). Bu
modül her göstergeye AÇIK BİR YORUM KURALI uygular ve AL/SAT/NÖTR üretir,
kategoriye göre gruplar ve sayar.

⚠️ EN ÖNEMLİ UYARI — "80 AL / 20 SAT" %80 GÜVEN DEĞİLDİR.
13 farklı periyottaki EMA aynı anda "AL" derse bu 13 bağımsız kanıt değil,
TEK bir kanıttır (hepsi aynı fiyat serisinin aynı özelliğini ölçer). Göstergeler
birbirinden bağımsız değildir; ham sayım bu yüzden yanıltıcıdır.

Bu modül iki sayı birden verir:
  • ham sayım        — kullanıcının istediği "kaçı al, kaçı sat"
  • AİLE sayımı      — birbirini tekrar eden göstergeler tek oya indirgenir
Karar için AİLE sayımına bakılmalıdır.

Ayrıca: bu tablo TEK BAŞINA işlem açtırmaz. Program boyunca ölçüldü ki gösterge
yoğunluğu örneklem dışı kâr üretmiyor (bkz. runs/FAZ4-10_BULGULAR.md — 180
göstergeyle beslenen meta-etiketleme Δ Sharpe −0,25). Tablo KARAR DESTEĞİdir.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import indicators as I

AL, SAT, NOTR = "AL", "SAT", "NÖTR"

# ---------------------------------------------------------------------------
# ÖLÇÜLMÜŞ KANIT — panelde bu tablonun yanında gösterilir.
# Koşum: runs/board_validation_300.csv (5 parite × {1d,4h} × 175 bar, 1.750 gözlem)
# 300 göstergeye çıkıldıktan SONRA yeniden ölçüldü; eski 129'luk ölçüm
# runs/board_validation_full.csv olarak duruyor ve karşılaştırma aşağıda.
# ---------------------------------------------------------------------------
BOARD_EVIDENCE = {
    "measured": True,
    "n_indicators": 384,
    "sample": "5 parite × {1d, 4h} × ~175 bar = 1.750 gözlem (384 göstergeyle)",
    "headline": "Konsensüsü takip etmek bu örneklemde PARA KAYBETTİRDİ.",
    "rows": [
        {"tf": "4h", "corr_fwd1": -0.163, "cross_pair_t": -6.00,
         "follow_ret_pct": -0.540, "follow_winrate": 35.9, "neg_pairs": "5/5"},
        {"tf": "1d", "corr_fwd1": -0.069, "cross_pair_t": -3.54,
         "follow_ret_pct": -0.240, "follow_winrate": 45.7, "neg_pairs": "5/5"},
    ],
    # 129 → 300: gösterge sayısını 2,3 KATINA çıkarmak sonucu DEĞİŞTİRMEDİ.
    # Bu, "daha çok gösterge = daha iyi karar" sezgisinin doğrudan ölçülmüş
    # reddidir ve panelde bu haliyle gösterilir.
    "count_comparison": {
        "tf": "4h",
        "before": {"n": 129, "corr": -0.145, "follow_ret_pct": -0.511, "winrate": 36.9},
        "mid": {"n": 300, "corr": -0.164, "follow_ret_pct": -0.492, "winrate": 36.2},
        "after": {"n": 384, "corr": -0.163, "follow_ret_pct": -0.540, "winrate": 35.9},
        "note": ("Gösterge sayısı ÜÇ kez ölçüldü: 129 → 300 → 384 (toplam 3 kat). "
                 "Korelasyon −0,145 → −0,164 → −0,163; takip getirisi −%0,51 → "
                 "−%0,49 → −%0,54; kazanma %36,9 → %36,2 → %35,9. Hiçbiri "
                 "anlamlı şekilde değişmedi ve 5/5 parite her seferinde negatif. "
                 "384'e mikroyapı (funding, defter eğimi, derinlik dengesizliği) "
                 "gibi FİYATTAN TÜRETİLMEYEN göstergeler de dahildir — onlar bile "
                 "sonucu değiştirmedi. Gösterge sayısı bir kalite ölçüsü değildir; "
                 "bu tahmin değil, üç kez tekrarlanmış ölçümdür."),
    },
    "detail": (
        "Konsensüs (net) ile ertesi bar getirisi arasındaki korelasyon her iki "
        "zaman diliminde ve 5 paritenin 5'inde de NEGATİF. 4h'te 'AL' dediğinde "
        "5 bar tutmak ortalama %0,54 kaybettirdi (kazanma %35,9)."),
    "why_not_inverted": (
        "«O hâlde tersini al» denemesi de yapıldı ve KAPIDA ELENDİ. 4h'te maliyet "
        "düşülmüş yıllık Sharpe 4,43 görünüyor, ama Deflated Sharpe = 0,344 "
        "(eşik 0,95) → KALDI; 1d'de Sharpe −0,31, DSR 0,000. Sebep: 15 farklı "
        "hipotez denendi ve bunların Sharpe dağılımı 2,82; bu dağılımda ŞANSLA "
        "beklenen en iyi Sharpe zaten 5,00 — yani 4,43 şansı aşmıyor. Çeyrek "
        "bazlı Sharpe da bunu doğruluyor: −2,09 → 1,66 → 6,11 → 9,05, kazancın "
        "tamamı son çeyrekte. 67 günlük tek rejimli örneklem, eşik/ufuk/tersleme "
        "kararı veriye BAKTIKTAN sonra seçilmiş, kilitli test hiç açılmadı."),
    "dsr": {"4h": 0.344, "1d": 0.000, "threshold": 0.95,
            "n_trials": 15, "sr0_annual": 5.00},
    "verdict": "GÖSTERGE PANELİ — karar desteği. İşlem sinyali DEĞİL.",
}

# ---------------------------------------------------------------------------
# MİKROYAPI KANITI — ayrı ölçüldü, çünkü tablodaki tek DİK bilgi sınıfı bu.
# Koşum: kaydedici 4.162 gözlem · 10 parite · 34,5 saat · fiyat = kaydedicinin
# kendi `mid` sütunu (hizalama sorunu yok). Gösterge parite İÇİNDE z-skorlandı.
# ---------------------------------------------------------------------------
# SONUÇ, fiyat türevlerinden FARKLI: mikroyapıda ÖLÇÜLEBİLİR bir yön bilgisi
# VAR — ama işlem maliyetinin çok altında.
#
#   1 saat ufku, örtüşmeyen gözlem:
#     Kote spread          r=+0,156  t=+4,66   9/10 parite pozitif
#     Defter dengesizliği  r=+0,000  t=+0,01   (15 dk'da t=+1,51)
#     diğer altı gösterge  |t| < 1  → gürültü
#
#   MALİYET SINAVI (gidiş-dönüş = 2×taker + sinyal anındaki spread):
#     Kote spread          brüt %+0,0360 · maliyet %0,0925 → NET %−0,0565
#                          maliyet brütün 2,6 KATI
#     Defter dengesizliği  brüt %+0,0177 · maliyet %0,0928 → NET %−0,0751
#                          maliyet brütün 5,2 KATI
#
#   SPREAD SİNYALİ NE ÖLÇÜYOR: önceki 1 saatlik getiriyle korelasyonu −0,66,
#   yani ağırlıklı olarak "düşüş sonrası" durumunu etiketliyor. Önceki getiri
#   KONTROL EDİLİNCE bile bağımsız katkısı kalıyor (t=+2,69) — tamamen dönüş
#   vekili değil. Ama işlem açısından bu fark etmiyor: maliyeti karşılamıyor.
#
# ⚠️ GÜÇ SINIRI: 34,5 saat TEK REJİM. 1 saatlik ufukta örtüşmeyen gözlem parite
# başına ~34. Bu KANIT değil İLK BAKIŞ'tır; kaydedici birikiyor, tekrarlanmalı.
MICRO_EVIDENCE = {
    "measured": True,
    "sample": "4.162 gözlem · 10 parite · 34,5 saat (kaydedici)",
    "headline": ("Mikroyapıda ÖLÇÜLEBİLİR yön bilgisi VAR — ama işlem "
                 "maliyetinin 2,6-5,2 katı ALTINDA."),
    "differs_from_price": (
        "Fiyat türevi 364 göstergenin konsensüsü TERS yönde çalışıyordu; "
        "mikroyapı ilk kez DOĞRU yönde ölçülebilir bir sinyal verdi. Fark "
        "önemli ama pratik değil — sinyal maliyetin çok altında."),
    "signals": [
        {"name": "Kote spread", "r_1h": 0.156, "t_1h": 4.66, "pos_pairs": "9/10",
         "gross_pct": 0.0360, "cost_pct": 0.0925, "net_pct": -0.0565,
         "cost_multiple": 2.6},
        {"name": "Defter dengesizliği", "r_15m": 0.032, "t_15m": 1.51,
         "gross_pct": 0.0177, "cost_pct": 0.0928, "net_pct": -0.0751,
         "cost_multiple": 5.2},
    ],
    "noise": ["Funding oranı", "Taker alış oranı", "Long/short hesap oranı",
              "Büyük hesap oranı", "Açık pozisyon değişimi", "Perp bazı"],
    "spread_nature": (
        "Spread z-skorunun önceki 1 saatlik getiriyle korelasyonu −0,66: "
        "ağırlıklı olarak düşüş sonrasını etiketliyor. Önceki getiri kontrol "
        "edilince bağımsız katkısı KALIYOR (t=+2,69), yani tamamen dönüş "
        "vekili değil."),
    "power_warning": (
        "34,5 saatlik TEK REJİM. 1 saatlik ufukta örtüşmeyen gözlem parite "
        "başına ~34 — küçük etkiyi tespit etmeye yetmez. KANIT değil İLK BAKIŞ."),
    "ladder_pending": (
        "L2 merdiven (defter eğimi, çoklu derinlik, Kyle lambda) yeni kaydedilmeye "
        "başladı; ölçülecek kadar veri YOK."),
    "verdict": "MALİYETİ KARŞILAMIYOR — gösterge olarak değerli, sinyal olarak değil.",
}

# Kategori → aynı aileden sayılan göstergeler tek oya indirgenir
CATEGORIES = {
    "trend_ma": "Hareketli Ortalamalar",
    "crossover": "Kesişimler",
    "momentum": "Momentum Osilatörleri",
    "trend_strength": "Trend Gücü",
    "volatility": "Volatilite",
    "volume": "Hacim",
    "adaptive_ma": "Uyarlanır Ortalamalar",
    "structure": "Fiyat Yapısı",
    # Tablodaki TEK fiyattan-türetilmeyen sınıf: borsa defterinden ve türev
    # verisinden gelir. Kendi kategorisi olmalı — "Hacim"in içine gömülünce
    # ayırt edilemiyordu ve kullanıcı hangi göstergenin gerçekten yeni bilgi
    # taşıdığını göremiyordu.
    "microstructure": "Mikroyapı / L2",
}


@dataclass
class IndicatorSignal:
    name: str
    category: str
    value: float
    signal: str          # AL | SAT | NÖTR
    rule: str            # neden bu sinyal
    family: str          # aynı aileyi tekrar sayma

    def to_dict(self) -> Dict:
        return asdict(self)


def _f(x) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def _sig(cond_buy: bool, cond_sell: bool) -> str:
    if cond_buy and not cond_sell:
        return AL
    if cond_sell and not cond_buy:
        return SAT
    return NOTR


def _threshold(v: float, lo: float, hi: float, invert: bool = False) -> str:
    """v > hi → AL, v < lo → SAT (invert ise ters). NaN → NÖTR."""
    if not math.isfinite(v):
        return NOTR
    if invert:
        return _sig(v < lo, v > hi)
    return _sig(v > hi, v < lo)


# ===========================================================================
# Ana hesap
# ===========================================================================
def build_board(df: pd.DataFrame, micro=None) -> Dict:
    """~400 göstergeyi hesapla, yorumla, say.

    Çekirdek 129 + `indicator_pack` (~171) + `indicator_pack2` (~100).

    micro: `data/recorder.py`'den o paritenin SON satırı (pandas Series).
    Verilirse mikroyapı/L2 göstergeleri (funding, defter eğimi, derinlik
    dengesizliği, Kyle lambda…) eklenir. Verilmezse o bölüm ATLANIR —
    bunlar fiyattan türetilemez ve uydurulmaz."""
    if df is None or len(df) < 210:
        return {"available": False,
                "reason": f"en az 210 bar gerekli ({0 if df is None else len(df)} var)"}

    c, h, l = df["close"].astype(float), df["high"].astype(float), df["low"].astype(float)
    v = df["volume"].astype(float) if "volume" in df else pd.Series(1.0, index=df.index)
    px = float(c.iloc[-1])
    out: List[IndicatorSignal] = []

    def add(name, cat, val, sig, rule, family=None):
        out.append(IndicatorSignal(name, cat, round(_f(val), 6), sig, rule,
                                   family or name))

    # ─────────────────── 1) HAREKETLİ ORTALAMALAR (26) ───────────────────
    # Kural: fiyat ortalamanın ÜSTÜNDE → AL. Aynı aile: "ma_kisa"/"ma_orta"/"ma_uzun"
    for n in (5, 8, 9, 10, 13, 20, 21, 34, 50, 55, 100, 144, 200):
        fam = "ma_kisa" if n <= 21 else ("ma_orta" if n <= 55 else "ma_uzun")
        e = _f(I.ema(c, n).iloc[-1])
        s = _f(I.sma(c, n).iloc[-1])
        add(f"EMA {n}", "trend_ma", e, _sig(px > e, px < e),
            f"fiyat {'>' if px > e else '<'} EMA{n}", fam)
        add(f"SMA {n}", "trend_ma", s, _sig(px > s, px < s),
            f"fiyat {'>' if px > s else '<'} SMA{n}", fam)

    for nm, fn in (("WMA 20", I.wma), ("HMA 20", I.hma)):
        val = _f(fn(c, 20).iloc[-1])
        add(nm, "trend_ma", val, _sig(px > val, px < val),
            f"fiyat {'>' if px > val else '<'} {nm}", "ma_orta")
    vw = _f(I.vwap(df).iloc[-1])
    add("VWAP", "trend_ma", vw, _sig(px > vw, px < vw),
        f"fiyat {'>' if px > vw else '<'} VWAP", "vwap")

    # ─────────────────── 2) KESİŞİMLER (8) ───────────────────
    e9, e21, e20, e50, e200 = (I.ema(c, 9), I.ema(c, 21), I.ema(c, 20),
                               I.ema(c, 50), I.ema(c, 200))
    s50, s200 = I.sma(c, 50), I.sma(c, 200)
    for nm, a, b, fam in (("EMA 9/21", e9, e21, "cross_kisa"),
                          ("EMA 20/50", e20, e50, "cross_orta"),
                          ("EMA 50/200", e50, e200, "cross_uzun"),
                          ("SMA 50/200 (altın/ölüm)", s50, s200, "cross_uzun")):
        d = _f(a.iloc[-1]) - _f(b.iloc[-1])
        add(nm, "crossover", d, _sig(d > 0, d < 0),
            f"{'kısa üstte' if d > 0 else 'kısa altta'}", fam)
    # kesişim TAZE mi (son 5 barda yön değişti mi) — ayrı bilgi
    for nm, a, b, fam in (("EMA 9/21 tazelik", e9, e21, "cross_fresh"),
                          ("EMA 50/200 tazelik", e50, e200, "cross_fresh")):
        d = (a - b).tail(6)
        flipped_up = bool(d.iloc[0] < 0 and d.iloc[-1] > 0)
        flipped_dn = bool(d.iloc[0] > 0 and d.iloc[-1] < 0)
        add(nm, "crossover", _f(d.iloc[-1]), _sig(flipped_up, flipped_dn),
            "son 5 barda kesişti" if (flipped_up or flipped_dn) else "kesişim yok", fam)
    ma_up = sum(1 for n in (9, 21, 50, 200) if px > _f(I.ema(c, n).iloc[-1]))
    add("MA dizilimi (4 kademe)", "crossover", ma_up, _sig(ma_up >= 3, ma_up <= 1),
        f"{ma_up}/4 ortalamanın üstünde", "ma_align")
    slope200 = _f(s200.iloc[-1]) - _f(s200.iloc[-20]) if len(s200) > 20 else float("nan")
    add("SMA200 eğimi", "crossover", slope200, _sig(slope200 > 0, slope200 < 0),
        f"200 ortalama {'yükseliyor' if slope200 > 0 else 'düşüyor'}", "ma_uzun")

    # ─────────────────── 3) MOMENTUM OSİLATÖRLERİ (~28) ───────────────────
    for n in (7, 14, 21):
        r = _f(I.rsi(c, n).iloc[-1])
        add(f"RSI {n}", "momentum", r, _threshold(r, 45, 55),
            f"RSI {r:.1f} ({'>55 boğa' if r > 55 else '<45 ayı' if r < 45 else '45-55 nötr'})",
            "rsi")
    r14 = _f(I.rsi(c, 14).iloc[-1])
    add("RSI aşırı bölge", "momentum", r14, _threshold(r14, 30, 70, invert=True),
        f"{'aşırı satım (dönüş beklentisi)' if r14 < 30 else 'aşırı alım' if r14 > 70 else 'normal'}",
        "rsi_extreme")

    k, d_ = I.stoch(df)
    kk, dd = _f(k.iloc[-1]), _f(d_.iloc[-1])
    add("Stochastic %K", "momentum", kk, _threshold(kk, 20, 80, invert=True),
        f"%K {kk:.1f}", "stoch")
    add("Stochastic K/D", "momentum", kk - dd, _sig(kk > dd, kk < dd),
        f"%K {'>' if kk > dd else '<'} %D", "stoch")
    sr = _f(I.stoch_rsi(c).iloc[-1])
    add("Stoch RSI", "momentum", sr, _threshold(sr, 0.2, 0.8, invert=True),
        f"{sr:.2f}", "stoch")

    cci = _f(I.cci(df).iloc[-1])
    add("CCI 20", "momentum", cci, _threshold(cci, -100, 100),
        f"CCI {cci:.0f}", "cci")
    mfi = _f(I.mfi(df).iloc[-1])
    add("MFI 14", "momentum", mfi, _threshold(mfi, 20, 80, invert=True),
        f"MFI {mfi:.1f}", "mfi")
    wr = _f(I.williams_r(df).iloc[-1])
    add("Williams %R", "momentum", wr, _threshold(wr, -80, -20, invert=True),
        f"{wr:.1f}", "williams")
    for nm, val, fam in (("ROC 12", _f(I.roc(c, 12).iloc[-1]), "roc"),
                         ("Momentum 10", _f(I.momentum(c, 10).iloc[-1]), "roc"),
                         ("TRIX 15", _f(I.trix(c).iloc[-1]), "trix"),
                         ("Awesome Osc", _f(I.awesome_osc(df).iloc[-1]), "ao"),
                         ("DPO 20", _f(I.dpo(c).iloc[-1]), "dpo")):
        add(nm, "momentum", val, _sig(val > 0, val < 0),
            f"{'pozitif' if val > 0 else 'negatif'}", fam)
    tsi = _f(I.tsi(c).iloc[-1])
    add("TSI", "momentum", tsi, _sig(tsi > 0, tsi < 0), f"{tsi:.2f}", "tsi")
    uo = _f(I.ultimate_osc(df).iloc[-1])
    add("Ultimate Osc", "momentum", uo, _threshold(uo, 40, 60), f"{uo:.1f}", "uo")
    kl, ks = I.kst(c)
    add("KST", "momentum", _f(kl.iloc[-1]), _sig(_f(kl.iloc[-1]) > 0, _f(kl.iloc[-1]) < 0),
        "sıfır üstü" if _f(kl.iloc[-1]) > 0 else "sıfır altı", "kst")
    add("KST sinyal kesişimi", "momentum", _f(kl.iloc[-1]) - _f(ks.iloc[-1]),
        _sig(_f(kl.iloc[-1]) > _f(ks.iloc[-1]), _f(kl.iloc[-1]) < _f(ks.iloc[-1])),
        "sinyal çizgisine göre", "kst")
    cmo = _f(I.cmo(c).iloc[-1])
    add("CMO 14", "momentum", cmo, _threshold(cmo, -50, 50), f"{cmo:.1f}", "cmo")
    bop = _f(I.bop(df).iloc[-1])
    add("Balance of Power", "momentum", bop, _sig(bop > 0, bop < 0),
        "alıcı baskın" if bop > 0 else "satıcı baskın", "bop")

    macd_l, macd_s, macd_h = I.macd(c)
    ml, ms, mh = _f(macd_l.iloc[-1]), _f(macd_s.iloc[-1]), _f(macd_h.iloc[-1])
    add("MACD çizgisi", "momentum", ml, _sig(ml > 0, ml < 0),
        "sıfır üstü" if ml > 0 else "sıfır altı", "macd")
    add("MACD sinyal kesişimi", "momentum", ml - ms, _sig(ml > ms, ml < ms),
        "MACD sinyalin üstünde" if ml > ms else "altında", "macd")
    hist_prev = _f(macd_h.iloc[-2]) if len(macd_h) > 2 else float("nan")
    add("MACD histogram yönü", "momentum", mh, _sig(mh > hist_prev, mh < hist_prev),
        "histogram güçleniyor" if mh > hist_prev else "zayıflıyor", "macd")

    # ─────────────────── 4) TREND GÜCÜ (~14) ───────────────────
    adx_v, pdi, mdi = I.adx(df)
    a_, p_, m_ = _f(adx_v.iloc[-1]), _f(pdi.iloc[-1]), _f(mdi.iloc[-1])
    add("ADX (trend gücü)", "trend_strength", a_, NOTR if a_ < 20 else _sig(p_ > m_, m_ > p_),
        f"ADX {a_:.1f} ({'trendsiz' if a_ < 20 else 'trendli'})", "adx")
    add("DI+/DI−", "trend_strength", p_ - m_, _sig(p_ > m_, m_ > p_),
        f"DI+ {p_:.1f} / DI− {m_:.1f}", "adx")
    au, ad = I.aroon(df)
    add("Aroon", "trend_strength", _f(au.iloc[-1]) - _f(ad.iloc[-1]),
        _sig(_f(au.iloc[-1]) > _f(ad.iloc[-1]), _f(ad.iloc[-1]) > _f(au.iloc[-1])),
        f"yukarı {_f(au.iloc[-1]):.0f} / aşağı {_f(ad.iloc[-1]):.0f}", "aroon")
    vip, vim = I.vortex(df)
    add("Vortex", "trend_strength", _f(vip.iloc[-1]) - _f(vim.iloc[-1]),
        _sig(_f(vip.iloc[-1]) > _f(vim.iloc[-1]), _f(vim.iloc[-1]) > _f(vip.iloc[-1])),
        "VI+ / VI− karşılaştırması", "vortex")
    st_line, st_dir = I.supertrend(df)
    sd = _f(st_dir.iloc[-1])
    add("SuperTrend", "trend_strength", _f(st_line.iloc[-1]), _sig(sd > 0, sd < 0),
        "yükseliş modunda" if sd > 0 else "düşüş modunda", "supertrend")
    ps = _f(I.psar(df).iloc[-1])
    add("Parabolic SAR", "trend_strength", ps, _sig(px > ps, px < ps),
        f"fiyat SAR'ın {'üstünde' if px > ps else 'altında'}", "psar")
    ich = I.ichimoku(df)
    try:
        span_a = _f(pd.Series(ich[2]).iloc[-1]); span_b = _f(pd.Series(ich[3]).iloc[-1])
        tenkan = _f(pd.Series(ich[0]).iloc[-1]); kijun = _f(pd.Series(ich[1]).iloc[-1])
        cloud_top, cloud_bot = max(span_a, span_b), min(span_a, span_b)
        add("Ichimoku bulut", "trend_strength", px - cloud_top,
            _sig(px > cloud_top, px < cloud_bot),
            "bulutun üstünde" if px > cloud_top else
            "bulutun altında" if px < cloud_bot else "bulut içinde", "ichimoku")
        add("Tenkan/Kijun", "trend_strength", tenkan - kijun,
            _sig(tenkan > kijun, tenkan < kijun), "tenkan/kijun kesişimi", "ichimoku")
        add("Bulut kalınlığı (yön)", "trend_strength", span_a - span_b,
            _sig(span_a > span_b, span_a < span_b),
            "bulut yükseliş renginde" if span_a > span_b else "düşüş renginde", "ichimoku")
    except Exception:
        pass

    # ─────────────────── 5) VOLATİLİTE (~10) ───────────────────
    bu, bm, bl = I.bollinger(c)
    bw = (_f(bu.iloc[-1]) - _f(bl.iloc[-1])) / (_f(bm.iloc[-1]) + 1e-12)
    pctb = (px - _f(bl.iloc[-1])) / (_f(bu.iloc[-1]) - _f(bl.iloc[-1]) + 1e-12)
    add("Bollinger %B", "volatility", pctb, _threshold(pctb, 0.0, 1.0, invert=True),
        f"%B {pctb:.2f} ({'bant üstü' if pctb > 1 else 'bant altı' if pctb < 0 else 'bant içi'})",
        "bollinger")
    add("Bollinger orta bant", "volatility", _f(bm.iloc[-1]),
        _sig(px > _f(bm.iloc[-1]), px < _f(bm.iloc[-1])),
        "orta bandın üstünde" if px > _f(bm.iloc[-1]) else "altında", "bollinger")
    bw_series = ((bu - bl) / (bm + 1e-12)).tail(120)
    squeeze = bool(bw <= float(bw_series.quantile(0.2))) if len(bw_series) > 20 else False
    add("Bollinger sıkışması", "volatility", bw, NOTR,
        "SIKIŞMA (kırılım yaklaşıyor, yön belirsiz)" if squeeze else "normal genişlik",
        "bollinger")
    ku, km, kl_ = I.keltner(df)
    add("Keltner kanalı", "volatility", px - _f(km.iloc[-1]),
        _sig(px > _f(ku.iloc[-1]), px < _f(kl_.iloc[-1])),
        "üst bandın üstünde" if px > _f(ku.iloc[-1]) else
        "alt bandın altında" if px < _f(kl_.iloc[-1]) else "kanal içinde", "keltner")
    du, dm, dl = I.donchian(df)
    add("Donchian kırılımı", "volatility", px - _f(du.iloc[-1]),
        _sig(px >= _f(du.iloc[-1]) * 0.999, px <= _f(dl.iloc[-1]) * 1.001),
        "20 bar zirvesi" if px >= _f(du.iloc[-1]) * 0.999 else
        "20 bar dibi" if px <= _f(dl.iloc[-1]) * 1.001 else "aralık içi", "donchian")
    atr14 = _f(I.atr(df).iloc[-1])
    atr_pct = atr14 / px * 100
    atr_med = float((I.atr(df) / c * 100).tail(120).median())
    add("ATR/fiyat", "volatility", atr_pct, NOTR,
        f"%{atr_pct:.2f} (medyan %{atr_med:.2f}) — "
        f"{'yüksek oynaklık' if atr_pct > atr_med * 1.3 else 'sakin' if atr_pct < atr_med * 0.7 else 'normal'}",
        "atr")

    # ─────────────────── 6) HACİM (~12) ───────────────────
    obv = I.obv(df)
    obv_sl = _f(obv.iloc[-1]) - _f(obv.iloc[-20]) if len(obv) > 20 else float("nan")
    add("OBV eğimi", "volume", obv_sl, _sig(obv_sl > 0, obv_sl < 0),
        "hacim birikimi" if obv_sl > 0 else "hacim dağıtımı", "obv")
    cmf = _f(I.cmf(df).iloc[-1])
    add("Chaikin Money Flow", "volume", cmf, _threshold(cmf, -0.05, 0.05),
        f"CMF {cmf:.3f}", "cmf")
    fi = _f(I.force_index(df).iloc[-1])
    add("Force Index", "volume", fi, _sig(fi > 0, fi < 0),
        "pozitif güç" if fi > 0 else "negatif güç", "force")
    eom = _f(I.eom(df).iloc[-1])
    add("Ease of Movement", "volume", eom, _sig(eom > 0, eom < 0), "EOM", "eom")
    vpt_s = I.vpt(df)
    vpt_sl = _f(vpt_s.iloc[-1]) - _f(vpt_s.iloc[-20]) if len(vpt_s) > 20 else float("nan")
    add("VPT eğimi", "volume", vpt_sl, _sig(vpt_sl > 0, vpt_sl < 0), "VPT trendi", "vpt")
    co = _f(I.chaikin_osc(df).iloc[-1])
    add("Chaikin Osc", "volume", co, _sig(co > 0, co < 0), "Chaikin", "chaikin")
    vol_z = _f(((v - v.rolling(50).mean()) / (v.rolling(50).std() + 1e-12)).iloc[-1])
    add("Hacim z-skoru", "volume", vol_z, NOTR,
        f"z {vol_z:+.2f} — {'olağandışı yüksek' if vol_z > 2 else 'düşük' if vol_z < -1 else 'normal'}",
        "vol_z")
    up_vol = float(v[c.diff() > 0].tail(20).sum())
    dn_vol = float(v[c.diff() < 0].tail(20).sum())
    add("Yukarı/aşağı hacim", "volume", up_vol - dn_vol, _sig(up_vol > dn_vol, dn_vol > up_vol),
        f"{'alıcı hacmi baskın' if up_vol > dn_vol else 'satıcı hacmi baskın'}", "updown_vol")

    # ─────────────────── 7) UYARLANIR ORTALAMALAR (~8) ───────────────────
    try:
        from .indicators_ext import (_kama, _zlema, _tema, _dema, _t3,
                                     _vidya, _mcginley, _ribbon_align)
        # bazıları (c), bazıları (c, n) imzalı — ikisini de dene
        for nm, fn in (("KAMA", _kama), ("ZLEMA", _zlema), ("TEMA", _tema),
                       ("DEMA", _dema), ("T3", _t3), ("VIDYA", _vidya),
                       ("McGinley", _mcginley)):
            val = float("nan")
            for call in (lambda: fn(c, 20), lambda: fn(c)):
                try:
                    val = _f(pd.Series(call()).iloc[-1])
                    if math.isfinite(val):
                        break
                except TypeError:
                    continue
                except Exception:
                    break
            if not math.isfinite(val):
                continue
            add(nm, "adaptive_ma", val, _sig(px > val, px < val),
                f"fiyat {'>' if px > val else '<'} {nm}", "adaptive")
        ra = _f(_ribbon_align(c))
        add("MA şerit hizalanması", "adaptive_ma", ra, _sig(ra > 0.6, ra < -0.6),
            f"hizalanma {ra:+.2f}", "ribbon")
    except Exception:
        pass

    # ─────────────────── 8) FİYAT YAPISI (~10) ───────────────────
    hi52 = float(h.tail(min(len(h), 365)).max()); lo52 = float(l.tail(min(len(l), 365)).min())
    pos52 = (px - lo52) / (hi52 - lo52 + 1e-12)
    add("52 bar aralık konumu", "structure", pos52, _threshold(pos52, 0.25, 0.75),
        f"aralığın %{pos52*100:.0f} seviyesinde", "range_pos")
    for n in (20, 50):
        hh = float(h.tail(n).max()); ll = float(l.tail(n).min())
        add(f"{n} bar zirve/dip", "structure", (px - ll) / (hh - ll + 1e-12),
            _sig(px >= hh * 0.995, px <= ll * 1.005),
            "zirveye yakın" if px >= hh * 0.995 else "dibe yakın" if px <= ll * 1.005 else "orta",
            "range_pos")
    ret5, ret20 = _f(c.pct_change(5).iloc[-1]), _f(c.pct_change(20).iloc[-1])
    add("5 bar getiri", "structure", ret5 * 100, _sig(ret5 > 0, ret5 < 0),
        f"%{ret5*100:+.2f}", "ret")
    add("20 bar getiri", "structure", ret20 * 100, _sig(ret20 > 0, ret20 < 0),
        f"%{ret20*100:+.2f}", "ret")
    # yüksek-dip yapısı (HH/HL mi LH/LL mi)
    try:
        hh1, hh2 = float(h.tail(10).max()), float(h.iloc[-30:-10].max())
        ll1, ll2 = float(l.tail(10).min()), float(l.iloc[-30:-10].min())
        struct = _sig(hh1 > hh2 and ll1 > ll2, hh1 < hh2 and ll1 < ll2)
        add("Zirve/dip yapısı", "structure", hh1 - hh2, struct,
            "yükselen zirve+dip" if struct == AL else
            "alçalan zirve+dip" if struct == SAT else "karışık", "structure_hl")
    except Exception:
        pass
    body = float((c.iloc[-1] - df["open"].astype(float).iloc[-1]))
    rng = float(h.iloc[-1] - l.iloc[-1]) + 1e-12
    add("Son mum gövdesi", "structure", body / rng,
        _sig(body / rng > 0.3, body / rng < -0.3),
        f"gövde/aralık {body/rng:+.2f}", "candle")
    add("Kapanış konumu", "structure", (px - float(l.iloc[-1])) / rng,
        _threshold((px - float(l.iloc[-1])) / rng, 0.3, 0.7),
        "barın üst kısmında kapandı" if (px - float(l.iloc[-1])) / rng > 0.7 else
        "alt kısmında kapandı" if (px - float(l.iloc[-1])) / rng < 0.3 else "ortada",
        "candle")

    # ─────────────────── 9) İLERİ OSİLATÖRLER (~14) ───────────────────
    bull, bear = I.elder_ray(df)
    bp, bep = _f(bull.iloc[-1]), _f(bear.iloc[-1])
    add("Elder Ray", "momentum", bp + bep, _sig(bp > 0 and bep > 0, bp < 0 and bep < 0),
        f"boğa gücü {bp:+.2f} / ayı gücü {bep:+.2f}", "elder")
    nvi, pvi = I.nvi_pvi(df)
    nv = _f(nvi.iloc[-1]) - _f(nvi.rolling(50).mean().iloc[-1])
    add("NVI (akıllı para)", "volume", nv, _sig(nv > 0, nv < 0),
        "NVI 50 ortalamasının üstünde" if nv > 0 else "altında", "nvi")
    mi = _f(I.mass_index(df).iloc[-1])
    add("Mass Index", "volatility", mi, NOTR,
        f"{mi:.1f} — {'DÖNÜŞ UYARISI (>27)' if mi > 27 else 'normal'}", "mass")
    cop = _f(I.coppock(c).iloc[-1])
    add("Coppock", "momentum", cop, _sig(cop > 0, cop < 0),
        "sıfır üstü" if cop > 0 else "sıfır altı", "coppock")
    ha = _f(I.heikin_ashi_signal(df))
    add("Heikin Ashi", "structure", ha, _sig(ha > 0, ha < 0),
        f"HA sinyali {ha:+.1f}", "heikin")
    ft = _f(I.fisher_transform(df).iloc[-1])
    add("Fisher Transform", "momentum", ft, _threshold(ft, -1.5, 1.5), f"{ft:+.2f}", "fisher")
    stc = _f(I.schaff_trend_cycle(c).iloc[-1])
    add("Schaff Trend Cycle", "momentum", stc, _threshold(stc, 25, 75), f"{stc:.1f}", "stc")
    rv = _f(I.rvi(df).iloc[-1])
    add("Relative Vigor", "momentum", rv, _sig(rv > 0, rv < 0), f"{rv:+.3f}", "rvi")
    qs = _f(I.qstick(df).iloc[-1])
    add("Qstick", "momentum", qs, _sig(qs > 0, qs < 0),
        "gövdeler pozitif" if qs > 0 else "gövdeler negatif", "qstick")
    ac = _f(I.accel_osc(df).iloc[-1])
    add("Acceleration Osc", "momentum", ac, _sig(ac > 0, ac < 0), f"{ac:+.4f}", "accel")
    try:
        pv = I.pivot_points(df)
        piv = _f(pv.get("pivot"))
        add("Pivot noktası", "structure", px - piv, _sig(px > piv, px < piv),
            f"fiyat pivotun {'üstünde' if px > piv else 'altında'}", "pivot")
        r1, s1 = _f(pv.get("r1")), _f(pv.get("s1"))
        add("Pivot R1/S1", "structure", px, _sig(px > r1, px < s1),
            "R1 kırıldı" if px > r1 else "S1 kırıldı" if px < s1 else "R1-S1 arasında", "pivot")
    except Exception:
        pass
    try:
        fb = I.fib_levels(df)
        f618, f382 = _f(fb.get("0.618")), _f(fb.get("0.382"))
        lo_, hi_ = min(f618, f382), max(f618, f382)
        add("Fibonacci 0,382-0,618", "structure", px, _sig(px > hi_, px < lo_),
            "altın orana göre üstte" if px > hi_ else "altta" if px < lo_ else "geri çekilme bölgesinde",
            "fib")
    except Exception:
        pass

    # ─────────────────── 10) EK / İSTATİSTİKSEL (~13) ───────────────────
    try:
        from . import indicators_ext as E
        add("Connors RSI", "momentum", _f(E._connors_rsi(c)),
            _threshold(_f(E._connors_rsi(c)), 20, 80, invert=True), "kısa vade aşırılık", "connors")
        stk = _f(E._streak(c).iloc[-1])
        add("Seri (streak)", "momentum", stk, _sig(stk <= -3, stk >= 3),
            f"{abs(int(stk))} bar {'düşüş → dönüş beklentisi' if stk < 0 else 'yükseliş → yorgunluk'}"
            if abs(stk) >= 3 else "seri yok", "streak")
        ppo = _f(E._ppo(c))
        add("PPO", "momentum", ppo, _sig(ppo > 0, ppo < 0), f"%{ppo:+.2f}", "ppo")
        ppoh = _f(E._ppo(c, hist=True))
        add("PPO histogram", "momentum", ppoh, _sig(ppoh > 0, ppoh < 0), "hız", "ppo")
        rmi = _f(E._rmi(c))
        add("RMI", "momentum", rmi, _threshold(rmi, 40, 60), f"{rmi:.1f}", "rmi")
        cv = _f(E._chaikin_vol(df))
        add("Chaikin volatilite", "volatility", cv, NOTR,
            f"%{cv:+.1f} — {'genişliyor' if cv > 0 else 'daralıyor'}", "chaikin_vol")
        ul = _f(E._ulcer(c))
        add("Ulcer Index", "volatility", ul, NOTR,
            f"{ul:.2f} — {'yüksek dip acısı' if ul > 5 else 'ılımlı'}", "ulcer")
        adl_s = E._adl(df)
        adl_sl = _f(adl_s.iloc[-1]) - _f(adl_s.iloc[-20]) if len(adl_s) > 20 else float("nan")
        add("A/D çizgisi eğimi", "volume", adl_sl, _sig(adl_sl > 0, adl_sl < 0),
            "birikim" if adl_sl > 0 else "dağıtım", "adl")
        kv = _f(E._klinger(df))
        add("Klinger", "volume", kv, _sig(kv > 0, kv < 0), f"{kv:+.3f}", "klinger")
        tw = _f(E._twiggs_mf(df))
        add("Twiggs Money Flow", "volume", tw, _threshold(tw, -0.01, 0.01), f"{tw:+.4f}", "twiggs")
        udv = _f(E._up_down_vol(df))
        add("Yukarı/aşağı hacim oranı", "volume", udv, _threshold(udv, 0.8, 1.2),
            f"oran {udv:.2f}", "updown_vol")
        ch = _f(E._choppiness(df))
        add("Choppiness Index", "volatility", ch, NOTR,
            f"{ch:.1f} — {'YATAY/ÇALKANTILI (>61, trend takibi zayıf)' if ch > 61 else 'TRENDLİ (<38)' if ch < 38 else 'ara'}",
            "choppiness")
        dp = _f(E._donchian_pos(df))
        add("Donchian konumu", "structure", dp, _threshold(dp, 0.25, 0.75),
            f"kanalın %{dp*100:.0f} seviyesinde", "donchian")
        sq = _f(E._squeeze_mom(df))
        add("Squeeze momentum", "volatility", sq, _sig(sq > 0, sq < 0),
            "sıkışma sonrası yukarı" if sq > 0 else "aşağı", "squeeze")
        z20 = _f(E._zscore(c, 20))
        add("Fiyat z-skoru (20)", "structure", z20, _threshold(z20, -2, 2, invert=True),
            f"z {z20:+.2f} — {'aşırı gerilmiş' if abs(z20) > 2 else 'normal'}", "zscore")
        sl20 = _f(E._slope(c, 20))
        add("Regresyon eğimi (20)", "trend_strength", sl20, _sig(sl20 > 0, sl20 < 0),
            "yukarı eğimli" if sl20 > 0 else "aşağı eğimli", "slope")
    except Exception:
        pass

    # ─────────────── 11) GENİŞLETME PAKETİ (~175) ───────────────
    # Ayrı modül: oynaklık tahmincileri, istatistiksel rejim ölçütleri, pivot
    # aileleri ve ÇOKLU ZAMAN DİLİMİ hizası. Paket çökerse çekirdek tablo yine
    # üretilsin diye izole edilir.
    try:
        from .indicator_pack import extend_board
        extend_board(df, add)
    except Exception as e:                      # pragma: no cover
        add("Genişletme paketi", "structure", float("nan"), NOTR,
            f"yüklenemedi: {type(e).__name__}", "pack_error")

    # ─────────────── 12) İKİNCİ PAKET (~100) ───────────────
    # Ehlers filtreleri · istatistiksel performans · hacim profili · MİKROYAPI.
    # `micro` = kaydediciden o paritenin son satırı; yoksa mikroyapı bölümü
    # atlanır (fiyattan üretilemez, uydurulmaz).
    micro_eklendi = False
    try:
        from .indicator_pack2 import extend_board2
        micro_eklendi = bool(extend_board2(df, add, micro=micro))
    except Exception as e:                      # pragma: no cover
        add("Genişletme paketi 2", "structure", float("nan"), NOTR,
            f"yüklenemedi: {type(e).__name__}", "pack2_error")

    # ─────────────────── SAYIM ───────────────────
    ozet = _summarize(out, px)
    ozet["microstructure"] = micro_eklendi
    ozet["micro_evidence"] = MICRO_EVIDENCE if micro_eklendi else None
    ozet["microstructure_note"] = (
        "Funding, defter eğimi, derinlik dengesizliği gibi göstergeler kaydedici "
        "verisinden gelir ve FİYATTAN TÜRETİLEMEZ."
        + ("" if micro_eklendi else " Bu parite için kayıt yok — bölüm ATLANDI."))
    return ozet


def _summarize(sigs: List[IndicatorSignal], price: float) -> Dict:
    raw = {AL: 0, SAT: 0, NOTR: 0}
    for s in sigs:
        raw[s.signal] += 1

    # AİLE oyu: aynı ailedeki göstergeler tek oya indirgenir (çoğunluk)
    fam_votes: Dict[str, List[str]] = {}
    for s in sigs:
        fam_votes.setdefault(s.family, []).append(s.signal)
    fam = {AL: 0, SAT: 0, NOTR: 0}
    for f, votes in fam_votes.items():
        a, sl = votes.count(AL), votes.count(SAT)
        fam[AL if a > sl else SAT if sl > a else NOTR] += 1

    by_cat: Dict[str, Dict] = {}
    for s in sigs:
        d = by_cat.setdefault(s.category, {AL: 0, SAT: 0, NOTR: 0, "n": 0})
        d[s.signal] += 1
        d["n"] += 1

    n = len(sigs)
    n_fam = sum(fam.values())
    # Net eğilim AİLE oylarından hesaplanır (ham sayım tekrar içerir)
    net = (fam[AL] - fam[SAT]) / max(1, n_fam)
    bias = "YUKARI" if net > 0.2 else "AŞAĞI" if net < -0.2 else "NÖTR"

    return {
        "available": True,
        "price": round(price, 8),
        "total": n,
        "raw": {"al": raw[AL], "sat": raw[SAT], "notr": raw[NOTR]},
        "family": {"al": fam[AL], "sat": fam[SAT], "notr": fam[NOTR], "total": n_fam},
        "by_category": {k: {"al": v[AL], "sat": v[SAT], "notr": v[NOTR],
                            "n": v["n"], "label": CATEGORIES.get(k, k)}
                        for k, v in by_cat.items()},
        "net": round(float(net), 3),
        "bias": bias,
        "indicators": [s.to_dict() for s in sigs],
        "redundancy_note": (
            f"Ham sayım {n} göstergedir ama bunlar BAĞIMSIZ DEĞİLDİR — "
            f"örneğin 13 periyottaki EMA aynı şeyi ölçer. Aile bazında "
            f"indirgendiğinde {n_fam} bağımsız oy kalır. Karar için AİLE "
            f"sayımına bakın; ham sayım yalnız kapsamı gösterir."),
        "usage_note": (
            "Bu tablo tek başına işlem açtırmaz. Ölçüldü: 180 göstergeyle beslenen "
            "meta-etiketleme örneklem dışında Sharpe'ı 0,25 DÜŞÜRDÜ "
            "(runs/FAZ4-10_BULGULAR.md). Gösterge yoğunluğu bir kalite ölçüsü değildir."),
    }
