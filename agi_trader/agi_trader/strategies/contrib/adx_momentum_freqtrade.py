"""
ADXMomentum — ADX + yönlü hareket + momentum ile trend takibi.

KAYNAK: freqtrade/freqtrade-strategies · `user_data/strategies/berlinguyinca/ADXMomentum.py`
        (GPL-3.0) — https://github.com/freqtrade/freqtrade-strategies

ÖZGÜN KURAL (kaynaktan doğrulandı, 2026-09-05):
    giriş : adx > 25  AND  mom > 0  AND  plus_di > 25  AND  plus_di > minus_di
    çıkış : adx > 25  AND  mom < 0  AND  minus_di > 25 AND  plus_di < minus_di
    göstergeler: ADX(14) · PLUS_DI(25) · MINUS_DI(25) · MOM(14) · SAR
    minimal_roi {"0": 0.01} · stoploss −0.25 · timeframe '1h'

KOD KOPYALANMADI; kural bu deponun kendi araçlarıyla bağımsız yazıldı. DI± ve MOM `f`
sözlüğünde bulunmadığı için bar çerçevesinden (`df`) hesaplanır — katkı arayüzünün
beşinci parametresi tam da bunun için var.

──────────────────────────────────────────────────────────────── SAPMALAR
1. ZAMAN DİLİMİ: özgün 1 saatlik; bu port komitenin bar zaman diliminde (1 dk) ölçülür.
2. ÇIKIŞ: özgün çıkış bir SİNYALDİR (momentum ters döner). Bu çerçevede çıkış motoru
   stop/hedefle çalışır; en yakın karşılık DYNAMIC_PEAK (tepe takibi) seçildi —
   "momentum bitene kadar sür" niyetinin çerçevedeki karşılığı budur. Hedef, özgün
   %1 ROI yerine 3×ATR (1 dk barda %1 ROI ulaşılamaz bir hedeftir).
3. STOP: özgün −%25 stop 1 dk ufkunda anlamsızdır (koşucu `max_stop_pct` ile reddeder);
   1,5×ATR kullanılır.
Yani ölçüm "ADXMomentum kârlı mı"yı DEĞİL, "bu GİRİŞ kuralı bu sistemin zaman dilimi ve
maliyet yapısında kenar üretiyor mu"yu yanıtlar.

DI± hesabı Wilder'ın özgün tanımıdır (yumuşatma = Wilder RMA), TA-Lib'in PLUS_DI/MINUS_DI
fonksiyonlarıyla aynı formül. MOM(n) = close − close.shift(n) (TA-Lib MOM ile aynı).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

META = {
    "name": "adx_momentum_freqtrade",
    "title_tr": "ADXMomentum (freqtrade örneği)",
    "author": "@alylmazzz (port) · berlinguyinca (özgün kural)",
    "source": "freqtrade/freqtrade-strategies · berlinguyinca/ADXMomentum.py (GPL-3.0) — "
              "https://github.com/freqtrade/freqtrade-strategies",
    "claim": "ADX > 25 ile trend güçlüyken, momentum pozitif ve +DI hem 25'in üstünde hem "
             "−DI'nin üstündeyse yükseliş devam eder.",
    "claim_evidence": "YOK — kaynak depo bu stratejileri 'hazır kullanım değil, başlangıç "
                      "noktası' olarak sunuyor ve 'kendi backtest'inizi koşun' diyor. "
                      "Yayımlanmış kâr kanıtı YOK. Özgün bağlam 1 saatlik barlar.",
    "mechanism": "adx(14) > 25 VE mom(14) > 0 VE plus_di(25) > 25 VE plus_di > minus_di. "
                 "Hedef 3×ATR, stop 1,5×ATR (çerçeve karşılıkları — bkz. SAPMALAR).",
    "exit_mode": "DYNAMIC_PEAK",
    "time_stop_min": 120,
    "urgency": 1,                    # trend girişi bekleyemez ama acil de değil
    "regimes": ["TREND YUKARI", "VOLATİL"],
}


def _wilder(s: pd.Series, n: int) -> pd.Series:
    """Wilder yumuşatması (RMA) — ADX/DI ailesinin özgün tanımı."""
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def _di(df: pd.DataFrame, n: int):
    """+DI ve −DI (Wilder). TA-Lib PLUS_DI/MINUS_DI ile aynı formül."""
    h, l, c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    up, down = h.diff(), -l.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = _wilder(tr, n)
    plus = 100.0 * _wilder(pd.Series(plus_dm, index=df.index), n) / atr.replace(0.0, np.nan)
    minus = 100.0 * _wilder(pd.Series(minus_dm, index=df.index), n) / atr.replace(0.0, np.nan)
    return plus, minus


def _adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    plus, minus = _di(df, n)
    dx = 100.0 * (plus - minus).abs() / (plus + minus).replace(0.0, np.nan)
    return _wilder(dx.fillna(0.0), n)


def fire(f: Dict, p, price: float, atr_abs: float, df=None) -> Optional[Dict]:
    if df is None or len(df) < 80:
        return None
    try:
        adx = float(_adx(df, 14).iloc[-1])
        plus, minus = _di(df, 25)                       # özgün: PLUS_DI/MINUS_DI timeperiod=25
        plus_di, minus_di = float(plus.iloc[-1]), float(minus.iloc[-1])
        mom = float(df["close"].astype(float).diff(14).iloc[-1])   # TA-Lib MOM(14)
    except Exception:
        return None
    if not all(np.isfinite(v) for v in (adx, plus_di, minus_di, mom)):
        return None

    # ——— ÖZGÜN GİRİŞ KOŞULU (kaynakla birebir) ———
    if not (adx > 25.0 and mom > 0.0 and plus_di > 25.0 and plus_di > minus_di):
        return None

    stop = price - 1.5 * atr_abs
    if stop >= price:
        return None
    if (price - stop) / price * 100.0 > float(getattr(p, "max_stop_pct", 2.0)):
        return None

    return {
        "direction": "LONG",
        "size": 0.5,
        "stop_hint": stop,
        "target_hint": price + 3.0 * atr_abs,
        "note": (f"ADX {adx:.0f} > 25 · +DI {plus_di:.0f} > −DI {minus_di:.0f} · "
                 f"MOM(14) {mom:+.6g} > 0 → trend devamı"),
    }

# ═════════════════════════════════════════════════════════════════════════
# BİZ ÖLÇTÜK — 2026-09-05T18:13:09Z · binance · 60 GÜN · 5 parite (1 dk bar)
# ═════════════════════════════════════════════════════════════════════════
# Bu blok SİLİNMEZ. Çürütülen ölçüm de kayıttır: bir kurulumun neden gölgede ya da
# reddedilmiş olduğu, sonradan bakan birinin yeniden ölçmek zorunda kalmaması için
# burada durur. Bütün katkılar AYNI 60 günlük pencerede, aynı maliyetle ölçüldü.
#
#   pencere 28680 · ateşleme 5070 · oran %17.678
#
# VERDİKT: REDDEDİLDİ — ateşleme oranı %17.7 > %15
#
# Ateşleme kapısı kenar ölçülmeden ÖNCE devreye girer: %17,5 oran, kurulumun
# piyasanın altıda birini 'giriş' saydığı anlamına gelir — seçici değildir.
# Bu, özgün 1 SAATLİK bağlam hakkında bir yargı DEĞİLDİR: ADX>25 ve +DI>25 gibi
# eşikler kısa barlarda çok daha sık sağlanır (bkz. SAPMALAR).
# KARARLI: 7 gün/MEXC'te %17,5 · 60 gün/Binance'te %17,7 — farklı borsa, 8,6 kat
# veri, neredeyse aynı oran. Bu bir örneklem kazası değil, kuralın özelliği.
#
# Yeniden ölçmek için:
#   python scripts/cm_verify_contribution.py --sleeve adx_momentum_freqtrade --days 60 --venue binance --step 15
MEASURED = {
    "window": "60 gün · binance · 1 dk",
    "n_windows": 28680, "n_fires": 5070, "n_effective": None,
    "fire_rate_pct": 17.678,
    "verdict": "REJECTED",
}
