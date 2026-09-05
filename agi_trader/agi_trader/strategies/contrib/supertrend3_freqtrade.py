"""
Supertrend (üçlü teyit) — üç farklı Supertrend aynı anda "yukarı" derse giriş.

KAYNAK: freqtrade/freqtrade-strategies · `user_data/strategies/Supertrend.py` (GPL-3.0)
        https://github.com/freqtrade/freqtrade-strategies

ÖZGÜN KURAL (kaynaktan doğrulandı, 2026-09-05):
    giriş : üç Supertrend de 'up'  —  (çarpan/periyot) = (4/8), (7/9), (1/8)   AND volume > 0
    çıkış : üç Supertrend de 'down' —  (1/16), (3/18), (6/18)
    minimal_roi 0,087'den başlayıp azalır · stoploss −0,265 · timeframe '1h'

KOD KOPYALANMADI; algoritma bu deponun kendi araçlarıyla bağımsız yazıldı.

════════════════════════════════════════════════════════ FİDELİTE UYARISI (önemli)
Kaynak dosya Supertrend'i KENDİ İÇİNDE hesaplamıyor; `technical.indicators.supertrend`
fonksiyonunu çağırıyor. O kütüphanenin gövdesini DOĞRULAYAMADIM — özellikle ATR'nin
Wilder yumuşatmasıyla mı yoksa TR'nin düz SMA'sıyla mı hesaplandığını. Bu port
**kanonik Supertrend**'i kullanır: Wilder ATR + standart bant özyinelemesi.
Dolayısıyla sonuçlar, referans uygulamadan ATR yumuşatması kadar sapabilir. Bu belirsizlik
ölçümün bir parçasıdır ve gizlenmemelidir: aşağıdaki ölçüm "kanonik üçlü Supertrend"i
ölçer, "freqtrade Supertrend.py'nin bit-birebir aynısını" değil.

──────────────────────────────────────────────────────────────── SAPMALAR
1. ZAMAN DİLİMİ: özgün 1 saatlik; bu port komitenin bar zaman diliminde (1 dk) ölçülür.
2. ÇIKIŞ: özgün çıkış bir SİNYALDİR (üç farklı Supertrend 'down'). Çerçevede çıkış motoru
   stop/hedefle çalışır → DYNAMIC_PEAK (tepe takibi) seçildi; hedef 3×ATR.
3. STOP: özgün −%26,5 stop 1 dk ufkunda anlamsızdır → en gevşek Supertrend bandının
   (7/9) altı stop olarak kullanılır; `max_stop_pct` aşılırsa TETİKLENMEZ. Bu, stratejinin
   kendi mantığına en yakın stoptur (Supertrend zaten bir takip-stop göstergesidir).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

META = {
    "name": "supertrend3_freqtrade",
    "title_tr": "Supertrend üçlü teyit (freqtrade örneği)",
    "author": "@alylmazzz (port) · freqtrade katkıcıları (özgün kural)",
    "source": "freqtrade/freqtrade-strategies · Supertrend.py (GPL-3.0) — "
              "https://github.com/freqtrade/freqtrade-strategies",
    "claim": "Farklı periyot/çarpandaki ÜÇ Supertrend aynı anda yukarı yönlüyse trend "
             "teyitlidir ve devam eder.",
    "claim_evidence": "YOK — kaynak depo bu stratejileri 'hazır kullanım değil, başlangıç "
                      "noktası' olarak sunuyor. Yayımlanmış kâr kanıtı YOK. Ayrıca referans "
                      "Supertrend uygulaması (technical kütüphanesi) doğrulanamadı; bu port "
                      "kanonik Supertrend kullanır (bkz. FİDELİTE UYARISI).",
    "mechanism": "Supertrend(8, 4) VE Supertrend(9, 7) VE Supertrend(8, 1) üçü de 'up'. "
                 "Hedef 3×ATR; stop Supertrend(9,7) bandı.",
    "exit_mode": "DYNAMIC_PEAK",
    "time_stop_min": 180,
    "urgency": 1,
    "regimes": ["TREND YUKARI", "VOLATİL"],
}


def _atr_wilder(df: pd.DataFrame, n: int) -> pd.Series:
    h, l, c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()


def _supertrend(df: pd.DataFrame, period: int, multiplier: float) -> Tuple[bool, float]:
    """Kanonik Supertrend. Döner: (yukarı_mı, band_seviyesi).

    Standart özyineleme: bantlar yalnız DARALDIĞINDA ya da fiyat karşı tarafa geçtiğinde
    güncellenir; yön, kapanışın nihai bandı aşmasıyla döner."""
    h, l, c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    hl2 = (h + l) / 2.0
    atr = _atr_wilder(df, period)
    ub = (hl2 + multiplier * atr).to_numpy()
    lb = (hl2 - multiplier * atr).to_numpy()
    cl = c.to_numpy()
    n = len(cl)
    if n < period + 2:
        return False, float("nan")
    f_ub = np.full(n, np.nan)
    f_lb = np.full(n, np.nan)
    f_ub[period] = ub[period]
    f_lb[period] = lb[period]
    yukari = np.zeros(n, dtype=bool)
    yukari[period] = cl[period] > f_ub[period]
    for i in range(period + 1, n):
        f_ub[i] = ub[i] if (ub[i] < f_ub[i - 1] or cl[i - 1] > f_ub[i - 1]) else f_ub[i - 1]
        f_lb[i] = lb[i] if (lb[i] > f_lb[i - 1] or cl[i - 1] < f_lb[i - 1]) else f_lb[i - 1]
        if yukari[i - 1]:
            yukari[i] = cl[i] >= f_lb[i]          # yukarı trend, alt bandı kırana dek sürer
        else:
            yukari[i] = cl[i] > f_ub[i]           # aşağı trend, üst bandı aşınca döner
    return bool(yukari[-1]), float(f_lb[-1] if yukari[-1] else f_ub[-1])


def fire(f: Dict, p, price: float, atr_abs: float, df=None) -> Optional[Dict]:
    if df is None or len(df) < 120:
        return None
    if "volume" in df and float(df["volume"].iloc[-1]) <= 0:
        return None                                # özgün koşul: volume > 0
    try:
        st1, _ = _supertrend(df, 8, 4.0)
        st2, band2 = _supertrend(df, 9, 7.0)
        st3, _ = _supertrend(df, 8, 1.0)
    except Exception:
        return None

    # ——— ÖZGÜN GİRİŞ KOŞULU (kaynakla birebir): üçü de yukarı ———
    if not (st1 and st2 and st3):
        return None
    if not np.isfinite(band2) or band2 >= price:
        return None

    stop = float(band2)                            # Supertrend zaten bir takip-stoptur
    if (price - stop) / price * 100.0 > float(getattr(p, "max_stop_pct", 2.0)):
        return None

    return {
        "direction": "LONG",
        "size": 0.5,
        "stop_hint": stop,
        "target_hint": price + 3.0 * atr_abs,
        "note": (f"üç Supertrend de yukarı (8/4 · 9/7 · 8/1) → trend teyitli; "
                 f"stop 9/7 bandı {stop:.6g}"),
    }

# ═════════════════════════════════════════════════════════════════════════
# BİZ ÖLÇTÜK — 2026-09-05T18:33:54Z · binance · 60 GÜN · 5 parite (1 dk bar)
# ═════════════════════════════════════════════════════════════════════════
# Bu blok SİLİNMEZ. Çürütülen ölçüm de kayıttır: bir kurulumun neden gölgede ya da
# reddedilmiş olduğu, sonradan bakan birinin yeniden ölçmek zorunda kalmaması için
# burada durur. Bütün katkılar AYNI 60 günlük pencerede, aynı maliyetle ölçüldü.
#
#   pencere 28660 · ateşleme 5912 · oran %20.628
#
# VERDİKT: REDDEDİLDİ — ateşleme oranı %20.6 > %15
#
# Ateşleme oranı %20,4 — üç Supertrend'in aynı anda yukarı olması 1 dakikalık
# barlarda nadir DEĞİL, olağan durumdur. Üçlü teyit burada seçicilik katmıyor.
# Ayrıca FİDELİTE UYARISI geçerli: referans uygulama (technical kütüphanesi)
# doğrulanamadı; bu ölçüm kanonik Supertrend'i ölçer.
# KARARLI: 7 gün/MEXC'te %20,4 · 60 gün/Binance'te %20,6.
#
# Yeniden ölçmek için:
#   python scripts/cm_verify_contribution.py --sleeve supertrend3_freqtrade --days 60 --venue binance --step 15
MEASURED = {
    "window": "60 gün · binance · 1 dk",
    "n_windows": 28660, "n_fires": 5912, "n_effective": None,
    "fire_rate_pct": 20.628,
    "verdict": "REJECTED",
}
