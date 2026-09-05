"""
HLHB — RSI'ın 50'yi kesmesi + EMA5/EMA10 kesişimi + ADX teyidi.

KAYNAK: freqtrade/freqtrade-strategies · `user_data/strategies/hlhb.py` (GPL-3.0)
        https://github.com/freqtrade/freqtrade-strategies

ÖZGÜN KURAL (kaynaktan doğrulandı, 2026-09-05):
    giriş : RSI 50'yi YUKARI KESTİ  VE  EMA5 EMA10'u YUKARI KESTİ  VE  ADX > 25  VE  hacim > 0
    çıkış : RSI 50'yi aşağı kesti   VE  EMA5 EMA10'u aşağı kesti   VE  ADX > 25  VE  hacim > 0
    göstergeler: RSI(10) · EMA(5) · EMA(10) · ADX
    minimal_roi 0:%62,25 → 5520dk:0 · stoploss −0,3211 · timeframe '4h'

KOD KOPYALANMADI; kural bu deponun kendi araçlarıyla bağımsız yazıldı.

═══════════════════════════════════════════════ FİDELİTE NOTU (RSI'ın fiyat serisi)
Kaynak, RSI'ı kapanış yerine bir ORTA FİYAT serisi üzerinden hesaplıyor. Elimdeki
tanımda bu seri "açılış ve kapanışın ortalaması" olarak geçiyor; bu port
(open + close)/2 kullanır. Kaynak bunun yerine (high+low)/2 kullanıyorsa sonuç biraz
kayabilir. Belirsizlik ölçümün parçasıdır, gizlenmemelidir.

──────────────────────────────────────────────────────────────── SAPMALAR
1. ZAMAN DİLİMİ: özgün 4 SAATLİK; bu port 1 dakikalık barlarda ölçülür. Aradaki fark
   bu ailedeki en büyüğü — 4 saatlik bir kesişim sistemi 1 dakikada çok daha sık ve
   çok daha gürültülü tetiklenir.
2. ÇIKIŞ: özgün çıkış SİNYALDİR → çerçevede DYNAMIC_PEAK (tepe takibi), hedef 3×ATR.
3. STOP: özgün −%32,11 stop 1 dk ufkunda anlamsızdır → oransal stop,
   giriş × (1 − min(%32, max_stop_pct)); pratikte max_stop_pct bağlar.
4. İKİ KESİŞİM AYNI BARDA: özgün kural her iki kesişimin de AYNI mumda olmasını ister.
   Bu kasıtlı olarak birebir korundu. Ateşleme oranı düşük çıkarsa bu kuralın kendi
   özelliğidir — port kusuru olmadığı ham koşul ayrıca taranarak doğrulanır.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

META = {
    "name": "hlhb_freqtrade",
    "title_tr": "HLHB — RSI/EMA kesişimi + ADX (freqtrade örneği)",
    "author": "@alylmazzz (port) · freqtrade katkıcıları (özgün kural)",
    "source": "freqtrade/freqtrade-strategies · hlhb.py (GPL-3.0) — "
              "https://github.com/freqtrade/freqtrade-strategies",
    "claim": "RSI 50'yi yukarı keserken EMA5 de EMA10'u yukarı kesiyorsa ve ADX > 25 ile "
             "trend güçlüyse yükseliş başlamıştır.",
    "claim_evidence": "YOK — kaynak depo bu stratejileri 'hazır kullanım değil, başlangıç "
                      "noktası' olarak sunuyor. Yayımlanmış kâr kanıtı YOK. Özgün bağlam "
                      "4 SAATLİK barlar; bu port 1 dakikalıkta ölçülür.",
    "mechanism": "RSI(10, (açılış+kapanış)/2) 50'yi yukarı kesti VE EMA5 EMA10'u yukarı "
                 "kesti (AYNI bar) VE ADX(14) > 25 VE hacim > 0. Hedef 3×ATR.",
    "exit_mode": "DYNAMIC_PEAK",
    "time_stop_min": 180,
    "urgency": 1,
    "regimes": ["TREND YUKARI", "VOLATİL"],
}


def _wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def _rsi(s: pd.Series, n: int = 10) -> pd.Series:
    d = s.diff()
    up = _wilder(d.clip(lower=0.0), n)
    dn = _wilder((-d).clip(lower=0.0), n)
    return 100.0 - 100.0 / (1.0 + up / dn.replace(0.0, np.nan))


def _adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    up, down = h.diff(), -l.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = _wilder(tr, n).replace(0.0, np.nan)
    plus = 100.0 * _wilder(plus_dm, n) / atr
    minus = 100.0 * _wilder(minus_dm, n) / atr
    dx = 100.0 * (plus - minus).abs() / (plus + minus).replace(0.0, np.nan)
    return _wilder(dx.fillna(0.0), n)


def fire(f: Dict, p, price: float, atr_abs: float, df=None) -> Optional[Dict]:
    if df is None or len(df) < 120:
        return None
    if "volume" in df and float(df["volume"].iloc[-1]) <= 0:
        return None
    try:
        c = df["close"].astype(float)
        o = df["open"].astype(float) if "open" in df else c
        orta = (o + c) / 2.0                       # bkz. FİDELİTE NOTU
        rsi = _rsi(orta, 10).to_numpy()
        ema5 = c.ewm(span=5, adjust=False).mean().to_numpy()
        ema10 = c.ewm(span=10, adjust=False).mean().to_numpy()
        adx = float(_adx(df, 14).iloc[-1])
    except Exception:
        return None
    if not (np.isfinite(rsi[-1]) and np.isfinite(rsi[-2]) and np.isfinite(adx)):
        return None

    # ——— ÖZGÜN GİRİŞ KOŞULU (kaynakla birebir): İKİ KESİŞİM AYNI BARDA ———
    rsi_kesti = rsi[-1] > 50.0 and rsi[-2] <= 50.0
    ema_kesti = ema5[-1] > ema10[-1] and ema5[-2] <= ema10[-2]
    if not (rsi_kesti and ema_kesti and adx > 25.0):
        return None

    stop_pct = min(32.0, float(getattr(p, "max_stop_pct", 2.0)))
    stop = price * (1.0 - stop_pct / 100.0)
    return {
        "direction": "LONG",
        "size": 0.5,
        "stop_hint": stop,
        "target_hint": price + 3.0 * atr_abs,
        "note": (f"RSI {rsi[-1]:.0f} 50'yi kesti + EMA5/EMA10 kesişimi AYNI barda · "
                 f"ADX {adx:.0f} > 25 · stop %{stop_pct:.1f}"),
    }

# ═════════════════════════════════════════════════════════════════════════
# BİZ ÖLÇTÜK — binance · 60 gün · BÜYÜK pariteler (1 dk)
# ═════════════════════════════════════════════════════════════════════════
#   pencere 28660 · ateşleme 159 · oran %0.555
#   örneklem: nominal 159 → ETKİN 153
#   ortalama net %-0.0953 · t -2.23 · CI95 [-0.1849, -0.0187] · kazanma %47.1
#   çıkış sebepleri: {'HEDEF': 128, 'ZAMAN': 24, 'STOP': 1}
#
# VERDİKT: GÖLGE
#
# Her iki grupta da NEGATİF ve küçük paritelerde DAHA KÖTÜ (−%0,095 → −%0,220).
# 4 saatlik bir kesişim sistemi 1 dakikada gürültüye dönüşüyor: RSI'ın 50'yi kesmesi
# saatlik grafikte bir rejim değişimi işareti olabilir, dakikalık grafikte olağan
# salınımdır. ADX > 25 teyidi bunu kurtarmıyor.
MEASURED = {
    "window": "60 gün · binance · 1 dk · büyük pariteler",
    "n_windows": 28660, "n_fires": 159, "n_effective": 153,
    "fire_rate_pct": 0.555,
    "mean_net_pct": -0.0953, "t_stat": -2.23, "ci95": [-0.1849, -0.0187],
    "win_rate": 0.471, "exit_reasons": {'HEDEF': 128, 'ZAMAN': 24, 'STOP': 1},
    "verdict": "SHADOW",
}

# ─────────────────────────────────────────────────────────────────────────
# KÜÇÜK/OYNAK PARİTELER — BONK, ORDI, PYTH, ARB, PEPE
# ─────────────────────────────────────────────────────────────────────────
#   pencere 28660 · ateşleme 362 · oran %1.263
#   örneklem: nominal 362 → ETKİN 303
#   ortalama net %-0.2201 · t -3.9 · CI95 [-0.3369, -0.1095] · kazanma %55.4
#   çıkış sebepleri: {'HEDEF': 169, 'ZAMAN': 129, 'STOP': 5}
#
# VERDİKT: GÖLGE
#
MEASURED_SMALL_CAPS = {
    "window": "60 gün · binance · 1 dk · küçük pariteler",
    "n_windows": 28660, "n_fires": 362, "n_effective": 303,
    "fire_rate_pct": 1.263,
    "mean_net_pct": -0.2201, "t_stat": -3.9, "ci95": [-0.3369, -0.1095],
    "win_rate": 0.554, "exit_reasons": {'HEDEF': 169, 'ZAMAN': 129, 'STOP': 5},
    "verdict": "SHADOW",
}
