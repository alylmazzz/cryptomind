"""
ClucMay72018 — derin sapma (Bollinger alt bandının %1,5 ALTINDA) ortalamaya dönüş.

KAYNAK: freqtrade/freqtrade-strategies · `user_data/strategies/berlinguyinca/ClucMay72018.py`
        (GPL-3.0) — https://github.com/freqtrade/freqtrade-strategies

ÖZGÜN KURAL (kaynaktan doğrulandı, 2026-09-05):
    giriş : close < ema100
            AND close < 0.985 * bb_lowerband
            AND volume < volume.rolling(30).mean().shift(1) * 20
    çıkış : close > bb_middleband
    göstergeler: Bollinger(20, 2σ) · EMA(100) · hacim ortalaması(30)
    minimal_roi {"0": 0.01} · stoploss −0.05 · timeframe '5m'

KOD KOPYALANMADI; kural bu deponun kendi araçlarıyla bağımsız yazıldı.

──────────────────────────────────────────────────────── BbandRsi'DEN FARKI
Aynı ailedendir ama BbandRsi bandın hemen altını yeterli sayar; Clucmay bandın **%1,5
altını** ister ve RSI yerine EMA100 trend filtresi kullanır. Yani daha seçici ve daha
derin bir sapma arar. İkisinin ayrı ölçülmesi bu yüzden anlamlıdır: "ne kadar derin sapma
yeterli" sorusunun ölçülmüş cevabı.

──────────────────────────────────────────────────────────────── SAPMALAR
1. ZAMAN DİLİMİ: özgün 5 dakikalık; bu port komitenin bar zaman diliminde (1 dk) ölçülür.
2. ÇIKIŞ: özgün çıkış `close > bb_middleband` — ÇERÇEVEDE BİREBİR karşılığı var
   (FIXED_TARGET, hedef = orta bant). Bu strateji çıkış açısından sadık portlanabildi.
3. STOP: özgün −%5 stop 1 dk ufkunda çok geniştir (koşucu `max_stop_pct` ile reddeder);
   alt bandın 0,8×ATR altı kullanılır ve tavanı aşarsa kurulum TETİKLENMEZ.
4. HACİM KOŞULU: `volume < 20 × ort(30)` pratikte neredeyse her zaman doğrudur (aşırı
   hacim patlamasını eleyen bir emniyet filtresidir, seçici bir koşul değil). Birebir
   uygulanır ama bağlayıcı olmadığı bilinmelidir.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

META = {
    "name": "cluc_may_freqtrade",
    "title_tr": "ClucMay72018 (freqtrade örneği)",
    "author": "@alylmazzz (port) · berlinguyinca (özgün kural)",
    "source": "freqtrade/freqtrade-strategies · berlinguyinca/ClucMay72018.py (GPL-3.0) — "
              "https://github.com/freqtrade/freqtrade-strategies",
    "claim": "Fiyat EMA100'ün altındayken Bollinger alt bandının %1,5 ALTINA sarkarsa "
             "aşırı sapmadır ve orta banda döner.",
    "claim_evidence": "YOK — kaynak depo bu stratejileri 'hazır kullanım değil, başlangıç "
                      "noktası' olarak sunuyor ve 'kendi backtest'inizi koşun' diyor. "
                      "Yayımlanmış kâr kanıtı YOK. Özgün bağlam 5 dakikalık barlar.",
    "mechanism": "close < ema100 VE close < 0,985×bb_lower(20,2σ) VE hacim < 20×ort(30). "
                 "Hedef: orta bant (özgün çıkışla birebir). Stop: bb_lower − 0,8×ATR.",
    "exit_mode": "FIXED_TARGET",
    "time_stop_min": 240,
    "urgency": 0,
    "regimes": ["RANGE / YATAY", "VOLATİL", "TREND AŞAĞI"],
}


def fire(f: Dict, p, price: float, atr_abs: float, df=None) -> Optional[Dict]:
    if df is None or len(df) < 120:
        return None
    try:
        c = df["close"].astype(float)
        v = df["volume"].astype(float) if "volume" in df else None
        sma20 = c.rolling(20).mean()
        sd20 = c.rolling(20).std(ddof=0)
        bb_lo = float(sma20.iloc[-1] - 2.0 * sd20.iloc[-1])
        bb_mid = float(sma20.iloc[-1])
        ema100 = float(c.ewm(span=100, adjust=False).mean().iloc[-1])
    except Exception:
        return None
    if not all(np.isfinite(x) for x in (bb_lo, bb_mid, ema100)):
        return None

    # ——— ÖZGÜN GİRİŞ KOŞULU (kaynakla birebir) ———
    if not (price < ema100 and price < 0.985 * bb_lo):
        return None
    if v is not None and len(v) > 31:
        hacim_ort = float(v.rolling(30).mean().shift(1).iloc[-1])
        if np.isfinite(hacim_ort) and not (float(v.iloc[-1]) < hacim_ort * 20.0):
            return None

    if bb_mid <= price:                     # orta bant hedefi fiyatın üstünde olmalı
        return None
    stop = bb_lo - 0.8 * atr_abs
    if stop >= price:
        return None
    if (price - stop) / price * 100.0 > float(getattr(p, "max_stop_pct", 2.0)):
        return None

    return {
        "direction": "LONG",
        "size": 0.5,
        "stop_hint": stop,
        "target_hint": bb_mid,
        "note": (f"EMA100 altı ve alt bandın %1,5 altı ({price:.6g} < {0.985 * bb_lo:.6g}) "
                 f"→ orta bant {bb_mid:.6g} hedefi"),
    }

# ═════════════════════════════════════════════════════════════════════════
# BİZ ÖLÇTÜK — 2026-09-05T18:25:05Z · binance · 60 GÜN · 5 parite (1 dk bar)
# ═════════════════════════════════════════════════════════════════════════
# Bu blok SİLİNMEZ. Çürütülen ölçüm de kayıttır: bir kurulumun neden gölgede ya da
# reddedilmiş olduğu, sonradan bakan birinin yeniden ölçmek zorunda kalmaması için
# burada durur. Bütün katkılar AYNI 60 günlük pencerede, aynı maliyetle ölçüldü.
#
#   pencere 28640 · ateşleme 0 · oran %0.0
#
# VERDİKT: REDDEDİLDİ — hiç ateşlemedi
#
# NEDEN HİÇ ATEŞLEMEDİ — ÖLÇÜLDÜ (ilk hipotez YANLIŞ çıktı):
# İlk açıklama 'zaman dilimi 5 dk'dan 1 dk'ya indiği için' olacaktı. Ölçüm bunu
# ÇÜRÜTTÜ. Aynı 7 günlük veride ham koşulların sıklığı:
#     close < bb_lower            → %5,47   (1 dk)   ·  %5,37 (5 dk)
#     close < 0,985 × bb_lower    → %0,000  (1 dk)   ·  %0,000 (5 dk)
# Yani koşul 5 DAKİKALIK barlarda da ulaşılamaz. Sebep ölçek: Bollinger yarım
# genişliği bu pariteler için %0,14–0,24 (1 dk) / %0,32–0,52 (5 dk) iken 0,985
# çarpanı bandın %1,5 ALTINI ister — bant genişliğinin 3–10 katı.
# Yani bu kurulum sakin bir piyasada büyük pariteler için TANIM GEREĞİ ölü;
# ancak çöküş günlerinde ya da yüksek oynaklıklı küçük paritelerde ateşler.
# Port doğrudur: ham koşul bağımsız olarak ölçüldü ve o da 0 verdi.
# 60 GÜNLÜK PENCEREDE DE 0: ilk ölçüm 7 gündü ve 'pencere kısa olabilir' itirazı
# meşruydu. 60 güne (Binance, 5×86.400 bar) çıkarıldı — sonuç yine 0 ateşleme.
# Yani bu kurulum büyük paritelerde pratikte ölüdür; ancak bir çöküş gününde ya da
# yüksek oynaklıklı küçük paritelerde ateşler. Bu pariteler+dönem için ölçülmüştür.
#
# Yeniden ölçmek için:
#   python scripts/cm_verify_contribution.py --sleeve cluc_may_freqtrade --days 60 --venue binance --step 15
MEASURED = {
    "window": "60 gün · binance · 1 dk",
    "n_windows": 28640, "n_fires": 0, "n_effective": None,
    "fire_rate_pct": 0.0,
    "verdict": "REJECTED",
}
