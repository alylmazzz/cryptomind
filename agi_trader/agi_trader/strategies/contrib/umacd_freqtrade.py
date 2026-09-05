"""
UniversalMACD — EMA12/EMA26 oranının DAR bir bandına giriş (hyperopt ile bulunmuş).

KAYNAK: freqtrade/freqtrade-strategies · `user_data/strategies/UniversalMACD.py` (GPL-3.0)
        https://github.com/freqtrade/freqtrade-strategies

ÖZGÜN KURAL (kaynaktan doğrulandı, 2026-09-05):
    umacd = (EMA12 / EMA26) − 1
    giriş : umacd ∈ [−0,01416 , −0,01176]
    çıkış : umacd ∈ [−0,02323 , −0,00707]
    minimal_roi 0:%21,3 → 164dk:0 · stoploss −0,318 · timeframe '5m'

KOD KOPYALANMADI; kural bu deponun kendi araçlarıyla bağımsız yazıldı.

═══════════════════════════════ NEDEN BU STRATEJİ ÖLÇÜLÜYOR (dürüstlük notu)
Bu kurulum diğerlerinden farklı bir sebeple seçildi: giriş bandı **0,0024 genişliğinde**
(−%1,416 ile −%1,176 arası) ve bu sayılar hyperopt ile bulunmuş. Bu, aşırı-uydurmanın
(overfitting) ders kitabı imzasıdır — bir parametre bandı ne kadar dar ve ne kadar
"özel" ise, geçmişe uydurulmuş olma olasılığı o kadar yüksektir. Buradaki ölçüm bu
yüzden bir strateji denemesi kadar bir HİPOTEZ TESTİdir: hyperopt ile bulunmuş dar bir
bant, kendi verisinin dışında da anlamlı mı?

BEKLENTİ (ölçümden ÖNCE yazıldı ki sonradan rasyonalize edilmesin): bu kadar dar bir
bant başka bir borsada, başka paritelerde ve başka bir dönemde kenar üretMEmeli.
Üretirse bu, bandın ekonomik bir mekanizmaya karşılık geldiğinin işareti olurdu ve
şaşırtıcı olurdu.

──────────────────────────────────────────────────────────────── SAPMALAR
1. ZAMAN DİLİMİ: özgün 5 dakikalık; bu port 1 dakikalık barlarda ölçülür.
2. ÇIKIŞ: özgün çıkış yine bir umacd bandıdır → çerçevede stop/hedef kullanılır
   (FIXED_TARGET, hedef 2×ATR — özgün %21,3 ROI 1 dk ufkunda ulaşılamaz).
3. STOP: özgün −%31,8 → oransal stop, giriş × (1 − min(%32, max_stop_pct));
   pratikte max_stop_pct bağlar.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

META = {
    "name": "umacd_freqtrade",
    "title_tr": "UniversalMACD — dar hyperopt bandı (freqtrade örneği)",
    "author": "@alylmazzz (port) · freqtrade katkıcıları (özgün kural)",
    "source": "freqtrade/freqtrade-strategies · UniversalMACD.py (GPL-3.0) — "
              "https://github.com/freqtrade/freqtrade-strategies",
    "claim": "(EMA12/EMA26 − 1) değeri −0,01416 ile −0,01176 arasındayken alım fırsatı vardır.",
    "claim_evidence": "YOK ve DAHASI VAR: bu eşikler hyperopt ile bulunmuş, 0,0024 "
                      "genişliğinde bir bant. Dar ve 'özel' parametre bandı aşırı-uydurma "
                      "imzasıdır. Kaynak depo zaten 'kendi backtest'inizi koşun' diyor.",
    "mechanism": "umacd = EMA12/EMA26 − 1; umacd ∈ [−0,01416, −0,01176]. "
                 "Hedef 2×ATR; stop giriş × (1 − min(%32, max_stop_pct)).",
    "exit_mode": "FIXED_TARGET",
    "time_stop_min": 120,
    "urgency": 0,
    "regimes": ["RANGE / YATAY", "VOLATİL", "TREND AŞAĞI"],
}


def fire(f: Dict, p, price: float, atr_abs: float, df=None) -> Optional[Dict]:
    if df is None or len(df) < 120:
        return None
    try:
        c = df["close"].astype(float)
        ema12 = float(c.ewm(span=12, adjust=False).mean().iloc[-1])
        ema26 = float(c.ewm(span=26, adjust=False).mean().iloc[-1])
    except Exception:
        return None
    if not (np.isfinite(ema12) and np.isfinite(ema26)) or ema26 == 0:
        return None
    umacd = ema12 / ema26 - 1.0

    # ——— ÖZGÜN GİRİŞ KOŞULU (kaynakla birebir, hyperopt değerleri dahil) ———
    if not (-0.01416 <= umacd <= -0.01176):
        return None

    stop_pct = min(32.0, float(getattr(p, "max_stop_pct", 2.0)))
    stop = price * (1.0 - stop_pct / 100.0)
    return {
        "direction": "LONG",
        "size": 0.5,
        "stop_hint": stop,
        "target_hint": price + 2.0 * atr_abs,
        "note": (f"umacd {umacd:+.5f} dar bandın içinde [−0,01416, −0,01176] · "
                 f"stop %{stop_pct:.1f}"),
    }

# ═════════════════════════════════════════════════════════════════════════
# BİZ ÖLÇTÜK (TAM TARAMA) — binance · 60 gün · BÜYÜK pariteler
# ═════════════════════════════════════════════════════════════════════════
#   ham koşul 8 bar · örtüşmeyen işlem 3
#   ortalama net %-0.2986 · t -0.19 · kazanma %66.7
#   çıkış sebepleri: {'STOP': 1, 'ZAMAN': 2}
#
# VERDİKT: ÖLÇÜLEMEDİ (n < 30)
#
# ⚠️ BU ÖLÇÜM TAM TARAMAYLA YAPILDI, doğrulayıcının adım örneklemesiyle DEĞİL.
# Sebebi: bant o kadar dar ki büyük paritelerde 54.000 barda bir, küçüklerde 4.700
# barda bir oluşuyor. Adım 15 ile beklenen yakalama büyüklerde ~1, küçüklerde ~6 —
# yani örnekleme bu kurulumu yapısal olarak göremez.
#
# ÖLÇÜMDEN ÖNCE YAZILAN BEKLENTİ DOĞRULANDI: dosyanın başındaki not, hyperopt ile
# bulunmuş 0,0024 genişliğindeki bandın kendi verisi dışında kenar üretmemesini
# bekliyordu. Sonuç: 60 günde 10 paritede toplam 18 bağımsız işlem, ort +%0,29,
# t +0,52 → n < 30 ve anlamlı değil. ÖLÇÜLEMEDİ.
#
# Buradaki asıl bulgu kâr/zarar değil TAŞINABİLİRLİK: bir parametre bandı, üzerinde
# optimize edildiği veriden çıkınca neredeyse hiç tetiklenmiyorsa, o bant bir
# piyasa mekanizmasını değil o verinin gürültüsünü tarif ediyordur.
MEASURED = {
    "window": "60 gün · binance · 1 dk · büyük pariteler · TAM TARAMA",
    "n_raw_bars": 8, "n_effective": 3,
    "mean_net_pct": -0.2986, "t_stat": -0.19,
    "verdict": "UNMEASURABLE",
}

# ─────────────────────────────────────────────────────────────────────────
# KÜÇÜK/OYNAK PARİTELER (TAM TARAMA) — BONK, ORDI, PYTH, ARB, PEPE
# ─────────────────────────────────────────────────────────────────────────
#   ham koşul 91 bar · örtüşmeyen işlem 15
#   ortalama net %+0.4075 · t +0.70 · kazanma %73.3
#   çıkış sebepleri: {'HEDEF': 9, 'ZAMAN': 4, 'STOP': 2}
#
# VERDİKT: ÖLÇÜLEMEDİ (n < 30)
#
MEASURED_SMALL_CAPS = {
    "window": "60 gün · binance · 1 dk · küçük pariteler · TAM TARAMA",
    "n_raw_bars": 91, "n_effective": 15,
    "mean_net_pct": 0.4075, "t_stat": 0.7,
    "verdict": "UNMEASURABLE",
}
