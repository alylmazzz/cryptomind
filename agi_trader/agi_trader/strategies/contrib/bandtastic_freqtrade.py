"""
Bandtastic — Bollinger alt bandı (1σ) kırılımında alım.

KAYNAK: freqtrade/freqtrade-strategies · `user_data/strategies/Bandtastic.py` (GPL-3.0)
        https://github.com/freqtrade/freqtrade-strategies

ÖZGÜN KURAL (kaynaktan doğrulandı, 2026-09-05):
    giriş : kapanış < bb_lower(20, 1σ)  VE  hacim > 0
            (RSI < 52, MFI < 30, EMA211 > EMA250 kapıları VARSAYILAN OLARAK KAPALI)
    çıkış : kapanış > bb_upper(20, 2σ)  VE  MFI > 46  VE  hacim > 0
    minimal_roi 0:%16,2 → 566dk:0 · stoploss −0,345 · timeframe '15m'

KOD KOPYALANMADI; kural bu deponun kendi araçlarıyla bağımsız yazıldı.

════════════════════════════════ NEDEN BU STRATEJİ ÖLÇÜLÜYOR (dürüstlük notu)
Bandtastic hyperopt için tasarlanmış bir İSKELETTİR: RSI/MFI/EMA kapılarının hepsi
açılıp kapanabilir ve varsayılanda KAPALIDIR. Yani varsayılan hâliyle strateji tek bir
koşula indirgenir: "kapanış 1 sigma alt bandın altında". Bu, BbandRsi'nin (2σ + RSI<30)
çok daha GEVŞEK bir akrabasıdır.

Buradaki ölçüm bu yüzden ikinci bir hipotez testidir: BbandRsi 2σ + RSI teyidiyle
kaybediyordu; teyit kaldırılıp eşik 1σ'ya gevşetilirse ne olur? Beklenti (ölçümden ÖNCE
yazıldı): daha gevşek eşik daha çok ateşleme + daha çok gürültü ⇒ en az BbandRsi kadar
kötü, muhtemelen daha kötü. Aksi çıkarsa "teyit eklemek zarar veriyor" gibi ilginç bir
sonuç olurdu.

──────────────────────────────────────────────────────────────── SAPMALAR
1. ZAMAN DİLİMİ: özgün 15 dakikalık; bu port 1 dakikalık barlarda ölçülür.
2. ÇIKIŞ: özgün çıkış "kapanış > bb_upper(2σ) VE MFI > 46". Çerçevede hedef olarak
   bb_upper(20, 2σ) kullanılır (FIXED_TARGET); MFI teyidi hedefe çevrilemediği için
   UYGULANMADI — bu, çıkışı özgününden bir miktar KOLAY yapar (lehte sapma).
3. STOP: özgün −%34,5 stop 1 dk ufkunda anlamsızdır → oransal stop,
   giriş × (1 − min(%34, max_stop_pct)); pratikte max_stop_pct bağlar.
4. VARSAYILAN KAPILAR: RSI/MFI/EMA kapıları kaynakta varsayılan olarak kapalı olduğu
   için burada da uygulanmadı. Kaynağı sadık biçimde temsil eden budur.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

META = {
    "name": "bandtastic_freqtrade",
    "title_tr": "Bandtastic — BB 1σ alt band (freqtrade örneği)",
    "author": "@alylmazzz (port) · freqtrade katkıcıları (özgün kural)",
    "source": "freqtrade/freqtrade-strategies · Bandtastic.py (GPL-3.0) — "
              "https://github.com/freqtrade/freqtrade-strategies",
    "claim": "Kapanış Bollinger'in 1 sigma alt bandının altına inerse ucuzlamıştır ve "
             "üst banda doğru toparlanır.",
    "claim_evidence": "YOK — kaynak depo bu stratejileri 'hazır kullanım değil' diye "
                      "sunuyor. Ayrıca strateji hyperopt İSKELETİDİR: varsayılan hâlinde "
                      "bütün teyit kapıları kapalıdır, geriye tek bir eşik kalır.",
    "mechanism": "kapanış < bb_lower(20, 1σ) VE hacim > 0. Hedef: bb_upper(20, 2σ). "
                 "Stop: giriş × (1 − min(%34, max_stop_pct)).",
    "exit_mode": "FIXED_TARGET",
    "time_stop_min": 180,
    "urgency": 0,
    "regimes": ["RANGE / YATAY", "VOLATİL", "TREND AŞAĞI"],
}


def fire(f: Dict, p, price: float, atr_abs: float, df=None) -> Optional[Dict]:
    if df is None or len(df) < 60:
        return None
    if "volume" in df and float(df["volume"].iloc[-1]) <= 0:
        return None
    try:
        c = df["close"].astype(float)
        sma20 = c.rolling(20).mean()
        sd20 = c.rolling(20).std(ddof=0)
        bb_lo1 = float(sma20.iloc[-1] - 1.0 * sd20.iloc[-1])     # 1σ (özgün giriş tetiği)
        bb_up2 = float(sma20.iloc[-1] + 2.0 * sd20.iloc[-1])     # 2σ (özgün çıkış tetiği)
    except Exception:
        return None
    if not (np.isfinite(bb_lo1) and np.isfinite(bb_up2)):
        return None

    # ——— ÖZGÜN GİRİŞ KOŞULU (varsayılan kapılarla birebir) ———
    if not (price < bb_lo1):
        return None
    if bb_up2 <= price:
        return None

    stop_pct = min(34.0, float(getattr(p, "max_stop_pct", 2.0)))
    stop = price * (1.0 - stop_pct / 100.0)
    return {
        "direction": "LONG",
        "size": 0.5,
        "stop_hint": stop,
        "target_hint": bb_up2,
        "note": (f"kapanış 1σ alt bandın altında ({price:.6g} < {bb_lo1:.6g}) → "
                 f"2σ üst bant {bb_up2:.6g} hedefi · stop %{stop_pct:.1f}"),
    }

# ═════════════════════════════════════════════════════════════════════════
# BİZ ÖLÇTÜK — binance · 60 gün · BÜYÜK pariteler (1 dk)
# ═════════════════════════════════════════════════════════════════════════
#   pencere 14330 · ateşleme 3569 · oran %24.906
#
# VERDİKT: REDDEDİLDİ — ateşleme oranı %24.9 > %15
#
# ÖLÇÜMDEN ÖNCE YAZILAN BEKLENTİ DOĞRULANDI: BbandRsi (2σ + RSI<30 teyidi) zaten
# kaybediyordu; teyit kaldırılıp eşik 1σ'ya gevşetilince ateşleme %0,93'ten %24,9'a
# fırladı ve kurulum seçicilik kapısında elendi. Daha gevşek eşik = daha çok gürültü.
# Not: bu strateji bir hyperopt İSKELETİDİR — varsayılan hâlinde bütün teyit kapıları
# kapalı olduğu için geriye tek bir eşik kalır. Ölçülen budur.
MEASURED = {
    "window": "60 gün · binance · 1 dk · büyük pariteler",
    "n_windows": 14330, "n_fires": 3569, "n_effective": None,
    "fire_rate_pct": 24.906,
    "verdict": "REJECTED",
}

# ─────────────────────────────────────────────────────────────────────────
# KÜÇÜK/OYNAK PARİTELER — BONK, ORDI, PYTH, ARB, PEPE
# ─────────────────────────────────────────────────────────────────────────
#   pencere 14330 · ateşleme 3102 · oran %21.647
#
# VERDİKT: REDDEDİLDİ — ateşleme oranı %21.6 > %15
#
MEASURED_SMALL_CAPS = {
    "window": "60 gün · binance · 1 dk · küçük/oynak pariteler",
    "n_windows": 14330, "n_fires": 3102, "n_effective": None,
    "fire_rate_pct": 21.647,
    "verdict": "REJECTED",
}
