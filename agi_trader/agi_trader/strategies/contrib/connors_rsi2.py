"""
Connors RSI(2) — trend içi aşırı satım (yayımlanmış kitap kuralı).

KAYNAK: Larry Connors & Cesar Alvarez, "Short Term Trading Strategies That Work".
        Kural kamuya açık ve çok sayıda bağımsız yerde aynı biçimde belgelenmiş:
        · StockCharts ChartSchool — https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models/trading-strategies/rsi-2
        · QuantifiedStrategies — https://www.quantifiedstrategies.com/rsi-2-strategy/

ÖZGÜN KURAL (doğrulandı, 2026-09-05):
    giriş : kapanış > SMA200  VE  kapanış < SMA5  VE  RSI(2) < 10
    çıkış : kapanış > SMA5  (tipik 2–4 gün; 10 günde çıkış olmazsa gün sonu kapat)
    STOP  : YOKTUR. Connors'ın kendi geri testinde sabit stop eklemek kârlılığı
            DÜŞÜRÜYOR; koruma 200 günlük ortalama filtresinden geliyor.
    zaman dilimi: GÜNLÜK barlar.

NEDEN BU STRATEJİ EKLENDİ: önceki yedi ölçümün hepsi TEK bir depodan
(freqtrade-strategies) geliyordu ve o deponun README'si stratejilerini zaten "hazır
kullanım değil, örnek" diye sunuyor. Tek kaynaklı örneklem yanlılığını kırmak için
farklı bir provenance seçildi: yayımlanmış, yaygın biçimde alıntılanan bir kitap kuralı.

═══════════════════════════════════════ SAPMALAR (bu ikisi BÜYÜK — dikkatle okuyun)
1. ZAMAN DİLİMİ: özgün kural **GÜNLÜK** barlar içindir. Bu port komitenin 1 dakikalık
   barlarında ölçülür; yani SMA200 = 200 DAKİKA (~3,3 saat), SMA5 = 5 dakika.
   "200 günlük trend filtresi" ile "200 dakikalık trend filtresi" aynı şey DEĞİLDİR.
   Bu ölçüm "Connors RSI(2) kârlı mı"yı DEĞİL, "aynı kural yapısı bu sistemin zaman
   diliminde kenar üretiyor mu"yu yanıtlar.
2. STOP: özgün sistemde stop YOKTUR ve Connors stop eklemenin zarar verdiğini yazar.
   Bu çerçeve stopsuz pozisyon taşıyamaz (koşucu stop ister). Oransal stop kullanılır:
   `giriş × (1 − max_stop_pct)`. Bu, stratejinin ASLINA AYKIRIDIR ve sonucu olumsuz
   yönde etkilemesi beklenir — sonuç negatif çıkarsa bu sapma akılda tutulmalıdır.
3. ÇIKIŞ: özgün çıkış "kapanış > SMA5". Çerçevede hedef olarak SMA5 kullanılır
   (FIXED_TARGET) — bu, özgün çıkışın en yakın karşılığıdır.

PENCERE NOTU: doğrulayıcı 240 barlık pencere verir; SMA200 için yalnız son ~40 bar
geçerli olur. Kurulum bu yüzden yalnız pencerenin sonunda değerlendirilebilir — zaten
tek bir bar için karar veriliyor, sorun değil.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

META = {
    "name": "connors_rsi2",
    "title_tr": "Connors RSI(2) — trend içi aşırı satım",
    "author": "@alylmazzz (port) · Larry Connors & Cesar Alvarez (özgün kural)",
    "source": "Connors & Alvarez, 'Short Term Trading Strategies That Work'; kural "
              "StockCharts ChartSchool ve QuantifiedStrategies'te aynı biçimde belgeli",
    "claim": "Fiyat 200 periyotluk ortalamanın ÜSTÜNDEyken (trend yukarı) RSI(2) 10'un "
             "altına inerse bu geçici bir aşırı satımdır ve fiyat kısa sürede toparlanır.",
    "claim_evidence": "VAR ama BAĞIMSIZ DEĞİL: Connors kendi kitabında geniş geri testler "
                      "yayımladı (ABD hisseleri/endeksleri, günlük barlar). Üçüncü taraf "
                      "denetimli bir kanıt YOK ve kripto 1 dakikalık barlarda hiç test "
                      "edilmedi. Kuralın kripto/dakikalık bağlama taşındığına dair kanıt YOK.",
    "mechanism": "kapanış > SMA200 VE kapanış < SMA5 VE RSI(2) < 10. Hedef: SMA5. "
                 "Stop: giriş × (1 − max_stop_pct) — ÖZGÜN SİSTEMDE STOP YOKTUR (bkz. SAPMALAR).",
    "exit_mode": "FIXED_TARGET",
    "time_stop_min": 60,
    "urgency": 0,
    "regimes": ["TREND YUKARI", "RANGE / YATAY"],
}


def _rsi(s: pd.Series, n: int = 2) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0.0).ewm(alpha=1.0 / n, adjust=False).mean()
    dn = (-d).clip(lower=0.0).ewm(alpha=1.0 / n, adjust=False).mean()
    return 100.0 - 100.0 / (1.0 + up / dn.replace(0.0, np.nan))


def fire(f: Dict, p, price: float, atr_abs: float, df=None) -> Optional[Dict]:
    if df is None or len(df) < 205:
        return None
    try:
        c = df["close"].astype(float)
        sma200 = float(c.rolling(200).mean().iloc[-1])
        sma5 = float(c.rolling(5).mean().iloc[-1])
        rsi2 = float(_rsi(c, 2).iloc[-1])
    except Exception:
        return None
    if not all(np.isfinite(x) for x in (sma200, sma5, rsi2)):
        return None

    # ——— ÖZGÜN GİRİŞ KOŞULU (kaynakla birebir) ———
    if not (price > sma200 and price < sma5 and rsi2 < 10.0):
        return None
    if sma5 <= price:                       # hedef girişin üstünde olmalı
        return None

    stop_pct = float(getattr(p, "max_stop_pct", 2.0))
    stop = price * (1.0 - stop_pct / 100.0)
    return {
        "direction": "LONG",
        "size": 0.5,
        "stop_hint": stop,
        "target_hint": sma5,
        "note": (f"SMA200 üstü trend + RSI(2) {rsi2:.0f} < 10 aşırı satım · "
                 f"hedef SMA5 {sma5:.6g} · stop %{stop_pct:.1f} (özgünde stop YOK)"),
    }

# ═════════════════════════════════════════════════════════════════════════
# BİZ ÖLÇTÜK — binance · 60 gün · BÜYÜK pariteler (1 dk)
# ═════════════════════════════════════════════════════════════════════════
#   pencere 14350 · ateşleme 792 · oran %5.519
#   örneklem: nominal 792 → ETKİN 719
#   ortalama net %-0.1373 · t -16.67 · CI95 [-0.1547, -0.1219] · kazanma %6.3
#   çıkış sebepleri: {'HEDEF': 634, 'ZAMAN': 85}
#   2× maliyette beklenti: %-0.2773
#
# VERDİKT: GÖLGE
#
# ‼️ BU ÖLÇÜMÜN ASIL BULGUSU: KALIP ÇALIŞIYOR, HAREKET MALİYETTEN KÜÇÜK.
#
# Kurulum hedefini (SMA5) BÜYÜK paritelerde %88,2, küçüklerde %90,5 oranında
# tutturuyor. Yani 'RSI(2) dipte alıp 5 periyot ortalamasına dönüş' kalıbı GERÇEK
# ve güvenilir. Sorun kalıbın yanlışlığı değil, ÖLÇEĞİ:
#
#     grup     hedef tutturma   BRÜT hareket   maliyet   NET
#     BÜYÜK        %88,2         +%0,0027      %0,14    −%0,1373
#     KÜÇÜK        %90,5         +%0,0617      %0,14    −%0,0783
#
# Büyük paritelerde yakalanan brüt hareket, gidiş-dönüş maliyetin 1/50'si kadar.
# Küçük (oynak) paritelerde brüt 23 KAT büyüyor (+%0,0027 → +%0,0617) ama hâlâ
# maliyetin yarısına ulaşamıyor. Oynaklık arttıkça kalıp daha çok kazandırıyor —
# yalnızca yeterince değil.
#
# BU, ÖNCEKİ DOKUZ ÖLÇÜMÜ DE YENİDEN OKUTUR: bu stratejilerin çoğu 'yanlış' değil,
# SÜRTÜNMENİN KENARDAN BÜYÜK OLDUĞU bir zaman diliminde çalıştırılıyor. Özgün kural
# GÜNLÜK barlar içindir; orada SMA5 mesafesi mertebe olarak daha büyüktür.
# Bu ölçüm 'Connors RSI(2) kârsızdır' DEMEZ; '1 dakikalık kripto barlarında, bu
# maliyet yapısında kârsızdır' der.
#
# AYRICA HATIRLATMA (SAPMALAR-2): özgün sistemde STOP YOKTUR ve Connors stop
# eklemenin kârlılığı düşürdüğünü yazar. Bu port stop kullanmak zorundaydı.
MEASURED = {
    "window": "60 gün · binance · 1 dk · büyük pariteler",
    "n_windows": 14350, "n_fires": 792, "n_effective": 719,
    "fire_rate_pct": 5.519,
    "mean_net_pct": -0.1373, "t_stat": -16.67, "ci95": [-0.1547, -0.1219],
    "win_rate": 0.063, "exit_reasons": {'HEDEF': 634, 'ZAMAN': 85},
    "expectancy_cost_x2_pct": -0.2773,
    "verdict": "SHADOW",
}

# ─────────────────────────────────────────────────────────────────────────
# KÜÇÜK/OYNAK PARİTELER — BONK, ORDI, PYTH, ARB, PEPE
# ─────────────────────────────────────────────────────────────────────────
#   pencere 14350 · ateşleme 692 · oran %4.822
#   örneklem: nominal 692 → ETKİN 629
#   ortalama net %-0.0783 · t -4.27 · CI95 [-0.1175, -0.0445] · kazanma %44.0
#   çıkış sebepleri: {'HEDEF': 569, 'ZAMAN': 56, 'STOP': 4}
#   2× maliyette beklenti: %-0.2183
#
# VERDİKT: GÖLGE
#
MEASURED_SMALL_CAPS = {
    "window": "60 gün · binance · 1 dk · küçük/oynak pariteler",
    "n_windows": 14350, "n_fires": 692, "n_effective": 629,
    "fire_rate_pct": 4.822,
    "mean_net_pct": -0.0783, "t_stat": -4.27, "ci95": [-0.1175, -0.0445],
    "win_rate": 0.44, "exit_reasons": {'HEDEF': 569, 'ZAMAN': 56, 'STOP': 4},
    "expectancy_cost_x2_pct": -0.2183,
    "verdict": "SHADOW",
}
