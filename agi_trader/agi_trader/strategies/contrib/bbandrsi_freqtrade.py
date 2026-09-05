"""
BbandRsi — freqtrade örnek stratejilerinden uyarlanan aşırı-satım ortalamaya dönüş kurulumu.

KAYNAK: freqtrade/freqtrade-strategies · `user_data/strategies/berlinguyinca/BbandRsi.py` (GPL-3.0)
        https://github.com/freqtrade/freqtrade-strategies

ÖZGÜN KURAL (kaynaktan doğrulandı, 2026-09-05):
    giriş : rsi < 30  AND  close < bb_lowerband
    çıkış : rsi > 70
    göstergeler: RSI timeperiod=14 · Bollinger window=20, stds=2
    minimal_roi {"0": 0.1} · stoploss −0.25 · timeframe '1h'

KOD KOPYALANMADI. Kaynak GPL-3.0'dır; buraya yalnızca *kural* (yayımlanmış, telif konusu
olmayan bir yöntem tanımı) bu deponun kendi özellik sözlüğü kullanılarak bağımsız biçimde
yazıldı. Künye yukarıdadır.

──────────────────────────────────────────────────────────────── SAPMALAR (dürüstlük notu)
Bu port, özgün stratejiyle AYNI ŞEY DEĞİLDİR. İki maddede bilerek ayrılır ve ölçüm bu
farkla birlikte okunmalıdır:

1. ZAMAN DİLİMİ. Özgün strateji **1 saatlik** barlarda çalışır. Burada komitenin bar
   zaman dilimi kullanılır (varsayılan **1 dakika**). RSI(14) < 30 koşulu 1 dakikalık
   barlarda saatlik barlara göre kıyasla ÇOK daha sık gerçekleşir; dolayısıyla bu ölçüm
   "BbandRsi kârlı mı" sorusunu DEĞİL, "BbandRsi'nin giriş kuralı bu sistemin zaman
   diliminde ve maliyet yapısında kenar üretiyor mu" sorusunu yanıtlar.

2. ÇIKIŞ. Özgün çıkış `rsi > 70`; ayrıca %10 ROI ve −%25 stop kullanır. −%25 stop bu
   sistemin 1 dakikalık ufkunda anlamsızdır (koşucu `max_stop_pct` ile reddeder). Burada
   çıkış, çerçevenin kendi motoruna bırakılır: hedef **orta bant** (ortalamaya dönüşün
   doğal hedefi), stop alt bandın biraz altında.

GÖSTERGE PARAMETRELERİ TUTUYOR: bu depoda `f["rsi"]` = RSI(14) ve `f["bb_lower"]`/`f["bb_mid"]`
= SMA20 ∓ 2σ — kaynağın 14 / 20 / 2 değerleriyle birebir aynı (ölçülerek doğrulandı).

──────────────────────────────────────────────────────────── MEVCUT SLEEVE'DEN FARKI
Depoda zaten `bb_lower_band` adlı bir video kurulumu var ama koşulu TERSİDİR:
`at_bb_lower = low <= bb_lower AND close > bb_lower` — yani alt bandı FİTİLLE delip
İÇERİDE kapanış ("dokun ve geri al"). BbandRsi ise kapanışın bandın ALTINDA olmasını
ister ve ek olarak RSI < 30 arar. Bunlar farklı kurulumlardır; ölçümleri karıştırılmamalı.
(`bb_lower_band` bu depoda ölçüldü: n 181 · ort −%0,093 · t −2,68.)
"""
from __future__ import annotations

from typing import Dict, Optional

META = {
    "name": "bbandrsi_freqtrade",
    "title_tr": "BbandRsi (freqtrade örneği)",
    "author": "@alylmazzz (port) · berlinguyinca (özgün kural)",
    "source": "freqtrade/freqtrade-strategies · berlinguyinca/BbandRsi.py (GPL-3.0) — "
              "https://github.com/freqtrade/freqtrade-strategies",
    "claim": "RSI(14) < 30 iken kapanış Bollinger(20,2) alt bandının ALTINDAysa piyasa aşırı "
             "satımdadır ve ortalamaya (orta banda) döner.",
    "claim_evidence": "YOK — kaynak deponun README'si bu stratejileri açıkça 'hazır kullanım "
                      "değil, kendi stratejiniz için başlangıç noktası' olarak sunuyor ve "
                      "'kendi backtest'inizi koşun' diyor. Yayımlanmış kâr kanıtı YOK. "
                      "Özgün bağlam 1 saatlik barlar; bu port 1 dakikalıkta ölçülür.",
    "mechanism": "rsi < 30 VE fiyat < bb_lower(20, 2σ). Hedef: orta bant (bb_mid). "
                 "Stop: bb_lower − 0,8×ATR; stop mesafesi max_stop_pct'i aşarsa TETİKLENMEZ.",
    "exit_mode": "FIXED_TARGET",
    "time_stop_min": 240,
    "urgency": 0,                      # ortalamaya dönüş acele istemez → maker giriş
    "regimes": ["RANGE / YATAY", "VOLATİL"],
}


def fire(f: Dict, p, price: float, atr_abs: float) -> Optional[Dict]:
    rsi = f.get("rsi")
    bb_lo = f.get("bb_lower")
    bb_mid = f.get("bb_mid")
    if rsi is None or bb_lo is None or bb_mid is None:
        return None

    # ——— ÖZGÜN GİRİŞ KOŞULU (kaynakla birebir) ———
    if not (float(rsi) < 30.0 and price < float(bb_lo)):
        return None

    # Orta bant hedefi fiyatın üstünde olmalı; değilse kurulum tutarsızdır.
    hedef = float(bb_mid)
    if hedef <= price:
        return None

    stop = float(bb_lo) - 0.8 * atr_abs
    if stop >= price:                                  # fiyat zaten stopun altındaysa geçersiz
        return None
    stop_pct = (price - stop) / price * 100.0
    if stop_pct > float(getattr(p, "max_stop_pct", 2.0)):
        return None                                    # çerçeve bu stopu zaten reddederdi

    return {
        "direction": "LONG",
        "size": 0.5,                                   # kanıtsız kurulum → küçük
        "stop_hint": stop,
        "target_hint": hedef,
        "note": (f"RSI {float(rsi):.0f} < 30 ve kapanış alt bandın altında "
                 f"({price:.6g} < {float(bb_lo):.6g}) → orta bant {hedef:.6g} hedefi"),
    }

# ═════════════════════════════════════════════════════════════════════════
# BİZ ÖLÇTÜK — 2026-09-05T18:18:57Z · binance · 60 GÜN · 5 parite (1 dk bar)
# ═════════════════════════════════════════════════════════════════════════
# Bu blok SİLİNMEZ. Çürütülen ölçüm de kayıttır: bir kurulumun neden gölgede ya da
# reddedilmiş olduğu, sonradan bakan birinin yeniden ölçmek zorunda kalmaması için
# burada durur. Bütün katkılar AYNI 60 günlük pencerede, aynı maliyetle ölçüldü.
#
#   pencere 28640 · ateşleme 265 · oran %0.925
#   örneklem: nominal 265 → ETKİN 230 (örtüşme şişmesi 1.15×)
#   ortalama net %-0.1379 · t -19.71 · CI95 [-0.1512, -0.1232] · kazanma %10.4
#   alt-dönem: ilk yarı %-0.135 · ikinci yarı %-0.1408
#   çıkış sebepleri: {'HEDEF': 33, 'STOP': 197}
#   2× maliyette beklenti: %-0.2779
#
# VERDİKT: GÖLGE
#
# GÖLGE bir RET DEĞİLDİR: kurulum sinyal üretir, EMİR VERMEZ ve ölçülmeye devam eder.
#
# İKİ PENCEREDE TUTARLI (her ikisi de örtüşme düzeltmesiyle):
#     7 gün · MEXC   → etkin n 62  · ort −%0,1323 · t −7,64  · CI95 [−0,163, −0,095]
#    60 gün · Binance → etkin n 230 · ort −%0,1379 · t −19,71 · CI95 [−0,151, −0,123]
# Farklı borsa, 8,6 kat veri; iki ortalama farkı 7 günlük ölçümün güven aralığı içinde.
# Alt dönemler de ayrışmıyor (−%0,135 / −%0,141). Kenarın yokluğu örneklem gürültüsü
# DEĞİL, kuralın bu zaman diliminde ve maliyet yapısındaki kararlı bir özelliği.
#
# NOT — ÖRTÜŞME DÜZELTMESİNİN ETKİSİ ÖLÇÜLDÜ: düzeltmeden önce 7 günlük ölçüm
# t = −10,63 diyordu; örtüşen ileri pencereler çıkarılınca t = −7,64'e indi
# (nominal 90 → etkin 62, şişme 1,5×). Yön değişmedi ama BÜYÜKLÜK %39 abartılıydı.
# Çıkışların 197/230'u stop: 1 dk barda 'aşırı satım' ortalamaya dönmüyor, düşüşün
# devamı oluyor.
# GÖLGE bir RET DEĞİLDİR: kurulum sinyal üretir, EMİR VERMEZ ve ölçülmeye devam
# eder. Kanıt pozitife dönerse terfi yolu açıktır.
#
# Yeniden ölçmek için:
#   python scripts/cm_verify_contribution.py --sleeve bbandrsi_freqtrade --days 60 --venue binance --step 15
MEASURED = {
    "window": "60 gün · binance · 1 dk",
    "n_windows": 28640, "n_fires": 265, "n_effective": 230,
    "fire_rate_pct": 0.925,
    "mean_net_pct": -0.1379, "t_stat": -19.71, "ci95": [-0.1512, -0.1232],
    "win_rate": 0.104, "exit_reasons": {'HEDEF': 33, 'STOP': 197},
    "expectancy_cost_x2_pct": -0.2779,
    "verdict": "SHADOW",
}
