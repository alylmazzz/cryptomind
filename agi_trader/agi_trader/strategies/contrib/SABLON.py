"""
ŞABLON — bu dosya YÜKLENMEZ (yükleyici 'SABLON' adını atlar). Kopyalayıp kendi adınızla
kaydedin: `contrib/benim_kurulumum.py`.

Katkı süreci (ayrıntı: depo kökündeki CONTRIBUTING.md):
  1. Bu dosyayı kopyalayın, META'yı ve `fire`'ı doldurun.
  2. `python scripts/cm_verify_contribution.py --sleeve benim_kurulumum --days 7`
     → statik kontroller + GERÇEK veride ateşleme oranı + maliyet düşülmüş kenar ölçümü.
  3. Çıktıyı PR açıklamasına yapıştırın. Ölçüm negatif çıkarsa da gönderin — çürütülen
     ölçüm de bilgidir ve bu depoda silinmez.
  4. Katkınız SHADOW olarak birleşir: sinyal üretir, EMİR VERMEZ. Kanıt kapılarını
     geçerse PAPER'a terfi eder.

`f` sözlüğünde bulabileceğiniz alanlardan bazıları (tamamı için
`committee.fast_features`, `sleeves_fast.extra_features`, `sleeves_video.video_features`):

    ok, price, close, rsi, atr_pct, ema20, ema50, dist_ema_pct, trend_up, trend_down,
    vol_ratio, z, breakout_near, range_hi, range_lo, cvd, book_imbalance, spread_bps,
    bb_lower, bb_mid, bb_width_pct_v, at_bb_lower, stoch_k, stoch_d,
    htf_range, opening_range, vprofile, near_val, near_poc, fvg_bull, fvg_bear,
    manip_bull, engulf_bull, mss_up, ob_lo, ob_hi, in_ob, swing_hi_piv, swing_lo_piv

KENDİ GÖSTERGENİZ (opsiyonel): `f`'te olmayan bir gösterge gerekiyorsa `fire`'a BEŞİNCİ
parametre olarak `df` ekleyin — özellikleri üreten AYNI bar çerçevesini alırsınız ve
göstergenizi kendiniz hesaplarsınız:

    def fire(f, p, price, atr_abs, df):
        adx = ...  # df["high"], df["low"], df["close"], df["volume"]

Bu, gerçek açık kaynak stratejilerin çoğu için ZORUNLUDUR: DI±, MACD, SAR, mum
formasyonları gibi göstergeler `f`'te yoktur. Beşinci parametrenin adı `df` OLMALIDIR;
başka bir ad ya da altıncı parametre yükleyici tarafından reddedilir.

KURALLAR (yükleyici bunları zorlar):
  * Ağ isteği, dosya okuma, `time.sleep`, global durum YASAK — yalnız `f` ve `df`'den okuyun.
  * Geleceğe bakmayın: `df` size sistemin geri kalanının gördüğü çerçevenin AYNISIDIR;
    `shift(-n)` gibi ileri kaydırma yapan yardımcı yazmayın (doğrulayıcı bunu arar ve
    reddeder). `df.iloc[-1]` son bardır — ondan sonrasına erişiminiz YOKTUR.
  * Spot hesapta SHORT üretilmez; `direction` "LONG" olmalıdır.
  * `size` 0–1 arasıdır ve zaten kelepçelenir.
"""
from __future__ import annotations

from typing import Dict, Optional

META = {
    "name": "sablon_kurulum",              # DEĞİŞTİRİN — [a-z][a-z0-9_]{2,39}
    "title_tr": "Şablon kurulum",          # panelde görünen ad (opsiyonel)
    "author": "@github_kullanici_adiniz",
    "source": "Kaynak: kitap/makale/video URL'si ya da 'kendi fikrim'",
    "claim": "Bu kurulumun ne yaptığı iddia ediliyor? (ör. 'düşük hacimli sıkışma "
             "sonrası kırılım, 1:2 R/R ile pozitif beklenti')",
    "claim_evidence": "İddianın kanıtı nedir? Geri test var mı, kaç işlem, hangi dönem? "
                      "KANIT YOKSA 'YOK' YAZIN — dürüstlük katkınızı reddettirmez, "
                      "abartılmış iddia reddettirir.",
    "mechanism": "Mekanik tanım: hangi koşullar aynı anda sağlanınca tetiklenir. "
                 "Öznel ifade ('bariz destek', 'güçlü momentum') kabul edilmez; "
                 "her koşul koda çevrilebilir olmalıdır.",
    "exit_mode": "PARTIAL_AND_RUN",        # FIXED_TARGET | PARTIAL_AND_RUN | DYNAMIC_PEAK
    "time_stop_min": 90,
    "urgency": 0,                          # 0 = acele yok → maker giriş (komisyon avantajı)
    "regimes": ["TREND YUKARI"],           # hangi rejimlerde değerlendirilsin
}


def fire(f: Dict, p, price: float, atr_abs: float) -> Optional[Dict]:
    """Tetiklenmiyorsa None döndürün. Tetikleniyorsa aşağıdaki sözlüğü döndürün.

    `p` komite parametreleridir (ör. `p.max_stop_pct`), `atr_abs` mutlak ATR'dir.
    """
    rsi = f.get("rsi")
    if rsi is None or not f.get("trend_up"):
        return None

    # ——— ÖRNEK KOŞUL (kendi mantığınızla değiştirin) ———
    if not (35.0 <= float(rsi) <= 55.0):
        return None

    stop = price - 1.2 * atr_abs
    if (price - stop) / price * 100.0 > float(getattr(p, "max_stop_pct", 2.0)):
        return None                         # stop çok uzaksa kurulum geçersiz

    return {
        "direction": "LONG",
        "size": 0.6,                        # 0–1; kanıtsız kurulumda küçük tutun
        "stop_hint": stop,
        "target_hint": price + 2.4 * atr_abs,
        "note": f"şablon: trend yukarı + RSI {float(rsi):.0f} geri çekilme bandında",
    }
