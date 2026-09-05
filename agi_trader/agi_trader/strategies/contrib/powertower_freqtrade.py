"""
PowerTower — kapanışın, önceki kapanışın 3,849. KUVVETİYLE karşılaştırılması.

KAYNAK: freqtrade/freqtrade-strategies · `user_data/strategies/PowerTower.py` (GPL-3.0)
        https://github.com/freqtrade/freqtrade-strategies

ÖZGÜN KURAL (kaynaktan doğrulandı, 2026-09-05):
    giriş : close[0] > close[2] ** buy_pow  VE  close[1] > close[3] ** buy_pow
            VE  close[2] > close[4] ** buy_pow      (buy_pow = 3,849)
    çıkış : aynı karşılaştırmaların herhangi biri ters dönerse (sell_pow = 3,798)
    gösterge KULLANMAZ — yalnız kapanış fiyatlarının kuvvetleri.
    minimal_roi 0:%21,3 → 159dk:0 · stoploss −0,288 · timeframe '5m'

KOD KOPYALANMADI; kural bu deponun kendi araçlarıyla bağımsız yazıldı.

═══════════════════════════════════════════ BU KURAL BOYUTSAL OLARAK ANLAMSIZDIR
`close > close[2] ** 3.849` karşılaştırması bir fiyatı, bir fiyatın 3,849. kuvvetiyle
karşılaştırır. Bunların BİRİMİ farklıdır: BTC 80.000 iken sağ taraf 10^18 mertebesinde,
PEPE 0,0000036 iken 10^-21 mertebesinde olur. Yani koşul, piyasanın ne yaptığından
bağımsız olarak yalnızca **fiyatın 1'den büyük mü küçük mü** olduğunu ölçer.

BU BİR TAHMİN DEĞİL, ÖLÇÜM: aynı 60 günlük veride ham koşulun ateşleme sıklığı —

    fiyat > 1 olan pariteler                fiyat < 1 olan pariteler
      BTC   79.754   → %0,0000                ARB   0,1616   → %99,9954
      ETH    2.478   → %0,0000                PYTH  0,0548   → %99,9954
      SOL      103   → %0,0000                DOGE  0,0908   → %99,9954
      AVAX     7,60  → %0,0000                BONK  3,3e-06  → %99,9954
      ORDI     4,21  → %0,0000                PEPE  3,6e-06  → %99,9954

Kusursuz ikili ayrım. Kural bir alım-satım kurulumu değil, bir FİYAT BÜYÜKLÜĞÜ
detektörüdür. Hyperopt "buy_pow = 3,849" değerini bulmuştur çünkü optimize edildiği
veri kümesinde bu sayı işe yarayan bir eşik gibi görünmüştür — oysa ölçtüğü şey
stratejinin kendisi değil, evrendeki paritelerin fiyat ölçeğidir.

NEDEN YİNE DE EKLENDİ: kapının bunu yakalayıp yakalamadığını görmek için. Ateşleme
oranı kapısı (%0–15) tam olarak bu tür "kurulum diye piyasanın kendisini seçen"
mantıkları kenar ölçülmeden ÖNCE elemek için var. Bu dosya o kapının çalıştığının
kanıtı olarak durur.

──────────────────────────────────────────────────────────────── SAPMALAR
1. ZAMAN DİLİMİ: özgün 5 dakikalık; bu port 1 dakikalık barlarda ölçülür. (Bu kuralın
   sonucunu değiştirmez — koşul fiyat ölçeğine bağlı, zaman dilimine değil.)
2. ÇIKIŞ/STOP: özgün çıkış yine kuvvet karşılaştırmasıdır → çerçevede hedef 2×ATR,
   stop giriş × (1 − min(%28, max_stop_pct)).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

BUY_POW = 3.849          # özgün hyperopt değeri

META = {
    "name": "powertower_freqtrade",
    "title_tr": "PowerTower — kuvvet karşılaştırması (freqtrade örneği)",
    "author": "@alylmazzz (port) · freqtrade katkıcıları (özgün kural)",
    "source": "freqtrade/freqtrade-strategies · PowerTower.py (GPL-3.0) — "
              "https://github.com/freqtrade/freqtrade-strategies",
    "claim": "Kapanış, iki bar önceki kapanışın 3,849. kuvvetinden büyükse (üç ardışık "
             "bar için) yükseliş başlamıştır.",
    "claim_evidence": "YOK — ve dahası, kural BOYUTSAL OLARAK ANLAMSIZDIR: bir fiyatı bir "
                      "fiyatın kuvvetiyle karşılaştırır. Ölçüldü: fiyatı 1'in altındaki her "
                      "paritede %99,9954, üstündeki her paritede %0,0000 ateşliyor. "
                      "Ölçtüğü şey piyasa değil, varlığın fiyat büyüklüğü.",
    "mechanism": "close[0] > close[2]^3,849 VE close[1] > close[3]^3,849 VE "
                 "close[2] > close[4]^3,849. Hedef 2×ATR; stop oransal.",
    "exit_mode": "FIXED_TARGET",
    "time_stop_min": 120,
    "urgency": 1,
    "regimes": ["TREND YUKARI", "VOLATİL"],
}


def fire(f: Dict, p, price: float, atr_abs: float, df=None) -> Optional[Dict]:
    if df is None or len(df) < 20:
        return None
    try:
        c = df["close"].astype(float).to_numpy()
    except Exception:
        return None
    if len(c) < 5 or not np.all(np.isfinite(c[-5:])) or np.any(c[-5:] <= 0):
        return None

    # ——— ÖZGÜN GİRİŞ KOŞULU (kaynakla birebir, hyperopt değeri dahil) ———
    with np.errstate(over="ignore"):
        try:
            k1 = c[-1] > c[-3] ** BUY_POW
            k2 = c[-2] > c[-4] ** BUY_POW
            k3 = c[-3] > c[-5] ** BUY_POW
        except (OverflowError, FloatingPointError):
            return None
    if not (k1 and k2 and k3):
        return None

    stop_pct = min(28.0, float(getattr(p, "max_stop_pct", 2.0)))
    stop = price * (1.0 - stop_pct / 100.0)
    return {
        "direction": "LONG",
        "size": 0.5,
        "stop_hint": stop,
        "target_hint": price + 2.0 * atr_abs,
        "note": (f"kuvvet karşılaştırması sağlandı (fiyat {price:.6g}) · "
                 f"stop %{stop_pct:.1f}"),
    }

# ═════════════════════════════════════════════════════════════════════════
# BİZ ÖLÇTÜK — binance · 60 gün · BÜYÜK pariteler (1 dk)
# ═════════════════════════════════════════════════════════════════════════
#   pencere 14340 · ateşleme 2868 · oran %20.0
#
# VERDİKT: REDDEDİLDİ — ateşleme oranı %20.0 > %15
#
# BOYUTSAL TEŞHİS ÖLÇÜMLE DOĞRULANDI — kapı da çalıştı.
#
# Büyük grup %20,0 · küçük grup %79,8 ateşledi. Bu sayılar tesadüf değil: büyük grupta
# 5 pariteden YALNIZ DOGE (0,09) 1'in altında → 1/5 = %20. Küçük grupta 5'ten 4'ü
# 1'in altında → 4/5 = %80. Yani ateşleme oranı, paritelerin fiyat büyüklüğü dağılımını
# birebir yansıtıyor; piyasa hakkında HİÇBİR ŞEY ölçmüyor.
#
# Ateşleme oranı kapısı bunu kenar ölçülmeden ÖNCE eledi — kapının var oluş sebebi tam
# olarak budur. Bu dosya, kapının çalıştığının kanıtı olarak durur.
MEASURED = {
    "window": "60 gün · binance · 1 dk · büyük pariteler",
    "n_windows": 14340, "n_fires": 2868, "n_effective": None,
    "fire_rate_pct": 20.0,
    "verdict": "REJECTED",
}

# ─────────────────────────────────────────────────────────────────────────
# KÜÇÜK/OYNAK PARİTELER — BONK, ORDI, PYTH, ARB, PEPE
# ─────────────────────────────────────────────────────────────────────────
#   pencere 14340 · ateşleme 11450 · oran %79.847
#
# VERDİKT: REDDEDİLDİ — ateşleme oranı %79.8 > %15
#
MEASURED_SMALL_CAPS = {
    "window": "60 gün · binance · 1 dk · küçük/oynak pariteler",
    "n_windows": 14340, "n_fires": 11450, "n_effective": None,
    "fire_rate_pct": 79.847,
    "verdict": "REJECTED",
}
