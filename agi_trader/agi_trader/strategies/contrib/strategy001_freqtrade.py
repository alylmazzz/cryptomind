"""
Strategy001 — Heikin-Ashi teyitli EMA kesişimi.

KAYNAK: freqtrade/freqtrade-strategies · `user_data/strategies/Strategy001.py` (GPL-3.0)
        https://github.com/freqtrade/freqtrade-strategies

ÖZGÜN KURAL (kaynaktan doğrulandı, 2026-09-05):
    giriş : EMA20, EMA50'yi YUKARI KESTİ  VE  HA_kapanış > EMA20  VE  HA yeşil (HA_açılış < HA_kapanış)
    çıkış : EMA50, EMA100'ü yukarı kesti  VE  HA_kapanış < EMA20  VE  HA kırmızı
    göstergeler: EMA 20/50/100 · Heikin-Ashi mumları
    minimal_roi 0:%5 → 60:%1 · stoploss −0,10 · timeframe '5m'

KOD KOPYALANMADI; kural bu deponun kendi araçlarıyla bağımsız yazıldı. Heikin-Ashi `f`
sözlüğünde yok, bar çerçevesinden hesaplanır.

──────────────────────────────────────────────────────────────── SAPMALAR
1. ZAMAN DİLİMİ: özgün 5 dakikalık; bu port 1 dakikalık barlarda ölçülür.
2. ÇIKIŞ: özgün çıkış bir SİNYALDİR (EMA50/EMA100 kesişimi + HA kırmızı). Çerçevede
   çıkış motoru stop/hedefle çalışır → PARTIAL_AND_RUN; hedef 3×ATR.
3. STOP: özgün −%10 stop 1 dk ufkunda anlamsızdır → ORANSAL stop,
   `giriş × (1 − min(%10, max_stop_pct))`. (ClucMay dersi: banda/göstergeye dayalı stop
   ikame etmek kurulumu sessizce imkânsız kılabiliyor; oransal stop bu riski taşımaz.)

Heikin-Ashi tanımı standarttır:
    HA_kapanış = (O+H+L+C)/4
    HA_açılış  = (önceki HA_açılış + önceki HA_kapanış)/2  (ilk bar: (O+C)/2)
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

META = {
    "name": "strategy001_freqtrade",
    "title_tr": "Strategy001 — HA + EMA kesişimi (freqtrade örneği)",
    "author": "@alylmazzz (port) · freqtrade katkıcıları (özgün kural)",
    "source": "freqtrade/freqtrade-strategies · Strategy001.py (GPL-3.0) — "
              "https://github.com/freqtrade/freqtrade-strategies",
    "claim": "EMA20 EMA50'yi yukarı kestiğinde ve Heikin-Ashi mumu yeşil olup kapanışı "
             "EMA20'nin üstündeyse yükseliş başlamıştır.",
    "claim_evidence": "YOK — kaynak depo bu stratejileri 'hazır kullanım değil, başlangıç "
                      "noktası' olarak sunuyor ve 'kendi backtest'inizi koşun' diyor. "
                      "Yayımlanmış kâr kanıtı YOK. Özgün bağlam 5 dakikalık barlar.",
    "mechanism": "EMA20 EMA50'yi yukarı kesti (bu barda) VE HA_kapanış > EMA20 VE HA yeşil. "
                 "Hedef 3×ATR; stop giriş × (1 − min(%10, max_stop_pct)).",
    "exit_mode": "PARTIAL_AND_RUN",
    "time_stop_min": 120,
    "urgency": 1,
    "regimes": ["TREND YUKARI", "VOLATİL"],
}


def _heikin_ashi(o, h, l, c):
    ha_c = (o + h + l + c) / 4.0
    ha_o = np.empty(len(c))
    ha_o[0] = (o[0] + c[0]) / 2.0
    for i in range(1, len(c)):
        ha_o[i] = (ha_o[i - 1] + ha_c[i - 1]) / 2.0
    return ha_o, ha_c


def fire(f: Dict, p, price: float, atr_abs: float, df=None) -> Optional[Dict]:
    if df is None or len(df) < 150:
        return None
    try:
        c = df["close"].astype(float)
        ema20 = c.ewm(span=20, adjust=False).mean().to_numpy()
        ema50 = c.ewm(span=50, adjust=False).mean().to_numpy()
        ha_o, ha_c = _heikin_ashi(df["open"].astype(float).to_numpy(),
                                  df["high"].astype(float).to_numpy(),
                                  df["low"].astype(float).to_numpy(), c.to_numpy())
    except Exception:
        return None
    if not (np.isfinite(ema20[-1]) and np.isfinite(ema50[-1])):
        return None

    # ——— ÖZGÜN GİRİŞ KOŞULU (kaynakla birebir) ———
    kesti = ema20[-1] > ema50[-1] and ema20[-2] <= ema50[-2]      # crossed_above
    if not (kesti and ha_c[-1] > ema20[-1] and ha_o[-1] < ha_c[-1]):
        return None

    stop_pct = min(10.0, float(getattr(p, "max_stop_pct", 2.0)))
    stop = price * (1.0 - stop_pct / 100.0)
    return {
        "direction": "LONG",
        "size": 0.5,
        "stop_hint": stop,
        "target_hint": price + 3.0 * atr_abs,
        "note": (f"EMA20 ({ema20[-1]:.6g}) EMA50'yi ({ema50[-1]:.6g}) yukarı kesti · "
                 f"HA yeşil ve kapanış EMA20 üstünde · stop %{stop_pct:.1f}"),
    }

# ═════════════════════════════════════════════════════════════════════════
# BİZ ÖLÇTÜK — binance · 60 gün · BÜYÜK pariteler (1 dk)
# ═════════════════════════════════════════════════════════════════════════
#   pencere 28680 · ateşleme 263 · oran %0.917
#   örneklem: nominal 263 → ETKİN 251
#   ortalama net %-0.1931 · t -5.43 · CI95 [-0.2691, -0.1265] · kazanma %40.6
#   çıkış sebepleri: {'HEDEF': 171, 'ZAMAN': 79, 'STOP': 1}
#
# VERDİKT: GÖLGE
#
# Her iki grupta da NEGATİF ve birbirine yakın (−%0,193 / −%0,210). Heikin-Ashi
# teyidi + EMA kesişimi, 1 dakikalık barlarda kenar üretmiyor. HA mumları gürültüyü
# yumuşatır ama gecikme ekler; kesişim zaten geciken bir sinyaldir.
MEASURED = {
    "window": "60 gün · binance · 1 dk · büyük pariteler",
    "n_windows": 28680, "n_fires": 263, "n_effective": 251,
    "fire_rate_pct": 0.917,
    "mean_net_pct": -0.1931, "t_stat": -5.43, "ci95": [-0.2691, -0.1265],
    "win_rate": 0.406, "exit_reasons": {'HEDEF': 171, 'ZAMAN': 79, 'STOP': 1},
    "verdict": "SHADOW",
}

# ─────────────────────────────────────────────────────────────────────────
# KÜÇÜK/OYNAK PARİTELER — BONK, ORDI, PYTH, ARB, PEPE
# ─────────────────────────────────────────────────────────────────────────
#   pencere 28680 · ateşleme 311 · oran %1.084
#   örneklem: nominal 311 → ETKİN 292
#   ortalama net %-0.2102 · t -4.34 · CI95 [-0.3068, -0.1184] · kazanma %55.5
#   çıkış sebepleri: {'HEDEF': 165, 'ZAMAN': 125, 'STOP': 2}
#
# VERDİKT: GÖLGE
#
MEASURED_SMALL_CAPS = {
    "window": "60 gün · binance · 1 dk · küçük pariteler",
    "n_windows": 28680, "n_fires": 311, "n_effective": 292,
    "fire_rate_pct": 1.084,
    "mean_net_pct": -0.2102, "t_stat": -4.34, "ci95": [-0.3068, -0.1184],
    "win_rate": 0.555, "exit_reasons": {'HEDEF': 165, 'ZAMAN': 125, 'STOP': 2},
    "verdict": "SHADOW",
}
