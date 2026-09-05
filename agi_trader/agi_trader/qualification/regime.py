"""Rejim sınıflandırıcı — şartname 20, 21, 77, 110.

NEDEN REJİM
Aynı BTC 4h modeli düşük ve yüksek oynaklıkta aynı eşiği kullanamaz: net +%1
hedef, oynaklık düşükken 3 sigma uzakta, yüksekken 0,4 sigma uzakta olabilir.
Aynı taban oranı iki dünyada aynı anlamı taşımaz.

İKİ EKSEN AYRI TUTULUR
  vol_regime  : LOW_VOL · NORMAL_VOL · HIGH_VOL · PANIC   (koşullama ekseni)
  structure   : TREND_UP · TREND_DOWN · SIDEWAYS · BREAKOUT · SQUEEZE

Neden ikisi tek etikete katlanmıyor: 4×5 = 20 hücre, her paritede ufuk ve yön
ile çarpılınca örneklem hücre başına anlamsızlaşır. Koşullama vol ekseninde
yapılır; yapı ekseni açıklama ve filtre olarak taşınır.

LIQUIDITY_STRESS — DÜRÜST BOŞLUK
Şartnamedeki bu rejim yalnız L2 defterinden (spread genişlemesi + derinlik
çöküşü) ölçülebilir. Tarihsel 5m OHLCV'de defter YOKTUR; bu yüzden geçmişte
bu etiket ASLA atanmaz ve `UNMEASURED_HISTORICALLY` olarak beyan edilir.
Canlı tarafta kaydedici verisiyle atanabilir. "Veri yok" ile "etki yok" aynı
şey değildir (şartname 33).

BÜTÜN ÖZELLİKLER NEDENSEL
Her değer yalnız GEÇMİŞ barlardan hesaplanır ve eşikler ilerleyen (expanding)
90 günlük pencereden alınır. Sabit "2022-2024'ten seçilmiş" eşik kullanılmaz;
bu, test döneminde bilgi sızıntısı olurdu.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

VOL_REGIMES = ["LOW_VOL", "NORMAL_VOL", "HIGH_VOL", "PANIC"]
STRUCTURES = ["TREND_UP", "TREND_DOWN", "SIDEWAYS", "BREAKOUT", "SQUEEZE"]
LIVE_ONLY_REGIMES = ["LIQUIDITY_STRESS"]
UNKNOWN = "UNKNOWN"

BARS_1H = 12
BARS_24H = 288
BARS_7D = 2016
BURN_IN = BARS_7D + BARS_24H           # eşik penceresi dolana kadar UNKNOWN
QUANT_WINDOW = 90 * BARS_24H           # eşikler son 90 günden
QUANT_STEP = BARS_24H                  # günde bir yenilenir, ileri taşınır


def _causal_quantiles(x: np.ndarray, qs=(0.20, 0.80, 0.99)) -> np.ndarray:
    """Her bar için, YALNIZ o barın öncesindeki 90 günden alınan kantiller.

    Günde bir kez hesaplanıp ileri taşınır (1.673 hesap vs 480.000). İleri
    taşıma nedenseldir: bugünün eşiği dünün verisinden gelir."""
    n = len(x)
    out = np.full((n, len(qs)), np.nan)
    son: Optional[np.ndarray] = None
    for i in range(0, n, QUANT_STEP):
        bas = max(0, i - QUANT_WINDOW)
        pencere = x[bas:i]
        pencere = pencere[np.isfinite(pencere)]
        if len(pencere) >= BARS_24H:
            son = np.quantile(pencere, qs)
        if son is not None:
            out[i: i + QUANT_STEP] = son
    return out


def classify(df: pd.DataFrame) -> pd.DataFrame:
    """5m OHLCV → rejim/yapı etiketleri + ham rejim özellikleri.

    Dönen sütunlar: vol_regime, structure, rv_24h_pct, vol_rank,
    trend_z, bb_width, target_distance_sigma_1h (şartname 21 için taban).
    """
    c = df["close"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    n = len(c)

    logret = np.zeros(n)
    logret[1:] = np.log(c[1:] / c[:-1])
    s = pd.Series(logret)

    # gerçekleşen oynaklık — 24 saatlik pencere, bar başına std (%)
    rv = s.rolling(BARS_24H, min_periods=BARS_24H // 2).std().to_numpy() * 100.0
    # 7 günlük trend z-skoru: (close − SMA7g) / (7g std)
    sma7 = pd.Series(c).rolling(BARS_7D, min_periods=BARS_7D // 2).mean().to_numpy()
    sd7 = pd.Series(c).rolling(BARS_7D, min_periods=BARS_7D // 2).std().to_numpy()
    trend_z = (c - sma7) / np.where(sd7 > 0, sd7, np.nan)
    # Bollinger genişliği (24h) — sıkışma göstergesi
    sma1 = pd.Series(c).rolling(BARS_24H, min_periods=BARS_24H // 2).mean().to_numpy()
    sd1 = pd.Series(c).rolling(BARS_24H, min_periods=BARS_24H // 2).std().to_numpy()
    bbw = np.where(sma1 > 0, 4.0 * sd1 / sma1, np.nan) * 100.0
    # 24 saatlik getiri (panik tespiti)
    r24 = np.full(n, np.nan)
    r24[BARS_24H:] = (c[BARS_24H:] / c[:-BARS_24H] - 1.0) * 100.0

    q = _causal_quantiles(rv)
    q20, q80, q99 = q[:, 0], q[:, 1], q[:, 2]
    qb = _causal_quantiles(bbw, qs=(0.10, 0.90, 0.99))
    bb10, bb90 = qb[:, 0], qb[:, 1]

    vol_reg = np.full(n, UNKNOWN, dtype=object)
    hazir = np.isfinite(rv) & np.isfinite(q20) & (np.arange(n) >= BURN_IN)
    vol_reg[hazir & (rv <= q20)] = "LOW_VOL"
    vol_reg[hazir & (rv > q20) & (rv < q80)] = "NORMAL_VOL"
    vol_reg[hazir & (rv >= q80)] = "HIGH_VOL"
    # PANİK: oynaklık en üst %1'de VE 24 saatte sert düşüş
    panik = hazir & np.isfinite(q99) & (rv >= q99) & np.isfinite(r24) & (r24 <= -5.0)
    vol_reg[panik] = "PANIC"

    yapi = np.full(n, UNKNOWN, dtype=object)
    hz = hazir & np.isfinite(trend_z) & np.isfinite(bbw) & np.isfinite(bb10)
    yapi[hz] = "SIDEWAYS"
    yapi[hz & (trend_z >= 1.0)] = "TREND_UP"
    yapi[hz & (trend_z <= -1.0)] = "TREND_DOWN"
    yapi[hz & (bbw <= bb10)] = "SQUEEZE"
    # KIRILMA sıkışmayı ezer: dar banttan çıkış
    kirilma = hz & np.isfinite(bb90) & (bbw >= bb90) & (np.abs(trend_z) >= 1.5)
    yapi[kirilma] = "BREAKOUT"

    out = pd.DataFrame({
        "vol_regime": vol_reg,
        "structure": yapi,
        "rv_24h_pct": rv,
        "vol_q20": q20, "vol_q80": q80,
        "trend_z": trend_z,
        "bb_width": bbw,
        "ret_24h_pct": r24,
    }, index=df.index)
    return out


def target_distance_sigma(target_pct: float, rv_bar_pct: np.ndarray,
                          horizon_bars: int) -> np.ndarray:
    """Şartname 21 — hedef, mevcut oynaklıkta kaç sigma uzakta?

    σ(H) = bar oynaklığı × √H  (bağımsız artışlar varsayımı).
    2,8 sigma uzaktaki bir hedefin o ufukta görülmesi FİZİKSEL olarak
    zordur; olasılık düşürülmeli. Bu, modelin değil geometrinin sonucudur.
    """
    sig = np.asarray(rv_bar_pct, dtype=float) * np.sqrt(max(1, horizon_bars))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sig > 0, target_pct / sig, np.nan)


def regime_note() -> Dict:
    return {
        "vol_regimes": VOL_REGIMES,
        "structures": STRUCTURES,
        "live_only": LIVE_ONLY_REGIMES,
        "liquidity_stress": "UNMEASURED_HISTORICALLY — L2 defteri gerektirir; "
                            "5m OHLCV geçmişinde atanmaz",
        "causality": "eşikler ilerleyen 90 günlük pencereden, günde bir "
                     "yenilenip ileri taşınarak alınır — sabit/geriye dönük "
                     "eşik kullanılmaz",
    }
