"""Özellik üretici — şartname 2, 34, 35, 70, 80, 97.

TASARIM KURALI: i barındaki HER özellik yalnız ≤ i barlarından hesaplanır.
Bu, yorum değil test edilen bir değişmezdir: `test_gelecek_bar_gecmis_ozelligi
_degistirmez` geleceği bozup özelliklerin bit-bit aynı kaldığını doğrular
(şartname 80). Sızıntı bulunursa test KIRILIR.

ÖZELLİK AİLELERİ (şartname 34 — her biri ayrı ablasyona girer)
  technical    fiyat/oynaklık/momentum türevleri
  flow         5m vadeli klinelerin taker alış hacmi ve işlem sayısı —
               GEÇMİŞİ VAR, bu yüzden gerçek bir mikroyapı vekilidir
  geometry     hedefin kaç sigma uzakta olduğu (şartname 21)
  cross_asset  piyasa faktörü ve idiyosinkratik kısım
  time         UTC saat / haftanın günü (şartname 97 — kanıtlanırsa kalır)

BULUNMAYAN AİLELER — "VERİ YOK" ≠ "ETKİ YOK" (şartname 33, 70)
  microstructure(L2) : kaydedici yalnız günlerdir topluyor, 4,5 yıllık
                       etiket kümesiyle hizalanamaz → UNMEASURED
  derivatives        : funding/OI geçmişi 8 saatlik ve kısmi → UNMEASURED
  patterns           : formasyon ailelerinin kenarı bu programda ölçüldü ve
                       çıkmadı; özellik olarak eklemek denemesi ayrı bir
                       çalışmadır → bu modelde YOK
  social / macro     : olay örneklemi yetersiz → UNMEASURED
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

BARS_1H = 12
BARS_4H = 48
BARS_24H = 288
BARS_7D = 2016

FAMILIES = ("technical", "flow", "geometry", "cross_asset", "time")

# Şartname 4: FEATURE TIMEFRAME ekseni. Her küme kümülatiftir — "1h" kümesi
# 5 dakikalık özellikleri de içerir. Ablasyon bu kümeler üzerinde koşulur ve
# hangi çözünürlüğün gerçekten katkı verdiği ÖLÇÜLÜR, varsayılmaz.
TF_SETS = {
    "5m":  ("5m", "derived", "time"),
    "1h":  ("5m", "1h", "derived", "time"),
    "4h":  ("5m", "1h", "4h", "derived", "time"),
    "1d":  ("5m", "1h", "4h", "1d", "derived", "time"),
    "all": ("5m", "1h", "4h", "1d", "1w", "derived", "time"),
}


def tf_mask(tfs: List[str], tf_set: str) -> np.ndarray:
    """Bir tf kümesine ait sütun maskesi."""
    izin = set(TF_SETS[tf_set])
    return np.array([t in izin for t in tfs], dtype=bool)


def _z(x: np.ndarray, win: int) -> np.ndarray:
    s = pd.Series(x)
    m = s.rolling(win, min_periods=win // 2).mean()
    sd = s.rolling(win, min_periods=win // 2).std()
    return ((s - m) / (sd + 1e-12)).to_numpy()


def _past_return(c: np.ndarray, lag: int) -> np.ndarray:
    out = np.full(len(c), np.nan)
    if lag < len(c):
        out[lag:] = np.log(c[lag:] / c[:-lag]) * 100.0
    return out


def _rsi(c: np.ndarray, win: int) -> np.ndarray:
    d = np.zeros(len(c))
    d[1:] = np.diff(c)
    up = pd.Series(np.where(d > 0, d, 0.0)).ewm(alpha=1.0 / win, adjust=False).mean()
    dn = pd.Series(np.where(d < 0, -d, 0.0)).ewm(alpha=1.0 / win, adjust=False).mean()
    rs = up / (dn + 1e-12)
    return (100.0 - 100.0 / (1.0 + rs)).to_numpy()


def build(df: pd.DataFrame,
          sigma_bar_pct: np.ndarray,
          market_ret_4h: Optional[np.ndarray] = None,
          market_rv: Optional[np.ndarray] = None
          ) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    """5m OHLCV(+akış) → (X, isimler, aileler).

    `market_ret_4h` / `market_rv`: aynı zaman ızgarasında hesaplanmış PİYASA
    faktörü (pariteler ortalaması). Kendi paritesi de içindeyse hafif bir
    kendi-kendine referans olur; bu yüzden çağıran taraf paritenin KENDİSİNİ
    dışlayarak (leave-one-out) hesaplar.
    """
    c = df["close"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    v = df["volume"].to_numpy(dtype=float)
    n = len(c)
    tb = (df["taker_buy_base"].to_numpy(dtype=float)
          if "taker_buy_base" in df else np.full(n, np.nan))
    tr = (df["trades"].to_numpy(dtype=float)
          if "trades" in df else np.full(n, np.nan))

    ad: List[str] = []
    ail: List[str] = []
    tfs: List[str] = []
    kol: List[np.ndarray] = []

    def ekle(isim, aile, arr, tf="5m"):
        """`tf` = özelliğin fiilen hangi zaman diliminden bilgi taşıdığı.
        Şartname 4'ün FEATURE TIMEFRAME ekseni bu etiketten türetilir:
        bir tf kümesi seçildiğinde yalnız o kümeye ait sütunlar kullanılır."""
        ad.append(isim); ail.append(aile); tfs.append(tf)
        kol.append(np.asarray(arr, dtype=float))

    # ── technical ─────────────────────────────────────────────────────────
    for lag, isim, tf in ((BARS_1H, "ret_1h", "1h"), (BARS_4H, "ret_4h", "4h"),
                          (BARS_24H, "ret_24h", "1d"), (BARS_7D, "ret_7d", "1w")):
        ekle(isim, "technical", _past_return(c, lag), tf)
    ekle("rv_24h", "technical", sigma_bar_pct, "1d")
    rv7 = pd.Series(np.concatenate([[0.0], np.diff(np.log(c))])).rolling(
        BARS_7D, min_periods=BARS_7D // 2).std().to_numpy() * 100.0
    ekle("rv_7d", "technical", rv7, "1w")
    ekle("vol_of_vol", "technical", sigma_bar_pct / (rv7 + 1e-12), "1w")
    sma7 = pd.Series(c).rolling(BARS_7D, min_periods=BARS_7D // 2).mean().to_numpy()
    sd7 = pd.Series(c).rolling(BARS_7D, min_periods=BARS_7D // 2).std().to_numpy()
    ekle("trend_z_7d", "technical", (c - sma7) / (sd7 + 1e-12), "1w")
    sma1 = pd.Series(c).rolling(BARS_24H, min_periods=BARS_24H // 2).mean().to_numpy()
    sd1 = pd.Series(c).rolling(BARS_24H, min_periods=BARS_24H // 2).std().to_numpy()
    ekle("bb_pos_24h", "technical", (c - sma1) / (sd1 + 1e-12), "1d")
    ekle("rsi_1h", "technical", _rsi(c, BARS_1H), "1h")
    ekle("rsi_24h", "technical", _rsi(c, BARS_24H), "1d")
    hi24 = pd.Series(h).rolling(BARS_24H, min_periods=BARS_24H // 2).max().to_numpy()
    lo24 = pd.Series(l).rolling(BARS_24H, min_periods=BARS_24H // 2).min().to_numpy()
    ekle("dist_high_24h", "technical", (c / hi24 - 1.0) * 100.0, "1d")
    ekle("dist_low_24h", "technical", (c / lo24 - 1.0) * 100.0, "1d")

    # ── flow (5m vadeli klinelerin gerçek taker verisi) ───────────────────
    with np.errstate(divide="ignore", invalid="ignore"):
        tbr = np.where(v > 0, tb / v, np.nan)
    ekle("taker_buy_ratio", "flow", tbr, "5m")
    ekle("taker_ratio_z_24h", "flow", _z(tbr, BARS_24H), "1d")
    ekle("volume_z_24h", "flow", _z(v, BARS_24H), "1d")
    ekle("trades_z_24h", "flow", _z(tr, BARS_24H), "1d")
    # kümülatif hacim deltası (CVD) eğimi — 4 saatlik
    cvd = np.cumsum(np.nan_to_num(2.0 * tb - v, nan=0.0))
    ekle("cvd_slope_4h", "flow",
         (cvd - np.concatenate([np.full(BARS_4H, np.nan), cvd[:-BARS_4H]]))
         / (pd.Series(v).rolling(BARS_4H, min_periods=BARS_4H // 2).sum().to_numpy() + 1e-9),
         "4h")

    # ── cross_asset ───────────────────────────────────────────────────────
    if market_ret_4h is not None:
        ekle("market_ret_4h", "cross_asset", market_ret_4h, "4h")
        ekle("idio_ret_4h", "cross_asset", _past_return(c, BARS_4H) - market_ret_4h, "4h")
    if market_rv is not None:
        ekle("market_rv_24h", "cross_asset", market_rv, "1d")
        ekle("rel_rv", "cross_asset", sigma_bar_pct / (market_rv + 1e-12), "1d")

    # ── time (şartname 97 — kanıtlanırsa kalır, keyfi seans kuralı YOK) ──
    idx = df.index
    saat = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
    ekle("hour_sin", "time", np.sin(2 * np.pi * saat / 24.0), "time")
    ekle("hour_cos", "time", np.cos(2 * np.pi * saat / 24.0), "time")
    gun = idx.dayofweek.to_numpy()
    ekle("dow_sin", "time", np.sin(2 * np.pi * gun / 7.0), "time")
    ekle("dow_cos", "time", np.cos(2 * np.pi * gun / 7.0), "time")

    X = np.column_stack(kol)
    return X, ad, ail, tfs


def add_geometry(X: np.ndarray, names: List[str], families: List[str],
                 tfs: List[str], sigma_bar_pct: np.ndarray, target_pct: float,
                 horizon_bars: int
                 ) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    """Ufka özgü geometri: hedef kaç sigma uzakta (şartname 21).

    Bu, modelin öğrenmesi gereken en temel ve en fiziksel bilgidir; ufuk
    değiştikçe değiştiği için ana özellik matrisine sonradan eklenir."""
    sig = np.asarray(sigma_bar_pct, dtype=float) * np.sqrt(max(1, horizon_bars))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(sig > 0, target_pct / sig, np.nan)
    return (np.column_stack([X, d, np.log1p(np.clip(d, 0, 50))]),
            names + ["target_distance_sigma", "log_target_distance"],
            families + ["geometry", "geometry"],
            tfs + ["derived", "derived"])


def market_factor(closes: Dict[str, "pd.Series"], symbol: str,
                  index: Optional["pd.Index"] = None,
                  lag: int = BARS_4H) -> np.ndarray:
    """Paritenin KENDİSİ hariç piyasa getirisi (leave-one-out).

    İKİ TUZAK BURADA KAPATILIR:

    1. SIZINTI — kendini içeren bir "piyasa" faktörü, paritenin kendi
       getirisini özellik olarak geri sızdırır.
    2. HİZALAMA — pariteler farklı sayıda bara sahip olabilir (listeleme
       tarihleri, borsa kesintileri). Dizileri KONUMA göre hizalamak zaman
       damgalarını kaydırır ve gerçek bir look-ahead üretebilir. Bu yüzden
       hizalama YALNIZ zaman damgası üzerinden yapılır; eşleşmeyen barlar
       NaN kalır ve o bar özellik matrisinden düşer.
    """
    import pandas as pd
    digerleri = {k: v for k, v in closes.items() if k != symbol}
    idx = index if index is not None else (
        closes[symbol].index if symbol in closes else None)
    if not digerleri or idx is None:
        return np.full(len(idx) if idx is not None else 0, np.nan)
    kolonlar = {}
    for k, s in digerleri.items():
        s = s if isinstance(s, pd.Series) else pd.Series(s)
        kolonlar[k] = np.log(s / s.shift(lag)) * 100.0
    M = pd.DataFrame(kolonlar).reindex(idx)
    return M.mean(axis=1, skipna=True).to_numpy()
