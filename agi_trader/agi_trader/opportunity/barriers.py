"""
Üçlü bariyer / ilk-geçiş (first-passage) — şartname 5, 6 ve 62.12.

Sorulan soru "fiyat yükselecek mi?" DEĞİL:

    P(net +%1 hedefi, stop bariyerinden ÖNCE görülür mü?)

⚠️ EN KRİTİK TUZAK — ŞARTNAME 6 VE 47'DE DE YAZILI
Bir 4H mumun içinde hem +%1 hem −%1 görülmüş olabilir. `high`/`low` hangisinin
ÖNCE geldiğini SÖYLEMEZ. Yalnız OHLC ile "hedefe ulaştı" demek look-ahead
hatasıdır ve backtest'i sistematik olarak iyimser gösterir.

Bu modül bunu iki şekilde ele alır:
  1. Mümkünse ALT zaman dilimi barlarıyla sıra çözülür (`resolve_with`).
  2. Çözülemiyorsa örnek `AMBIGUOUS` etiketi alır ve ölçümden DIŞLANIR —
     "yukarı saydım" varsayımı yapılmaz.

Etiketler:
    +1  hedef bariyeri önce
    -1  stop bariyeri önce
     0  süre doldu (hiçbiri)
  None  belirsiz (aynı barda ikisi de, alt veri yok) → ÖLÇÜME GİRMEZ
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

AMBIGUOUS = None


@dataclass
class BarrierResult:
    label: Optional[int]
    bars_to_hit: Optional[int]
    mfe_pct: float            # maximum favorable excursion
    mae_pct: float            # maximum adverse excursion
    resolved_by: str          # "ana_bar" | "alt_bar" | "belirsiz" | "zaman"


def _scan(highs: Sequence[float], lows: Sequence[float],
          up: float, dn: float, direction: str) -> Optional[int]:
    """Bar bar tara; hangi bariyer önce vurulmuş, aynı barda ikisi de varsa None."""
    for k, (h, l) in enumerate(zip(highs, lows)):
        hit_up = h >= up
        hit_dn = l <= dn
        if hit_up and hit_dn:
            return None                    # aynı bar — sıra bilinmiyor
        if hit_up:
            return +1 if direction == "LONG" else -1
        if hit_dn:
            return -1 if direction == "LONG" else +1
    return 0


def first_passage(entry: float,
                  target_pct: float,
                  stop_pct: float,
                  direction: str,
                  future: pd.DataFrame,
                  resolve_with: Optional[pd.DataFrame] = None) -> BarrierResult:
    """Giriş sonrası ilk hangi bariyere ulaşıldığını belirler.

    entry      : giriş fiyatı
    target_pct : hedef (pozitif sayı, yönü `direction` belirler)
    stop_pct   : stop (pozitif sayı)
    future     : giriş barından SONRAKİ barlar (high/low), zaman ufku kadar
    resolve_with: aynı aralığı kapsayan DAHA KISA zaman dilimi barları
    """
    if future is None or len(future) == 0:
        return BarrierResult(AMBIGUOUS, None, 0.0, 0.0, "veri_yok")

    if direction == "LONG":
        up = entry * (1.0 + target_pct / 100.0)
        dn = entry * (1.0 - stop_pct / 100.0)
    else:
        up = entry * (1.0 + stop_pct / 100.0)      # SHORT'ta yukarısı STOP
        dn = entry * (1.0 - target_pct / 100.0)

    fh = future["high"].to_numpy(dtype=float)
    fl = future["low"].to_numpy(dtype=float)

    lab = _scan(fh, fl, up, dn, direction)
    resolved = "ana_bar"

    if lab is None and resolve_with is not None and len(resolve_with):
        # Belirsiz bar(lar)ı alt zaman dilimiyle çöz
        sh = resolve_with["high"].to_numpy(dtype=float)
        sl = resolve_with["low"].to_numpy(dtype=float)
        lab = _scan(sh, sl, up, dn, direction)
        resolved = "alt_bar" if lab is not None else "belirsiz"
    elif lab is None:
        resolved = "belirsiz"
    elif lab == 0:
        resolved = "zaman"

    # MFE / MAE — yön düzeltilmiş
    if direction == "LONG":
        mfe = float((fh.max() / entry - 1.0) * 100.0)
        mae = float((fl.min() / entry - 1.0) * 100.0)
    else:
        mfe = float((1.0 - fl.min() / entry) * 100.0)
        mae = float((1.0 - fh.max() / entry) * 100.0)

    bars = None
    if lab in (+1, -1):
        hedef = up if (direction == "LONG") == (lab == +1) else dn
        for k, (h, l) in enumerate(zip(fh, fl)):
            if (lab == +1 and direction == "LONG" and h >= up) or \
               (lab == +1 and direction == "SHORT" and l <= dn) or \
               (lab == -1 and direction == "LONG" and l <= dn) or \
               (lab == -1 and direction == "SHORT" and h >= up):
                bars = k + 1
                break

    return BarrierResult(lab, bars, round(mfe, 4), round(mae, 4), resolved)


def base_rate(df: pd.DataFrame,
              target_pct: float,
              stop_pct: float,
              horizon_bars: int,
              direction: str = "LONG",
              resolve_df: Optional[pd.DataFrame] = None,
              step: int = 1) -> Dict:
    """Bir seride hedef-önce-stop TABAN ORANI.

    Bu, "günde %1 bulunur mu?" sorusunun ÖLÇÜLEBİLİR hâlidir: model olmadan,
    rastgele bir anda girildiğinde hedefin stop'tan önce gelme sıklığı.
    Model ancak bu taban oranını AŞARSA bilgi taşıyor demektir."""
    n = len(df)
    hits = stops = timeouts = ambiguous = 0
    mfes, maes, bars_list = [], [], []
    for i in range(0, n - horizon_bars - 1, step):
        entry = float(df["close"].iloc[i])
        fut = df.iloc[i + 1: i + 1 + horizon_bars]
        sub = None
        if resolve_df is not None:
            t0, t1 = df.index[i], df.index[min(i + horizon_bars, n - 1)]
            sub = resolve_df.loc[(resolve_df.index > t0) & (resolve_df.index <= t1)]
        r = first_passage(entry, target_pct, stop_pct, direction, fut, sub)
        if r.label is AMBIGUOUS:
            ambiguous += 1
            continue
        mfes.append(r.mfe_pct); maes.append(r.mae_pct)
        if r.label == +1:
            hits += 1
            if r.bars_to_hit:
                bars_list.append(r.bars_to_hit)
        elif r.label == -1:
            stops += 1
        else:
            timeouts += 1
    kesin = hits + stops + timeouts
    return {
        "n_evaluated": kesin,
        "n_ambiguous": ambiguous,
        "ambiguous_pct": round(ambiguous / max(1, kesin + ambiguous) * 100, 1),
        "hit_first_pct": round(hits / max(1, kesin) * 100, 2),
        "stop_first_pct": round(stops / max(1, kesin) * 100, 2),
        "timeout_pct": round(timeouts / max(1, kesin) * 100, 2),
        "median_bars_to_hit": (float(np.median(bars_list)) if bars_list else None),
        "mfe_p50": (round(float(np.percentile(mfes, 50)), 3) if mfes else None),
        "mfe_p90": (round(float(np.percentile(mfes, 90)), 3) if mfes else None),
        "mae_p50": (round(float(np.percentile(maes, 50)), 3) if maes else None),
        "mae_p90": (round(float(np.percentile(maes, 90)), 3) if maes else None),
    }
