"""
Parite seçim hattı (FAZ 10) — "hangi paritelerde işlem yapılmalı?"

Sabit bir liste DEĞİL, çeyreklik yeniden koşan bir SKORLAMA hattıdır. Bugünkü
17 varlıklı evren de bu mantığın çıktısıdır (HYG/FXB/FXF/FXE eklenince
Calmar 1,59 → 1,80).

Altı kriter (plan tablosundan):
  likidite · trend kalitesi · çeşitlendirme · taşıma maliyeti · kapasite · istikrar

TASARIM KURALI: seçim yalnız TRAIN+VALIDATION verisiyle yapılır. Kilitli test
penceresinde iyi görünen pariteyi eklemek, test setini validasyona çevirir.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import dataset as ds

# Eşikler — plan tablosundan
MIN_TREND_SHARPE = 0.40
MAX_BOOK_CORR = 0.70
MIN_HISTORY_DAYS = 500


def trend_returns(close: pd.Series, cost: float = 0.0006) -> pd.Series:
    """Tek varlıkta Trend200+Mom20 (vol-hedefli) net getirisi."""
    c = close.astype(float)
    r = c.pct_change().fillna(0.0)
    raw = ((c > c.rolling(200).mean()) & (c.pct_change(20) > 0)).astype(float)
    vol = r.rolling(30).std()
    pos = (raw * (0.025 / (vol + 1e-9)).clip(0, 1.0)).fillna(0.0)
    held = pos.shift(1).fillna(0.0)
    turn = held.diff().abs().fillna(0.0)
    return held * r - turn * cost


@dataclass
class AssetScore:
    symbol: str
    ok: bool
    trend_sharpe: float
    trend_cagr: float
    corr_to_book: Optional[float]
    ann_vol: float
    history_days: int
    turnover: float
    reasons: List[str]

    def to_dict(self) -> Dict:
        return asdict(self)


def score_asset(symbol: str, close: pd.Series,
                book_returns: Optional[pd.Series] = None) -> AssetScore:
    """Tek varlığı altı kriterle değerlendir (likidite verisi yoksa atlanır)."""
    reasons: List[str] = []
    r = trend_returns(close)
    m = ds.annualized(r)
    n = int(len(close))

    corr = None
    if book_returns is not None and len(book_returns) > 50:
        a, b = r.align(book_returns, join="inner")
        if len(a) > 50:
            corr = float(np.corrcoef(a.values, b.values)[0, 1])

    pos = ((close > close.rolling(200).mean()) &
           (close.pct_change(20) > 0)).astype(float)
    turnover = float(pos.diff().abs().sum() / max(1, len(pos)) * 365)

    ok = True
    if n < MIN_HISTORY_DAYS:
        ok = False
        reasons.append(f"❌ geçmiş kısa ({n} gün < {MIN_HISTORY_DAYS})")
    if m["sharpe"] < MIN_TREND_SHARPE:
        ok = False
        reasons.append(f"❌ trend kalitesi düşük (Sharpe {m['sharpe']:.2f} < {MIN_TREND_SHARPE})")
    if corr is not None and abs(corr) > MAX_BOOK_CORR:
        ok = False
        reasons.append(f"❌ kitapla korelasyon {corr:+.2f} > ±{MAX_BOOK_CORR}")
    if ok:
        reasons.append(f"✅ Sharpe {m['sharpe']:.2f}"
                       + (f" · kitapla ρ {corr:+.2f}" if corr is not None else ""))

    return AssetScore(symbol=symbol, ok=ok, trend_sharpe=m["sharpe"],
                      trend_cagr=m["cagr"], corr_to_book=(round(corr, 3) if corr is not None else None),
                      ann_vol=m["vol"], history_days=n,
                      turnover=round(turnover, 2), reasons=reasons)


def forward_select(candidates: Dict[str, pd.Series], base: List[str],
                   metric: str = "calmar", max_add: int = 6) -> Dict:
    """İleri seçim: kitabı GERÇEKTEN iyileştiren adayları tek tek ekle.

    Her adımda, mevcut kitaba eklenince metriği en çok artıran aday seçilir;
    hiçbir aday artırmıyorsa durulur. Bu, "en yüksek tekil Sharpe'lı N varlığı
    seç" yaklaşımından üstündür: portföy etkisini ölçer, tekil performansı değil.
    (Ölçüldü: 28-varlık genişletme Sharpe'ı 1,36→1,15 düşürmüştü — çok varlık
    çeşitlendirme değil, gürültü seyreltmesi yapıyordu.)"""
    def book_metric(syms: List[str]) -> float:
        rets = [trend_returns(candidates[s]) for s in syms if s in candidates]
        if not rets:
            return -np.inf
        R = pd.concat(rets, axis=1).fillna(0.0)
        port = R.mean(axis=1)
        return ds.annualized(port).get(metric, -np.inf)

    cur = [s for s in base if s in candidates]
    best = book_metric(cur)
    history = [{"step": 0, "added": None, metric: round(best, 3), "book": list(cur)}]

    pool = [s for s in candidates if s not in cur]
    for step in range(1, max_add + 1):
        gains: List[Tuple[float, str]] = []
        for s in pool:
            v = book_metric(cur + [s])
            if np.isfinite(v):
                gains.append((v, s))
        if not gains:
            break
        gains.sort(reverse=True)
        v, s = gains[0]
        if v <= best + 1e-6:
            history.append({"step": step, "added": None, metric: round(best, 3),
                            "note": "hiçbir aday iyileştirmedi — durduruldu"})
            break
        cur.append(s)
        pool.remove(s)
        best = v
        history.append({"step": step, "added": s, metric: round(v, 3),
                        "book": list(cur)})

    return {"selected": cur, "metric": metric, "final": round(best, 3),
            "history": history}


def screen(symbols: List[str], book: Optional[List[str]] = None) -> Dict:
    """Aday listesini eler ve raporlar. Yalnız train+validation verisi kullanılır."""
    series: Dict[str, pd.Series] = {}
    for s in symbols:
        try:
            c = (ds.load_crypto_daily(s) if ("USDT" in s.upper() or "/" in s)
                 else ds.load_noncrypto_daily(s))
            series[s] = ds.train_val(c)
        except FileNotFoundError:
            continue

    book_ret = None
    if book:
        rets = [trend_returns(series[b]) for b in book if b in series]
        if rets:
            book_ret = pd.concat(rets, axis=1).fillna(0.0).mean(axis=1)

    scores = [score_asset(s, c, book_ret).to_dict() for s, c in series.items()]
    scores.sort(key=lambda x: -x["trend_sharpe"])
    return {"n_candidates": len(scores),
            "passed": [s for s in scores if s["ok"]],
            "rejected": [s for s in scores if not s["ok"]],
            "all": scores}
