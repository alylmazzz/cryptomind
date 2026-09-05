#!/usr/bin/env python3
"""
FAZ 4 ölçümü — meta-etiketleme trend sinyalini gerçekten iyileştiriyor mu?

Kurgu:
  Birincil (yön)  : Trend200 + Mom20 (değişmez)
  Etiket          : üçlü bariyer → "birincil model kârlı mıydı?"
  İkincil (boyut) : GBM/lojistik → P(kâr) → kesirli Kelly
  Doğrulama       : purged K-fold + embargo (örtüşen etiket sızıntısı temizlenir)

Yalnız TRAIN+VALIDATION (2022-2025). KİLİTLİ TEST (2026) AÇILMAZ.
  python research_meta.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).parent))

from agi_trader.research import dataset as ds
from agi_trader.research.labeling import (
    triple_barrier_labels, uniqueness_weights, label_summary,
)
from agi_trader.research.validation import (
    purged_kfold_splits, deflated_sharpe, sharpe, trial_log,
)
from agi_trader.ai.meta_label import MetaLabeler, build_features
from agi_trader.analysis.chart_patterns import detect_chart_patterns

CRYPTO = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "AVAXUSDT"]
OUT = Path(__file__).parent / "runs" / "meta_research.json"
COST = 0.0006


def primary_signal(close: pd.Series) -> pd.Series:
    """Birincil model: trend yönü (+1 long / 0 nakit). DEĞİŞTİRİLMEZ."""
    above = close > close.rolling(200).mean()
    mom = close.pct_change(20) > 0
    return (above & mom).astype(float)


def pattern_feature_frame(df: pd.DataFrame, step: int = 20) -> pd.DataFrame:
    """Formasyon skorlarını zaman serisine çevir (her `step` barda bir tara).

    Her bar için taramak çok pahalı; formasyonlar yavaş değiştiği için
    aralıklı tarama + ileri doldurma yeterlidir (look-ahead yok: t'de yalnız
    t'ye kadarki veriyle tarama yapılır)."""
    keys = ["ascending_triangle", "descending_triangle", "symmetric_triangle",
            "rising_wedge", "falling_wedge", "bull_flag", "bear_flag",
            "cup_handle", "rectangle", "ascending_channel", "descending_channel",
            "horizontal_channel", "head_shoulders", "inverse_head_shoulders",
            "double_top", "double_bottom"]
    out = pd.DataFrame(0.0, index=df.index, columns=keys)
    for i in range(300, len(df), step):
        win = df.iloc[max(0, i - 320):i]           # yalnız GEÇMİŞ
        try:
            pats = detect_chart_patterns(win, top_n=8)
        except Exception:
            continue
        for p in pats:
            if p["key"] in out.columns:
                sign = 1.0 if p["direction"] == "LONG" else -1.0
                out.iloc[i, out.columns.get_loc(p["key"])] = sign * p["score"]
    return out.replace(0.0, np.nan).ffill().fillna(0.0)


def evaluate(sym: str, use_patterns: bool) -> dict:
    d = ds.train_val(ds.load_crypto_ohlcv_daily(sym))
    if len(d) < 500:
        return {}
    close = d["close"]
    side = primary_signal(close)

    lab = triple_barrier_labels(close, side=side, pt_mult=2.0, sl_mult=2.0,
                                max_hold=10, vol_span=50)
    pats = pattern_feature_frame(d) if use_patterns else None
    X = build_features(d, pats)
    y = lab["bin"].values
    w = uniqueness_weights(lab["t1"].values)

    valid = X.notna().all(axis=1).values & (side.values > 0)
    idx = np.flatnonzero(valid)
    if len(idx) < 200:
        return {}

    # Purged CV: her katta eğit, örneklem-dışı olasılık üret
    proba = np.full(len(d), np.nan)
    t1_rel = lab["t1"].values[idx]
    t1_local = np.searchsorted(idx, np.clip(t1_rel, 0, len(d) - 1))
    for tr, te in purged_kfold_splits(t1_local, n_splits=5, embargo_pct=0.01):
        tr_i, te_i = idx[tr], idx[te]
        m = MetaLabeler().fit(X.iloc[tr_i], y[tr_i], w[tr_i])
        proba[te_i] = m.predict(X.iloc[te_i]).proba

    # Getiri karşılaştırması
    r = close.pct_change().fillna(0.0)
    pos_base = side.shift(1).fillna(0.0)
    size = np.clip(2 * np.nan_to_num(proba, nan=0.5) - 1, 0.0, 0.5) * 2.0  # 0..1
    pos_meta = pd.Series(size, index=d.index).where(side > 0, 0.0).shift(1).fillna(0.0)

    def net(pos):
        turn = pos.diff().abs().fillna(0.0)
        return pos * r - turn * COST

    base, meta = net(pos_base), net(pos_meta)
    cover = float(np.isfinite(proba[idx]).mean())
    return {
        "symbol": sym, "use_patterns": use_patterns,
        "label": label_summary(lab),
        "oos_proba_coverage": round(cover, 3),
        "base": ds.annualized(base), "meta": ds.annualized(meta),
        "mean_proba": round(float(np.nanmean(proba)), 4),
        "n_events": int(len(idx)),
    }


def main() -> None:
    print("=" * 78)
    print("FAZ 4 — META-ETİKETLEME ÖLÇÜMÜ (train+validation, test KİLİTLİ)")
    print("=" * 78)
    results = []
    for use_pat in (False, True):
        tag = "formasyon ÖZELLİKLERİ İLE" if use_pat else "yalnız teknik özellikler"
        print(f"\n### {tag}")
        print(f"{'parite':10s} {'baz Sharpe':>11s} {'meta Sharpe':>12s} {'Δ':>7s} "
              f"{'baz CAGR':>9s} {'meta CAGR':>10s} {'olay':>6s}")
        agg_b, agg_m = [], []
        for sym in CRYPTO:
            try:
                res = evaluate(sym, use_pat)
            except Exception as e:
                print(f"{sym:10s} HATA {type(e).__name__}: {e}")
                continue
            if not res:
                continue
            b, m = res["base"], res["meta"]
            print(f"{sym:10s} {b['sharpe']:>11.2f} {m['sharpe']:>12.2f} "
                  f"{m['sharpe']-b['sharpe']:>+7.2f} {b['cagr']:>8.1f}% "
                  f"{m['cagr']:>9.1f}% {res['n_events']:>6d}")
            results.append(res)
            agg_b.append(b["sharpe"]); agg_m.append(m["sharpe"])
        if agg_b:
            print(f"{'ORTALAMA':10s} {np.mean(agg_b):>11.2f} {np.mean(agg_m):>12.2f} "
                  f"{np.mean(agg_m)-np.mean(agg_b):>+7.2f}")
            trial_log("meta_label", {"use_patterns": use_pat, "period": "train_val"},
                      {"sharpe": float(np.mean(agg_m))})

    if results:
        ex = results[0]
        print(f"\nEtiket dağılımı örneği ({ex['symbol']}):")
        for k, v in ex["label"].items():
            print(f"  {k}: {v}")
        print(f"\nModel: {'sklearn GBM' if __import__('agi_trader.ai.meta_label', fromlist=['_HAS_SK'])._HAS_SK else 'saf-numpy lojistik'}")

    print("\n" + "=" * 78)
    b_all = [r["base"]["sharpe"] for r in results]
    m_all = [r["meta"]["sharpe"] for r in results]
    delta = float(np.mean(m_all) - np.mean(b_all)) if b_all else 0.0
    if delta > 0.15:
        print(f"HÜKÜM: meta-etiketleme KATKI SAĞLIYOR (ortalama Δ Sharpe {delta:+.2f}).")
        print("Sıradaki adım: kilitli testte doğrulama (konfig dondurulduktan sonra).")
    else:
        print(f"HÜKÜM: meta-etiketleme anlamlı katkı SAĞLAMIYOR (Δ Sharpe {delta:+.2f}).")
        print("Model dağıtılmaz; formasyon/gösterge özellikleri bu haliyle boyut")
        print("kararını iyileştirmiyor. Fiyat-dışı özellikler biriktikten sonra")
        print("(recorder → 6 ay) tekrar denenecek.")

    OUT.write_text(json.dumps({"results": results, "delta_sharpe": round(delta, 3)},
                              indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
