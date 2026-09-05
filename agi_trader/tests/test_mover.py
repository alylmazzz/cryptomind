# -*- coding: utf-8 -*-
"""'%1 hareket adayı' modülü testleri.

En kritik test: doğrulamayı GEÇEMEYEN paritenin panelde "geçti" gibi
görünmemesi. Sembol biçimi uyuşmazlığı (AVAXUSDT ↔ AVAX/USDT) bu korumayı
sessizce devre dışı bırakmıştı — regresyon testi aşağıda.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agi_trader.analysis.mover import (  # noqa: E402
    build_mover_features, move_labels, MoverModel, rank_movers,
    auc_score, brier_skill, calibration_table, _norm_symbol,
    MIN_TRUSTED_AUC, FEATURES,
)


def make_ohlcv(n=400, seed=0, vol=0.02):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, vol, n)
    close = 100 * np.exp(np.cumsum(r))
    high = close * (1 + np.abs(rng.normal(0, vol / 2, n)))
    low = close * (1 - np.abs(rng.normal(0, vol / 2, n)))
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    return pd.DataFrame({"open": close, "high": high, "low": low,
                         "close": close, "volume": 1.0}, index=idx)


# ───────────────────────── sembol normalizasyonu ─────────────────────────
def test_norm_symbol():
    assert _norm_symbol("AVAX/USDT") == "AVAXUSDT"
    assert _norm_symbol("AVAX/USDT:USDT") == "AVAXUSDT"
    assert _norm_symbol("avaxusdt") == "AVAXUSDT"
    assert _norm_symbol("BTC-USDT") == "BTCUSDT"


def test_dogrulamayi_gecemeyen_parite_isaretlenir():
    """REGRESYON: doğrulama 'AVAXUSDT', canlı sembol 'AVAX/USDT' — eşleşme
    normalize edilmezse AVAX 'model geçerli' görünüyordu."""
    panel = {"AVAX/USDT": make_ohlcv(seed=1), "BTC/USDT": make_ohlcv(seed=2)}
    validation = {"AVAXUSDT": {"auc": 0.481}, "BTCUSDT": {"auc": 0.587}}
    out = rank_movers(panel, validation=validation)
    by = {p["symbol"]: p for p in out["picks"]}
    assert by["AVAX/USDT"]["model_trusted"] is False
    assert by["BTC/USDT"]["model_trusted"] is True
    assert "TABAN ORANIDIR" in by["AVAX/USDT"]["note"]


def test_guvenilmeyen_paritede_taban_orani_gosterilir():
    panel = {"AVAX/USDT": make_ohlcv(seed=3)}
    out = rank_movers(panel, validation={"AVAXUSDT": {"auc": 0.40}})
    p = out["picks"][0]
    assert abs(p["probability"] - p["base_rate"]) < 1e-9   # tahmin değil, taban
    assert p["evidence"] == []                             # kanıt gösterilmez


def test_guvenilir_paritede_kanit_gosterilir():
    panel = {"BTC/USDT": make_ohlcv(seed=4)}
    out = rank_movers(panel, validation={"BTCUSDT": {"auc": 0.62}})
    p = out["picks"][0]
    assert p["model_trusted"] is True
    assert len(p["evidence"]) > 0
    assert all("contribution" in e for e in p["evidence"])


# ───────────────────────── look-ahead koruması ───────────────────────────
def test_ozellikler_gelecegi_gormez():
    """Tüm özellikler .shift(1) ile ötelenmiş olmalı: t satırındaki değerler
    yalnız t-1 ve öncesinden türemeli."""
    d = make_ohlcv(seed=5)
    X = build_mover_features(d)
    d2 = d.copy()
    d2.iloc[-1, d2.columns.get_loc("close")] *= 1.5      # SON barı değiştir
    d2.iloc[-1, d2.columns.get_loc("high")] *= 1.5
    X2 = build_mover_features(d2)
    # son satırdaki özellikler DEĞİŞMEMELİ (bugünün verisini kullanmıyorlar)
    a, b = X.iloc[-1].values, X2.iloc[-1].values
    m = np.isfinite(a) & np.isfinite(b)
    assert np.allclose(a[m], b[m]), "özellikler bugünün barını görüyor → look-ahead"


def test_etiket_esigi():
    d = make_ohlcv(seed=6)
    y = move_labels(d, 0.01)
    r = d["close"].pct_change().abs()
    assert ((r >= 0.01).astype(float).dropna() == y.dropna()).all()


# ───────────────────────────── metrikler ─────────────────────────────────
def test_auc_mukemmel_ve_rastgele():
    y = np.array([0, 0, 1, 1])
    assert auc_score(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc_score(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert abs(auc_score(y, np.array([0.5, 0.5, 0.5, 0.5])) - 0.5) < 1e-9


def test_brier_beceri_isaretleri():
    y = np.array([1.0] * 50 + [0.0] * 50)
    base = 0.5
    iyi = np.array([0.9] * 50 + [0.1] * 50)
    kotu = np.array([0.1] * 50 + [0.9] * 50)
    assert brier_skill(y, iyi, base) > 0
    assert brier_skill(y, kotu, base) < 0
    assert abs(brier_skill(y, np.full(100, base), base)) < 1e-9


def test_model_taban_orani_ogrenir():
    d = make_ohlcv(n=600, seed=7)
    X, y = build_mover_features(d), move_labels(d)
    m = MoverModel().fit(X, y)
    assert 0.0 < m.base_rate < 1.0
    p = m.predict_proba(X)
    assert len(p) == len(X) and np.all((p >= 0) & (p <= 1))


def test_model_yetersiz_veride_tabana_duser():
    d = make_ohlcv(n=60, seed=8)
    m = MoverModel().fit(build_mover_features(d), move_labels(d))
    p = m.predict_proba(build_mover_features(d))
    assert np.allclose(p, m.base_rate)


def test_nan_ozellikte_tabana_duser():
    d = make_ohlcv(n=500, seed=9)
    X, y = build_mover_features(d), move_labels(d)
    m = MoverModel().fit(X, y)
    bad = X.copy()
    bad.iloc[-1] = np.nan
    p = m.predict_proba(bad.iloc[[-1]])
    assert abs(float(p[0]) - m.base_rate) < 1e-9


# ───────────────────────── çıktı sözleşmesi ──────────────────────────────
def test_yon_uyarisi_her_zaman_var():
    out = rank_movers({"BTC/USDT": make_ohlcv(seed=10)})
    assert "YÖN TAHMİN EDİLMEZ" in out["direction_warning"]
    assert "taban" in out["base_rate_warning"].lower()


def test_siralama_olasiliga_gore():
    panel = {"A/USDT": make_ohlcv(seed=11, vol=0.005),
             "B/USDT": make_ohlcv(seed=12, vol=0.04)}
    out = rank_movers(panel)
    probs = [p["probability"] for p in out["picks"]]
    assert probs == sorted(probs, reverse=True)
    assert [p["rank"] for p in out["picks"]] == list(range(1, len(probs) + 1))


def test_kisa_veri_atlanir():
    out = rank_movers({"X/USDT": make_ohlcv(n=100, seed=13)})
    assert out["picks"] == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
