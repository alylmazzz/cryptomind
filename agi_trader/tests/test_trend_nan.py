# -*- coding: utf-8 -*-
"""Trend-takip paper motorunun NaN dayanıklılığı — 2026-09-04 canlı arızasının regresyonu.

ARIZA: 2026-09-03'te 12 kripto-dışı varlığın (GLD/SPY/QQQ/TLT/UUP/USO/SLV/DBC/HYG/FXB/FXF/FXE)
fiyatı yfinance'ten gelmedi. `fetch_live_daily` hatayı SESSİZCE yuttu (`except: pass`), eksik
varlık `data` sözlüğünde olmadı, `step()` NaN getiri üretti ve `equity *= (1+NaN)` ile özsermaye
KALICI olarak NaN oldu — üstelik diske yazıldı. Sonuç: 45 günlük gerçek paper kaydı (+%4,87)
panelde `equity: null` olarak okunamaz hâle geldi ve iki gün kimse fark etmedi.

Kilitlenen davranış: (1) NaN fiyat özsermayeyi bozamaz, (2) eksik varlık raporlanır,
(3) diske asla NaN yazılmaz, (4) bozuk durum yüklenirse son sağlam güne kurtarılır.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agi_trader.auto.trend_engine import TrendTrader  # noqa: E402
from agi_trader.config import load_config  # noqa: E402

PAIRS = ["BTC/USDT", "ETH/USDT", "GLD", "SPY"]


def _series(n=300, start=100.0, drift=0.0006, seed=1):
    rng = np.random.default_rng(seed)
    px = start * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.DataFrame({"open": px, "high": px * 1.01, "low": px * 0.99,
                         "close": px, "volume": np.full(n, 1000.0)}, index=idx)


def _data(pairs=PAIRS, seed0=1):
    return {p: _series(seed=seed0 + i) for i, p in enumerate(pairs)}


def _tt():
    t = TrendTrader(load_config(), pairs=list(PAIRS), initial=10_000.0)
    return t


def test_eksik_varlik_ozsermayeyi_bozmaz():
    """Bir varlığın verisi HİÇ gelmezse: equity sonlu kalır, eksik varlık raporlanır."""
    tt = _tt()
    full = _data()
    tt.step(full, date_str="2026-09-01")
    eq1 = tt.equity
    assert math.isfinite(eq1)
    partial = {k: v for k, v in _data().items() if k not in ("GLD", "SPY")}   # 2 varlık düştü
    ev = tt.step(partial, date_str="2026-09-02")
    assert math.isfinite(tt.equity), "eksik veri özsermayeyi NaN yapamaz"
    assert set(ev["missing_prices"]) == {"GLD", "SPY"} and ev["data_ok"] is False
    assert math.isfinite(ev["equity"]) and math.isfinite(ev["return_pct"])


def test_nan_fiyat_ozsermayeyi_bozmaz():
    """Veri GELİR ama son kapanış NaN olursa (yfinance'in tipik davranışı) equity bozulmamalı."""
    tt = _tt()
    tt.step(_data(), date_str="2026-09-01")
    d = _data()
    for s in ("GLD", "SPY"):
        d[s] = d[s].copy()
        d[s].iloc[-1, d[s].columns.get_loc("close")] = float("nan")
    ev = tt.step(d, date_str="2026-09-02")
    assert math.isfinite(tt.equity) and math.isfinite(ev["equity"])
    assert set(ev["missing_prices"]) == {"GLD", "SPY"}


def test_diske_asla_nan_yazilmaz(tmp_path):
    tt = _tt()
    tt.step(_data(), date_str="2026-09-01")
    d = {k: v for k, v in _data().items() if k == "BTC/USDT"}     # 3 varlık düştü
    tt.step(d, date_str="2026-09-02")
    p = tmp_path / "trend_state.json"
    tt.save_state(str(p))
    raw = p.read_text(encoding="utf-8")
    assert "NaN" not in raw and "Infinity" not in raw, "durum dosyasında NaN olamaz"
    st = json.loads(raw)
    assert math.isfinite(st["equity"])
    assert all(math.isfinite(v) for v in st["last_close"].values())


def test_bozuk_durum_son_saglam_gune_kurtarilir(tmp_path):
    """Canlı senaryo: diskteki equity NaN. Yüklenince son sağlam geçmiş değerine dönülür."""
    p = tmp_path / "trend_state.json"
    p.write_text(json.dumps({
        "pairs": PAIRS, "initial": 10000.0, "equity": float("nan"),
        "weights": {"BTC/USDT": 0.2, "GLD": float("nan")},
        "last_close": {"BTC/USDT": 81037.8, "GLD": float("nan")},
        "last_signals": {}, "last_rebalance": "2026-09-04",
        "history": [{"date": "2026-09-01", "equity": 10400.0},
                    {"date": "2026-09-02", "equity": 10486.96},
                    {"date": "2026-09-03", "equity": float("nan")},
                    {"date": "2026-09-04", "equity": float("nan")}],
    }), encoding="utf-8")
    tt = _tt()
    assert tt.load_state(str(p)) is True
    assert tt.equity == pytest.approx(10486.96), "son SAĞLAM güne dönmeli"
    assert tt.recovered_from_nan is True
    assert "GLD" not in tt.last_close and "BTC/USDT" in tt.last_close   # NaN fiyat elenmeli
    assert all(math.isfinite(v) for v in tt.weights.values())
    # kurtarılan durumla adım atılabilmeli
    ev = tt.step(_data(), date_str="2026-09-05")
    assert math.isfinite(ev["equity"]) and ev["equity"] > 0


def test_tum_veri_gelmezse_ozsermaye_sabit_kalir():
    """Hiçbir fiyat gelmezse: portföy işaretlenemez ama özsermaye korunur (sıfırlanmaz/NaN olmaz)."""
    tt = _tt()
    tt.step(_data(), date_str="2026-09-01")
    eq = tt.equity
    ev = tt.step({}, date_str="2026-09-02")
    assert math.isfinite(tt.equity) and tt.equity == pytest.approx(eq, rel=0.02)
    assert len(ev["missing_prices"]) == len(PAIRS) and ev["data_ok"] is False


def test_mark_nan_getiriyi_atlar_ve_raporlar():
    tt = _tt()
    tt.weights = {"BTC/USDT": 0.5, "GLD": 0.5}
    tt.equity = 10_000.0
    out = tt.mark({"BTC/USDT": 0.02, "GLD": float("nan")})
    assert tt.equity == pytest.approx(10_100.0)      # yalnız BTC katkısı: 0,5 × %2
    assert out["skipped"] == ["GLD"]


# ═══════════════════════ üç katmanlı veri hattı ═══════════════════════
def test_onbellek_yaz_oku(tmp_path, monkeypatch):
    """Kaynak düşerse son başarılı çekim kullanılabilmeli (portföy körleşmesin)."""
    import trend_daemon as td
    monkeypatch.setattr(td, "PRICE_CACHE", tmp_path / "price_cache")
    df = _series(n=60)
    td._cache_write("GLD", df)
    back = td._cache_read("GLD")
    assert back is not None and len(back) == 60
    assert float(back["close"].iloc[-1]) == pytest.approx(float(df["close"].iloc[-1]))
    assert td._cache_read("YOKTUR") is None


def test_onbellek_bayat_veri_vermez(tmp_path, monkeypatch):
    import os, time as _t
    import trend_daemon as td
    monkeypatch.setattr(td, "PRICE_CACHE", tmp_path / "price_cache")
    td._cache_write("GLD", _series(n=30))
    f = (tmp_path / "price_cache" / "GLD.csv")
    old = _t.time() - 40 * 86400
    os.utime(f, (old, old))
    assert td._cache_read("GLD", max_age_days=10) is None, "40 günlük fiyat kullanılmamalı"
    assert td._cache_read("GLD", max_age_days=90) is not None


def test_chart_api_bozuk_yanitta_none_doner(monkeypatch):
    """Yedek kaynak hata/boş yanıtta patlamamalı — üst katman önbelleğe düşebilsin."""
    import trend_daemon as td

    class _R:
        def __init__(self, body): self.body = body.encode()
        def read(self): return self.body
        def __enter__(self): return self
        def __exit__(self, *a): return False

    import urllib.request as ur
    monkeypatch.setattr(ur, "urlopen", lambda *a, **k: _R('{"chart":{"result":[]}}'))
    assert td._fetch_chart_api("GLD") is None
    monkeypatch.setattr(ur, "urlopen", lambda *a, **k: _R("bozuk json"))
    assert td._fetch_chart_api("GLD") is None

    def _boom(*a, **k):
        raise OSError("ağ yok")
    monkeypatch.setattr(ur, "urlopen", _boom)
    assert td._fetch_chart_api("GLD") is None
