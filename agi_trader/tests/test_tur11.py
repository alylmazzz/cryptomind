# -*- coding: utf-8 -*-
"""11. tur — devre kesici, devam olasılığı, rotasyon, gün-içi tepe geri-verme, swing sleeve, evren, keşif, deneme kaydı."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agi_trader.auto import live_runner as LR  # noqa: E402
from agi_trader.auto import simulator as SIM  # noqa: E402
from agi_trader.auto import replay as RP  # noqa: E402
from agi_trader.learn.allocator import MetaAllocator, BREAKER_MIN_N  # noqa: E402
from agi_trader.strategies import committee as CM  # noqa: E402
from agi_trader.strategies import exit_engine as XE  # noqa: E402
from agi_trader.strategies import light_context as LC  # noqa: E402
from agi_trader.strategies import portfolio_mode as PM  # noqa: E402
from agi_trader.strategies import sleeves_fast as SF  # noqa: E402
from agi_trader.server import secure_keys as V  # noqa: E402

from test_live_trading import FakeExchange, factory, _path  # noqa: E402
from test_committee import _slow, _ctx, _ctx_provider, _df  # noqa: E402

MASTER = "test-master-key-" + "q" * 40


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv(V.MASTER_ENV, MASTER)
    FakeExchange.path = {"BTC/USDT": _path(), "ETH/USDT": _path(seed=5)}
    FakeExchange.orders = []
    yield


def test_devre_kesici_kaybeden_sleevei_duraklatir(tmp_path):
    a = MetaAllocator(tmp_path / "a.json", SF.REGIME_SLEEVES)
    for i in range(BREAKER_MIN_N):
        a.record("dip_moderate", None, i % 3 == 0, -0.1, net_pct=(-0.4 if i % 3 else 0.2))
    ev = a.check_breakers(now=1000.0)
    assert ev and ev[0]["sleeve"] == "dip_moderate" and "dip_moderate" in a.paused_sleeves(1000.0) and ev[0]["t_stat"] < -1.0
    assert a.check_breakers(now=1001.0) == [] and "dip_moderate" not in a.paused_sleeves(1000.0 + 7 * 3600)
    for i in range(BREAKER_MIN_N):
        a.record("dip", None, i % 3 != 0, 0.3, net_pct=0.3 if i % 3 else -0.2)       # kazanan sleeve duraklatılmaz
    assert not a.check_breakers(now=2000.0) and a.status()["paused"] == {}
    b = MetaAllocator(tmp_path / "a.json", SF.REGIME_SLEEVES)
    assert "dip_moderate" in b.paused_until


def test_komite_duraklatilan_sleevei_kullanmaz_ve_golgeler():
    v = CM.evaluate(_ctx(), CM.CommitteeParams(), {"paused_sleeves": ["dip", "dip_moderate"]})
    assert v.trigger != "dip"
    d = v.to_dict()
    assert any(t.get("gate") == "SLEEVE_DURAKLATILDI" for t in (d.get("silenced") or []))


def test_devam_olasiligi_sezgisel_sinirlar():
    good = XE.continuation_probability({"trend_score": 0.9, "cvd_ratio": 0.4, "obi": 0.7, "ema_slope_pct": 0.05, "rsi": 55}, "LONG", 0.9, 1.0)
    bad = XE.continuation_probability({"trend_score": 0.1, "cvd_ratio": -0.4, "obi": 0.3, "ema_slope_pct": -0.05, "rsi": 80}, "LONG", 0.1, 4.0)
    assert 0.05 <= bad["p"] < 0.5 < good["p"] <= 0.95 and "kalibre" in good["note"]
    assert XE.continuation_probability({}, "SHORT", 0.5)["p"] == pytest.approx(1 / (1 + np.exp(0.3)), abs=0.01)


def test_swing_ozellikleri_ve_sleeve():
    idx4 = pd.date_range("2026-01-01", periods=320, freq="4h", tz="UTC")
    c4 = np.linspace(90, 110, 320)
    df4 = pd.DataFrame({"open": c4, "high": c4 * 1.01, "low": c4 * 0.99, "close": c4, "volume": 1000.0}, index=idx4)
    idx1 = pd.date_range("2026-02-01", periods=120, freq="1h", tz="UTC")
    c1 = np.linspace(100, 112, 120); c1[-1] = c1[-2] * 1.002
    df1 = pd.DataFrame({"open": c1, "high": c1 * 1.005, "low": c1 * 0.995, "close": c1, "volume": 500.0}, index=idx1)
    ctx = LC.build_light_context("X/USDT", df4, df1, None, 1_000_000.0)
    sw = ctx["swing"]
    assert sw and sw["trend_4h_up"] and sw["atr_pct_4h"] and sw["swing_low_4h"] < c4[-1] and sw["ret_7d_pct"] is not None
    f = {"ok": True, "rsi": 55.0, "bar_up": True, "atr_pct": 0.3, "vol_ratio": 1.0, "obi": None,
         "swing": {**sw, "pullback_1h": True, "regime": "TREND YUKARI"}}
    out = SF.fire_sleeves(f, ["swing_trend"], None, CM.CommitteeParams())
    assert out and out[0]["kind"] == "swing_trend" and out[0]["exit_mode"] == "DYNAMIC_PEAK" and out[0]["atr_hint"] == sw["atr_pct_4h"]
    assert SF.SLEEVE_TIME_STOP_MIN["swing_trend"] == 4320 and "swing_trend" in SF.allowed_sleeves("TREND YUKARI")
    assert SF.fire_sleeves({**f, "swing": {**sw, "pullback_1h": False, "regime": "TREND YUKARI"}}, ["swing_trend"], None, CM.CommitteeParams()) == []


def _runner(tmp_path, symbols, **over):
    ctxp = _ctx_provider(lambda s: _slow(FakeExchange.path[s][-1], age=30.0))
    cfg = LR.RunnerConfig.from_dict({**SIM.default_config("mexc").to_dict(), "symbols": symbols, "symbols_mode": "fixed", **over})
    reg = LR.RunnerRegistry(output_dir=str(tmp_path), ctx=ctxp, client_factory=factory)
    return reg, reg.create(0, cfg)


def test_kalan_ev_ve_rotasyon(tmp_path):
    syms = ["BTC/USDT", "ETH/USDT"]
    for s in syms:
        FakeExchange.path[s] = _path(dip_last=12, dip_pct=3.0); FakeExchange.path[s][-1] = FakeExchange.path[s][-2] * 1.002
    reg, r = _runner(tmp_path, syms, max_open=1, params={**SIM.default_config("mexc").params, "maker_wait_bars": 0, "min_hold_sec": 0},
                     exit={**SIM.default_config("mexc").exit, "edge_decay_enabled": False})   # edge-decay rotasyondan önce kapatmasın
    r._update_portfolio_mode = lambda *a, **k: None
    r.run_cycle(now=1_000_000.0)
    assert len(r.positions) == 1
    pos = next(iter(r.positions.values()))
    # pozisyon kötüye gitti (kalan EV düşük): fiyat stop'a yaklaştı, trend aşağı
    FakeExchange.path[pos.symbol] = FakeExchange.path[pos.symbol] + [pos.entry * (1 - 0.4 * pos.stop_pct / 100)] * 3   # 0,6 = erken iptal eşiği; altında kal
    for k in range(1, 6):
        r.run_cycle(now=1_000_000.0 + 30 * k)
    assert pos.remaining_ev_pct is not None and pos.cont_prob is not None
    # diğer aday güçlü dip → rotasyon
    other = [s for s in syms if s != pos.symbol][0]
    p2 = _path(dip_last=12, dip_pct=4.0); p2[-1] = p2[-2] * 1.003
    FakeExchange.path[other] = p2
    pos.remaining_ev_pct = -0.5
    r.run_cycle(now=1_000_000.0 + 30 * 7)
    rot = [t for t in r.trades if t["reason"] == "ROTATION"]
    assert rot and rot[0]["symbol"] == pos.symbol and other in r.positions and getattr(r, "_rotations_today", 0) >= 1,         ([(t["symbol"], t["reason"]) for t in r.trades], [e["msg"][:100] for e in list(r.events)[:10]], list(r.positions), r.last_decisions.get(other, {}).get("result"))


def test_gun_ici_tepe_geri_verme_savunmaya_gecirir(tmp_path):
    reg, r = _runner(tmp_path, ["BTC/USDT"])
    r.guard.state.day_start_equity = 1000.0
    r.day = "2026-09-03"
    r.equity = lambda: 1004.0
    r._update_portfolio_mode({}, 1_000_000.0)
    assert r._day_peak == 1004.0 and r.portfolio["mode"] in (PM.RISK_ON, PM.SELECTIVE)
    r.equity = lambda: 1001.0                                   # +4 → +1: %75 geri verildi
    for k in range(1, 4):
        r._update_portfolio_mode({}, 1_000_000.0 + 30 * k)
    assert r.portfolio["mode"] == PM.DEFENSIVE and not r.portfolio["actions"]["new_entries"] and "tepe geri-verme" in r.portfolio["reasons"][0]


def test_evren_tablosu_ve_ek_pariteler(tmp_path, monkeypatch):
    reg, r = _runner(tmp_path, ["BTC/USDT", "ETH/USDT"])
    r.run_cycle(now=time.time())
    rows = r.universe()
    assert len(rows) == 2 and {"symbol", "freshness", "interest", "result", "position", "shadows_open"} <= set(rows[0])
    p = ROOT / "runs" / "live" / "universe_extra.json"
    existed = p.exists(); backup = p.read_text(encoding="utf-8") if existed else None
    monkeypatch.setenv("CRYPTOMIND_EXTRA_SYMBOLS", "1")
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"ts": time.time(), "symbols": ["NEWA/USDT", "BTC/USDT", "NEWB/USDT"]}), encoding="utf-8")
        assert SIM.extra_symbols() == ["NEWA/USDT", "NEWB/USDT"]
        assert "NEWA/USDT" in SIM.default_config("mexc").symbols
        p.write_text(json.dumps({"ts": time.time() - 30 * 86400, "symbols": ["OLD/USDT"]}), encoding="utf-8")
        assert SIM.extra_symbols() == []
    finally:
        if existed:
            p.write_text(backup, encoding="utf-8")
        else:
            p.unlink(missing_ok=True)


def test_deneme_kaydi_ve_validation_report(tmp_path, monkeypatch):
    monkeypatch.setattr(RP, "trials_registry_path", lambda: tmp_path / "research" / "trials.jsonl")
    res = {"trades": [], "equity_curve": [], "capital": 1000.0, "symbols": ["A/USDT"], "n_cycles": 5, "config": {"params": {}, "exit": {}}, "limits": ["x"]}
    an = RP.analyze(res)
    RP.save_result(res, an, tmp_path / "out")
    assert RP.trials_count() == 1
    vr = json.loads((tmp_path / "research" / "VALIDATION_REPORT.json").read_text(encoding="utf-8"))
    assert vr["status"] == "UNVERIFIED" and vr["n_trials_registry"] == 1
    RP.save_result(res, an, tmp_path / "out")
    assert RP.trials_count() == 2


def test_parite_kesfi_sahte_borsayla():
    sys.path.insert(0, str(ROOT / "scripts"))
    import cm_pair_scout as PS

    class FakeClient:
        def load_markets(self):
            return {s: {"active": True, "spot": True, "limits": {"cost": {"min": 1.0}}} for s in ("AAA/USDT", "BBB/USDT", "USDC/USDT", "CCC3L/USDT", "BTC/USDT")}

        def fetch_tickers(self):
            return {"AAA/USDT": {"quoteVolume": 50e6}, "BBB/USDT": {"quoteVolume": 20e6}, "USDC/USDT": {"quoteVolume": 900e6},
                    "CCC3L/USDT": {"quoteVolume": 30e6}, "BTC/USDT": {"quoteVolume": 1e9}}

        def fetch_order_book(self, s, limit=20):
            px = 1.0
            return {"bids": [[px * 0.9999, 200_000]], "asks": [[px * 1.0001, 200_000]]}

        def fetch_ohlcv(self, s, tf, limit=300):
            rng = np.random.default_rng(1 if s == "AAA/USDT" else 2)
            c = np.cumprod(1 + rng.normal(0, 0.004 if s == "AAA/USDT" else 0.001, limit))
            return [[i * 60000, x, x, x, x, 1.0] for i, x in enumerate(c)]
    res = PS.scout("fake", top=2, client=FakeClient())
    assert res["symbols"] == ["AAA/USDT", "BBB/USDT"] and all(r["eligible"] for r in res["rows"])
