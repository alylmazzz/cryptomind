# -*- coding: utf-8 -*-
"""Video Dip-Scalp stratejisi — saf fonksiyon testleri.

Videonun ÖLÇÜLMÜŞ dersleri (komisyon yer, 0-15 dk zarar, 15-60 dk kâr) burada
kural olarak kilitlenir. Kilitlenmemiş bir ders, bir sonraki düzenlemede
sessizce kaybolur.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agi_trader.strategies import video_scalp as VS  # noqa: E402


def _bars(n=150, base=100.0, dip_last=0, dip_pct=3.0, seed=3, noise=0.0005):
    """Yatay seyir + son `dip_last` barda `dip_pct` düşüş (video: 'düşmüş coin')."""
    rng = np.random.default_rng(seed)
    c = base * np.exp(np.cumsum(rng.normal(0.0, noise, n)))
    if dip_last:
        ramp = np.linspace(0, dip_pct / 100.0, dip_last)
        c[-dip_last:] = c[-dip_last:] * (1.0 - ramp)
    idx = pd.date_range("2026-01-01", periods=n, freq="min")
    return pd.DataFrame({"open": c, "high": c * 1.001, "low": c * 0.999,
                         "close": c, "volume": 1.0}, index=idx)


# ─────────────────────────── parametreler ───────────────────────────
def test_video_preset_birebir():
    p = VS.ScalpParams.from_dict(VS.VIDEO_PRESET)
    assert p.rr == 1.6 and p.giveback == 0.5
    assert p.min_hold_sec == 900 and p.max_hold_sec == 3600 and p.loop_sec == 30


def test_parametreler_kelepcelenir():
    p = VS.ScalpParams.from_dict({"rr": 99, "giveback": 5, "loop_sec": 1,
                                  "min_hold_sec": 7200, "max_hold_sec": 10})
    assert p.rr == 5.0 and p.giveback == 0.9 and p.loop_sec == 10
    assert p.max_hold_sec > p.min_hold_sec


def test_varsayilan_gunluk_limit_videodan_kucuk():
    """Videodaki %20 tek gecede sermayenin beşte biri — varsayılan OLAMAZ."""
    assert VS.DEFAULT_DAILY_LOSS_PCT < VS.VIDEO_DAILY_LOSS_PCT


# ─────────────────────────── komisyon ───────────────────────────
def test_bnb_indirimi_yalniz_acikken():
    p = VS.ScalpParams(fee_bps=10.0, bnb_discount=False)
    assert VS.effective_fee_bps(p) == 10.0
    p.bnb_discount = True
    assert VS.effective_fee_bps(p) == pytest.approx(7.5)


def test_komisyon_kapisi_videodaki_hesap():
    """Video: 1.000 $ işlem, 3 $ komisyon (gidiş-dönüş ≈ %0,3). Hedef %1,6 geçer,
    %0,3'lük 'azıcık kâr' hedefi GEÇMEZ — bot kazanırken hesap erimesin."""
    p = VS.ScalpParams(fee_bps=10.0, min_gross_to_cost=2.0)
    cost = VS.roundtrip_cost_pct(p, spread_bps=2.0, slippage_bps=2.0)
    assert 0.2 < cost < 0.35
    assert VS.fee_gate(1.6, cost, p)["ok"]
    g = VS.fee_gate(0.3, cost, p)
    assert not g["ok"] and "İŞLEM YOK" in g["note"]


# ─────────────────────────── sinyal ───────────────────────────
def test_dip_long_sinyali_uretir():
    df = _bars(dip_last=12, dip_pct=3.0)
    s = VS.signal(df, VS.ScalpParams())
    assert s["direction"] == "LONG" and s["z"] < -1.5 and s["rsi"] <= 35


def test_yatay_seyirde_sinyal_yok():
    s = VS.signal(_bars(), VS.ScalpParams())
    assert s["direction"] is None and s["reasons"]


def test_spotta_short_yok():
    df = _bars(dip_last=12, dip_pct=-3.0)         # yükseliş
    s = VS.signal(df, VS.ScalpParams(allow_short=True), market_type="spot")
    assert s["direction"] is None
    s2 = VS.signal(df, VS.ScalpParams(allow_short=True), market_type="future")
    assert s2["direction"] == "SHORT"


def test_yetersiz_bar_gerekce_verir():
    s = VS.signal(_bars(n=10), VS.ScalpParams())
    assert s["direction"] is None and "yetersiz" in s["reasons"][0]


def test_sinyal_ileriye_bakmaz():
    """Son barı değiştirmek önceki barın sinyalini değiştiremez."""
    df = _bars(dip_last=12)
    a = VS.compute_features(df.iloc[:-1], VS.ScalpParams())
    df2 = df.copy()
    df2.iloc[-1, df2.columns.get_loc("close")] *= 1.5
    b = VS.compute_features(df2.iloc[:-1], VS.ScalpParams())
    assert a == b


# ─────────────────────────── plan ───────────────────────────
def test_plan_rr_ve_ufka_bagli_stop():
    p = VS.ScalpParams(rr=1.6, stop_sigma_mult=1.0, min_stop_pct=0.0, max_stop_pct=50.0)
    a = VS.plan_trade("LONG", 100.0, 0.1, p)
    assert a["target_pct"] == pytest.approx(a["stop_pct"] * 1.6, rel=1e-3)
    assert a["stop"] < 100.0 < a["target"]
    # ufuk 4 katına çıkınca stop √4 = 2 katına
    p2 = VS.ScalpParams(rr=1.6, stop_sigma_mult=1.0, min_stop_pct=0.0, max_stop_pct=50.0,
                        max_hold_sec=p.max_hold_sec * 4)
    b = VS.plan_trade("LONG", 100.0, 0.1, p2)
    assert b["stop_pct"] == pytest.approx(a["stop_pct"] * 2.0, rel=0.02)


def test_plan_short_ayna():
    p = VS.ScalpParams()
    a = VS.plan_trade("SHORT", 100.0, 0.1, p)
    assert a["target"] < 100.0 < a["stop"]


# ─────────────────────────── çıkış ───────────────────────────
def _st(entry=100.0, stop_pct=1.0, rr=1.6, opened=1000.0, d="LONG"):
    s = 1 if d == "LONG" else -1
    return VS.ExitState(d, entry, entry * (1 - s * stop_pct / 100), entry * (1 + s * stop_pct * rr / 100),
                        opened, 0.0, stop_pct)


def test_stop_asgari_tutmaya_bakmaz():
    p = VS.ScalpParams(min_hold_sec=900)
    st = _st()
    d = VS.exit_decision(st, 98.9, p, now=1000.0 + 10)
    assert d and d["reason"] == "STOP"


def test_tp_asgari_tutmadan_once_tetiklenmez():
    """Video: 0-15 dk işlemler zarar bölgesi → hedef bile 15 dk'dan önce alınmaz."""
    p = VS.ScalpParams(min_hold_sec=900)
    st = _st()
    assert VS.exit_decision(st, 101.7, p, now=1000.0 + 300) is None
    d = VS.exit_decision(st, 101.7, p, now=1000.0 + 901)
    assert d and d["reason"] == "TP"


def test_giveback_tepe_karin_yarisinda_cikar():
    """Video: 1.000 → 1.100'e çıkıp dönerse 1.050'de sat."""
    p = VS.ScalpParams(min_hold_sec=0, giveback=0.5, giveback_activate_r=0.5)
    st = _st(stop_pct=1.0)
    assert VS.exit_decision(st, 101.0, p, now=1001.0) is None        # tepe %1 (1R)
    assert st.peak_pnl_pct == pytest.approx(1.0)
    assert VS.exit_decision(st, 100.6, p, now=1002.0) is None        # %0,6 > %0,5
    d = VS.exit_decision(st, 100.49, p, now=1003.0)
    assert d and d["reason"] == "GIVEBACK" and d["peak_pnl_pct"] == pytest.approx(1.0)


def test_giveback_silahlanmadan_tetiklenmez():
    p = VS.ScalpParams(min_hold_sec=0, giveback=0.5, giveback_activate_r=0.5)
    st = _st(stop_pct=1.0)
    VS.exit_decision(st, 100.2, p, now=1001.0)      # tepe %0,2 < 0,5R
    assert VS.exit_decision(st, 100.05, p, now=1002.0) is None


def test_zaman_stopu_60dk():
    p = VS.ScalpParams(min_hold_sec=900, max_hold_sec=3600)
    st = _st()
    assert VS.exit_decision(st, 100.1, p, now=1000.0 + 3599) is None
    d = VS.exit_decision(st, 100.1, p, now=1000.0 + 3600)
    assert d and d["reason"] == "TIME_STOP"


def test_short_cikis_ayna():
    p = VS.ScalpParams(min_hold_sec=0)
    st = _st(d="SHORT")
    d = VS.exit_decision(st, 101.1, p, now=1001.0)
    assert d and d["reason"] == "STOP"
    st = _st(d="SHORT")
    d = VS.exit_decision(st, 98.3, p, now=1001.0)
    assert d and d["reason"] == "TP"


def test_tutma_kovalari():
    assert VS.hold_bucket(5 * 60) == "0-15 dk"
    assert VS.hold_bucket(30 * 60) == "15-60 dk"
    assert VS.hold_bucket(90 * 60) == "60+ dk"


def test_kunye_videoyu_ve_dersleri_tasir():
    d = VS.describe()
    assert d["source"] == VS.VIDEO_URL
    assert any("komisyon" in x.lower() for x in d["measured_lessons"])
    assert any("KOMİSYON KAPISI" in x for x in d["cryptomind_gates"])
