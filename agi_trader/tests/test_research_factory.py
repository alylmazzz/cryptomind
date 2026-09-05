# -*- coding: utf-8 -*-
"""7. tur — kaçırılan-fırsat atıf motoru, strateji kütüphanesi (250 → 13. turda 260) + hasatçı, yeni sleeve'ler,
meta-tahsisçi, gölge araştırma modülleri (çift / carry / üçgen / piyasa yapıcı), koşucu kör noktaları."""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agi_trader.learn.missed import MissedEngine, normalize_gate, GATE_PROPOSAL  # noqa: E402
from agi_trader.learn.allocator import MetaAllocator, MIN_N as ALLOC_MIN_N  # noqa: E402
from agi_trader.research import library as LIB  # noqa: E402
from agi_trader.research.harvester import Harvester, parse_strategy  # noqa: E402
from agi_trader.research import pairs as PR  # noqa: E402
from agi_trader.research import carry as CR  # noqa: E402
from agi_trader.research import triangular as TR  # noqa: E402
from agi_trader.research import market_making as MM  # noqa: E402
from agi_trader.strategies import sleeves_fast as SF  # noqa: E402
from agi_trader.strategies import committee as CM  # noqa: E402
from agi_trader.strategies import portfolio_mode as PM  # noqa: E402
from agi_trader.strategies.lifecycle import Lifecycle, DEFAULT_SLEEVES  # noqa: E402
from agi_trader.auto import live_runner as LR  # noqa: E402
from agi_trader.auto import simulator as SIM  # noqa: E402
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


def _bars(closes, hi=None, lo=None):
    c = np.array(closes, dtype=float)
    idx = pd.date_range("2026-01-01", periods=len(c), freq="min")
    return pd.DataFrame({"open": c, "high": (c * 1.001 if hi is None else hi), "low": (c * 0.999 if lo is None else lo),
                         "close": c, "volume": np.full(len(c), 100.0)}, index=idx)


# ═══════════════════════════ kaçırılan-fırsat motoru ═══════════════════════════
def _trace(sym="SOL/USDT", vetoes=("KOMİSYON brüt/maliyet 1.8 < 2",), fast=None, votes=None, trigger="dip"):
    return {"symbol": sym, "direction": "LONG", "trigger": trigger, "allowed": False, "vetoes": list(vetoes),
            "plan": {"entry": 100.0, "stop": 99.0, "target": 101.6, "stop_pct": 1.0, "target_pct": 1.6, "rr": 1.6},
            "fast": fast or {"z": -2.3, "rsi": 28.0, "vol_ratio": 1.8, "rs_rank": 0.9, "trend_up": False, "ema_cross_up": False,
                             "breakout_up": False, "bb_prev_pctile": 60, "swept_low": False, "dist_vwap_pct": -0.2, "move_4h_pct": -0.5},
            "votes": votes or [{"role": "formasyon", "score": 0.0, "confidence": 0.0, "data_ok": False},
                               {"role": "piyasa_yapisi", "score": 0.4, "confidence": 0.6, "data_ok": True}],
            "ticket": {"ev_pct": 0.12}, "score": 0.3, "confidence": 0.5, "competition": [{"kind": "dip"}]}


def test_gate_normalizasyonu():
    assert normalize_gate("KOMİSYON brüt/maliyet 1.8 < 2 | SPREAD 20 bps > 15") == "KOMİSYON"
    assert normalize_gate("OY +0.10 < eşik 0.25") == "OY" and normalize_gate("GÜVEN 0.30 < 0.35") == "GÜVEN"
    assert normalize_gate("NEGATİF NET EV (%-0.020) — NO TRADE") == "NEGATİF_EV"
    assert normalize_gate("MAX CHASE aşıldı (101.2)") == "MAX_CHASE" and normalize_gate("DENETÇİ: hedef < 1.2×stop") == "DENETÇİ"
    assert normalize_gate("AÇIK POZİSYON tavanı (5)") == "AÇIK_POZİSYON" and normalize_gate("SAĞLIK RED") == "SAĞLIK"


def test_veto_gölgesi_hedefe_ulasinca_kacirilan_kazanc_ve_atif(tmp_path):
    m = MissedEngine(tmp_path / "m.json", journal_md=tmp_path / "K.md", journal_jsonl=tmp_path / "m.jsonl")
    rec = m.on_vetoed(_trace(), {"slow_ctx": False, "tier": "light", "news": False, "book_depth": True, "fees_verified": False, "qual_cell": True},
                      {"spread_bps": 2.0, "bid_depth_usd": 5e5, "ask_depth_usd": 5e5}, None, 3600.0, 1000.0, {"portfolio_mode": "RISK_ON"})
    assert rec and rec["gate"] == "KOMİSYON" and rec["info"]["roles_no_data"] == ["formasyon"] and not rec["info"]["roles_full"]
    assert m.on_vetoed(_trace(), {}, None, None, 3600.0, 1001.0) is None           # aynı parite açıkken tekrar yok
    # hedef vuruldu (stop görülmeden)
    out = m.update({"SOL/USDT": _bars([100.0, 100.5, 101.0], hi=[100.2, 100.9, 101.7], lo=[99.5, 100.1, 100.6])}, 1500.0, cost_pct=0.14)
    assert len(out) == 1 and out[0]["outcome"] == "TARGET"
    a = out[0]["attribution"]
    assert a["verdict"] == "KAÇIRILAN KAZANÇ" and a["gate"] == "KOMİSYON" and a["net_missed_pct"] == pytest.approx(1.46)
    assert "hacim_patlamasi" in a["ignored_supportive"] and "goreli_guc_ust" in a["ignored_supportive"] and "derin_dip_z" in a["ignored_supportive"]
    assert "slow_ctx" in a["missing_info"] and "fees_verified" in a["missing_info"] and "roles_full" in a["missing_info"]
    assert "komisyon kapısı" in a["how"] and "NASIL DÜZELİR" in a["how"]
    assert out[0]["mfe_pct"] == pytest.approx(1.7, rel=1e-3)
    md = (tmp_path / "K.md").read_text(encoding="utf-8")
    assert "KAÇIRILAN KAZANÇ" in md and "Neden yapılmadı" in md and (tmp_path / "m.jsonl").exists()


def test_stop_olan_aday_dogru_kacinma_ve_uyaranlar(tmp_path):
    m = MissedEngine(tmp_path / "m.json")
    fast = {"z": -2.3, "rsi": 28.0, "vol_ratio": 0.5, "trend_up": False, "breakdown": True, "spread_bps": 12.0}
    m.on_vetoed(_trace(fast=fast), {"slow_ctx": True}, {"spread_bps": 12.0, "bid_depth_usd": 1e5, "ask_depth_usd": 1e5},
                {"severe_risk": True, "confirmed": False}, 3600.0, 1000.0)
    out = m.update({"SOL/USDT": _bars([100.0, 99.2, 98.8], hi=[100.3, 99.6, 99.0], lo=[99.4, 98.9, 98.5])}, 1100.0)
    a = out[0]["attribution"]
    assert out[0]["outcome"] == "STOP" and a["verdict"] == "DOĞRU KAÇINMA"
    assert {"haber_riski", "trend_ters", "kirilma_ters", "spread_genis", "hacim_yok"} <= set(a["warnings_present"])


def test_belirsiz_ve_ufuk_dolmasi(tmp_path):
    m = MissedEngine(tmp_path / "m.json")
    m.on_vetoed(_trace("A/USDT"), {}, None, None, 100.0, 1000.0)
    m.on_vetoed(_trace("B/USDT"), {}, None, None, 100.0, 1000.0)
    out = m.update({"A/USDT": _bars([100.0, 100.0], hi=[100.1, 102.0], lo=[99.9, 98.0]), "B/USDT": _bars([100.0, 100.3])}, 1200.0)
    o = {r["symbol"]: r["outcome"] for r in out}
    assert o["A/USDT"] == "AMBIGUOUS" and o["B/USDT"] == "TIMEOUT"
    rep = m.report()
    assert rep["n_missed"] == 0 and rep["gates"][0]["ambiguous"] + rep["gates"][0]["timeout"] == 2


def test_rapor_kapi_isabeti_ozellik_lifti_ve_bilgi_yoklugu(tmp_path):
    m = MissedEngine(tmp_path / "m.json")
    now = 1000.0
    # 14 kaçırılan (hepsinde hacim patlaması + ağır bağlam YOK), 4 kaçınılan (hacim yok + ağır bağlam VAR)
    for i in range(14):
        sym = f"W{i}/USDT"
        m.on_vetoed(_trace(sym, fast={"z": -2.2, "rsi": 30.0, "vol_ratio": 1.9}), {"slow_ctx": False}, None, None, 100.0, now + i)
    for i in range(4):
        sym = f"L{i}/USDT"
        m.on_vetoed(_trace(sym, fast={"z": -2.2, "rsi": 30.0, "vol_ratio": 0.6}), {"slow_ctx": True}, None, None, 100.0, now + 50 + i)
    bars = {f"W{i}/USDT": _bars([100.0, 101.0], hi=[100.5, 101.8], lo=[99.6, 100.5]) for i in range(14)}
    bars.update({f"L{i}/USDT": _bars([100.0, 99.0], hi=[100.2, 99.4], lo=[99.5, 98.7]) for i in range(4)})
    m.update(bars, now + 60)
    rep = m.report(now + 61)
    g = next(x for x in rep["gates"] if x["gate"] == "KOMİSYON")
    assert g["missed"] == 14 and g["avoided"] == 4 and g["verdict"].startswith("ZARARLI") and g["precision"] == pytest.approx(4 / 18, abs=1e-3)
    f = next(x for x in rep["features"] if x["feature"] == "hacim_patlamasi")
    assert f["in_missed"] == 14 and f["in_avoided"] == 0 and f["lift"] > 3 and "göz ardı" in f["note"]
    inf = next(x for x in rep["info"] if x["info"] == "slow_ctx")
    assert inf["missed_rate_without"] == 1.0 and inf["missed_rate_with"] == 0.0 and "YOKKEN" in inf["note"]
    # n ≥ 20 değil → henüz öneri yok
    assert rep["proposals"] == {}
    for i in range(3):
        m.on_vetoed(_trace(f"X{i}/USDT"), {}, None, None, 100.0, now + 100 + i)
    m.update({f"X{i}/USDT": _bars([100.0, 101.0], hi=[100.5, 101.8], lo=[99.6, 100.5]) for i in range(3)}, now + 200)
    props = m.proposals(now + 201, {"min_gross_to_cost": 2.0})
    assert props == {"min_gross_to_cost": 1.5}
    assert m.proposals(now + 300, {"min_gross_to_cost": 2.0}) == {"min_gross_to_cost": 1.5}    # soğuma: aynı öneri korunur


def test_olculmemis_kapi_oneri_uretmez_ve_koruyan_kapi(tmp_path):
    m = MissedEngine(tmp_path / "m.json")
    for i in range(25):
        m.on_vetoed(_trace(f"S{i}/USDT", vetoes=("OY +0.10 < eşik 0.25",)), {}, None, None, 100.0, 1000.0 + i)
    m.update({f"S{i}/USDT": _bars([100.0, 99.0], hi=[100.2, 99.3], lo=[99.5, 98.5]) for i in range(25)}, 1100.0)
    rep = m.report(1101.0)
    g = rep["gates"][0]
    assert g["gate"] == "OY" and g["verdict"].startswith("KORUYOR") and rep["proposals"] == {}


def test_degerlendirilmeyen_ve_yurutme_kaynakli_kayitlar(tmp_path):
    m = MissedEngine(tmp_path / "m.json")
    r1 = m.on_unevaluated("ARB/USDT", "TOP_K", {"z20": -1.8, "vol_ratio": 1.4, "bar_up": True}, 1.0, "LONG", 1.0, 1.6, 100.0, 1000.0,
                          info={"slow_ctx": False}, detail="Top-K dışı")
    r2 = m.on_execution_miss(_trace("OP/USDT"), {"entry": 100.0, "stop": 99.0, "target": 101.6, "stop_pct": 1.0, "target_pct": 1.6, "rr": 1.6},
                             "MAKER_DOLMADI", "maker dolmadı", 100.0, 1000.0)
    assert r1["kind"] == "unevaluated" and r2["kind"] == "execution"
    m.count_blind("BAYAT", 3)
    out = m.update({"ARB/USDT": _bars([1.0, 1.02], hi=[1.005, 1.02], lo=[0.995, 1.0]), "OP/USDT": _bars([100.0, 101.7], hi=[100.5, 101.8], lo=[99.6, 100.9])}, 1050.0)
    verdicts = {r["symbol"]: r["attribution"]["how"] for r in out}
    assert "hiç DEĞERLENDİRİLMEDİ" in verdicts["ARB/USDT"] and "YÜRÜTME kaynaklı" in verdicts["OP/USDT"]
    rep = m.report()
    assert rep["blind"] == {"BAYAT": 3} and rep["kinds"] == {"unevaluated": 1, "execution": 1}


# ═══════════════════════════ meta-tahsisçi ═══════════════════════════
def test_meta_tahsisci_olculmeden_agirlik_1_sonra_sinirli(tmp_path):
    a = MetaAllocator(tmp_path / "a.json", SF.REGIME_SLEEVES)
    assert a.weight("dip", "RANGE / YATAY") == 1.0 and a.sleeve_reliability() == {}
    for i in range(ALLOC_MIN_N):
        a.record("dip", "RANGE / YATAY", i % 4 != 0, 1.0)      # %75 kazanma
    r = a.reliability("dip", "RANGE / YATAY")
    assert r["measured"] and 0.6 < r["mean"] < 0.8 and a.weight("dip", "RANGE / YATAY") == pytest.approx(min(1.2, 2 * r["mean"]), abs=1e-6)
    for i in range(ALLOC_MIN_N):
        a.record("momentum", None, False, -1.0)
    assert a.weight("momentum") == 0.5 and a.sleeve_reliability()["momentum"] < 0.3
    assert a.regime_fit("momentum", "RANGE / YATAY") == 0.5 and a.regime_fit("pullback", "TREND YUKARI") == 1.0
    assert a.score("dip", 0.2, "RANGE / YATAY", 0.8) is not None and a.score("dip", None, None) is None
    rk = a.thompson_ranking(["dip", "momentum", "pullback"])
    assert len(rk) == 3 and rk[0]["sleeve"] != "momentum"
    b = MetaAllocator(tmp_path / "a.json", SF.REGIME_SLEEVES)
    assert b.reliability("dip", "RANGE / YATAY")["n"] == ALLOC_MIN_N


# ═══════════════════════════ kütüphane + hasatçı ═══════════════════════════
def test_kutuphane_kayit_durumlar_ve_huni():
    assert len(LIB.LIBRARY) == 260
    st = LIB.by_status()
    assert st["IMPLEMENTED"] >= 60 and st["SHADOW"] >= 15 and st["NO_DATA"] >= 15 and st["DATA_NOT_WIRED"] >= 10
    assert {"adaptive_trend", "rs_momentum", "failed_breakdown", "obi_momentum", "bos_retest", "donchian_breakout"} <= set(LIB.sleeves_implemented())
    for k in LIB.sleeves_implemented():
        assert k in SF.SLEEVE_TR and k in DEFAULT_SLEEVES, k
    onchain = [r for r in LIB.LIBRARY if r["family"] == "L"]
    assert all(r["status"] == "NO_DATA" and r["blocker"] for r in onchain)
    fun = LIB.funnel()
    assert fun[0]["n"] == 260 and fun[-1]["n"] == 0                # hiçbiri QUALIFIED değil — dürüst
    lc = Lifecycle()
    s = LIB.summary(lc.status())
    row = next(r for r in s["rows"] if r["impl_key"] == "dip")
    assert row["lifecycle_stage"] == "PAPER" and row["pipeline_stage"] == "PAPER"


FREQTRADE_SAMPLE = '''
# SPDX-License-Identifier: MIT
from freqtrade.strategy import IStrategy
import talib.abstract as ta

class SampleStrategy(IStrategy):
    timeframe = "5m"
    stoploss = -0.03
    minimal_roi = {"0": 0.02}

    def populate_indicators(self, dataframe, metadata):
        dataframe["rsi"] = ta.RSI(dataframe)
        dataframe["ema20"] = ta.EMA(dataframe, timeperiod=20)
        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        dataframe.loc[(dataframe["rsi"] < 30) & (dataframe["close"] > dataframe["ema20"]), "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        dataframe.loc[(dataframe["rsi"] > 70), "exit_long"] = 1
        return dataframe
'''


def test_hasatci_freqtrade_dosyasini_hipoteze_cevirir_uretime_gecirmez(tmp_path):
    h = Harvester(tmp_path / "inbox")
    rec = h.harvest_text(FREQTRADE_SAMPLE, "sample.py")
    assert rec["format"] == "FREQTRADE" and rec["name"] == "SampleStrategy" and rec["timeframe"] == "5m" and rec["stoploss"] == -0.03
    assert rec["license"]["license"] == "MIT" and rec["static_review"]["ok"] and rec["status"] == "RESEARCH_INBOX"
    assert set(rec["feature_families"]) == {"ema", "rsi"} and rec["implementable"]
    assert rec["entry_conditions"] and "rsi" in rec["entry_conditions"][0] and rec["exit_conditions"]
    assert "yeniden OOS test" in rec["hypothesis"]
    assert (tmp_path / "inbox" / f"{rec['sha']}.json").exists() and h.inbox_summary()["n"] == 1
    bad = parse_strategy("import subprocess\nclass X(IStrategy):\n    timeframe='1m'\n", "bad.py")
    assert bad["status"] == "REJECTED" and "subprocess" in bad["reject_reason"]
    nolic = parse_strategy("class Y:\n    pass\n", "y.py")
    assert nolic["status"] == "LICENSE_UNKNOWN" and nolic["format"] == "GENERIC"


# ═══════════════════════════ yeni sleeve'ler ═══════════════════════════
def _f(**k):
    d = dict(ok=True, z=0.0, rsi=55.0, bar_up=True, trend_up=True, dist_ema_pct=-0.3, atr_pct=0.3, breakout_up=False, vol_ratio=1.1,
             ema_cross_up=False, ema_slope_pct=0.02, bb_prev_pctile=50, swept_low=False, lower_wick_ratio=0.0, range_ok=False,
             range_pos=0.5, range_low=95.0, range_high=105.0, range_width_pct=10.0, vwap=100.0, dist_vwap_pct=0.1,
             prior_swing_low=97.0, prior_swing_high=103.0, move_4h_pct=0.0, rs_rank=0.7, trend_score=0.8, extended=False,
             pullback_atr=1.0, donchian_break=False, donchian_hi55=104.0, bos_retest_up=False, bos_level=102.0, failed_breakdown=False,
             failed_breakout=False, fb_low=96.0, fb_high=104.0, obi=0.5, microprice_dev_bps=0.0, spread_bps=2.0, ema_fast=99.7, price=100.0, adx=25.0)
    d.update(k)
    return d


def test_yeni_sleeveler_ateslenir_ve_rejim_secici():
    p = CM.CommitteeParams()
    a = SF.fire_sleeves(_f(), ["adaptive_trend"], None, p)
    assert a and a[0]["kind"] == "adaptive_trend" and a[0]["exit_mode"] == "DYNAMIC_PEAK" and a[0]["stop_hint"] < 97.0
    assert SF.fire_sleeves(_f(extended=True), ["adaptive_trend"], None, p) == []
    assert SF.fire_sleeves(_f(pullback_atr=0.1), ["adaptive_trend"], None, p) == []
    d = SF.fire_sleeves(_f(donchian_break=True, vol_ratio=1.6), ["donchian_breakout"], None, p)
    assert d and d[0]["stop_hint"] == 99.7
    b = SF.fire_sleeves(_f(bos_retest_up=True), ["bos_retest"], None, p)
    assert b and b[0]["stop_hint"] == pytest.approx(102.0 * (1 - 0.003))
    fb = SF.fire_sleeves(_f(failed_breakdown=True, trend_up=False), ["failed_breakdown"], None, p)
    assert fb and fb[0]["target_hint"] == 103.0 and fb[0]["stop_hint"] < 96.0
    assert SF.fire_sleeves(_f(failed_breakout=True, bar_up=False), ["failed_breakout"], None, p) == []      # spot: short yok
    fo = SF.fire_sleeves(_f(failed_breakout=True, bar_up=False), ["failed_breakout"], None, p, allow_short=True)
    assert fo and fo[0]["direction"] == "SHORT"
    o = SF.fire_sleeves(_f(obi=0.72, microprice_dev_bps=1.8), ["obi_momentum"], None, p)
    assert o and o[0]["kind"] == "obi_momentum" and SF.SLEEVE_TIME_STOP_MIN["obi_momentum"] == 45
    assert SF.fire_sleeves(_f(obi=0.72, microprice_dev_bps=1.8, spread_bps=12.0), ["obi_momentum"], None, p) == []
    assert "adaptive_trend" in SF.allowed_sleeves("TREND YUKARI") and "adaptive_trend" not in SF.allowed_sleeves("RANGE / YATAY")
    assert "failed_breakdown" in SF.allowed_sleeves("TREND AŞAĞI")


def test_ek_ozellikler_gercek_df_uzerinde_hesaplanir():
    p = CM.CommitteeParams()
    closes = list(np.linspace(95, 105, 150))
    closes[-1] = closes[-2] * 1.001
    df = _df(closes, up_last=False)
    f = CM.fast_features(df, p)
    f["price"] = float(df["close"].iloc[-1])
    book = {"bid": 104.9, "ask": 105.0, "bid_depth_usd": 700_000.0, "ask_depth_usd": 300_000.0, "spread_bps": 1.0}
    f = SF.extra_features(df, f, 0.9, book, {"chart": {"regime": {"label": "TREND YUKARI"}}})
    assert f["adx"] is not None and f["trend_score"] >= 0.8 and f["regime_4h"] == "TREND YUKARI"
    assert f["obi"] == pytest.approx(0.7) and f["microprice_dev_bps"] > 0 and f["spread_bps"] == 1.0
    assert isinstance(f["donchian_break"], bool) and isinstance(f["failed_breakdown"], bool) and isinstance(f["bos_retest_up"], bool)
    v = CM.evaluate(_ctx(), CM.CommitteeParams())
    assert "regime" in v.to_dict() and v.to_dict()["fast"].get("trend_score") is not None


# ═══════════════════════════ gölge araştırma modülleri ═══════════════════════════
def test_kointegre_cift_bulunur_kalman_ve_yari_omur():
    rng = np.random.default_rng(3)
    n = 400
    x = np.cumsum(rng.normal(0, 0.01, n)) + 5.0
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.85 * spread[t - 1] + rng.normal(0, 0.004)
    y = 1.3 * x + 0.2 + spread
    r = PR.analyze_pair(np.exp(y), np.exp(x))
    assert r["cointegrated"] and abs(r["beta_ols"] - 1.3) < 0.1 and abs(r["beta_kalman"] - 1.3) < 0.15
    assert r["half_life_bars"] is not None and 2 < r["half_life_bars"] < 15
    z = np.cumsum(rng.normal(0, 0.01, n)) + 3.0           # bağımsız rastgele yürüyüş → kointegre değil
    r2 = PR.analyze_pair(np.exp(y), np.exp(z))
    assert not r2["cointegrated"]
    found = PR.scan_pairs({"A/USDT": np.exp(y), "B/USDT": np.exp(x), "C/USDT": np.exp(z)})
    assert len(found) == 1 and {found[0]["a"], found[0]["b"]} == {"A/USDT", "B/USDT"}


def test_cift_golgesi_acar_kapatir_emir_vermez(tmp_path):
    rng = np.random.default_rng(5)
    n = 300
    x = np.cumsum(rng.normal(0, 0.01, n)) + 5.0
    sp = np.zeros(n)
    for t in range(1, n):
        sp[t] = 0.85 * sp[t - 1] + rng.normal(0, 0.003)
    y = 1.1 * x + sp
    sh = PR.PairsShadow(tmp_path / "p.json")
    closes = {"A/USDT": np.exp(y), "B/USDT": np.exp(x)}
    assert sh.rescan(closes, 1000.0)
    # spread'i yapay olarak açalım → |z| ≥ 2 giriş
    y2 = y.copy(); y2[-1] += 0.03
    ev = sh.step({"A/USDT": np.exp(y2), "B/USDT": np.exp(x)}, 2000.0)
    assert ev and "opened" in ev[0] and sh.open and "GÖLGE" in list(sh.open.values())[0]["note"]
    y3 = y.copy()                                          # geri kapandı → çıkış (z ya da 3×yarı-ömür zaman)
    ev2 = sh.step({"A/USDT": np.exp(y3), "B/USDT": np.exp(x)}, 2000.0 + 12 * 4 * 3600)
    assert ev2 and ev2[0].get("reason") in ("Z_EXIT", "TIME") and not sh.open and sh.status()["n_closed"] == 1


def test_carry_ev_ve_dispersiyon():
    ev = CR.carry_ev(0.0003, 3.0, 5.0, 5.0)                 # %0,03/8s → 3 gün %0,27 brüt
    assert ev["side"] == "SPOT_LONG_PERP_SHORT" and ev["gross_pct"] == pytest.approx(0.27) and ev["net_pct"] < ev["gross_pct"]
    assert ev["net_pct"] < 0                                 # 2 bacak taker + spread + rezerv %0,35 > brüt → tipik funding carry TAŞIMAZ
    ev2 = CR.carry_ev(0.00001, 3.0, 5.0, 5.0)
    assert ev2["net_pct"] < 0                                # maliyet funding'i yer → NEGATİF EV → fırsat değil
    rows = CR.scan_funding({"BTC/USDT": {"rate": 0.0008, "venue": "binance"}, "ETH/USDT": {"rate": 0.00001, "venue": "binance"}})
    assert rows[0]["symbol"] == "BTC/USDT" and rows[0]["qualifies"] and not rows[1]["qualifies"]
    d = CR.dispersion({"binance": {"SOL/USDT": 0.0004}, "bybit": {"SOL/USDT": 0.0001}})
    assert d[0]["long_venue"] == "bybit" and d[0]["short_venue"] == "binance"
    assert CR.fetch_funding_ccxt("nonexistent_exchange", ["BTC/USDT"]) == {}


def test_ucgen_arbitraj_ucret_sonrasi_bulur_ve_yanlis_pozitif_vermez():
    t = {"BTC/USDT": {"bid": 100.0, "ask": 100.1}, "ETH/USDT": {"bid": 10.0, "ask": 10.01}, "ETH/BTC": {"bid": 0.1, "ask": 0.1001}}
    assert TR.find_triangles(t, "USDT", 5.0, 5.0) == []     # tutarlı fiyatlar → fırsat yok
    t2 = dict(t); t2["ETH/BTC"] = {"bid": 0.1030, "ask": 0.1031}   # ETH/BTC pahalı: USDT→BTC→ETH? hayır; USDT→ETH→BTC→USDT
    found = TR.find_triangles(t2, "USDT", 5.0, 5.0)
    assert found and found[0]["path"] == ["USDT", "ETH", "BTC", "USDT"] and found[0]["net_bps"] > 5
    cyc = TR.bellman_ford_negative_cycle(t2, 5.0, 5.0)
    assert cyc is not None and len(cyc) >= 4
    assert TR.bellman_ford_negative_cycle(t, 5.0, 5.0) is None


def test_avellaneda_kotasyon_envanter_egimi_ve_golge_dolum(tmp_path):
    q0 = MM.as_quotes(100.0, 0.05, 0.0)
    qp = MM.as_quotes(100.0, 0.05, 2.0)                      # uzun envanter → kotasyon aşağı kayar
    qs = MM.as_quotes(100.0, 0.20, 0.0)                      # yüksek σ → spread açılır
    assert qp["reservation"] < q0["reservation"] and qs["spread"] > q0["spread"] and q0["bid"] < 100.0 < q0["ask"]
    sh = MM.MMShadow(tmp_path / "mm.json", maker_fee_bps=0.0)
    book = {"spread_bps": 1.0, "bid_depth_usd": 1e6, "ask_depth_usd": 1e6}
    r = sh.step("BTC/USDT", 100.0, 100.02, 99.90, 0.05, book, 1000.0, obi=0.5)
    assert r["filled"] and r["filled"][0] == "BUY" and sh.state["BTC/USDT"]["inv_usd"] == 100.0
    r2 = sh.step("BTC/USDT", 100.0, 100.20, 100.00, 0.05, book, 1060.0, obi=0.5)
    assert r2["filled"] and r2["filled"][0] == "SELL" and sh.state["BTC/USDT"]["inv_usd"] == 0.0 and sh.status()["rows"][0]["n_fills"] == 2
    assert sh.step("SOL/USDT", 100.0, 101.0, 99.0, 0.3, book, 1000.0) is None
    assert sh.step("ETH/USDT", 100.0, 101.0, 99.0, 0.3, {"spread_bps": 10.0, "bid_depth_usd": 1e6, "ask_depth_usd": 1e6}, 1000.0)["skipped"]


# ═══════════════════════════ koşucu entegrasyonu ═══════════════════════════
def _runner(tmp_path, symbols, **over):
    ctxp = _ctx_provider(lambda s: _slow(FakeExchange.path[s][-1], age=30.0) if s == "BTC/USDT" else None)
    cfg = LR.RunnerConfig.from_dict({**SIM.default_config("mexc").to_dict(), "symbols": symbols, "symbols_mode": "fixed", **over})
    reg = LR.RunnerRegistry(output_dir=str(tmp_path), ctx=ctxp, client_factory=factory)
    return reg, reg.create(0, cfg)


def test_kosucu_veto_kaydini_kacirilanlara_yazar_ve_tavan_kor_noktasi(tmp_path):
    syms = ["BTC/USDT", "ETH/USDT"] + [f"S{i}/USDT" for i in range(6)]
    for s in syms:
        FakeExchange.path[s] = _path(dip_last=12, dip_pct=3.0)
        FakeExchange.path[s][-1] = FakeExchange.path[s][-2] * 1.002
    reg, r = _runner(tmp_path, syms, max_open=1, top_k=4, params={**SIM.default_config("mexc").params, "maker_wait_bars": 0})
    r._update_portfolio_mode = lambda *a, **k: None          # 8 parite aynı anda dipte → genişlik SAVUNMA derdi; burada tavanı ölçüyoruz
    r.run_cycle(now=1_000_000.0)
    kinds = {}
    for rec in r.missed.records:
        kinds[rec["gate"]] = kinds.get(rec["gate"], 0) + 1
    assert len(r.positions) == 1
    assert kinds.get("MAX_OPEN", 0) >= 1 or kinds.get("TOP_K", 0) >= 1, kinds
    st = r.full_state()
    assert st["missed"]["n_open"] >= 1 and st["allocator"] is not None and st["research_brief"] is not None
    assert (Path(tmp_path) / "live" / "missed_0_mexc.json").exists()


def test_kosucu_giris_kapaliyken_kor_nokta_gate_ile_kaydeder(tmp_path):
    reg, r = _runner(tmp_path, ["BTC/USDT", "ETH/USDT"])
    for s in ("BTC/USDT", "ETH/USDT"):
        FakeExchange.path[s] = _path(dip_last=12, dip_pct=3.0); FakeExchange.path[s][-1] = FakeExchange.path[s][-2] * 1.002
    r.manage_only = True
    r.run_cycle(now=1_000_000.0)
    gates = {rec["gate"] for rec in r.missed.records}
    assert not r.positions and "MANAGE_ONLY" in gates


def test_kosucu_kacirilan_kazanci_cozer_ve_loglar(tmp_path):
    reg, r = _runner(tmp_path, ["ETH/USDT"])
    r._fees = lambda: {"maker_bps": 40.0, "taker_bps": 60.0, "verified": True}     # pahalı borsa → KOMİSYON vetosu
    p = _path(dip_last=12, dip_pct=3.0); p[-1] = p[-2] * 1.002
    FakeExchange.path["ETH/USDT"] = p
    r.run_cycle(now=1_000_000.0)
    vet = [x for x in r.missed.records if x["gate"] in ("KOMİSYON", "NEGATİF_EV")]
    assert not r.positions and vet                                   # susturulan sleeve gölgeleri de eklenebilir (REJİM_SEÇİCİ)
    rec = vet[-1]
    FakeExchange.path["ETH/USDT"] = p + [rec["plan"]["target"] * 1.01] * 2
    r.run_cycle(now=1_000_030.0)
    assert rec["outcome"] == "TARGET" and rec["attribution"]["verdict"] == "KAÇIRILAN KAZANÇ"
    assert any("KAÇIRILAN KAZANÇ" in e["msg"] for e in r.events)
    assert r.full_state()["missed"]["n_missed"] >= 1 and (Path(tmp_path) / "live" / "KACIRILANLAR_0_mexc.md").exists()   # veto + susturulan gölgesi


def test_kapasite_onerisi_kanitla_uygulanir(tmp_path):
    reg, r = _runner(tmp_path, ["BTC/USDT"])
    now = 1_000_000.0
    for i in range(22):
        r.missed.on_unevaluated(f"Q{i}/USDT", "TOP_K", {"z20": -2.0}, 100.0, "LONG", 1.0, 1.6, 100.0, now + i)
    r.missed.update({f"Q{i}/USDT": _bars([100.0, 101.0], hi=[100.5, 101.8], lo=[99.6, 100.5]) for i in range(22)}, now + 50)
    old = r.cfg.top_k
    r._challenger_cycle()
    assert r.cfg.top_k == old + 2 and any("Kapasite düzeltmesi" in e["msg"] for e in r.events)
    assert GATE_PROPOSAL["TOP_K"][0] == "top_k"


def test_venue_kiyasi_paylasimli_istemci_kullanir(tmp_path):
    reg, r = _runner(tmp_path, ["BTC/USDT"])
    assert r._public_broker is not None
    b1 = reg.public_broker("binance"); b2 = reg.public_broker("binance")
    assert b1 is b2
    out = r._compare_venues("BTC/USDT", 100.0)
    assert out["symbol"] == "BTC/USDT" and isinstance(out["rows"], list)


def test_saglik_unknown_kor_nokta_golgeye_alinir(tmp_path):
    reg, r = _runner(tmp_path, ["BTC/USDT"])
    r.ctx.system_health = lambda: {"overall": "UNKNOWN"}
    p = _path(dip_last=12, dip_pct=3.0); p[-1] = p[-2] * 1.002
    FakeExchange.path["BTC/USDT"] = p
    r.run_cycle(now=1_000_000.0)
    assert not r.positions and r.last_decisions["BTC/USDT"]["result"] == "VETO: SAĞLIK UNKNOWN"
    assert any(rec["gate"] == "SAĞLIK" and rec["kind"] == "unevaluated" for rec in r.missed.records)


def test_nan_ozellikler_json_uyumlu_temizlenir(tmp_path):
    m = MissedEngine(tmp_path / "m.json")
    rec = m.on_unevaluated("X/USDT", "TOP_K", {"z20": float("nan"), "vol_ratio": np.float64("inf"), "bar_up": True}, 1.0, "LONG", 1.0, 1.6, 100.0, 1000.0)
    assert rec["features"]["z20"] is None and rec["features"]["vol_ratio"] is None and rec["features"]["bar_up"] is True
    json.dumps(m.report(), allow_nan=False)


# ═══════════════════════════ 8. tur — gece hiç işlem açılmamasının kök nedenleri ═══════════════════════════
def test_secici_mod_olculmemis_sleeveleri_kapatmaz():
    """SEÇİCİ mod kilidi: hiç işlem yok → hiç ölçüm yok → hiç sleeve yok. Ölçülmemiş açık kalmalı."""
    v = CM.evaluate(_ctx(reliable_only=True), CM.CommitteeParams(), {"sleeve_reliability": {}})
    assert v.allowed and v.trigger == "dip"
    v2 = CM.evaluate(_ctx(reliable_only=True), CM.CommitteeParams(), {"sleeve_reliability": {"dip": 0.3}})
    assert v2.trigger != "dip"


def test_sleeve_oz_oyu_hafif_katmanda_oy_esigini_gecer():
    """Hafif katman: ağır bağlam yok → roller veri vermez → oy 0 → OY vetosu (yapısal). Sleeve'in kanaati bir oydur."""
    ctx = _ctx(slow={"tier": "light", "chart": {"regime": {"label": "RANGE / YATAY"}}, "age_sec": 30.0}, qual_cell=None)
    v = CM.evaluate(ctx, CM.CommitteeParams())
    roles = {x["role"]: x for x in v.votes}
    assert "sleeve_sinyali" in roles and roles["sleeve_sinyali"]["score"] > 0
    assert v.score >= 0.25 and v.allowed, (v.result, v.vetoes)


def test_kapanmis_bar_sinyali_devam_eden_bar_titresince_kurtarir():
    """Kapanmış bar dip+yeşil; devam eden bar hafif kırmızı (fiyat kaçmadı) → sinyal yine geçerli."""
    closes = _path(dip_last=12, dip_pct=3.0); closes[-1] = closes[-2] * 1.002        # kapanmış bar yeşil
    closes = closes + [closes[-1] * 0.9995]                                           # devam eden bar hafif kırmızı
    df = _df(closes, up_last=False)
    ctx = _ctx(); ctx["df"] = df; ctx["price"] = float(df["close"].iloc[-1])
    v = CM.evaluate(ctx, CM.CommitteeParams())
    assert v.trigger == "dip" and any("kapanmış bar" in n for n in v.notes), v.result
    p2 = CM.CommitteeParams(); p2.closed_bar_fallback = False
    v2 = CM.evaluate(ctx, p2)
    assert v2.trigger is None


def test_portfoy_esikleri_ve_histerezis(tmp_path):
    br_mild = {"n": 12, "pct_above_ema20": 0.3, "pct_above_ema50": 0.5, "pct_pos_1h": 0.4, "new_lows_pct": 0.0}
    m = PM.decide_mode(br_mild, 0.3, None, 0, 0.0, "GREEN")
    assert m["mode"] == PM.SELECTIVE and m["actions"]["new_entries"]                   # ılımlı zayıf → giriş KAPANMAZ
    br_bad = {"n": 12, "pct_above_ema20": 0.1, "pct_above_ema50": 0.2, "pct_pos_1h": 0.1, "new_lows_pct": 0.4}
    assert PM.decide_mode(br_bad, 0.3, None, 0, 0.0, "GREEN")["mode"] == PM.DEFENSIVE
    assert PM.decide_mode(br_mild, 0.3, None, 2, 0.0, "GREEN")["mode"] == PM.CASH
    reg, r = _runner(tmp_path, ["BTC/USDT"])
    seq = iter([PM.RISK_ON, PM.DEFENSIVE, PM.RISK_ON, PM.DEFENSIVE, PM.DEFENSIVE, PM.DEFENSIVE, PM.CASH])
    orig = PM.decide_mode

    def fake(*a, **k):
        d = orig(*a, **k); mode = next(seq)
        return {**d, "mode": mode, "actions": {**d["actions"], "new_entries": mode in (PM.RISK_ON, PM.SELECTIVE), "flatten": mode == PM.CASH}}
    PM.decide_mode = fake
    try:
        modes = []
        for i in range(7):
            r._update_portfolio_mode({}, 1_000_000.0 + 30 * i)
            modes.append(r.portfolio["mode"])
    finally:
        PM.decide_mode = orig
    assert modes == [PM.RISK_ON, PM.RISK_ON, PM.RISK_ON, PM.RISK_ON, PM.RISK_ON, PM.DEFENSIVE, PM.CASH]


def test_kazanan_ve_stop_profilleri_ve_ogrenilmis_agirlik(tmp_path):
    m = MissedEngine(tmp_path / "m.json")
    now = 1000.0
    for i in range(6):
        m.on_unevaluated(f"D{i}/USDT", "TOP_K", {"z20": -2.0, "kind": "dip", "vol_ratio": 1.7}, 100.0, "LONG", 1.0, 1.6, 3600.0, now + i)
    for i in range(10):
        m.on_unevaluated(f"B{i}/USDT", "TOP_K", {"z20": 2.0, "kind": "breakout", "vol_ratio": 0.5, "trend_up": True}, 100.0, "LONG", 1.0, 1.6, 3600.0, now + 20 + i)
    # bar bar ilerle (motor her döngüde SON barı görür): önce %0,7 aleyhe, sonra hedef / hedefin %56'sına gidip stop
    steps = [({f"D{i}/USDT": _bars([100.0, 99.3], hi=[100.2, 99.6], lo=[99.5, 99.3]) for i in range(6)}, now + 600),
             ({f"D{i}/USDT": _bars([99.3, 101.0], hi=[99.6, 101.7], lo=[99.3, 100.4]) for i in range(6)}, now + 1800)]
    stepsB = [({f"B{i}/USDT": _bars([100.0, 100.9], hi=[100.3, 100.9], lo=[99.6, 100.2]) for i in range(10)}, now + 600),
              ({f"B{i}/USDT": _bars([100.9, 98.9], hi=[100.9, 99.2], lo=[100.2, 98.8]) for i in range(10)}, now + 1800)]
    for (b, t), (bb, tt) in zip(steps, stepsB):
        m.update({**b, **bb}, t)
    rep = m.report(now + 1801)
    wp, sp = rep["winner_profile"], rep["stop_profile"]
    assert wp["n"] == 6 and wp["median_mae_over_stop"] == pytest.approx(0.7, abs=0.05) and wp["near_stop_share"] == 1.0 and "dip" in wp["by_setup"]
    assert sp["n"] == 10 and sp["median_mfe_over_target"] == pytest.approx(0.56, abs=0.05) and sp["near_target_share"] == 1.0 and "hacim yok" in sp["common_warnings"]
    assert "kurtarır" in sp["lesson"] and "medyan" in wp["lesson"]
    iw = rep["interest_weights"]
    assert iw["breakout"] == 0.5 and iw["breakout_n"] == 10 and iw["dip"] == 1.0 and iw["dip_n"] == 6            # dip n<8 → 1,0
    a = [r for r in m.records if r["symbol"] == "D0/USDT"][0]["attribution"]
    assert "hedefe" in a["path"] and "neredeyse stop" in a["path"]


# ═══════════════════════════ veto incelemesi ═══════════════════════════
from agi_trader.strategies import veto_review as VRW  # noqa: E402


def test_veto_incelemesi_hafif_katman_indikator_konsensusu_ve_asma():
    f = {"trend_up": True, "rsi": 35.0, "dist_vwap_pct": 0.2, "z": -1.4, "adx": 25.0, "ema_cross_up": True, "obi": 0.7, "vol_ratio": 1.5, "bar_up": True,
         "swept_low": True, "trend_score": 0.8}
    ind = VRW.light_indicator_consensus(f)
    assert ind["al"] >= 6 and ind["sat"] == 0 and ind["bias"] == "YUKARI" and len(ind["rows"]) == 8
    r = VRW.review("LONG", ["OY +0.10 < eşik 0.25"], {"tier": "light"}, f, {"data_ok": True, "score": 0.1, "n_items": 3}, "dip", ["dip"], "mean_reversion", "RANGE / YATAY")
    assert r["decision"] == "AÇ" and r["size_mult"] == 0.6 and r["formations"]["structural"] == ["likidite süpürme"] and "AÇ ×0,6" in r["summary_tr"]
    r2 = VRW.review("LONG", ["NEGATİF NET EV (%-0.3) — NO TRADE"], {"tier": "light"}, f, None, "dip", ["dip"], "mean_reversion", None)
    assert r2["decision"] == "VETO" and "sert veto aşılmaz" in r2["why"][0]
    r3 = VRW.review("LONG", ["OY +0.10 < eşik 0.25"], {"tier": "light"}, f, {"data_ok": True, "score": -0.6, "severe_risk": True, "n_items": 5}, "dip", ["dip"], "mean_reversion", None)
    assert r3["decision"] == "VETO" and any("haber" in w for w in r3["why"])
    f_bad = {**f, "trend_up": False, "rsi": 65.0, "dist_vwap_pct": -0.3, "z": 1.2, "adx": 25.0, "ema_cross_up": False, "obi": 0.3}
    r4 = VRW.review("LONG", ["GÜVEN 0.30 < 0.35"], {"tier": "light"}, f_bad, None, "dip", ["dip"], "mean_reversion", None)
    assert r4["decision"] == "VETO" and any("indikatör" in w for w in r4["why"])
    r5 = VRW.review("LONG", ["OY -0.20 < eşik 0.25"], {"tier": "light"}, f, None, "dip", ["dip"], "mean_reversion", None, vote_score=-0.2)
    assert r5["decision"] == "VETO" and any("karşı oy" in w for w in r5["why"])       # roller karşıysa inceleme aşmaz


def test_veto_incelemesi_agir_baglam_aile_sayimi_ve_formasyon_karsi():
    slow = {"indicators": {"available": True, "family": {"al": 12, "sat": 30, "notr": 3}, "bias": "AŞAĞI"},
            "patterns": {"consensus": {"bias": "AŞAĞI", "score": -0.4, "n": 2, "long": 0, "short": 2}}}
    r = VRW.review("LONG", ["OY +0.05 < eşik 0.25"], slow, {"trend_up": True}, None, "dip", ["dip"], "mean_reversion", "RANGE / YATAY")
    assert r["decision"] == "VETO" and r["indicators"]["source"].startswith("ağır") and r["formations"]["against"]
    assert "indikatör yetersiz (12 al / 30 sat)" in r["why"] and "formasyon karşı yönde" in r["why"]


def test_komite_yumusak_vetoyu_incelemeyle_asar_serti_asmaz():
    ctx = _ctx(slow={"tier": "light", "chart": {"regime": {"label": "RANGE / YATAY"}}, "age_sec": 30.0}, qual_cell=None)
    p = CM.CommitteeParams(); p.theta = 0.9                                     # oy eşiği ulaşılamaz → OY vetosu
    v = CM.evaluate(ctx, p)
    d = v.to_dict()
    assert d["veto_review"] is not None and "İNCELEME" in " ".join(v.notes)
    if d["veto_review"]["decision"] == "AÇ":
        assert v.allowed and v.size_mult <= 0.6
    else:
        assert not v.allowed
    ctx2 = _ctx(fees={"maker_bps": 60.0, "taker_bps": 120.0, "verified": True})
    v2 = CM.evaluate(ctx2, CM.CommitteeParams())
    assert not v2.allowed and v2.to_dict()["veto_review"]["decision"] == "VETO"


def test_p_win_onculu_yalniz_qualified_hucreden(tmp_path):
    reg, r = _runner(tmp_path, ["BTC/USDT"])
    seen = {}
    orig = r.lessons.p_win

    def spy(prior=0.5, prior_n=10.0):
        seen["prior"] = prior
        return orig(prior=prior, prior_n=prior_n)
    r.lessons.p_win = spy
    r.ctx.qual_cell = lambda *a: {"status": "NOT_QUALIFIED", "p_model_live": 0.3}
    p = _path(dip_last=12, dip_pct=3.0); p[-1] = p[-2] * 1.002
    FakeExchange.path["BTC/USDT"] = p
    r.run_cycle(now=1_000_000.0)
    assert seen.get("prior") == 0.5
    r.ctx.qual_cell = lambda *a: {"status": "QUALIFIED", "p_model_live": 0.62}
    r.positions.clear(); r.pending.clear()
    r.run_cycle(now=1_000_030.0)
    assert seen.get("prior") == 0.62
    assert LR.VENUE_COMPARE is False and r.venue_compare == {}


# ═══════════════════════════ 9. tur — kazanç kaybolmasın / stop öngörüsü / boyut ═══════════════════════════
from agi_trader.strategies import exit_engine as XE9  # noqa: E402


def _trk(mode=XE9.FIXED_TARGET, cost=0.2, atr=0.3, stop_pct=1.0):
    return XE9.PositionTrack("LONG", 100.0, 100.0 * (1 - stop_pct / 100), 101.6, 1000.0, mode, stop_pct, cost, atr)


def test_basabas_kilidi_kazanci_kayba_dondurmez():
    p = XE9.ExitParams(min_hold_sec=0, time_stop_sec=3600)
    t = _trk()
    assert XE9.decide_exit(t, 100.35, 100.35, 100.0, p, now=1100.0) is None and not t.be_locked   # net %0,15 < 1,5×maliyet(0,2)=0,3 → henüz kilit yok
    assert XE9.decide_exit(t, 100.55, 100.55, 100.3, p, now=1200.0) is None and t.be_locked and t.hard_stop == pytest.approx(100.2)
    d = XE9.decide_exit(t, 100.15, 100.3, 100.1, p, now=1300.0)                              # başabaşın altına döndü → BE_LOCK çıkışı
    assert d and d["reason"] == "BE_LOCK" and d["net_pct"] >= -0.05


def test_zaman_kilidi_ve_erken_iptal():
    p = XE9.ExitParams(min_hold_sec=0, time_stop_sec=1000)
    t = _trk()
    assert XE9.decide_exit(t, 100.25, 100.25, 100.0, p, now=1000.0 + 650) is None and t.be_locked   # ufkun %65'i, kârda → kilit
    assert t.hard_stop == pytest.approx(100.2)
    t2 = _trk(stop_pct=1.0)
    d = XE9.decide_exit(t2, 99.35, 100.0, 99.35, p, now=1000.0 + 200)                   # ilk %20'de MAE %0,65 ≥ 0,6×stop, tepe ≤ 0
    assert d and d["reason"] == "EARLY_ABORT" and d["mae_pct"] == pytest.approx(0.65)
    t3 = _trk(stop_pct=1.0)
    assert XE9.decide_exit(t3, 99.35, 100.0, 99.35, p, now=1000.0 + 700) is None           # pencere dışı → erken iptal yok
    t4 = _trk(stop_pct=1.0)
    XE9.decide_exit(t4, 100.5, 100.5, 100.0, p, now=1010.0)                               # önce kâra geçti
    assert XE9.decide_exit(t4, 99.35, 100.0, 99.35, p, now=1020.0) is None or t4.be_locked  # tepe > 0 → erken iptal devre dışı


def test_komite_boyut_tabani_ve_stop_risk():
    v = CM.evaluate(_ctx(), CM.CommitteeParams())
    assert v.allowed and v.size_mult >= 0.5 and v.to_dict()["stop_risk"]["n"] <= 1
    p = CM.CommitteeParams(); p.stop_risk_veto_warnings = 2
    ctx = _ctx(); ctx["book"] = {"spread_bps": 12.0, "bid_depth_usd": 1e6, "ask_depth_usd": 1e6}   # geniş spread uyaranı
    ctx["news"] = {"data_ok": True, "severe_risk": True, "score": -0.3, "n_items": 2}                # haber riski uyaranı
    v2 = CM.evaluate(ctx, p)
    assert not v2.allowed and any("STOP RİSKİ" in x or "HABER RİSKİ" in x for x in v2.vetoes)
    assert SF.SLEEVE_TIME_STOP_MIN["dip_moderate"] == 90 and SF.SLEEVE_TIME_STOP_MIN["dip"] == 90


def test_kosucu_be_kilidini_pozisyona_yansitir(tmp_path):
    reg, r = _runner(tmp_path, ["BTC/USDT"], params={**SIM.default_config("mexc").params, "maker_wait_bars": 0})
    p = _path(dip_last=12, dip_pct=3.0); p[-1] = p[-2] * 1.002
    FakeExchange.path["BTC/USDT"] = p
    r.run_cycle(now=1_000_000.0)
    assert r.positions
    pos = r.positions["BTC/USDT"]
    up = pos.entry * (1 + 3.5 * pos.cost_pct_roundtrip / 100.0)      # net = 2,5×maliyet ≥ 1,5×maliyet → kilit
    FakeExchange.path["BTC/USDT"] = p + [up] * 2
    r.run_cycle(now=1_000_000.0 + 20 * 60)
    assert pos.be_locked and pos.hard_stop >= pos.entry
    FakeExchange.path["BTC/USDT"] = p + [up] * 2 + [pos.hard_stop * 0.9999] * 2      # kilitli stop'un hemen altına iner (boşluksuz)
    r.run_cycle(now=1_000_000.0 + 21 * 60)
    assert not r.positions and r.trades[-1]["reason"] == "BE_LOCK" and r.trades[-1]["net_pnl"] >= -0.05 * r.trades[-1]["notional"] / 100


def test_simulator_gunluk_tavan_ve_emir_tavani_senkron(tmp_path):
    cfg = SIM.default_config("mexc")
    # 200/200 baz çizgisiydi; şimdi pay ile ölçekleniyor (SCALP_BASELINE → scalp_limits)
    lim = SIM.scalp_limits()
    assert cfg.max_trades_per_day == lim["max_trades_per_day"] and cfg.max_order_usdt == lim["max_order_usdt"]
    assert SIM.SCALP_BASELINE["max_trades_per_day"] == 200 and SIM.SCALP_BASELINE["max_order_usdt"] == 200.0
    reg = LR.RunnerRegistry(output_dir=str(tmp_path), ctx=_ctx_provider(lambda s: None), client_factory=factory)
    old = SIM.default_config("mexc"); old.max_trades_per_day = 30
    reg.create(0, old)
    out = SIM.ensure_simulator(reg, venue="mexc", start=False)
    assert "max_trades_per_day" in out["synced"] and SIM.get_simulator(reg).cfg.max_trades_per_day == SIM.scalp_limits()["max_trades_per_day"]
