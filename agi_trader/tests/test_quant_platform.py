# -*- coding: utf-8 -*-
"""6. tur — quant platform katmanları: MarketStateStore/RateLimit, sleeve'ler + EV yarışması,
giriş optimizasyonu, çıkış motoru (NET yarı-tepe, chandelier, edge-decay, zaman), portföy modu,
ücret adaptörü + venue router, lifecycle, challenger, haber taksonomi/dedup, değişmezler,
no-LLM hot path, no-duplicate fetch, replay determinizmi, kaynak/döngü süresi."""
from __future__ import annotations

import ast
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agi_trader.data.market_state import MarketStateStore, RateLimitCoordinator, RateLimited, CircuitOpen  # noqa: E402
from agi_trader.strategies import sleeves_fast as SF  # noqa: E402
from agi_trader.strategies import committee as CM  # noqa: E402
from agi_trader.strategies import exit_engine as XE  # noqa: E402
from agi_trader.strategies import portfolio_mode as PM  # noqa: E402
from agi_trader.strategies import entry_optimizer as EO  # noqa: E402
from agi_trader.strategies.lifecycle import Lifecycle  # noqa: E402
from agi_trader.learn.challenger import Challenger  # noqa: E402
from agi_trader.execution import fee_adapter as FA  # noqa: E402
from agi_trader.execution import venue_router as VR  # noqa: E402
from agi_trader.sentiment import news_scanner as NS  # noqa: E402
from agi_trader.auto import live_runner as LR  # noqa: E402
from agi_trader.auto import simulator as SIM  # noqa: E402
from agi_trader.server import secure_keys as V  # noqa: E402

from test_live_trading import FakeExchange, factory, _path  # noqa: E402
from test_committee import _slow, _ctx, _ctx_provider  # noqa: E402

MASTER = "test-master-key-" + "q" * 40


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv(V.MASTER_ENV, MASTER)
    FakeExchange.path = {"BTC/USDT": _path(), "ETH/USDT": _path(seed=5)}
    FakeExchange.orders = []
    FA.clear_cache()
    yield


def _df(closes, up_last=True, vol=None):
    c = np.array(closes, dtype=float)
    if up_last:
        c[-1] = c[-2] * 1.002
    idx = pd.date_range("2026-01-01", periods=len(c), freq="min")
    v = np.full(len(c), 100.0) if vol is None else np.array(vol, dtype=float)
    return pd.DataFrame({"open": c, "high": c * 1.001, "low": c * 0.999, "close": c, "volume": v}, index=idx)


# ═══════════════════════════ MarketStateStore / RateLimit ═══════════════════════════
def test_store_ayni_veriyi_bir_kez_ceker():
    calls = {"n": 0}

    def fetch(ex, s, tf, lim):
        calls["n"] += 1
        return _df(_path())
    st = MarketStateStore(fetch_ohlcv=fetch)
    a = st.get_ohlcv("mexc", "BTC/USDT", "1m", 150, now=1000.0)
    b = st.get_ohlcv("mexc", "BTC/USDT", "1m", 150, now=1010.0)
    assert calls["n"] == 1 and a is b and st.stats["hits"] == 1
    st.get_ohlcv("mexc", "BTC/USDT", "1m", 150, now=1100.0)        # TTL (20 sn) doldu
    assert calls["n"] == 2
    f1 = st.cheap_features("mexc", "BTC/USDT"); f2 = st.cheap_features("mexc", "BTC/USDT")
    assert f1 is f2 and st.stats["feature_recomputes"] == 1 and "z20" in f1


def test_store_tazelik_stale():
    old = _df(_path()); old.index = old.index - pd.Timedelta(hours=3)
    st = MarketStateStore(fetch_ohlcv=lambda *a: old)
    st.get_ohlcv("mexc", "X/USDT", "1m", 150, now=time.time())
    assert st.freshness("mexc", "X/USDT", "1m")["state"] == "STALE"


def test_store_hata_durumunda_eski_veriyi_dondurur():
    state = {"fail": False}

    def fetch(ex, s, tf, lim):
        if state["fail"]:
            raise RuntimeError("ağ")
        return _df(_path())
    st = MarketStateStore(fetch_ohlcv=fetch)
    st.get_ohlcv("mexc", "BTC/USDT", "1m", 150, now=1000.0)
    state["fail"] = True
    df = st.get_ohlcv("mexc", "BTC/USDT", "1m", 150, now=2000.0)
    assert len(df) and st.stats["errors"] == 1


def test_rate_limit_ve_devre_kesici():
    rl = RateLimitCoordinator(rps=2.0, burst=2, max_wait_sec=0.01, breaker_errors=2, breaker_open_sec=30)
    rl.acquire("x", now=100.0); rl.acquire("x", now=100.0)
    with pytest.raises(RateLimited):
        rl.acquire("x", now=100.0)
    rl.report("x", False, now=100.0); rl.report("x", False, now=100.0)
    assert rl.state("x", now=101.0) == "OPEN"
    with pytest.raises(CircuitOpen):
        rl.acquire("x", now=101.0)
    assert rl.state("x", now=140.0) == "CLOSED"


# ═══════════════════════════ sleeve'ler / rejim seçici ═══════════════════════════
def _f(**k):
    d = dict(ok=True, z=0.0, rsi=50.0, bar_up=True, trend_up=True, dist_ema_pct=0.0, atr_pct=0.3,
             breakout_up=False, vol_ratio=1.0, ema_cross_up=False, sigma_bar_pct=0.1, ema_slope_pct=0.0,
             bb_prev_pctile=50, swept_low=False, lower_wick_ratio=0.0, range_ok=False, range_pos=0.5,
             range_low=95.0, range_high=105.0, range_width_pct=10.0, vwap=100.0, dist_vwap_pct=0.0,
             prior_swing_low=97.0, move_4h_pct=0.0, rs_rank=None)
    d.update(k); return d


def test_rejim_secici_zit_sleeveleri_kapatir():
    assert "range_edge" not in SF.allowed_sleeves("TREND YUKARI") and "pullback" not in SF.allowed_sleeves("RANGE / YATAY")
    assert "dip" in SF.allowed_sleeves("TREND YUKARI")            # trend-içi derin dip kapatılmaz (7. tur düzeltmesi)
    # 13. turda video kaynaklı kurulumlar da rejim listelerine katıldı; ÇEKİRDEK küme korunmalı ve
    # zıt sistemler (range_edge/pullback) kapalı kalmalı — liste artık kapalı bir eşitlik DEĞİL.
    vol = set(SF.allowed_sleeves("VOLATİL"))
    assert {"dip", "sweep_reversal", "catalyst", "failed_breakdown"} <= vol
    assert not ({"range_edge", "pullback", "vwap_reversion", "vwap_continuation"} & vol)
    from agi_trader.strategies import sleeves_video as _SV
    assert vol - {"dip", "sweep_reversal", "catalyst", "failed_breakdown"} <= set(_SV.ALL_VIDEO_SLEEVES)


def test_yeni_sleeveler_ateslenir():
    p = CM.CommitteeParams()
    a = SF.fire_sleeves(_f(breakout_up=True, bb_prev_pctile=10, vol_ratio=1.6, rsi=60), ["squeeze_breakout"], None, p)
    assert a and a[0]["kind"] == "squeeze_breakout" and a[0]["exit_mode"] == XE.DYNAMIC_PEAK
    b = SF.fire_sleeves(_f(swept_low=True, lower_wick_ratio=0.5, vol_ratio=1.3), ["sweep_reversal"], None, p)
    assert b and b[0]["stop_hint"] < 97.0
    c = SF.fire_sleeves(_f(range_ok=True, range_pos=0.1, rsi=40), ["range_edge"], None, p)
    assert c and c[0]["target_hint"] == 100.0
    d = SF.fire_sleeves(_f(dist_vwap_pct=-0.5, rsi=40, trend_up=False), ["vwap_reversion"], None, p)
    assert d and d[0]["kind"] == "vwap_reversion"
    e = SF.fire_sleeves(_f(dist_vwap_pct=0.05, rsi=55), ["vwap_continuation"], None, p)
    assert e and e[0]["kind"] == "vwap_continuation"
    g = SF.fire_sleeves(_f(rs_rank=0.9, ema_cross_up=True, rsi=60), ["rs_momentum"], None, p)
    assert g and g[0]["kind"] == "rs_momentum"
    h = SF.fire_sleeves(_f(move_4h_pct=-1.0, vol_ratio=0.8, trend_up=False),
                        ["news_overreaction"], {"data_ok": True, "score": -0.7, "severe_risk": False}, p)
    assert h and h[0]["size"] == 0.5
    assert SF.fire_sleeves(_f(range_ok=True, range_pos=0.5, rsi=40), ["range_edge"], None, p) == []   # range ortası → yok


def test_goreli_guc_siralamasi():
    cross = {f"S{i}/USDT": {"ret_1h_pct": i * 0.1, "vol_ratio": 1.0} for i in range(10)}
    r = SF.relative_strength_ranks(cross)
    assert r["S9/USDT"] == 1.0 and r["S0/USDT"] == 0.0


def test_ev_yarismasi_en_yuksek_evi_secer_ve_giris_optimizasyonu_uretir():
    ctx = _ctx()
    v = CM.evaluate(ctx, CM.CommitteeParams())
    assert v.allowed, v.result
    d = v.to_dict()
    assert d["competition"] and d["exit_mode"] in XE.MODES and d["entry"]["optimal"]
    assert d["entry"]["max_chase"] is not None and d["valid_until"] > time.time()
    assert "EV" in v.result
    best = max(d["competition"], key=lambda c: c["ev_pct"] if c["ev_pct"] is not None else -99)
    assert best["kind"] == v.trigger


def test_negatif_ev_islem_yok():
    ctx = _ctx(fees={"maker_bps": 60.0, "taker_bps": 120.0, "verified": True})
    v = CM.evaluate(ctx, CM.CommitteeParams())
    assert not v.allowed and any("NEGATİF NET EV" in x or "KOMİSYON" in x for x in v.vetoes)


def test_ucret_dogrulanmamissa_guven_kirpilir():
    v1 = CM.evaluate(_ctx(fees={"maker_bps": 0.0, "taker_bps": 5.0, "verified": True}), CM.CommitteeParams())
    v2 = CM.evaluate(_ctx(fees={"maker_bps": 0.0, "taker_bps": 5.0, "verified": False}), CM.CommitteeParams())
    assert v2.confidence < v1.confidence


def test_tier3_haber_katalizor_tetiklemez():
    news = {"data_ok": True, "confirmed": True, "score": 0.9, "severe_risk": False, "vol_ratio": 2.0,
            "tier12_items": 0, "n_items": 3, "bull": 3, "bear": 0}
    ctx = _ctx(dip=False, news=news)
    v = CM.evaluate(ctx, CM.CommitteeParams())
    assert v.trigger != "catalyst"


def test_lifecycle_kapisi_sleeve_kapatir():
    lc = Lifecycle()
    lc.reg["dip"]["stage"] = "RETIRED"
    v = CM.evaluate(_ctx(lifecycle=lc, mode="paper"), CM.CommitteeParams())
    assert v.trigger != "dip"


# ═══════════════════════════ giriş optimizasyonu ═══════════════════════════
def test_giris_optimizasyonu_maker_ucuzsa_bid_secer_ve_chase_sinir():
    r = EO.optimize_entry("LONG", 100.0, {"bid": 99.98, "ask": 100.02, "spread_bps": 4.0}, {"vwap": 99.9, "ema_fast": 99.95},
                          0.3, 1.0, 1.6, 0.0, 5.0, 0.55, 0.6)
    assert r["optimal"] and r["optimal"]["order_type"] == "maker" and r["max_chase"] > 100.02
    assert all(c["price"] <= 100.02 * 1.002 for c in r["candidates"])
    r2 = EO.optimize_entry("LONG", 100.0, {"bid": 99.98, "ask": 100.02, "spread_bps": 4.0}, {}, 0.3, 1.0, 1.6, 0.0, 5.0, 0.55, 0.05)
    assert r2["optimal"]["order_type"] == "taker"        # maker dolum çok düşükse taker


# ═══════════════════════════ çıkış motoru ═══════════════════════════
def _track(mode=XE.PARTIAL_AND_RUN, cost=0.2, atr=0.3):
    return XE.PositionTrack("LONG", 100.0, 99.0, 101.6, 1000.0, mode, 1.0, cost, atr)


def test_hard_stop_her_moda_ustun():
    p = XE.ExitParams(min_hold_sec=900)
    for m in XE.MODES:
        t = _track(m)
        d = XE.decide_exit(t, 98.9, 99.0, 98.8, p, now=1010.0)
        assert d and d["reason"] == "STOP"


def test_yari_tepe_net_uzerinden_silahlanir_ve_cikar():
    p = XE.ExitParams(min_hold_sec=0, retain_fraction=0.5)
    t = _track(XE.DYNAMIC_PEAK, cost=0.2, atr=0.3)
    assert XE.decide_exit(t, 100.4, 100.4, 100.0, p, now=1001.0) is None and not t.armed   # net %0,2 < eşik
    assert XE.decide_exit(t, 102.0, 102.0, 101.0, p, now=1002.0) is None and t.armed        # net tepe %1,8
    lvl = t.giveback_level(p)
    assert 100.0 < lvl < 102.0
    d = XE.decide_exit(t, lvl - 0.01, lvl, lvl - 0.02, p, now=1003.0)
    assert d and d["reason"] == "GIVEBACK" and d["peak_net_pct"] == pytest.approx(1.8)


def test_chandelier_trailing_ve_zaman_stopu():
    p = XE.ExitParams(min_hold_sec=0, chandelier_k=2.0, time_stop_sec=600)
    t = _track(XE.DYNAMIC_PEAK, atr=0.5)
    XE.decide_exit(t, 103.0, 103.0, 102.5, p, now=1001.0)
    assert t.armed and t.trail_stop is not None and t.trail_stop == pytest.approx(103.0 - 2.0 * 0.5, rel=1e-3)
    d = XE.decide_exit(t, 101.9, 102.0, 101.8, p, now=1002.0)
    assert d and d["reason"] in ("TRAIL", "GIVEBACK")
    t2 = _track(XE.FIXED_TARGET)
    d2 = XE.decide_exit(t2, 100.1, 100.2, 100.0, p, now=1000.0 + 601)
    assert d2 and d2["reason"] == "TIME_STOP"


def test_edge_decay_ve_model_cikisi():
    p = XE.ExitParams(min_hold_sec=100)
    t = _track(XE.DYNAMIC_PEAK)
    d = XE.decide_exit(t, 100.1, 100.1, 100.0, p, now=1250.0, current_ev_pct=-0.2)
    assert d and d["reason"] == "EDGE_DECAY"
    t2 = _track(XE.DYNAMIC_PEAK, atr=0.3)
    XE.decide_exit(t2, 102.0, 102.0, 101.0, p, now=1200.0)
    d2 = XE.decide_exit(t2, 101.9, 102.0, 101.8, p, now=1300.0, cont_prob=0.2)
    assert d2 and d2["reason"] in ("MODEL_EXIT", "GIVEBACK", "TRAIL")


def test_peak_capture_ve_cikis_kalitesi():
    assert XE.peak_capture_ratio(0.9, 1.8) == pytest.approx(0.5)
    q = XE.exit_quality({"net_pct_realized": 0.5, "peak_net_pct": 2.0}, None)
    assert "GEÇ" in q["verdict"]
    q2 = XE.exit_quality({"net_pct_realized": 0.8, "peak_net_pct": 0.9}, 2.1)
    assert "ERKEN" in q2["verdict"]


# ═══════════════════════════ portföy modu ═══════════════════════════
def test_portfoy_modu_genislik_ve_korelasyon():
    good = {f"S{i}": {"above_ema20": True, "above_ema50": True, "ret_1h_pct": 0.3, "dist_hi20_pct": 1, "dist_lo20_pct": 1} for i in range(12)}
    assert PM.decide_mode(PM.breadth(good), 0.3, "RANGE / YATAY", 0, 0.0, "GREEN")["mode"] == PM.RISK_ON
    bad = {f"S{i}": {"above_ema20": False, "above_ema50": False, "ret_1h_pct": -0.5, "dist_hi20_pct": 5, "dist_lo20_pct": 0.01} for i in range(12)}
    m = PM.decide_mode(PM.breadth(bad), 0.85, "TREND AŞAĞI", 0, 0.0, "GREEN")
    assert m["mode"] in (PM.DEFENSIVE, PM.CASH) and not m["actions"]["new_entries"]
    assert PM.decide_mode(PM.breadth(good), 0.3, None, 2, 0.0, "GREEN")["mode"] == PM.CASH
    assert PM.decide_mode(PM.breadth(good), 0.3, None, 0, 12.0, "GREEN", max_dd_pct=15)["mode"] != PM.RISK_ON
    rets = pd.DataFrame({"a": np.arange(50.0), "b": np.arange(50.0) * 2, "c": np.arange(50.0) * 3})
    assert PM.correlation_shock(rets) == pytest.approx(1.0)


# ═══════════════════════════ ücret adaptörü / venue router ═══════════════════════════
class FeeEx:
    fail = False

    def fetch_trading_fee(self, symbol):
        if FeeEx.fail:
            raise RuntimeError("api")
        return {"maker": 0.0002, "taker": 0.0006, "info": {"feeCurrency": "USDT"}}


def test_gercek_ucret_ve_yedek():
    FeeEx.fail = False
    f = FA.fetch_account_fee(FeeEx(), "binance", "BTC/USDT", now=1000.0)
    assert f["verified"] and f["maker_bps"] == 2.0 and f["taker_bps"] == 6.0
    FeeEx.fail = True
    f2 = FA.fetch_account_fee(FeeEx(), "binance", "BTC/USDT", now=1000.0 + 7 * 3600)   # TTL doldu, API hata → bayat+doğrulanmış
    assert f2["stale"] and f2["verified"]
    FA.clear_cache()
    f3 = FA.fetch_account_fee(FeeEx(), "binance", "ETH/USDT", now=1000.0)
    assert not f3["verified"] and f3["source"] == "static_table"
    f4 = FA.fetch_account_fee(None, "mexc", "BTC/USDT")
    assert f4["maker_bps"] == 0.0 and not f4["verified"]


def test_venue_router_tum_dahil_maliyet():
    books = {"mexc": {"spread_bps": 2.0, "bid_depth_usd": 50000, "ask_depth_usd": 50000},
             "binance": {"spread_bps": 1.0, "bid_depth_usd": 500000, "ask_depth_usd": 500000},
             "kraken": None}
    out = VR.compare("BTC/USDT", ["mexc", "binance", "kraken"], lambda v, s: books[v], VR.static_fee, 150.0, 0.5, now=1000.0)
    assert out["best"] in ("mexc", "binance") and any(r["venue"] == "kraken" and not r["available"] for r in out["rows"])
    assert all("total_bps" in r for r in out["rows"] if r.get("available"))


# ═══════════════════════════ lifecycle / challenger ═══════════════════════════
def test_lifecycle_kanitsiz_terfi_yok(tmp_path):
    lc = Lifecycle(tmp_path / "lc.json")
    assert lc.can_trade("dip", "paper") and not lc.can_trade("dip", "live")
    r = lc.promote("dip", "LIMITED_LIVE")
    assert not r["ok"]
    lc.record_evidence("dip", {"oos_expectancy": 0.2, "ci_lower": 0.05, "expectancy_cost_x2": 0.1,
                               "subperiod_consistent": True, "dsr": 0.3, "pbo": 0.2, "n_trades": 40})
    assert lc.promote("dip", "LIMITED_LIVE")["ok"] and lc.can_trade("dip", "live")


def test_challenger_gölgede_kanit_toplar_ve_terfi_eder(tmp_path):
    ch = Challenger(tmp_path / "ch.json")
    assert ch.propose({"theta": 0.2})
    for i in range(35):
        ch.record(f"S{i}/USDT", False, True, {"entry": 100, "target": 102, "stop": 99}, now=1000.0 + i)
    ch.resolve({f"S{i}/USDT@{1000 + i}": ("TARGET" if i % 10 != 0 else "STOP") for i in range(35)})
    ev = ch.evaluate()
    assert ev["n_challenger_only"] == 35 and ev["promote"]
    won = ch.conclude(True)
    assert won == {"theta": 0.2} and not ch.params


# ═══════════════════════════ haber taksonomi / dedup ═══════════════════════════
def test_haber_taksonomi_ve_kaynaklar_arasi_dedup():
    assert NS.classify_event("Binance will list Solana perpetual")["event_type"] == "LISTING"
    assert NS.classify_event("Protocol exploit drains $5M")["event_type"] in ("HACK", "EXPLOIT")
    now = 1_000_000.0
    items = [{"source": "coindesk", "weight": 1.0, "title": "Solana ETF approved by SEC — SOL surges", "ts": now - 60},
             {"source": "reddit", "weight": 0.4, "title": "Solana ETF approved by SEC: SOL surges!", "ts": now - 30},
             {"source": "reddit", "weight": 0.4, "title": "Bonk to the moon lol", "ts": now - 30}]
    d = NS.scan(["SOL/USDT", "BONK/USDT"], items_override=items, with_social=False, now=now)
    sol = d["symbols"]["SOL/USDT"]
    assert sol["n_items"] == 1 and sol["tier12_items"] == 1 and sol["top_event"] in ("ETF", "REGULATORY")
    assert d["sources"]["n_unique"] == 2
    assert d["symbols"]["BONK/USDT"]["tier12_items"] == 0


# ═══════════════════════════ değişmezler / mimari kuralları ═══════════════════════════
def test_degismezler_long_stop_giris_alti_hedef_maliyet_ustu():
    for seed in range(5):
        ctx = _ctx()
        v = CM.evaluate(ctx, CM.CommitteeParams())
        if v.plan:
            assert v.plan["stop"] < v.plan["entry"] < v.plan["target"]
            if v.allowed:
                assert v.ticket["expected_profit_pct"] > 0 and v.ticket["fee_usdt"] >= 0


def test_hot_path_llm_cagirmaz():
    hot = ["agi_trader/auto/live_runner.py", "agi_trader/strategies/committee.py", "agi_trader/strategies/roles.py",
           "agi_trader/strategies/sleeves_fast.py", "agi_trader/strategies/exit_engine.py", "agi_trader/execution/broker.py",
           "agi_trader/sentiment/news_scanner.py", "agi_trader/data/market_state.py"]
    bad = []
    for f in hot:
        src = (ROOT / f).read_text(encoding="utf-8")
        tree = ast.parse(src)
        for n in ast.walk(tree):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in n.names] + ([n.module] if isinstance(n, ast.ImportFrom) and n.module else [])
                if any(x and re.search(r"openai|anthropic|deepseek|openrouter|llm", x, re.I) for x in names):
                    bad.append(f"{f}: {names}")
        if re.search(r"api\.openai\.com|api\.anthropic\.com|openrouter\.ai", src):
            bad.append(f"{f}: LLM URL")
    assert not bad, bad


def test_kosucu_ayni_dongude_ayni_veriyi_iki_kez_cekmez(tmp_path):
    calls = {"n": 0}
    orig = FakeExchange.fetch_ohlcv

    def counting(self, symbol, timeframe, limit=150):
        calls["n"] += 1
        return orig(self, symbol, timeframe, limit)
    FakeExchange.fetch_ohlcv = counting
    try:
        ctxp = _ctx_provider(lambda s: _slow(FakeExchange.path[s][-1], age=30.0))
        cfg = LR.RunnerConfig.from_dict({**SIM.default_config("mexc").to_dict(), "symbols": ["BTC/USDT", "ETH/USDT"], "symbols_mode": "fixed"})
        reg = LR.RunnerRegistry(output_dir=str(tmp_path), ctx=ctxp, client_factory=factory)
        r = reg.create(0, cfg)
        r.run_cycle(now=1_000_000.0)
        n1 = calls["n"]
        r.run_cycle(now=1_000_010.0)                   # 10 sn sonra: 1m TTL 20 sn → yeniden fetch YOK
        assert calls["n"] == n1
        assert n1 <= 2 + 2 * 2                          # parite başına 1m (+ hafif bağlam 4h/1h en fazla)
    finally:
        FakeExchange.fetch_ohlcv = orig


def test_replay_ayni_girdi_ayni_karar(tmp_path):
    ctxp = _ctx_provider(lambda s: _slow(FakeExchange.path[s][-1], age=30.0))
    outs = []
    for k in range(2):
        cfg = LR.RunnerConfig.from_dict({**SIM.default_config("mexc").to_dict(), "symbols": ["BTC/USDT"], "symbols_mode": "fixed"})
        reg = LR.RunnerRegistry(output_dir=str(tmp_path / str(k)), ctx=ctxp, client_factory=factory)
        r = reg.create(0, cfg)
        p = _path(dip_last=12, dip_pct=3.0); p[-1] = p[-2] * 1.002
        FakeExchange.path["BTC/USDT"] = p
        r.run_cycle(now=1_000_000.0)
        d = r.last_decisions["BTC/USDT"]
        outs.append((d.get("result"), d.get("trigger"), (d.get("plan") or {}).get("stop_pct"), (d.get("ticket") or {}).get("ev_pct")))
    assert outs[0] == outs[1]


def test_kaynak_dongu_suresi_40_parite(tmp_path):
    syms = [f"S{i}/USDT" for i in range(40)]
    for s in syms:
        FakeExchange.path[s] = _path(seed=hash(s) % 1000)
    ctxp = _ctx_provider(lambda s: None)
    cfg = LR.RunnerConfig.from_dict({**SIM.default_config("mexc").to_dict(), "symbols": syms, "symbols_mode": "fixed"})
    reg = LR.RunnerRegistry(output_dir=str(tmp_path), ctx=ctxp, client_factory=factory)
    r = reg.create(0, cfg)
    t0 = time.time(); r.run_cycle(now=1_000_000.0); dt = time.time() - t0
    assert dt < 20.0 and len(r.scan) == 40 and r.resource["state"] in ("GREEN", "YELLOW", "RED")
    assert r.full_state()["top_k"] <= 10


def test_kosucu_nakit_modu_ve_stop_sikilastirma(tmp_path):
    level = {"v": 0}
    ctxp = _ctx_provider(lambda s: _slow(FakeExchange.path[s][-1], age=30.0))
    ctxp.market_news = lambda: {"level": level["v"], "risk_off_score": 3.0 if level["v"] else 0.0, "items": []}
    cfg = LR.RunnerConfig.from_dict({**SIM.default_config("mexc").to_dict(), "symbols": ["BTC/USDT"], "symbols_mode": "fixed",
                                     "params": {**SIM.default_config("mexc").params, "maker_wait_bars": 0}})
    reg = LR.RunnerRegistry(output_dir=str(tmp_path), ctx=ctxp, client_factory=factory)
    r = reg.create(0, cfg)
    p = _path(dip_last=12, dip_pct=3.0); p[-1] = p[-2] * 1.002
    FakeExchange.path["BTC/USDT"] = p
    r.run_cycle(now=1_000_000.0)
    assert r.positions and r.portfolio["mode"] in (PM.RISK_ON, PM.SELECTIVE)
    level["v"] = 2
    r.run_cycle(now=1_000_030.0)
    assert not r.positions and r.cash_mode and r.trades[-1]["reason"] == "NAKİT MODU"
    fs = r.full_state()
    assert fs["portfolio_mode"]["mode"] == PM.CASH and fs["best_action"]["action"] == "CASH"


def test_simulator_konfig_senkron_geri_yuklemede(tmp_path):
    reg = LR.RunnerRegistry(output_dir=str(tmp_path), ctx=_ctx_provider(lambda s: None), client_factory=factory)
    old = SIM.default_config("mexc"); old.symbols = ["BTC/USDT"]; old.max_open = 2
    reg.create(0, old)
    out = SIM.ensure_simulator(reg, venue="mexc", start=False)
    s = SIM.get_simulator(reg)
    assert out["created"] is False and "symbols" in out["synced"] and len(s.cfg.symbols) == 40
    assert s.cfg.max_open == SIM.scalp_limits()["max_open"]      # tahsisten türetilir


def test_ucuz_ozellikler_nan_sizdirmaz():
    import json as _json
    flat = _df([100.0] * 150, up_last=False)                   # σ = 0, sd = 0
    st = MarketStateStore(fetch_ohlcv=lambda *a: flat)
    st.get_ohlcv("mexc", "FLAT/USDT", "1m", 150, now=1000.0)
    f = st.cheap_features("mexc", "FLAT/USDT")
    _json.dumps(f, allow_nan=False)
    assert f["z20"] == 0.0 and all(not (isinstance(v, float) and v != v) for v in f.values())


def test_nan_guvenli_json_yaniti():
    import json as _json
    import numpy as _np
    from agi_trader.server.safe_json import SafeJSONResponse, clean
    body = SafeJSONResponse({"a": float("nan"), "b": _np.float64("inf"), "c": [1.5, float("-inf"), {"d": _np.int64(3)}], "e": "ok"}).body
    d = _json.loads(body)
    assert d == {"a": None, "b": None, "c": [1.5, None, {"d": 3}], "e": "ok"}
    assert clean(True) is True and clean(None) is None
