# -*- coding: utf-8 -*-
"""10. tur — kurumsal katman: replay/bilimsel kabul, kalibrasyon + Monte Carlo, uyarı kanalı, defter zinciri,
aciliyete göre maker/taker, sleeve tavanı, korelasyon bütçesi, CVD, tazelik, betikler."""
from __future__ import annotations

import importlib
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

from agi_trader.auto import replay as RP  # noqa: E402
from agi_trader.auto import live_runner as LR  # noqa: E402
from agi_trader.auto import simulator as SIM  # noqa: E402
from agi_trader.learn import calibration as CAL  # noqa: E402
from agi_trader.notify.alerts import AlertBus  # noqa: E402
from agi_trader.strategies import sleeves_fast as SF  # noqa: E402
from agi_trader.strategies.lifecycle import Lifecycle  # noqa: E402
from agi_trader.server import secure_keys as V  # noqa: E402

from test_live_trading import FakeExchange, factory, _path  # noqa: E402
from test_committee import _slow, _ctx_provider  # noqa: E402

MASTER = "test-master-key-" + "q" * 40


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv(V.MASTER_ENV, MASTER)
    FakeExchange.path = {"BTC/USDT": _path(), "ETH/USDT": _path(seed=5)}
    FakeExchange.orders = []
    yield


# ═══════════════════════════ replay ═══════════════════════════
def _hist(symbols, n_min=900, seed=1):
    rng = np.random.default_rng(seed)
    end = pd.Timestamp("2026-08-01 00:00", tz="UTC")
    out = {}
    for k, s in enumerate(symbols):
        t = np.arange(n_min)
        base = 100.0 * (1 + 0.02 * np.sin(t / 90.0 + k)) * np.exp(np.cumsum(rng.normal(0, 0.0006, n_min)))
        hi = base * (1 + np.abs(rng.normal(0, 0.0008, n_min))); lo = base * (1 - np.abs(rng.normal(0, 0.0008, n_min)))
        vol = rng.uniform(50, 150, n_min) * (1 + 2 * (np.sin(t / 30.0) > 0.9))
        idx = pd.date_range(end - pd.Timedelta(minutes=n_min - 1), periods=n_min, freq="min")
        df1 = pd.DataFrame({"open": base, "high": hi, "low": lo, "close": base, "volume": vol}, index=idx)
        out[(s, "1m")] = df1
        out[(s, "1h")] = df1.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna().tail(120)
        h4 = np.arange(320)
        b4 = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 320)))
        idx4 = pd.date_range(end - pd.Timedelta(hours=4 * 319), periods=320, freq="4h")
        out[(s, "4h")] = pd.DataFrame({"open": b4, "high": b4 * 1.01, "low": b4 * 0.99, "close": b4, "volume": np.full(320, 1000.0)}, index=idx4)
    return out


def test_replay_lookahead_yok_ve_imlec():
    hist = _hist(["A/USDT"])
    cur = RP.Cursor(int(hist[("A/USDT", "1m")].index[300].timestamp() * 1000))
    ex = RP.ReplayExchange(hist, cur)
    rows = ex.fetch_ohlcv("A/USDT", "1m", 150)
    assert len(rows) == 150 and rows[-1][0] == cur.ms and all(r[0] <= cur.ms for r in rows)
    assert ex.fetch_ticker("A/USDT")["last"] == pytest.approx(rows[-1][4])
    ob = ex.fetch_order_book("A/USDT"); assert ob["bids"][0][0] < ob["asks"][0][0]
    cur.ms += 60_000
    assert ex.fetch_ohlcv("A/USDT", "1m", 1)[0][0] == cur.ms
    assert ex.fetch_trades("A/USDT") == [] and ex.fetch_ohlcv("A/USDT", "4h", 300)[-1][0] <= cur.ms


def test_replay_kosumu_ve_analiz_ve_kanit(tmp_path):
    hist = _hist(["A/USDT", "B/USDT"], n_min=700)
    res = RP.run_replay(hist, ["A/USDT", "B/USDT"], tmp_path / "rp", step_sec=60, warmup_bars=200,
                        cfg_overrides={"params": {**SIM.default_config("mexc").params, "maker_wait_bars": 0, "dip_z_moderate": 0.8, "rsi_max_moderate": 60.0}},
                        max_cycles=260)
    assert res["n_cycles"] == 260 and res["symbols"] == ["A/USDT", "B/USDT"] and "missed" in res and res["limits"]
    an = RP.analyze(res, n_trials=10)
    assert an["n_trades"] == len(res["trades"]) and "capital" in an
    if an["n_trades"]:
        assert "expectancy_pct" in an and "ci95" in an and "per_sleeve" in an and an["exit_reasons"]
        lc = Lifecycle(tmp_path / "lc.json")
        rows = RP.write_evidence(lc, an)
        assert rows and all("passed" in r and "checks" in r for r in rows)
        assert all(not r["passed"] or r["n_trades"] >= 30 for r in rows)          # n<30 ile kapı geçilemez
    p = RP.save_result(res, an, tmp_path / "out")
    assert p.exists() and json.loads(p.read_text(encoding="utf-8"))["analysis"]["n_trades"] == an["n_trades"]


def _fake_trades(n=40, seed=3, sleeves=("dip", "breakout")):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        s = sleeves[i % len(sleeves)]
        r = float(rng.normal(0.3 if s == "dip" else -0.1, 0.8))
        out.append({"symbol": f"S{i%5}/USDT", "sleeve": s, "trigger": s, "net_pct_realized": r, "net_pnl": r, "gross_pnl": r + 0.1, "fees": 0.1,
                    "notional": 100.0, "closed_ts": 1000.0 + i * 600, "win": r > 0, "reason": "TP" if r > 0 else "STOP", "peak_capture": 0.5,
                    "decision": {"ticket": {"p_win": 0.55 if s == "dip" else 0.5}}})
    return out


def test_analiz_dsr_pbo_ve_alt_donem():
    res = {"trades": _fake_trades(60), "equity_curve": [{"ts": i, "equity": 1000 + i} for i in range(60)], "capital": 1000.0}
    an = RP.analyze(res, n_trials=10)
    assert an["n_trades"] == 60 and an["pbo"] is not None and 0.0 <= an["pbo"] <= 1.0 and an["dsr"] is not None
    assert set(an["per_sleeve"]) == {"dip", "breakout"} and an["per_sleeve"]["dip"]["oos_expectancy"] > an["per_sleeve"]["breakout"]["oos_expectancy"]
    assert an["subperiod"]["consistent"] in (True, False) and an["expectancy_cost_x2_pct"] < an["expectancy_pct"]


# ═══════════════════════════ kalibrasyon / Monte Carlo ═══════════════════════════
def test_kalibrasyon_tablosu_ve_monte_carlo():
    tr = _fake_trades(80)
    cal = CAL.reliability_table(tr)
    assert cal["n"] == 80 and cal["brier"] is not None and cal["bins"] and "dip" in cal["per_sleeve"] and cal["verdict"]
    mc = CAL.monte_carlo(tr, capital=1000.0, n_paths=300)
    assert mc["paths"] == 300 and mc["final_p5"] <= mc["final_p50"] <= mc["final_p95"] and 0 <= mc["p_ruin"] <= 1 and 0 <= mc["p_breach_daily_limit"] <= 1
    assert CAL.monte_carlo(tr[:3])["note"]
    assert CAL.reliability_table([{"win": True}])["n"] == 0


# ═══════════════════════════ uyarılar ═══════════════════════════
def test_uyari_kanali_hiz_siniri_ve_yapilandirmasiz():
    b = AlertBus(env={}, min_interval=60.0)
    assert not b.configured and b.send("x", "merhaba", now=1000.0) and b.log[0]["delivered"] is False
    assert b.send("x", "tekrar", now=1010.0) is False                 # 60 sn içinde aynı anahtar
    assert b.send("x", "zorla", now=1010.0, force=True)
    sent = []
    b2 = AlertBus(env={"CRYPTOMIND_ALERT_WEBHOOK": "https://example.invalid/hook"}, min_interval=0)
    b2._sender = lambda rec: sent.append(rec) or True
    assert b2.configured and b2.send("open:BTC", "AÇILDI", now=1.0)
    time.sleep(0.2)
    assert sent and sent[0]["key"] == "open:BTC" and b2.status()["sent"] == 1


# ═══════════════════════════ koşucu entegrasyonu ═══════════════════════════
def _runner(tmp_path, symbols, **over):
    ctxp = _ctx_provider(lambda s: _slow(FakeExchange.path[s][-1], age=30.0))
    cfg = LR.RunnerConfig.from_dict({**SIM.default_config("mexc").to_dict(), "symbols": symbols, "symbols_mode": "fixed", **over})
    reg = LR.RunnerRegistry(output_dir=str(tmp_path), ctx=ctxp, client_factory=factory)
    return reg, reg.create(0, cfg)


def test_defter_zinciri_ve_kurcalama(tmp_path):
    reg, r = _runner(tmp_path, ["BTC/USDT"], params={**SIM.default_config("mexc").params, "maker_wait_bars": 0})
    p = _path(dip_last=12, dip_pct=3.0); p[-1] = p[-2] * 1.002
    FakeExchange.path["BTC/USDT"] = p
    r.run_cycle(now=1_000_000.0)
    pos = r.positions["BTC/USDT"]
    FakeExchange.path["BTC/USDT"] = p + [pos.stop * 0.99] * 2
    r.run_cycle(now=1_000_060.0)
    assert r.trades and r.trades[-1]["hash"] and r.trades[-1]["prev_hash"] == ""
    assert r.trades[-1].get("p_win") is not None and CAL.reliability_table(r.trades)["n"] == 1
    v = LR.verify_ledger(r.trades); assert v["ok"] and v["chained"] == 1
    st = r.full_state(); assert st["ledger"]["ok"] and "alerts" in st and st["alerts"]["recent"]
    tam = json.loads(json.dumps(r.trades)); tam[-1]["net_pnl"] = 999.0
    assert not LR.verify_ledger(tam)["ok"]


def test_aciliyet_kirilim_anında_taker_dip_maker_bekler(tmp_path):
    reg, r = _runner(tmp_path, ["BTC/USDT"])
    p = _path(dip_last=12, dip_pct=3.0); p[-1] = p[-2] * 1.002
    FakeExchange.path["BTC/USDT"] = p
    r.run_cycle(now=1_000_000.0)
    assert "BTC/USDT" in r.pending and r.pending["BTC/USDT"]["max_bars"] == 2          # dip: sabırlı maker
    assert SF.SLEEVE_URGENCY["breakout"] == 0 and SF.SLEEVE_URGENCY["catalyst"] == 0


def test_sleeve_tavani_ve_korelasyon_butcesi(tmp_path):
    syms = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    for s in syms:
        FakeExchange.path[s] = _path(dip_last=12, dip_pct=3.0); FakeExchange.path[s][-1] = FakeExchange.path[s][-2] * 1.002
    reg, r = _runner(tmp_path, syms, max_open=5, max_open_per_sleeve=1, params={**SIM.default_config("mexc").params, "maker_wait_bars": 0})
    r._update_portfolio_mode = lambda *a, **k: None
    r.run_cycle(now=1_000_000.0)
    assert len(r.positions) == 1
    blocked = [d for d in r.last_decisions.values() if str(d.get("result", "")).startswith("SLEEVE TAVANI")]
    assert blocked and any(rec["gate"] == "SLEEVE_TAVANI" for rec in r.missed.records)
    corr = r._corr_with_open(blocked[0]["symbol"])
    assert corr is not None and corr > 0.99                                            # aynı yol → ρ≈1


def test_cvd_hesabi_ve_obi_sleeve():
    now = 1_000_000_000.0
    trades = [{"timestamp": now - i * 1000, "side": "buy" if i % 4 else "sell", "price": 100.0, "amount": 1.0} for i in range(120)]
    c = SF.cvd_from_trades(trades, now)
    assert c["cvd_n"] == 120 and c["cvd_ratio"] == pytest.approx(0.5) and c["cvd_burst"] is not None
    assert SF.cvd_from_trades([], now)["cvd_ratio"] is None
    f = dict(ok=True, z=0.0, rsi=55.0, bar_up=True, trend_up=True, dist_ema_pct=-0.3, atr_pct=0.3, vol_ratio=1.1, ema_cross_up=False,
             obi=0.58, microprice_dev_bps=0.2, spread_bps=2.0, cvd_ratio=0.4)
    from agi_trader.strategies import committee as CM
    out = SF.fire_sleeves(f, ["obi_momentum"], None, CM.CommitteeParams())
    assert out and "CVD" in out[0]["note"]
    f["cvd_ratio"] = -0.3
    assert SF.fire_sleeves(f, ["obi_momentum"], None, CM.CommitteeParams()) == []


def test_tazelik_taramada_ve_kararda(tmp_path):
    reg, r = _runner(tmp_path, ["BTC/USDT"])
    r.run_cycle(now=time.time())
    assert r.scan and r.scan[0]["freshness"] in ("LIVE", "DELAYED", "STALE")
    d = r.last_decisions.get("BTC/USDT") or {}
    assert d.get("freshness") in ("LIVE", "DELAYED", "STALE", None)
    assert LR.RunnerConfig.from_dict({"max_open_per_sleeve": 99}).max_open_per_sleeve == 10


def test_betikler_ice_aktarilir():
    sys.path.insert(0, str(ROOT / "scripts"))
    for m in ("cm_replay", "cm_daily_report", "cm_selfcheck", "cm_ledger_verify"):
        mod = importlib.import_module(m)
        assert hasattr(mod, "main")
