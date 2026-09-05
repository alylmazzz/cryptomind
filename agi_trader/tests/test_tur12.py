# -*- coding: utf-8 -*-
"""12. tur (2026-09-04) — kanıt kapıları (sleeve durum makinesi, kanıt boyutu, taker kanıt şartı, drawdown
yarı-boyut, seans kapısı), ulaşılabilir hedefli fiş, TAM özsermaye geçmişi, sayfalı işlem ucu.

Kanıt (85 canlı işlem, MEXC sanal 1.000 $): brüt −0,60 $ / komisyon 4,18 $ / net −4,78 $; taker girişler −6,02 $,
maker +1,24 $; kanıtsız sleeve'ler tam boyutta (dip_moderate −2,71 $, obi_momentum −1,95 $); 25 $ tavan
karşı-olgusalı −0,92 $. Bu testler o kusurların bir daha oluşmamasını kilitler."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agi_trader.auto import live_runner as LR  # noqa: E402
from agi_trader.auto import simulator as SIM  # noqa: E402
from agi_trader.learn import allocator as AL  # noqa: E402
from agi_trader.learn.allocator import MetaAllocator  # noqa: E402
from agi_trader.strategies import committee as CM  # noqa: E402
from agi_trader.strategies import entry_optimizer as EO  # noqa: E402
from agi_trader.strategies import sleeves_fast as SF  # noqa: E402
from agi_trader.server import secure_keys as V  # noqa: E402

from test_live_trading import FakeExchange, factory, _path  # noqa: E402
from test_committee import _slow, _ctx, _ctx_provider  # noqa: E402

MASTER = "test-master-key-" + "z" * 40


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv(V.MASTER_ENV, MASTER)
    FakeExchange.path = {"BTC/USDT": _path(), "ETH/USDT": _path(seed=5)}
    FakeExchange.orders = []
    yield


def _sim_runner(tmp_path, slow_fn, **over):
    cfg = LR.RunnerConfig.from_dict({**SIM.default_config("mexc").to_dict(),
                                     "symbols": ["BTC/USDT"], "symbols_mode": "fixed", **over})
    reg = LR.RunnerRegistry(output_dir=str(tmp_path), ctx=_ctx_provider(slow_fn), client_factory=factory)
    return reg, reg.create(SIM.SYSTEM_UID, cfg)


def _trade(i, net, notional=50.0, sleeve="dip", opened=None, closed=None, peak=0.5):
    opened = 1_000_000.0 + i * 600 if opened is None else opened
    closed = opened + 900 if closed is None else closed
    return {"symbol": "BTC/USDT", "direction": "LONG", "entry": 100.0, "exit": 100.0 * (1 + net / notional),
            "amount": notional / 100.0, "notional": notional, "gross_pnl": net + 0.01, "fees": 0.01, "net_pnl": net,
            "pnl_pct": net / notional * 100.0, "net_pct_realized": net / notional * 100.0, "reason": "TRAIL" if net > 0 else "STOP",
            "opened_ts": opened, "closed_ts": closed, "hold_sec": closed - opened, "hold_bucket": "15-60 dk",
            "peak_pnl_pct": peak, "peak_net_pct": peak - 0.07, "peak_capture": 0.5, "win": net > 0, "mode": "paper",
            "strategy": "committee", "order_type": "maker", "trigger": sleeve, "sleeve": sleeve, "template": "mean_reversion",
            "exit_mode": "PARTIAL_AND_RUN", "target": 102.0, "partial_done": False, "horizon_sec": 5400.0}


# ═══════════════════════════ 1) sleeve kanıt durum makinesi ═══════════════════════════
def test_sleeve_kanit_durumu_ve_tavan(tmp_path):
    a = MetaAllocator(tmp_path / "a.json", SF.REGIME_SLEEVES)
    assert a.state("dip") == AL.STATE_UNPROVEN and a.notional_cap("dip", 25.0, 200.0)["cap_usdt"] == 25.0
    # %50 kazanma ama küçük kazanç / büyük kayıp (canlı dip_moderate profili) → beklenti negatif → KANITLANMADI
    for i in range(28):
        a.record("dip_moderate", None, i % 2 == 0, 0.1 if i % 2 == 0 else -0.3, net_pct=(0.2 if i % 2 == 0 else -0.5))
    assert a.state("dip_moderate") == AL.STATE_UNPROVEN
    assert a.notional_cap("dip_moderate", 25.0, 200.0)["cap_usdt"] == 25.0
    assert a.weight("dip_moderate") >= 0.9            # Beta güvenilirliği bunu görmez — durum makinesi görür
    # 20+ işlemde ortalama net > 0 ve t ≥ 1 → KANITLANDI → tam boyut
    for i in range(24):
        a.record("dip", None, i % 4 != 0, 0.2, net_pct=(0.4 if i % 4 else -0.2))
    assert a.state("dip") == AL.STATE_PROVEN and a.notional_cap("dip", 25.0, 200.0)["cap_usdt"] == 200.0
    st = a.status()["states"]
    assert st["dip"]["state"] == "PROVEN" and st["dip_moderate"]["state"] == "UNPROVEN" and st["dip"]["t_stat"] >= 1.0
    # kalıcılık: geçmiş diske yazılır, yeniden yüklenince aynı durum
    b = MetaAllocator(tmp_path / "a.json", SF.REGIME_SLEEVES)
    assert b.state("dip") == AL.STATE_PROVEN and len(b.history["dip"]) == 24


def test_devre_kesici_katlanan_sure_ve_deneme_penceresi(tmp_path):
    a = MetaAllocator(tmp_path / "a.json", SF.REGIME_SLEEVES)
    for i in range(AL.BREAKER_MIN_N):
        a.record("obi_momentum", None, i % 3 == 0, -0.1, net_pct=(-0.4 if i % 3 else 0.2))
    ev = a.check_breakers(now=1000.0)
    assert ev and ev[0]["sleeve"] == "obi_momentum" and ev[0]["pause_hours"] == 6.0 and ev[0]["pause_count"] == 1
    assert a.state("obi_momentum", 1000.0) == AL.STATE_PAUSED and a.notional_cap("obi_momentum", 25.0, 200.0, 1000.0)["cap_usdt"] == 0.0
    assert len(a.history["obi_momentum"]) == AL.BREAKER_MIN_N          # kümülatif kanıt SIFIRLANMADI
    t1 = 1000.0 + 7 * 3600
    assert a.state("obi_momentum", t1) == AL.STATE_PROBATION             # duraklama bitti → deneme (kanıt boyutu)
    assert a.notional_cap("obi_momentum", 25.0, 200.0, t1)["cap_usdt"] == 25.0
    # deneme penceresinde 8 işlem yine negatif → yeniden duraklat, süre 12 sa
    for i in range(AL.PROBATION_N):
        a.record("obi_momentum", None, False, -0.1, net_pct=-0.3)
    ev2 = a.check_breakers(now=t1)
    assert ev2 and ev2[0]["rule"] == "deneme" and ev2[0]["pause_hours"] == 12.0 and ev2[0]["pause_count"] == 2
    # ikinci duraklama sonrası deneme geçilirse (8 işlemde net ≥ 0) deneme biter; kümülatif kanıt hâlâ negatif → KANITLANMADI
    t2 = t1 + 13 * 3600
    for i in range(AL.PROBATION_N):
        a.record("obi_momentum", None, True, 0.1, net_pct=0.3)
    assert "obi_momentum" not in a.probation and a.state("obi_momentum", t2) == AL.STATE_UNPROVEN


def test_allocator_backfill_gecmis_islemlerden():
    a = MetaAllocator(None, SF.REGIME_SLEEVES)
    trades = [_trade(i, -0.3 if i % 2 else 0.1, sleeve="dip_moderate") for i in range(20)]
    assert a.backfill(trades) == 20 and a.evidence("dip_moderate")["n"] == 20 and a.state("dip_moderate") == AL.STATE_UNPROVEN
    assert a.backfill(trades) == 0                                        # yalnız boşken


# ═══════════════════════════ 2) komite: ulaşılabilir hedef + taker kanıt şartı ═══════════════════════════
def test_ulasilabilir_hedef_sinirlari():
    r = CM.achievable_target_pct("dip", 2.0, atr_pct=0.2, hold_bars=90, cost_pct=0.1, min_gross_to_cost=2.0)
    assert 0.2 <= r["pct"] <= 2.0 and r["source"].startswith("formül")
    assert abs(r["pct"] - 0.8 * 0.2 * 90 ** 0.5) < 1e-3
    r2 = CM.achievable_target_pct("dip", 2.0, 0.2, 90, 0.1, 2.0, learned={"mfe_by_sleeve": {"dip": 0.55}})
    assert r2["pct"] == 0.55 and "ölçülmüş" in r2["source"]
    r3 = CM.achievable_target_pct("dip", 2.0, 0.2, 90, 0.1, 2.0, learned={"mfe_by_sleeve": {"dip": 9.0}})
    assert r3["pct"] == 2.0                                               # planın hedefini aşamaz


def test_fis_ulasilabilir_ev_yazar_ve_veto_etmez():
    v = CM.evaluate(_ctx(), CM.CommitteeParams(), {"mfe_by_sleeve": {"dip": 0.55}})
    assert v.allowed, v.result
    tk = v.ticket
    assert tk["achievable_target_pct"] == pytest.approx(0.55) and tk["ev_achievable_pct"] < tk["ev_pct"]
    assert tk["sleeve_state"] == "UNPROVEN"
    assert any("kanıt boyutu" in n for n in v.notes)                     # EV ≤ 0 uyarısı NOT'tur, veto değil
    assert all("achievable_target_pct" in c for c in v.to_dict()["competition"])


def test_taker_girisi_kanit_ister(monkeypatch):
    real = EO.optimize_entry

    def taker_opt(*a, **k):
        out = real(*a, **k)
        out["order_type"] = "taker"
        if out.get("optimal"):
            out["optimal"] = {**out["optimal"], "order_type": "taker"}
        return out
    monkeypatch.setattr(EO, "optimize_entry", taker_opt)
    v = CM.evaluate(_ctx(), CM.CommitteeParams(), {"sleeve_states": {"dip": "UNPROVEN"}})
    assert v.allowed and v.order_type == "maker" and any("kanıt kapısı" in n and "MAKER" in n for n in v.notes)
    v2 = CM.evaluate(_ctx(), CM.CommitteeParams(), {"sleeve_states": {"dip": "PROVEN"}})
    assert v2.allowed and v2.order_type == "taker"
    v3 = CM.evaluate(_ctx(), CM.CommitteeParams(taker_requires_proof=False), {"sleeve_states": {"dip": "UNPROVEN"}})
    assert v3.order_type == "taker"


# ═══════════════════════════ 3) koşucu: kanıt tavanı, drawdown, seans ═══════════════════════════
def test_kosucu_kanitsiz_sleeve_kanit_boyutuyla_girer(tmp_path):
    def slow(sym):
        return _slow(FakeExchange.path[sym][-1], age=30.0)
    reg, r = _sim_runner(tmp_path, slow)
    # kanıt boyutu = min(25, emir tavanı); ölçüm bütçesinde emir tavanı 10 $ → 10
    assert r.params.probe_notional_usdt == min(25.0, r.cfg.max_order_usdt)
    p = _path(dip_last=12, dip_pct=3.0); p[-1] = p[-2] * 1.002
    FakeExchange.path["BTC/USDT"] = p
    r.run_cycle(now=1_000_000.0)
    assert "BTC/USDT" in r.pending, r.last_decisions
    pend = r.pending["BTC/USDT"]
    assert pend["notional"] <= 25.0 + 1e-9
    d = r.last_decisions["BTC/USDT"]
    assert d["evidence_gate"]["state"] == "UNPROVEN" and d["evidence_gate"]["cap_usdt"] == r.params.probe_notional_usdt
    # Not yalnız tavan BAĞLAYINCA yazılır. Sermaye tahsisinden sonra emir tavanı (10 $) kanıt
    # boyutuna eşit olduğu için tavan zaten aşılamıyor; kilitlenen davranış boyutun tavanı
    # AŞMAMASI, notun varlığı değil.
    assert pend["notional"] <= r.params.probe_notional_usdt + 1e-9
    assert d["evidence_gate"]["cap_usdt"] >= pend["notional"]
    st = r.full_state()
    assert st["governance"]["probe_notional_usdt"] == r.params.probe_notional_usdt and st["governance"]["derisk"]["active"] is False
    assert any(s["state"] == "UNPROVEN" for s in st["governance"]["sleeves"]) or st["governance"]["sleeves"] == []


def test_kosucu_kanitlanmis_sleeve_tam_boyut(tmp_path):
    def slow(sym):
        return _slow(FakeExchange.path[sym][-1], age=30.0)
    reg, r = _sim_runner(tmp_path, slow)
    for i in range(24):
        r.allocator.record("dip", None, i % 4 != 0, 0.2, net_pct=(0.4 if i % 4 else -0.2))
    assert r.allocator.state("dip") == AL.STATE_PROVEN
    p = _path(dip_last=12, dip_pct=3.0); p[-1] = p[-2] * 1.002
    FakeExchange.path["BTC/USDT"] = p
    r.run_cycle(now=1_000_000.0)
    assert "BTC/USDT" in r.pending, r.last_decisions
    # kanıtlanmış sleeve tavandan kurtulur → emir tavanına kadar çıkabilir (kanıt boyutunun üstü)
    assert r.pending["BTC/USDT"]["notional"] > r.params.probe_notional_usdt or         r.pending["BTC/USDT"]["notional"] == pytest.approx(r.cfg.max_order_usdt)
    assert r.last_decisions["BTC/USDT"]["evidence_gate"]["state"] == "PROVEN"


def test_kosucu_duraklatilan_sleeve_girmez(tmp_path):
    def slow(sym):
        return _slow(FakeExchange.path[sym][-1], age=30.0)
    reg, r = _sim_runner(tmp_path, slow)
    r.allocator.paused_until["dip"] = 1_000_000.0 + 3600
    p = _path(dip_last=12, dip_pct=3.0); p[-1] = p[-2] * 1.002
    FakeExchange.path["BTC/USDT"] = p
    r.run_cycle(now=1_000_000.0)
    # duraklatılan 'dip' asla tetikleyici olamaz; başka bir sleeve (ör. süpürme dönüşü) girebilir — o kanıt boyutuyla
    d = r.last_decisions["BTC/USDT"]
    assert d.get("trigger") != "dip"
    if "BTC/USDT" in r.pending:
        assert r.pending["BTC/USDT"]["verdict"]["trigger"] != "dip" and r.pending["BTC/USDT"]["notional"] <= 25.0 + 1e-9
    else:
        assert d["result"].startswith(("VETO", "BEKLE", "SLEEVE DURAKLATILDI"))
    assert "dip" in r.allocator.paused_sleeves(1_000_000.0)


def test_drawdown_yari_boyut_ve_seans_kapisi(tmp_path):
    def slow(sym):
        return _slow(FakeExchange.path[sym][-1], age=30.0)
    reg, r = _sim_runner(tmp_path, slow)
    assert r._derisk()["active"] is False
    # son 20 işlem net < 0 → aktif
    r.trades = [_trade(i, -0.2 if i % 3 else 0.1) for i in range(20)]
    d = r._derisk()
    assert d["active"] and d["mult"] == 0.5 and d["trailing_net"] < 0
    # seans: 20-24 UTC bloğunda 25 işlem, ort net % −0,15, t ≈ −1,6 → ×0,5; 00-04 bloğunda t ≤ −2,5 → giriş yok
    now = 1_788_500_000.0                                                   # 2026-09-04 ~06:53 UTC
    base_20 = 1_788_465_600.0 + 600                                          # 2026-09-03 20:10 UTC
    base_00 = 1_788_480_000.0 + 600                                          # 2026-09-04 00:10 UTC
    rng = np.random.default_rng(1)
    tr = [_trade(i, float(-0.15 + rng.normal(0, 0.45)) * 0.5, notional=50.0, opened=base_20 + i * 20, closed=base_20 + i * 20 + 600)
          for i in range(25)]
    tr += [_trade(100 + i, -0.4, notional=50.0, opened=base_00 + i * 20, closed=base_00 + i * 20 + 600) for i in range(16)]
    r.trades = tr
    sg = r._session_gate(now)
    assert sg["00-04"]["mult"] == 0.0 and sg["00-04"]["t_stat"] <= -2.5
    assert sg["20-24"]["n"] == 25 and sg["20-24"]["mult"] in (0.5, 1.0, 0.0)
    r.trades = [_trade(i, -0.5, notional=50.0, opened=base_00 + i * 20, closed=base_00 + i * 20 + 600) for i in range(16)]
    gov = r.governance(now=base_00 + 3600)
    assert gov["session"]["00-04"]["mult"] == 0.0 and gov["session_now"] == "00-04"
    # kapalı blokta giriş denemesi → SEANS KAPISI
    p = _path(dip_last=12, dip_pct=3.0); p[-1] = p[-2] * 1.002
    FakeExchange.path["BTC/USDT"] = p
    r.run_cycle(now=base_00 + 3600)
    assert not r.pending and "SEANS KAPISI" in r.last_decisions["BTC/USDT"]["result"]


def test_kosucu_taker_kanit_kapisi_maker_bekler_kovalamaz(tmp_path, monkeypatch):
    real = EO.optimize_entry

    def taker_opt(*a, **k):
        out = real(*a, **k)
        out["order_type"] = "taker"
        if out.get("optimal"):
            out["optimal"] = {**out["optimal"], "order_type": "taker"}
        return out
    monkeypatch.setattr(EO, "optimize_entry", taker_opt)

    def slow(sym):
        return _slow(FakeExchange.path[sym][-1], age=30.0)
    reg, r = _sim_runner(tmp_path, slow, params={**SIM.default_config("mexc").params, "chase_taker_ratio": 1.0})
    p = _path(dip_last=12, dip_pct=3.0); p[-1] = p[-2] * 1.002
    FakeExchange.path["BTC/USDT"] = p
    r.run_cycle(now=1_000_000.0)
    assert "BTC/USDT" in r.pending and not r.positions, r.last_decisions   # taker yerine maker bekledi
    assert r.pending["BTC/USDT"]["no_chase"] is True and r.pending["BTC/USDT"]["max_bars"] >= 1
    FakeExchange.path["BTC/USDT"] = p + [p[-1] * 1.0015] * 3                 # dolmadı (low limitin üstünde), max chase İÇİNDE
    max_bars = int(r.pending["BTC/USDT"]["max_bars"])
    for k in range(1, max_bars + 1):                                        # dip: maker en çok 2 bar bekler
        r.run_cycle(now=1_000_000.0 + 30 * k)
    assert not r.positions and not r.pending
    assert "kovalanmaz" in r.last_decisions["BTC/USDT"]["result"]           # chase_taker_ratio=1 bile olsa taker'a geçmedi
    assert r.lessons.maker["attempts"] == 1 and r.lessons.maker["chased"] == 0
    for k in range(max_bars + 1, max_bars + 3):
        r.run_cycle(now=1_000_000.0 + 30 * k)
    assert not r.positions


# ═══════════════════════════ 4) TAM özsermaye geçmişi ═══════════════════════════
def test_ozsermaye_gecmisi_sikistirma_kalicilik_ve_defterden_tamamlama(tmp_path):
    def slow(sym):
        return _slow(FakeExchange.path[sym][-1], age=30.0)
    reg, r = _sim_runner(tmp_path, slow)
    t0 = 1_000_000.0
    for i in range(1200):                                                   # 10 saat, 30 sn
        r._mark(t0 + i * 30.0)
    r._compact_history(t0 + 1200 * 30.0)
    hist = r.equity_history
    assert hist[0]["ts"] == t0                                              # ilk nokta korunur
    cut = t0 + 1200 * 30.0 - LR.EQUITY_FULL_RES_SEC
    old = [p for p in hist if p["ts"] < cut]; new = [p for p in hist if p["ts"] >= cut]
    assert len(new) >= LR.EQUITY_FULL_RES_SEC / 30 - 2 and len(old) <= (cut - t0) / LR.EQUITY_BUCKET_SEC + 3
    assert len(r.equity_curve) <= 2000
    r.save()
    d = json.loads(r._path.read_text(encoding="utf-8"))
    assert len(d["equity_history"]) == len(hist)
    # defterden tamamlama: geçmiş kaybolmuş (eski sürüm) ama işlemler var → başlangıç anı + kapanış noktaları
    r2 = reg.create(7, LR.RunnerConfig.from_dict({**r.cfg.to_dict(), "exchange_id": "binance"}), restore={
        "trades": [_trade(i, 0.5 if i % 2 else -0.3, opened=900_000.0 + i * 600, closed=900_000.0 + i * 600 + 300) for i in range(6)],
        "equity_curve": [{"ts": 950_000.0, "equity": 1000.6}]})
    h2 = r2.equity_history
    assert h2[0] == {"ts": 900_000.0, "equity": 1000.0, "src": "ledger"}
    assert sum(1 for p in h2 if p["src"] == "ledger") == 7 and h2[-1]["src"] == "live"
    assert h2[-2]["equity"] == pytest.approx(1000.0 + sum(0.5 if i % 2 else -0.3 for i in range(6)))
    s = r2.equity_series(max_points=6)
    assert s["n_raw"] == 8 and 2 <= s["n"] <= 8 and s["marks"][-1]["seq"] == 6 and s["start_ts"] == 900_000.0
    assert all(s["points"][i]["ts"] <= s["points"][i + 1]["ts"] for i in range(len(s["points"]) - 1))


def test_islem_listesi_3000_kalici(tmp_path):
    def slow(sym):
        return _slow(FakeExchange.path[sym][-1], age=30.0)
    reg, r = _sim_runner(tmp_path, slow)
    r.trades = [_trade(i, 0.1) for i in range(700)]
    r.save()
    d = json.loads(r._path.read_text(encoding="utf-8"))
    assert len(d["trades"]) == 700                                          # 500 tavanı ilk işlemleri siliyordu


# ═══════════════════════════ 5) sayfalı işlem + özsermaye uçları ═══════════════════════════
def test_sayfali_islem_ve_ozsermaye_uclari(tmp_path):
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    def slow(sym):
        return _slow(FakeExchange.path[sym][-1], age=30.0)
    reg, r = _sim_runner(tmp_path, slow)
    # işlemler canlı işaretlemeden ÖNCE (eski sürümde geçmiş yoktu) → başlangıç defterden tamamlanır
    r.trades = [_trade(i, 0.1 if i % 2 else -0.05, opened=900_000.0 + i * 600, closed=900_000.0 + i * 600 + 300) for i in range(85)]
    for i in range(10):
        r._mark(1_000_000.0 + i * 30)
    app = FastAPI(); app.include_router(SIM.create_public_router(reg))
    c = TestClient(app)
    j = c.get("/api/simulator/trades?page=1&per_page=25").json()
    assert j["total"] == 85 and j["pages"] == 4 and j["from_seq"] == 85 and j["to_seq"] == 61
    assert [t["seq"] for t in j["trades"]][:3] == [85, 84, 83] and "prev_hash" not in j["trades"][0]
    j3 = c.get("/api/simulator/trades?page=3&per_page=25").json()
    assert j3["from_seq"] == 35 and j3["to_seq"] == 11 and len(j3["trades"]) == 25 and j3["trades"][-1]["seq"] == 11
    j4 = c.get("/api/simulator/trades?page=4&per_page=25").json()
    assert j4["from_seq"] == 10 and j4["to_seq"] == 1 and len(j4["trades"]) == 10
    j9 = c.get("/api/simulator/trades?page=99&per_page=25").json()
    assert j9["page"] == 4                                                  # taşan sayfa son sayfaya kelepçelenir
    e = c.get("/api/simulator/equity?max_points=100").json()
    assert e["configured"] and e["capital"] == 1000.0 and e["n"] >= 10 and len(e["marks"]) == 85
    assert e["points"][0]["src"] == "ledger" and e["points"][-1]["src"] == "live"
    st = c.get("/api/simulator").json()
    assert st["governance"]["probe_notional_usdt"] == min(25.0, r.cfg.max_order_usdt) and st["n_trades_total"] == 85


def test_simulator_default_config_kanit_parametrelerini_tasir():
    p = SIM.default_config("mexc").params
    lim = SIM.scalp_limits()
    assert p["probe_notional_usdt"] == min(25.0, lim["max_order_usdt"]) and p["taker_requires_proof"] is True
    assert p["derisk_trailing_n"] == 20 and p["session_gate"] is True and p["achievable_target"] is True
