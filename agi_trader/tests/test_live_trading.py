# -*- coding: utf-8 -*-
"""Otopilot katmanı — broker, karar zinciri, koşucu, kayıt defteri, anahtar
kapsamı ve /account/trading uçları.

Bu katman GERÇEK PARA gönderebilir. Test edilmemiş bir kapı, açık bir kapıdır.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agi_trader.execution.broker import Broker, BrokerError, make_client_order_id  # noqa: E402
from agi_trader.auto import decision_chain as DC  # noqa: E402
from agi_trader.auto import live_runner as LR  # noqa: E402
from agi_trader.strategies import video_scalp as VS  # noqa: E402
from agi_trader.server import secure_keys as V  # noqa: E402
from agi_trader.server import auth as A  # noqa: E402
from agi_trader.risk.live_guard import LiveGuard  # noqa: E402

pytest.importorskip("cryptography", reason="kasa cryptography olmadan çalışmaz")
MASTER = "test-master-key-" + "y" * 40


# ═══════════════════════════ sahte borsa ═══════════════════════════
class FakeExchange:
    """ccxt arayüzünün bu katmanda kullanılan alt kümesi. Fiyat yolu dışarıdan
    kurulur; para çekme ile ilgili HİÇBİR metodu yoktur (çağrılırsa AttributeError)."""
    path = {}                 # symbol -> list[float] (son kapanışlar)
    withdraw = False
    trade = True
    sandbox_calls = 0
    orders = []
    fail_order = False

    def __init__(self, params=None):
        self.params = params or {}
        self.markets = {"BTC/USDT": {"limits": {"cost": {"min": 5.0}, "amount": {"min": 1e-6}},
                                     "precision": {"amount": 6}},
                        "ETH/USDT": {"limits": {"cost": {"min": 5.0}, "amount": {"min": 1e-5}},
                                     "precision": {"amount": 5}}}
        self.urls = {"test": "https://testnet"}

    # public
    def load_markets(self):
        return self.markets

    def market(self, s):
        return self.markets[s]

    def amount_to_precision(self, s, a):
        return f"{a:.6f}"

    def fetch_ohlcv(self, symbol, timeframe, limit=150):
        closes = list(FakeExchange.path.get(symbol, [100.0] * limit))[-limit:]
        base = int(time.time() * 1000) - 60_000 * len(closes)
        return [[base + i * 60_000, c, c * 1.001, c * 0.999, c, 1.0] for i, c in enumerate(closes)]

    def fetch_ticker(self, symbol):
        return {"last": FakeExchange.path.get(symbol, [100.0])[-1]}

    def fetch_order_book(self, symbol, limit=20):
        px = FakeExchange.path.get(symbol, [100.0])[-1]
        return {"bids": [[px * 0.9999, 1000.0]], "asks": [[px * 1.0001, 1000.0]]}

    # private
    def set_sandbox_mode(self, on):
        FakeExchange.sandbox_calls += 1

    def fetch_balance(self):
        return {"USDT": {"free": 1000.0, "total": 1000.0}, "total": {"USDT": 1000.0}}

    def sapi_get_account_apirestrictions(self):
        return {"enableWithdrawals": FakeExchange.withdraw,
                "enableSpotAndMarginTrading": FakeExchange.trade, "ipRestrict": True}

    def fetch_positions(self):
        return []

    def create_order(self, symbol, typ, side, amount, price=None, params=None):
        if FakeExchange.fail_order:
            raise RuntimeError("InsufficientFunds")
        px = FakeExchange.path.get(symbol, [100.0])[-1]
        o = {"id": f"ex{len(FakeExchange.orders)+1}", "filled": amount, "average": px,
             "status": "closed", "fee": {"cost": amount * px * 0.001, "currency": "USDT"},
             "clientOrderId": (params or {}).get("clientOrderId")}
        FakeExchange.orders.append(o)
        return o


def factory(exchange_id, params):
    return FakeExchange(params)


def _path(n=150, base=100.0, dip_last=0, dip_pct=3.0, seed=3):
    rng = np.random.default_rng(seed)
    c = base * np.exp(np.cumsum(rng.normal(0.0, 0.0005, n)))
    if dip_last:
        c[-dip_last:] *= (1.0 - np.linspace(0, dip_pct / 100.0, dip_last))
    return list(c)


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setenv(V.MASTER_ENV, MASTER)
    A.reset_rate_limits()
    FakeExchange.path = {"BTC/USDT": _path(), "ETH/USDT": _path(seed=5)}
    FakeExchange.withdraw = False
    FakeExchange.trade = True
    FakeExchange.orders = []
    FakeExchange.fail_order = False
    yield


# ═══════════════════════════ broker ═══════════════════════════
def test_paper_emir_komisyon_ve_defter():
    b = Broker("binance", "paper", fee_bps=10.0, max_order_usdt=100.0, paper_capital=1000.0,
               client_factory=factory)
    o = b.market_order("BTC/USDT", "buy", 100.0, "cid1", ref_price=100.0)
    assert o["mode"] == "paper" and o["fee_usdt"] == pytest.approx(0.1)
    assert b.paper_cash == pytest.approx(1000.0 - 100.0 - 0.1)
    assert b.paper_holdings["BTC/USDT"] == pytest.approx(1.0)


def test_emir_tavani_asilamaz_ama_kapanis_engellenmez():
    b = Broker("binance", "paper", max_order_usdt=50.0, client_factory=factory)
    with pytest.raises(BrokerError):
        b.market_order("BTC/USDT", "buy", 51.0, "c1", ref_price=100.0)
    # reduce_only: pozisyon kapatmak riski azaltır, tavan onu engellemez
    o = b.market_order("BTC/USDT", "sell", 500.0, "c2", ref_price=100.0,
                       reduce_only=True, amount=5.0)
    assert o["amount"] == 5.0


def test_yinelenen_emir_kimligi_reddedilir():
    b = Broker("binance", "paper", client_factory=factory)
    b.market_order("BTC/USDT", "buy", 20.0, "same", ref_price=100.0)
    with pytest.raises(BrokerError):
        b.market_order("BTC/USDT", "buy", 20.0, "same", ref_price=100.0)


def test_asgari_notional_alti_emir_yok():
    b = Broker("binance", "paper", client_factory=factory)
    with pytest.raises(BrokerError):
        b.market_order("BTC/USDT", "buy", 1.0, "c", ref_price=100.0)


def test_client_order_id_deterministik_ve_kisa():
    a = make_client_order_id(1, "binance", "BTC/USDT", 7, "buy")
    assert a == make_client_order_id(1, "binance", "BTC/USDT", 7, "buy")
    assert a != make_client_order_id(1, "binance", "BTC/USDT", 8, "buy")
    assert len(a) <= 32 and a.isalnum()


def test_testnet_anahtarsiz_kurulamaz():
    with pytest.raises(BrokerError):
        Broker("binance", "testnet", client_factory=factory)


def test_testnet_sandbox_acilir_ve_gercek_emir_gider():
    b = Broker("binance", "testnet", creds={"apiKey": "k", "secret": "s"},
               max_order_usdt=100.0, client_factory=factory)
    before = FakeExchange.sandbox_calls
    o = b.market_order("BTC/USDT", "buy", 50.0, "cid", ref_price=100.0)
    assert FakeExchange.sandbox_calls == before + 1
    assert o["mode"] == "testnet" and FakeExchange.orders[-1]["clientOrderId"] == "cid"


def test_borsa_reddederse_kimlik_serbest_kalir():
    b = Broker("binance", "testnet", creds={"apiKey": "k", "secret": "s"},
               client_factory=factory)
    FakeExchange.fail_order = True
    with pytest.raises(BrokerError):
        b.market_order("BTC/USDT", "buy", 50.0, "cid", ref_price=100.0)
    FakeExchange.fail_order = False
    assert b.market_order("BTC/USDT", "buy", 50.0, "cid", ref_price=100.0)["ok"]


def test_broker_close_anahtari_dusurur():
    b = Broker("binance", "live", creds={"apiKey": "k", "secret": "s"}, client_factory=factory)
    b.close()
    assert b._creds == {} and b._private is None


# ═══════════════════════════ karar zinciri ═══════════════════════════
def _inp(**kw):
    d = dict(symbol="BTC/USDT", direction="LONG", entry=100.0, target_gross_pct=1.6,
             stop_pct=1.0, cost_pct=0.24, notional=100.0,
             system_health={"overall": "GREEN"}, bid_depth_usd=1e6, ask_depth_usd=1e6)
    d.update(kw)
    return DC.ChainInputs(**d)


def test_saglik_red_veto():
    d = DC.decide(_inp(system_health={"overall": "RED"}), DC.ChainConfig())
    assert not d.allowed and any("SAĞLIK" in v for v in d.vetoes)


def test_saglik_bilinmiyor_veto():
    d = DC.decide(_inp(system_health=None), DC.ChainConfig())
    assert not d.allowed


def test_komisyon_kapisi_veto():
    d = DC.decide(_inp(target_gross_pct=0.3, cost_pct=0.24), DC.ChainConfig())
    assert not d.allowed and "KOMİSYON" in d.vetoes


def test_konsensus_ters_guclu_veto_ayni_yon_boost():
    cfg = DC.ChainConfig(use_qualification=False, use_regime=False)
    d = DC.decide(_inp(cm_signal={"direction": "SHORT", "confidence": 0.8}), cfg)
    assert not d.allowed
    d2 = DC.decide(_inp(cm_signal={"direction": "LONG", "confidence": 0.7}), cfg)
    assert d2.allowed and d2.size_mult == pytest.approx(cfg.consensus_boost)
    d3 = DC.decide(_inp(cm_signal={"direction": "SHORT", "confidence": 0.3}), cfg)
    assert d3.allowed and d3.size_mult == pytest.approx(cfg.flat_mult)


def test_acil_cikis_sinifi_veto():
    d = DC.decide(_inp(cm_signal={"direction": "LONG", "confidence": 0.9,
                                  "signal_class": "acil_cikis"}), DC.ChainConfig())
    assert not d.allowed


def test_nitelendirme_olcekler_kati_modda_veto():
    cfg = DC.ChainConfig(use_consensus=False, use_regime=False)
    d = DC.decide(_inp(qual_cell={"status": "NO_EDGE"}), cfg)
    assert d.allowed and d.size_mult == pytest.approx(0.5)
    d2 = DC.decide(_inp(qual_cell={"status": "QUALIFIED"}), cfg)
    assert d2.size_mult == pytest.approx(1.0)
    d3 = DC.decide(_inp(qual_cell={"status": "NO_EDGE"}),
                   DC.ChainConfig(use_consensus=False, use_regime=False,
                                  strict_qualification=True))
    assert not d3.allowed
    d4 = DC.decide(_inp(qual_cell={"status": "DEGRADED"}), cfg)
    assert not d4.allowed


def test_firsat_kapisi_net_esik_veto():
    cfg = DC.ChainConfig(use_consensus=False, use_qualification=False, use_regime=False,
                         min_gross_to_cost=1.0, min_net_return_pct=2.0)
    d = DC.decide(_inp(target_gross_pct=1.0, cost_pct=0.2), cfg)
    assert not d.allowed and any("FIRSAT" in v for v in d.vetoes)
    assert d.opportunity and d.opportunity["action"] == "NO_TRADE"


def test_firsat_kapisi_olculmemis_derinlikte_VETO_eder():
    """Eskiden derinlik ölçülmediğinde notional'ın 50 katı VARSAYILIP geçiriliyordu:
    veri kaybı, en cömert likidite varsayımına dönüşüyordu. Artık veto."""
    cfg = DC.ChainConfig(use_consensus=False, use_qualification=False, use_regime=False)
    d = DC.decide(_inp(bid_depth_usd=0.0, ask_depth_usd=0.0), cfg)
    assert not d.allowed and any("DERİNLİK" in v for v in d.vetoes)
    # ölçülmüş derinlikte kapı normal çalışır
    d2 = DC.decide(_inp(bid_depth_usd=500_000.0, ask_depth_usd=500_000.0), cfg)
    assert d2.allowed and d2.opportunity["depth_assumed"] is False
    # geriye dönük mod: veto kapatılırsa eski davranış (beyan et, geçir)
    cfg2 = DC.ChainConfig(use_consensus=False, use_qualification=False, use_regime=False,
                          veto_on_assumed_depth=False)
    d3 = DC.decide(_inp(bid_depth_usd=0.0, ask_depth_usd=0.0), cfg2)
    assert d3.allowed and d3.opportunity["depth_assumed"] is True


def test_rejim_trende_karsi_yariya_indirir():
    cfg = DC.ChainConfig(use_consensus=False, use_qualification=False,
                         use_opportunity_gates=False)
    d = DC.decide(_inp(regime={"label": "TREND AŞAĞI", "multiplier": 1.0}), cfg)
    assert d.allowed and d.size_mult == pytest.approx(0.5)


def test_zincir_izi_her_kapiyi_yazar():
    d = DC.decide(_inp(), DC.ChainConfig())
    gates = [s["gate"] for s in d.steps]
    for g in ("SAĞLIK", "KOMİSYON", "KONSENSÜS", "NİTELENDİRME", "FIRSAT", "REJİM"):
        assert g in gates


# ═══════════════════════════ koşucu ═══════════════════════════
def _ctx(health="GREEN"):
    return LR.Context(cm_signal=lambda s: None, qual_cell=lambda *a: None,
                      system_health=lambda: {"overall": health}, regime=lambda df: None)


def _runner(tmp_path, mode="paper", **cfg_over):
    cfg = LR.RunnerConfig.from_dict({
        "exchange_id": "binance", "mode": mode, "symbols": ["BTC/USDT"],
        "capital_usdt": 5000.0, "max_order_usdt": 100.0, "max_open": 2,
        "params": {"min_hold_sec": 900, "max_hold_sec": 3600},
        "chain": {"use_consensus": False, "use_qualification": False, "use_regime": False},
        **cfg_over})
    reg = LR.RunnerRegistry(output_dir=str(tmp_path), ctx=_ctx(), client_factory=factory)
    creds = {"apiKey": "k", "secret": "s"} if mode != "paper" else None
    return reg, reg.create(1, cfg, creds=creds)


def test_kosucu_dip_gorunce_paper_pozisyon_acar(tmp_path):
    reg, r = _runner(tmp_path)
    FakeExchange.path["BTC/USDT"] = _path(dip_last=12, dip_pct=3.0)
    out = r.run_cycle(now=1_000_000.0)
    assert out["open"] == 1 and "BTC/USDT" in r.positions
    pos = r.positions["BTC/USDT"]
    assert pos.direction == "LONG" and pos.notional <= 100.0 + 1e-6
    assert pos.entry_fee > 0
    assert r.last_decisions["BTC/USDT"]["result"].startswith("AÇILDI")


def test_kosucu_merdivenle_kar_alir_ve_kilitler_komisyon_dahil(tmp_path):
    """v2: hedefin üstüne çıkan koşucuda stop KÂR KİLİDİNE çekilir (tepe × retain),
    başabaşa DEĞİL. Fiyat kilide geri gelse bile pozisyon ARTIDA kapanır — v1'de aynı
    senaryo `stop = entry` ile eksiye dönebiliyordu (giriş fiyatı net başabaş değildir).
    Merdiven varsayılan KAPALI (ölçüldü, reddedildi); burada açılıp basamağı da sınanır."""
    reg, r = _runner(tmp_path, exit={**LR.XE.ExitParams().__dict__, "ladder_enabled": True})
    FakeExchange.path["BTC/USDT"] = _path(dip_last=12, dip_pct=3.0)
    r.run_cycle(now=1_000_000.0)
    pos = r.positions["BTC/USDT"]
    entry = pos.entry
    # fiyat hedefin üstüne, tutma 20 dk (> asgari 15)
    FakeExchange.path["BTC/USDT"] = _path() + [pos.target * 1.01] * 3
    r.run_cycle(now=1_000_000.0 + 20 * 60)
    pos = r.positions.get("BTC/USDT")
    assert pos is not None and pos.partial_done, "kısmi kâr alınmalı"
    assert pos.realized > 0 and pos.amount < pos.amount_initial
    assert pos.be_locked and pos.lock_price is not None
    # KİLİT NET BAŞABAŞIN ÜSTÜNDE olmalı (giriş fiyatı net başabaş değildir)
    assert (pos.lock_price - entry) * pos.sign() > 0
    assert pos.locked_net_pct > 0
    lock = float(pos.lock_price)
    # fiyat kilide geri gelirse pozisyon ARTIDA kapanır (koşucu aynı döngüde yeniden
    # girebilir — burada bakılan, KAPANAN işlemin artıda olması)
    FakeExchange.path["BTC/USDT"] = _path() + [lock * 0.999] * 3
    r.run_cycle(now=1_000_000.0 + 40 * 60)
    assert len(r.trades) == 1
    t = r.trades[0]
    assert t["net_pnl"] == pytest.approx(t["gross_pnl"] - t["fees"])
    assert t["fees"] > 0 and t["net_pnl"] > 0, "kâr kilidi ARTIDA kapatmalı"
    assert t["levels_hit"] >= 1
    st = r.stats()
    # fees_paid AÇIK pozisyonların giriş komisyonunu da içerebilir (yeniden giriş olduysa)
    assert st["closed_trades"] == 1 and st["fees_paid"] >= t["fees"] - 0.01
    assert len(r.paper_history) == 1


def test_gunluk_zarar_limiti_halt_ve_flatten(tmp_path):
    reg, r = _runner(tmp_path, capital_usdt=200.0, daily_loss_limit_pct=0.01)
    FakeExchange.path["BTC/USDT"] = _path(dip_last=12, dip_pct=3.0)
    r.run_cycle(now=1_000_000.0)
    pos = r.positions["BTC/USDT"]
    FakeExchange.path["BTC/USDT"] = _path() + [pos.stop * 0.97] * 3       # stop altı
    out = r.run_cycle(now=1_000_000.0 + 60)
    assert out["halted"] and not r.positions and r.trades[0]["reason"] == "STOP"
    # HALT: dip yeniden görünse de giriş YOK
    FakeExchange.path["BTC/USDT"] = _path(dip_last=12, dip_pct=3.0)
    r.run_cycle(now=1_000_000.0 + 120)
    assert not r.positions
    # otomatik devam yok; elle resume
    r.resume()
    assert not r.guard.state.halted


def test_saglik_red_iken_giris_yok(tmp_path):
    reg, r = _runner(tmp_path)
    r.ctx = _ctx("RED")
    FakeExchange.path["BTC/USDT"] = _path(dip_last=12, dip_pct=3.0)
    r.run_cycle(now=1_000_000.0)
    assert not r.positions and "VETO" in r.last_decisions["BTC/USDT"]["result"]


def test_manage_only_giris_yapmaz_cikisi_yonetir(tmp_path):
    reg, r = _runner(tmp_path)
    FakeExchange.path["BTC/USDT"] = _path(dip_last=12, dip_pct=3.0)
    r.run_cycle(now=1_000_000.0)
    r.manage_only = True
    pos = r.positions["BTC/USDT"]
    FakeExchange.path["BTC/USDT"] = _path() + [pos.stop * 0.99] * 3
    r.run_cycle(now=1_000_000.0 + 30)
    assert not r.positions                          # çıkış yönetildi
    FakeExchange.path["BTC/USDT"] = _path(dip_last=12, dip_pct=3.0)
    r.run_cycle(now=1_000_000.0 + 60)
    assert not r.positions                          # giriş yok


def test_gunluk_islem_tavani(tmp_path):
    reg, r = _runner(tmp_path, max_trades_per_day=1)
    FakeExchange.path["BTC/USDT"] = _path(dip_last=12, dip_pct=3.0)
    r.run_cycle(now=1_000_000.0)
    r.close_all()
    r.run_cycle(now=1_000_000.0 + 60)
    assert not r.positions and r.day_trades == 1


def test_durum_diske_yazilir_ve_geri_yuklenir(tmp_path):
    reg, r = _runner(tmp_path)
    FakeExchange.path["BTC/USDT"] = _path(dip_last=12, dip_pct=3.0)
    r.run_cycle(now=1_000_000.0)
    p = LR.state_path(str(tmp_path), 1, "binance")
    assert p.exists()
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["positions"][0]["symbol"] == "BTC/USDT"
    # yeni süreç: kayıt defteri geri yükler, pozisyon yönetilmeye devam eder
    reg2 = LR.RunnerRegistry(output_dir=str(tmp_path), ctx=_ctx(), client_factory=factory)
    reg2.restore_all()
    r2 = reg2.get(1, "binance")
    assert r2 is not None and "BTC/USDT" in r2.positions and not r2.manage_only


def test_canli_geri_yukleme_yalniz_pozisyon_yonetimi(tmp_path):
    reg, r = _runner(tmp_path, mode="live")
    r.positions["BTC/USDT"] = LR.Position("BTC/USDT", "LONG", 100.0, 99.0, 101.6, 1.0, 1.6,
                                          0.5, 50.0, 1_000_000.0, mode="live")
    r.running = True
    r.save()
    reg2 = LR.RunnerRegistry(output_dir=str(tmp_path), ctx=_ctx(), client_factory=factory,
                             creds_lookup=lambda uid, ex: {"apiKey": "k", "secret": "s"})
    out = reg2.restore_all()
    r2 = reg2.get(1, "binance")
    assert out[0]["manage_only"] is True and r2.manage_only and r2.running
    r2.stop()


def test_canli_geri_yukleme_anahtarsiz_olmaz(tmp_path):
    reg, r = _runner(tmp_path, mode="live")
    r.save()
    reg2 = LR.RunnerRegistry(output_dir=str(tmp_path), ctx=_ctx(), client_factory=factory,
                             creds_lookup=lambda uid, ex: None)
    out = reg2.restore_all()
    assert out[0]["restored"] is False and reg2.get(1, "binance") is None


def test_readiness_paper_kaniti_ister(tmp_path):
    reg, r = _runner(tmp_path, paper_proof_trades=2)
    assert not r.readiness()["ok"]
    r.paper_history = [{"net_pnl": 1.0, "fees": 0.1, "win": True},
                       {"net_pnl": 2.0, "fees": 0.1, "win": True}]
    assert r.readiness()["ok"]
    r.paper_history[1]["net_pnl"] = -5.0
    assert not r.readiness()["ok"]


def test_canli_preflight_operator_kapisi_olmadan_emir_yok(tmp_path):
    class Cfg:
        def get(self, k, d=None):
            return {"execution.mode": "paper", "execution.allow_live": False}.get(k, d)
    reg, r = _runner(tmp_path, mode="live")
    r.server_config = Cfg()
    FakeExchange.path["BTC/USDT"] = _path(dip_last=12, dip_pct=3.0)
    n = len(FakeExchange.orders)
    r.run_cycle(now=1_000_000.0)
    assert not r.positions and len(FakeExchange.orders) == n
    assert "CANLI KAPALI" in r.last_decisions["BTC/USDT"]["result"]


def test_kayit_defteri_ems_ready_ve_degistirme(tmp_path):
    reg = LR.RunnerRegistry(output_dir=str(tmp_path), ctx=_ctx(), client_factory=factory)
    assert not reg.ems_ready()
    reg.create(1, LR.RunnerConfig.from_dict({"exchange_id": "binance"}))
    assert reg.ems_ready() and not reg.live_running()
    assert reg.remove(1, "binance") and not reg.ems_ready()


def test_live_guard_kosucu_esikleri():
    g = LiveGuard(None, output_dir=".", state_file="tmp_guard_test.json",
                  daily_loss_limit=0.02, max_drawdown=0.05)
    g.state.day = ""
    g.check(1000.0)
    assert g.check(985.0)["halted"] is False
    assert g.check(979.0)["halted"] is True
    try:
        g.path.unlink()
    except Exception:
        pass


# ═══════════════════════════ anahtar kapsamı ═══════════════════════════
def test_para_cekme_acik_anahtar_her_kapsamda_ret():
    FakeExchange.withdraw = True
    for req in (False, True):
        p = V.exchange_permissions("binance", "k", "s", require_trade=req,
                                   client_factory=FakeExchange)
        assert not p["ok"] and "PARA ÇEKME" in p["reason"]


def test_okuma_kapsami_emir_izinli_anahtari_reddeder():
    FakeExchange.trade = True
    p = V.exchange_permissions("binance", "k", "s", client_factory=FakeExchange)
    assert not p["ok"] and "EMİR" in p["reason"]
    FakeExchange.trade = False
    assert V.exchange_permissions("binance", "k", "s", client_factory=FakeExchange)["ok"]


def test_islem_kapsami_emir_izni_ister():
    FakeExchange.trade = False
    p = V.exchange_permissions("binance", "k", "s", require_trade=True,
                               client_factory=FakeExchange)
    assert not p["ok"] and "EMİR izni YOK" in p["reason"]
    FakeExchange.trade = True
    p = V.exchange_permissions("binance", "k", "s", require_trade=True,
                               client_factory=FakeExchange)
    assert p["ok"] and p["scope"] == "trade" and p["withdraw_verified"]


def test_kapsam_meta_kasada_saklanir(tmp_path):
    V.create_user("t@t.com", "cok-guclu-parola-1", str(tmp_path))
    uid = V.verify_user("t@t.com", "cok-guclu-parola-1", str(tmp_path))["id"]
    V.put_secret(uid, "binance", "apiKey", "AAAA1111", str(tmp_path))
    V.put_secret(uid, "binance", "secret", "BBBB2222", str(tmp_path))
    assert V.key_scope(uid, "binance", str(tmp_path)) == "read"
    V.set_meta(uid, "binance", "scope", "trade", str(tmp_path))
    assert V.key_scope(uid, "binance", str(tmp_path)) == "trade"
    assert all(not k["field"].startswith("meta:") for k in V.list_secrets(uid, str(tmp_path)))
    assert V.exchange_creds(uid, "binance", str(tmp_path)) == {"apiKey": "AAAA1111",
                                                               "secret": "BBBB2222"}


# ═══════════════════════════ API ═══════════════════════════
@pytest.fixture
def client(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from agi_trader.server.account_api import create_account_app
    from agi_trader.server.trading_api import create_trading_router

    class Cfg:
        def __init__(self, live=False):
            self.live = live
        def get(self, k, d=None):
            return {"execution.mode": "live" if self.live else "paper",
                    "execution.allow_live": self.live}.get(k, d)

    app = FastAPI()
    acc = create_account_app(output_dir=str(tmp_path), secure_cookies=False, cookie_path="/")
    app.include_router(acc.router)
    reg = LR.RunnerRegistry(output_dir=str(tmp_path), ctx=_ctx(), client_factory=factory)
    cfgbox = {"cfg": Cfg()}
    app.include_router(create_trading_router(reg, output_dir=str(tmp_path),
                                             server_config=cfgbox["cfg"]))
    c = TestClient(app)
    r = c.post("/account/register", json={"email": "a@b.com", "password": "parola-uzun-12345"})
    assert r.status_code == 200
    r = c.post("/account/login", json={"email": "a@b.com", "password": "parola-uzun-12345"})
    csrf = r.json()["csrf"]
    return c, csrf, reg, str(tmp_path)


def _post(c, csrf, path, body):
    return c.post(path, json=body, headers={"X-CSRF-Token": csrf})


def test_api_oturumsuz_401(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from agi_trader.server.trading_api import create_trading_router
    app = FastAPI()
    reg = LR.RunnerRegistry(output_dir=str(tmp_path), ctx=_ctx(), client_factory=factory)
    app.include_router(create_trading_router(reg, output_dir=str(tmp_path)))
    c = TestClient(app)
    assert c.get("/account/trading/state").status_code == 401
    assert c.post("/account/trading/start", json={}).status_code == 401
    cat = c.get("/account/trading/catalog").json()
    assert cat["logged_in"] is False and len(cat["exchanges"]) >= 10


def test_api_csrfsiz_baslatma_403(client):
    c, csrf, reg, _ = client
    assert c.post("/account/trading/start", json={"exchange": "binance"}).status_code == 403


def test_api_paper_baslat_durum_durdur(client, monkeypatch):
    c, csrf, reg, _ = client
    r = _post(c, csrf, "/account/trading/start",
              {"exchange": "binance", "mode": "paper",
               "config": {"symbols": ["BTC/USDT"], "capital_usdt": 5000}})
    assert r.status_code == 200, r.text
    st = r.json()["state"]
    assert st["running"] and st["mode"] == "paper" and st["strategy_name"] == VS.STRATEGY_NAME
    assert c.get("/account/trading/state?exchange=binance").json()["configured"]
    cat = c.get("/account/trading/catalog").json()
    b = next(e for e in cat["exchanges"] if e["id"] == "binance")
    assert b["runner"]["running"] and cat["strategy"]["video_preset"]["rr"] == 1.6
    assert _post(c, csrf, "/account/trading/stop", {"exchange": "binance"}).status_code == 200
    assert not reg.get(1, "binance").running


def test_api_testnet_anahtarsiz_400(client):
    c, csrf, reg, _ = client
    r = _post(c, csrf, "/account/trading/start", {"exchange": "binance", "mode": "testnet"})
    assert r.status_code == 400 and "anahtar" in r.json()["error"]


_ORIG_PERM = V.exchange_permissions


def _perm(ex_id, key, sec, pwd=None, timeout_ms=15000, *, require_trade=False, client_factory=None):
    """Gerçek izin mantığı, sahte borsayla."""
    return _ORIG_PERM(ex_id, key, sec, pwd, timeout_ms, require_trade=require_trade,
                      client_factory=FakeExchange)


def test_api_islem_kapsamli_anahtar_kaydi_ve_okuma_kapsami_reddi(client, monkeypatch):
    c, csrf, reg, out = client
    monkeypatch.setattr(V, "exchange_permissions", _perm)
    # onay kutusu olmadan işlem kapsamı reddedilir
    r = _post(c, csrf, "/account/keys", {"provider": "binance", "exchange_id": "binance",
                                         "scope": "trade",
                                         "fields": {"apiKey": "KEY12345", "secret": "SEC12345"}})
    assert r.status_code == 400 and "onay" in r.json()["error"]
    r = _post(c, csrf, "/account/keys", {"provider": "binance", "exchange_id": "binance",
                                         "scope": "trade", "withdraw_disabled_ack": True,
                                         "fields": {"apiKey": "KEY12345", "secret": "SEC12345"}})
    assert r.status_code == 200 and r.json()["scope"] == "trade"
    assert "KEY12345" not in r.text and "SEC12345" not in r.text
    ks = c.get("/account/keys").json()
    assert ks["scopes"]["binance"] == "trade"
    # testnet artık başlar
    r = _post(c, csrf, "/account/trading/start", {"exchange": "binance", "mode": "testnet",
                                                  "config": {"symbols": ["BTC/USDT"]}})
    assert r.status_code == 200, r.text
    assert reg.get(1, "binance").cfg.mode == "testnet"
    reg.get(1, "binance").stop()


def test_api_canli_tum_kapilar(client, monkeypatch):
    c, csrf, reg, out = client
    monkeypatch.setattr(V, "exchange_permissions", _perm)
    _post(c, csrf, "/account/keys", {"provider": "binance", "exchange_id": "binance",
                                     "scope": "trade", "withdraw_disabled_ack": True,
                                     "fields": {"apiKey": "KEY12345", "secret": "SEC12345"}})
    r = _post(c, csrf, "/account/trading/start", {"exchange": "binance", "mode": "live"})
    assert r.status_code == 403
    bl = r.json()["blockers"]
    assert any("operatör" in b for b in bl) and any("onay cümlesi" in b for b in bl)
    assert any("paper kanıtı" in b for b in bl)
    # yanıtta anahtar yok
    assert "KEY12345" not in r.text
    # okuma kapsamlı anahtarla canlı/testnet hiç açılmaz
    _post(c, csrf, "/account/keys", {"provider": "bybit", "exchange_id": "bybit",
                                     "scope": "read",
                                     "fields": {"apiKey": "RKEY1234", "secret": "RSEC1234"}})
    FakeExchange.trade = False
    r = _post(c, csrf, "/account/trading/start", {"exchange": "bybit", "mode": "testnet"})
    assert r.status_code == 403 and "SADECE-OKUMA" in r.json()["error"]


def test_api_mod_degisimi_acik_pozisyonla_409(client):
    c, csrf, reg, out = client
    _post(c, csrf, "/account/trading/start", {"exchange": "binance", "mode": "paper",
                                              "config": {"symbols": ["BTC/USDT"]}})
    r = reg.get(1, "binance")
    r.stop()
    FakeExchange.path["BTC/USDT"] = _path(dip_last=12, dip_pct=3.0)
    r.run_cycle(now=1_000_000.0)
    assert r.positions
    resp = _post(c, csrf, "/account/trading/start", {"exchange": "binance", "mode": "testnet"})
    assert resp.status_code in (400, 409)          # anahtar yok (400) ya da açık pozisyon (409)
    resp = _post(c, csrf, "/account/trading/remove", {"exchange": "binance"})
    assert resp.status_code == 409
    assert _post(c, csrf, "/account/trading/close_all", {"exchange": "binance"}).status_code == 200
    assert _post(c, csrf, "/account/trading/remove", {"exchange": "binance"}).status_code == 200
    assert reg.get(1, "binance") is None


def test_api_params_guncelle(client):
    c, csrf, reg, out = client
    _post(c, csrf, "/account/trading/start", {"exchange": "binance", "mode": "paper"})
    r = _post(c, csrf, "/account/trading/params",
              {"exchange": "binance", "params": {"rr": 2.0}, "chain": {"strict_qualification": True}})
    assert r.status_code == 200
    st = r.json()["state"]["config"]
    assert st["params"]["rr"] == 2.0 and st["chain"]["strict_qualification"] is True
    reg.get(1, "binance").stop()
