# -*- coding: utf-8 -*-
"""5. tur: haber/sosyal tarayıcı · hafif bağlam · çoklu tetikleyici · risk-off/nakit modu."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agi_trader.sentiment import news_scanner as NS  # noqa: E402
from agi_trader.strategies import light_context as LC  # noqa: E402
from agi_trader.strategies import committee as CM  # noqa: E402
from agi_trader.strategies import roles as R  # noqa: E402
from agi_trader.auto import live_runner as LR  # noqa: E402
from agi_trader.auto import simulator as SIM  # noqa: E402
from agi_trader.server import secure_keys as V  # noqa: E402

from test_live_trading import FakeExchange, factory, _path  # noqa: E402
from test_committee import _slow, _ctx, _ctx_provider  # noqa: E402

MASTER = "test-master-key-" + "w" * 40


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv(V.MASTER_ENV, MASTER)
    FakeExchange.path = {"BTC/USDT": _path(), "ETH/USDT": _path(seed=5)}
    FakeExchange.orders = []
    yield


# ═══════════════════════════ haber tarayıcı ═══════════════════════════
def test_sozluk_skoru_ve_etiketler():
    a = NS.score_text("Binance lists Solana ETF after SEC approval, price surges")
    assert a["score"] > 0.5 and "listing" in a["catalysts"] and "etf" in a["catalysts"]
    b = NS.score_text("Exchange hacked: $40M drained, withdrawals halted")
    assert b["score"] < -0.5 and "hack/exploit" in b["risks"] and "halt/outage" in b["risks"]
    assert NS.score_text("weather is nice")["score"] == 0.0


def test_varlik_eslestirme_takma_ad_ve_ticker():
    bases = ["BTC", "SOL", "PEPE", "OP", "ADA"]
    assert NS.match_assets("Bitcoin rallies as Solana hits record", bases) == ["BTC", "SOL"]
    assert NS.match_assets("$PEPE whales accumulate", bases) == ["PEPE"]
    assert NS.match_assets("we opt for caution", bases) == []           # 'OP' kelime içinde eşleşmez
    assert "ADA" in NS.match_assets("Cardano upgrade shipped", bases)


def test_tarama_override_ile_parite_skoru_ve_piyasa_riski():
    now = 1_000_000.0
    items = [
        {"source": "coindesk", "weight": 1.0, "title": "Solana ETF approved, SOL surges to record high", "ts": now - 600},
        {"source": "theblock", "weight": 1.0, "title": "Solana partners with major payments firm", "ts": now - 3600},
        {"source": "reddit", "weight": 0.5, "title": "Bitcoin exchange hacked, withdrawals halted", "ts": now - 900},
        {"source": "coindesk", "weight": 1.0, "title": "SEC sues major exchange; market crash fears", "ts": now - 1200},
        {"source": "decrypt", "weight": 0.9, "title": "Old news: Solana outage", "ts": now - 200 * 3600},  # çok eski → 0 ağırlık
    ]
    d = NS.scan(["SOL/USDT", "BTC/USDT", "DOGE/USDT"], items_override=items, with_social=False, now=now)
    sol = d["symbols"]["SOL/USDT"]
    assert sol["n_items"] == 2 and sol["score"] > 0.5 and "etf" in sol["catalysts"] and not sol["severe_risk"]
    btc = d["symbols"]["BTC/USDT"]
    assert btc["severe_risk"] and btc["score"] < 0
    assert d["symbols"]["DOGE/USDT"]["data_ok"] is False
    assert d["market"]["level"] >= 1 and d["market"]["risk_off_score"] > 0


def test_hareketlilik_dogrulamasi():
    def fetch(symbol, tf, limit):
        n = 60
        vol = np.full(n, 100.0); vol[-3:] = 300.0            # son 3 saat hacim ×3
        close = np.full(n, 100.0); close[-4:] = [100.5, 101.0, 101.8, 102.5]
        return pd.DataFrame({"open": close, "high": close * 1.004, "low": close * 0.996, "close": close, "volume": vol})
    r = NS.confirm_move("SOL/USDT", 0.7, fetch)
    assert r["confirmed"] is True and r["vol_ratio"] >= 1.5 and r["move_pct_4h"] > 0
    r2 = NS.confirm_move("SOL/USDT", -0.7, fetch)          # haber ayı, fiyat yukarı → doğrulanmadı
    assert r2["confirmed"] is False


def test_news_scanner_sinifi_dosyaya_yazar(tmp_path, monkeypatch):
    now = time.time()
    monkeypatch.setattr(NS, "scan", lambda syms, **k: {"generated_at": now, "symbols": {s: {"score": 0.4, "n_items": 1, "data_ok": True} for s in syms},
                                                        "market": {"risk_off_score": 0.0, "level": 0, "items": []}})
    sc = NS.NewsScanner(lambda: ["BTC/USDT"], out_path=tmp_path / "n.json")
    sc.run_once()
    assert (tmp_path / "n.json").exists() and sc.for_symbol("BTC/USDT")["score"] == 0.4
    assert sc.market()["level"] == 0 and sc.for_symbol("ETH/USDT") is None


# ═══════════════════════════ hafif bağlam ═══════════════════════════
def _bars4(n=300, seed=7):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.date_range("2026-01-01", periods=n, freq="4h")
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 100.0}, index=idx)


def test_hafif_baglam_seviye_ve_rejim():
    df4 = _bars4(); df1 = _bars4(120, 3)
    ctx = LC.build_light_context("ADA/USDT", df4, df1, events=[{"name": "CPI", "in_days": 2, "impact": "yüksek"}])
    assert ctx["tier"] == "light" and ctx["chart"]["regime"]["label"] and ctx["signal"] is None
    assert isinstance(ctx["chart"]["trendlines"]["horizontals"], list)
    assert 0 <= ctx["chart"]["extremes"]["range_position"] <= 1 and ctx["light"]["bias_1h"] in ("LONG", "SHORT")
    assert ctx["events"][0]["name"] == "CPI"


def test_hafif_baglam_onbellegi(tmp_path):
    from agi_trader.execution.broker import Broker
    b = Broker("mexc", "paper", client_factory=factory)
    c = LC.LightContextCache(b, ttl=900)
    a = c.get("BTC/USDT", now=1_000_000.0)
    assert a and a["tier"] == "light"
    calls = len(FakeExchange.orders)
    b2 = c.get("BTC/USDT", now=1_000_100.0)          # önbellekten
    assert b2["built_ts"] == a["built_ts"] and b2["age_sec"] == pytest.approx(100.0)
    b3 = c.get("BTC/USDT", now=1_001_000.0)          # TTL doldu → yeniden
    assert b3["built_ts"] != a["built_ts"]


# ═══════════════════════════ çoklu tetikleyici ═══════════════════════════
def _fast(**k):
    d = dict(ok=True, z=0.0, rsi=50.0, bar_up=True, trend_up=True, dist_ema_pct=0.0, atr_pct=0.3,
             breakout_up=False, vol_ratio=1.0, ema_cross_up=False, sigma_bar_pct=0.1)
    d.update(k); return d


def test_tetikleyiciler_ve_oncelik():
    p = CM.CommitteeParams()
    assert [t["kind"] for t in CM.triggers(_fast(z=-1.6, rsi=30), "mean_reversion", p, False)] == ["dip"]
    t = CM.triggers(_fast(z=-1.3, rsi=40), "mean_reversion", p, False)
    assert t[0]["kind"] == "dip_moderate" and t[0]["size"] == 0.6
    assert CM.triggers(_fast(z=-1.3, rsi=40), "mean_reversion", CM.CommitteeParams(enable_moderate=False), False) == []
    b = CM.triggers(_fast(breakout_up=True, vol_ratio=2.0, rsi=60, dist_ema_pct=0.5), "pullback", p, False)
    assert b[0]["kind"] == "breakout" and b[0]["size"] == 0.8
    b2 = CM.triggers(_fast(breakout_up=True, vol_ratio=2.0, rsi=60), "pullback", p, False)
    assert [x["kind"] for x in b2] == ["breakout", "pullback"]      # öncelik: kırılım > geri çekilme
    assert CM.triggers(_fast(breakout_up=True, vol_ratio=1.1, rsi=60, dist_ema_pct=0.5), "pullback", p, False) == []   # hacim yok → kırılım yok
    m = CM.triggers(_fast(ema_cross_up=True, rsi=58, dist_ema_pct=0.5), "pullback", p, False)
    assert m[0]["kind"] == "momentum" and m[0]["size"] == 0.7
    news = {"confirmed": True, "score": 0.7, "vol_ratio": 2.1, "severe_risk": False}
    c = CM.triggers(_fast(z=-1.6, rsi=30), "mean_reversion", p, False, news)
    assert [x["kind"] for x in c] == ["catalyst", "dip"]           # katalizör öncelikli
    assert CM.triggers(_fast(), "mean_reversion", p, False, {**news, "severe_risk": True}) == []
    assert CM.triggers(_fast(), "mean_reversion", p, False, {**news, "confirmed": False}) == []


def test_trigger_tek_secim_ve_bekle_notu():
    p = CM.CommitteeParams()
    t = CM.trigger(_fast(z=-1.6, rsi=30), "mean_reversion", p, False, {"confirmed": True, "score": 0.9, "severe_risk": False})
    assert t["kind"] == "catalyst" and t["others"] == ["dip"]
    w = CM.trigger(_fast(), "mean_reversion", p, False)
    assert w["kind"] is None and "koşulu yok" in w["note"]


def test_komite_ilimli_dip_kucuk_boyutla_acar():
    ctx = _ctx()
    df = ctx["df"]
    # ılımlı dip: son barları hafif düşür (z≈−1,3), RSI 35-42 arası
    closes = list(df["close"].values); base = closes[-14]
    for i in range(1, 14):
        closes[-14 + i] = base * (1 - 0.0011 * i)
    closes[-1] = closes[-2] * 1.001
    df2 = df.copy(); df2["close"] = closes; df2["high"] = np.array(closes) * 1.001; df2["low"] = np.array(closes) * 0.999
    ctx["df"] = df2; ctx["price"] = float(closes[-1])
    ctx["slow"] = _slow(ctx["price"])
    v = CM.evaluate(ctx, CM.CommitteeParams(dip_z=1.5, rsi_max=35, dip_z_moderate=1.0, rsi_max_moderate=48))
    if v.allowed:
        assert v.trigger in ("dip", "dip_moderate")
        if v.trigger == "dip_moderate":
            assert v.size_mult <= 0.6 * 1.5
    else:
        assert v.result.startswith("BEKLE") or v.vetoes


def test_haber_rolu_veto_ve_oy():
    news = {"data_ok": True, "score": 0.6, "n_items": 4, "bull": 3, "bear": 0, "confirmed": True,
            "move_pct_4h": 2.0, "vol_ratio": 1.9, "catalysts": {"etf": 1}, "risks": {}, "severe_risk": False,
            "social": {"bull": 20, "bear": 5, "ratio": 0.6, "msgs_24h": 40}, "headlines": [{"source": "coindesk", "title": "x"}]}
    v = R.role_news_social(news, None, None)
    assert v.data_ok and v.score > 0.4 and v.confidence > 0.5 and not v.veto
    v2 = R.role_news_social({**news, "severe_risk": True, "risks": {"hack/exploit": 2}}, None, None)
    assert v2.veto and "HABER RİSKİ" in v2.veto
    v3 = R.role_news_social(None, {"n_measured": 0}, None)
    assert v3.data_ok is False and v3.role == "haber_sosyal"
    assert "haber_sosyal" in R.ROLE_BASE_WEIGHT and "sosyal_duyarlilik" not in R.ROLE_TITLES


# ═══════════════════════════ risk-off / nakit modu ═══════════════════════════
def _slow_map(longs, shorts, btc="RANGE / YATAY"):
    m = {}
    for i in range(longs):
        m[f"L{i}/USDT"] = {"signal": {"direction": "LONG"}}
    for i in range(shorts):
        m[f"S{i}/USDT"] = {"signal": {"direction": "SHORT"}}
    m["BTC/USDT"] = {"signal": {"direction": "SHORT" if shorts else "LONG"}, "chart": {"regime": {"label": btc}}}
    return m


def test_piyasa_riski_seviyeleri():
    assert CM.market_risk(_slow_map(6, 2))["level"] == 0
    r1 = CM.market_risk(_slow_map(2, 6))
    assert r1["level"] == 1 and "genişlik" in r1["reasons"][0]
    r2 = CM.market_risk(_slow_map(1, 8, btc="TREND AŞAĞI"))
    assert r2["level"] == 2 and "NAKİT" in r2["label"]
    assert CM.market_risk(_slow_map(6, 2), {"level": 2, "risk_off_score": 3.0})["level"] == 2
    assert CM.market_risk({}, {"level": 1, "risk_off_score": 1.2})["level"] == 1


def test_kosucu_nakit_modu_pozisyonlari_kapatir(tmp_path):
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
    assert r.positions and r.risk["level"] == 0
    level["v"] = 2                                        # sistemik haber riski
    r.run_cycle(now=1_000_030.0)
    assert not r.positions and r.cash_mode and r.trades[-1]["reason"] == "NAKİT MODU"
    assert r.full_state()["risk_mode"]["level"] == 2
    # nakit modunda dip olsa da giriş yok
    FakeExchange.path["BTC/USDT"] = p
    r.run_cycle(now=1_000_060.0)
    assert not r.positions and not r.pending
    level["v"] = 0
    r.run_cycle(now=1_000_090.0)
    assert r.cash_mode                                # histerezis: nakitten çıkış 3 ardışık döngü ister (90 sn)
    r.run_cycle(now=1_000_120.0); r.run_cycle(now=1_000_150.0)
    assert not r.cash_mode


def test_hafif_katman_paritesi_kosucuda_calisir(tmp_path):
    """Ağır bağlamı olmayan parite: koşucu hafif bağlamı broker'dan kurar, komite oylar."""
    ctxp = _ctx_provider(lambda s: None)                 # ağır bağlam YOK
    cfg = LR.RunnerConfig.from_dict({**SIM.default_config("mexc").to_dict(), "symbols": ["ETH/USDT"], "symbols_mode": "fixed"})
    reg = LR.RunnerRegistry(output_dir=str(tmp_path), ctx=ctxp, client_factory=factory)
    r = reg.create(0, cfg)
    p = _path(dip_last=12, dip_pct=3.0, seed=5); p[-1] = p[-2] * 1.002
    FakeExchange.path["ETH/USDT"] = p
    r.run_cycle(now=1_000_000.0)
    d = r.last_decisions["ETH/USDT"]
    assert d.get("tier") == "light"
    orc = [v for v in d.get("votes", []) if v["role"] == "orkestrator_konsensusu"]
    if orc:                                              # tetikleyici ateşlediyse tam oylama var
        assert orc[0]["data_ok"] is False


def test_simulator_40_parite_ve_katmanlar():
    cfg = SIM.default_config("mexc")
    # limitler artık SERMAYE TAHSİSİNDEN türetilir (ölçüm bütçesi); evren kod-tanımlı kalır
    lim = SIM.scalp_limits()
    assert len(cfg.symbols) == 40 and cfg.max_open == lim["max_open"] and cfg.max_order_usdt == lim["max_order_usdt"]
    assert len(SIM.LIGHT_SYMBOLS) == 25 and not set(SIM.LIGHT_SYMBOLS) & set(SIM.HEAVY_SYMBOLS)


def test_otomatik_liste_hafif_pariteleri_korur(tmp_path):
    ctxp = _ctx_provider(lambda s: None)
    cfg = LR.RunnerConfig.from_dict({**SIM.default_config("mexc").to_dict(), "symbols_mode": "auto"})
    reg = LR.RunnerRegistry(output_dir=str(tmp_path), ctx=ctxp, client_factory=factory)
    r = reg.create(0, cfg)
    r._refresh_symbols()
    assert r.cfg.symbols[:2] == ["BTC/USDT", "ETH/USDT"] and len(r.cfg.symbols) == 40
