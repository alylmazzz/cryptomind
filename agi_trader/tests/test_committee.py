# -*- coding: utf-8 -*-
"""Komite stratejisi · ücret tablosu · ders motoru · 1.000 $ simülatör.

Her rolün oyu, her vetonun gerekçesi ve her dersin kanıt eşiği burada kilitlenir.
Kanıtsız ders yazan bir motor, uydurma bilgi üreten bir motordur.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agi_trader.strategies import fees as FE  # noqa: E402
from agi_trader.strategies import roles as R  # noqa: E402
from agi_trader.strategies import committee as CM  # noqa: E402
from agi_trader.learn.lessons import LessonEngine, BOUNDS  # noqa: E402
from agi_trader.auto import live_runner as LR  # noqa: E402
from agi_trader.auto import simulator as SIM  # noqa: E402
from agi_trader.server import secure_keys as V  # noqa: E402

from test_live_trading import FakeExchange, factory, _path  # noqa: E402

pytest.importorskip("cryptography")
MASTER = "test-master-key-" + "z" * 40


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv(V.MASTER_ENV, MASTER)
    FakeExchange.path = {"BTC/USDT": _path(), "ETH/USDT": _path(seed=5)}
    FakeExchange.orders = []
    FakeExchange.fail_order = False
    yield


def _df(closes, up_last=True):
    c = np.array(closes, dtype=float)
    if up_last:
        c[-1] = c[-2] * 1.002
    idx = pd.date_range("2026-01-01", periods=len(c), freq="min")
    return pd.DataFrame({"open": c, "high": c * 1.001, "low": c * 0.999, "close": c, "volume": 1.0}, index=idx)


def _slow(price, regime="RANGE / YATAY", sig_dir="LONG", sig_conf=0.7, age=60.0, events=None):
    return {
        "signal": {"direction": sig_dir, "confidence": sig_conf, "signal_class": "zayif_al",
                   "forecast": {"buy_threshold": price * 0.998, "sell_threshold": price * 1.03},
                   "volatility": "medium", "reasons": ["test"],
                   "layer_breakdown": [{"layer": "fear_greed", "contribution": 0.02}]},
        "chart": {"levels": {"expected_high": price * 1.03, "expected_low": price * 0.98},
                  "trendlines": {"horizontals": [{"price": price * 0.997, "touches": 3, "kind": "destek"},
                                                 {"price": price * 1.028, "touches": 2, "kind": "direnç"}],
                                 "channel": False},
                  "extremes": {"range_position": 0.1, "recent_low_20": price * 0.99, "recent_high_20": price * 1.04},
                  "regime": {"label": regime, "multiplier": 0.8, "confidence": 0.7, "method": "HMM"},
                  "smc": {"swings": []}},
        "patterns": {"consensus": {"bias": "YUKARI", "score": 0.5, "n": 2, "long": 2, "short": 0},
                     "recommendation": {"available": True, "direction": "LONG", "target_pct": 2.5,
                                        "stop_pct": 1.0, "rr": 2.5, "confidence": 0.5, "verdict": "ok"}},
        "harmonics": {"patterns": []}, "candles": {"summary": {"n": 0}},
        "indicators": {"available": True, "family": {"al": 30, "sat": 10, "notr": 5, "total": 45},
                       "net": 0.44, "bias": "YUKARI"},
        "mover_pick": {"symbol": "BTCUSDT", "probability": 0.7, "base_rate": 0.5, "lift": 1.4},
        "events": events if events is not None else [{"name": "CPI", "in_days": 3.0, "impact": "yüksek"}],
        "corr": {"symbols": ["BTC/USDT", "ETH/USDT"], "matrix": [[1, 0.9], [0.9, 1]]},
        "social": {"n_measured": 0}, "age_sec": age,
    }


def _ctx(price=100.0, dip=True, slow=None, **over):
    closes = _path(dip_last=12, dip_pct=3.0) if dip else _path()
    df = _df(closes, up_last=True)
    price = float(df["close"].iloc[-1])
    d = dict(symbol="BTC/USDT", price=price, df=df, slow=_slow(price) if slow is None else slow,
             qual_cell={"status": "QUALIFIED", "p_model_live": 0.6, "base_rate": 0.5},
             book={"spread_bps": 2.0, "bid_depth_usd": 1e6, "ask_depth_usd": 1e6},
             fees={"maker_bps": 0.0, "taker_bps": 5.0}, open_positions={}, max_open=3,
             exposure_room=700.0, capital=1000.0, max_order=250.0,
             notional_fn=lambda stop_pct: min(250.0, 10.0 / (stop_pct / 100.0)),
             p_win=0.55, halted=False, paused_reason=None, daily_loss_left_pct=5.0,
             market_type="spot")
    d.update(over)
    return d


# ═══════════════════════════ ücret tablosu ═══════════════════════════
def test_en_ucuz_borsa_mexc():
    rows = FE.ranked_venues()
    assert rows[0]["exchange_id"] == "mexc" and rows[-1]["exchange_id"] == "coinbase"
    assert FE.venue_fee("mexc").maker_bps == 0.0


def test_en_ucuz_borsa_yoklama_ile_atlar():
    v = FE.cheapest_venue(["mexc", "binance"], probe=lambda ex: ex != "mexc")
    assert v["exchange_id"] == "binance" and v["skipped"] == ["mexc"] and v["probed"]
    v2 = FE.cheapest_venue(["mexc", "binance"], probe=lambda ex: False)
    assert v2["exchange_id"] == "mexc" and v2["probed"] is False


# ═══════════════════════════ roller ═══════════════════════════
def test_piyasa_yapisi_destege_yakin_pozitif():
    s = _slow(100.0)
    v = R.role_market_structure(100.0, 0.3, s["chart"], s["signal"])
    assert v.data_ok and v.score > 0 and v.levels["support"] and v.levels["resistance"]


def test_piyasa_yapisi_dirence_yakin_negatif():
    s = _slow(100.0)
    s["chart"]["trendlines"]["horizontals"] = [{"price": 100.1, "kind": "direnç"}]
    s["chart"]["extremes"]["range_position"] = 0.95
    s["signal"]["forecast"] = {}
    v = R.role_market_structure(100.0, 0.3, s["chart"], s["signal"])
    assert v.score < 0


def test_formasyon_oneriyi_kullanir_harmonigi_oylatmaz():
    s = _slow(100.0)
    s["harmonics"] = {"patterns": [{"name": "Gartley"}]}
    v = R.role_formations(s["patterns"], s["harmonics"], s["candles"])
    assert v.score > 0 and any("yön oyu YOK" in n for n in v.notes)


def test_gosterge_konsensusu_dusuk_taban_agirlik():
    assert R.ROLE_BASE_WEIGHT["gosterge_konsensusu"] < R.ROLE_BASE_WEIGHT["piyasa_yapisi"]
    v = R.role_indicator_consensus(_slow(100.0)["indicators"])
    assert v.score > 0.4


def test_rejim_trende_karsi_yariya_indirir():
    v = R.role_regime({"label": "TREND AŞAĞI", "multiplier": 1.0, "confidence": 0.8}, "medium", "LONG", 0.5)
    assert v.size_mult == pytest.approx(0.5) and getattr(v, "template") == "pullback"
    v2 = R.role_regime({"label": "RANGE / YATAY", "multiplier": 0.8, "confidence": 0.8}, "extreme", "LONG")
    assert getattr(v2, "template") == "mean_reversion" and v2.size_mult == pytest.approx(0.48)


def test_nitelendirme_degraded_veto_kati_mod():
    assert R.role_qualification({"status": "DEGRADED"}, "LONG").veto
    assert R.role_qualification({"status": "NO_EDGE"}, "LONG", strict=True).veto
    v = R.role_qualification({"status": "NO_EDGE", "p_model_live": 0.6, "base_rate": 0.5}, "LONG")
    assert not v.veto and v.size_mult == 0.5 and v.score > 0


def test_mover_yon_oyu_vermez():
    v = R.role_mover({"probability": 0.8})
    assert v.score == 0.0 and v.confidence == pytest.approx(0.8)


def test_maliyet_vetosu_ve_maker_secimi():
    v = R.role_cost_execution(0.3, 0.3, 0.1, 2.0, 1e6, 1e6, 100.0, 2.0)   # oran 1,5 < 2
    assert v.veto and "KOMİSYON" in v.veto
    v2 = R.role_cost_execution(2.0, 0.3, 0.1, 2.0, 1e6, 1e6, 100.0, 2.0)
    assert not v2.veto and getattr(v2, "order_type") == "maker"
    v3 = R.role_cost_execution(2.0, 0.3, 0.1, 40.0, 1e6, 1e6, 100.0, 2.0, max_spread_bps=15)
    assert v3.veto and "SPREAD" in v3.veto


def test_risk_korelasyon_ve_tavan():
    corr = {"symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
            "matrix": [[1, 0.9, 0.85], [0.9, 1, 0.8], [0.85, 0.8, 1]]}
    v = R.role_risk("BTC/USDT", "LONG", {"ETH/USDT": {"direction": "LONG"}}, corr, 3, 500, 100, None, 0.55, 1.6, False)
    assert not v.veto and v.size_mult == pytest.approx(0.5)
    v2 = R.role_risk("BTC/USDT", "LONG", {"ETH/USDT": {"direction": "LONG"}, "SOL/USDT": {"direction": "LONG"}},
                     corr, 5, 500, 100, None, 0.55, 1.6, False)
    assert v2.veto and "KORELASYON" in v2.veto
    assert R.role_risk("BTC/USDT", "LONG", {}, None, 3, 500, 100, None, 0.5, 1.6, True).veto == "KILL-SWITCH aktif"
    assert "DURAKLATILDI" in R.role_risk("BTC/USDT", "LONG", {}, None, 3, 500, 100, "4 zarar", 0.5, 1.6, False).veto


def test_makro_yakin_olay_veto():
    v = R.role_macro([{"name": "FOMC", "in_days": 0.05, "impact": "çok yüksek"}], None)
    assert v.veto and "FOMC" in v.veto
    v2 = R.role_macro([{"name": "CPI", "in_days": 0.3, "impact": "yüksek"}], None)
    assert not v2.veto and v2.size_mult == pytest.approx(0.5)


def test_sosyal_olculmus_hesap_yoksa_oy_yok():
    assert R.role_social({"n_measured": 0}, None).data_ok is False


def test_orkestrator_acil_cikis_veto():
    assert R.role_orchestrator({"direction": "LONG", "confidence": 0.9, "signal_class": "acil_cikis"}).veto


def test_denetci_degismezler():
    assert R.role_auditor("LONG", 100, 101, 105, 5.0, 0.2, 50, 100, 10, 1800, 1.2).veto  # stop yanlış taraf
    assert R.role_auditor("LONG", 100, 99, 105, 5.0, 0.2, 50, 100, 5000, 1800, 1.2).veto  # bayat
    assert R.role_auditor("LONG", 100, 99, 101, 1.0, 0.2, 50, 100, 10, 1800, 1.2).veto  # R/R düşük
    assert not R.role_auditor("LONG", 100, 99, 102, 2.0, 0.2, 50, 100, 10, 1800, 1.2).veto


# ═══════════════════════════ komite ═══════════════════════════
def test_komite_destekleyici_baglamda_acar_ve_fis_uretir():
    v = CM.evaluate(_ctx(), CM.CommitteeParams())
    assert v.allowed, v.result
    assert v.direction == "LONG" and v.trigger == "dip" and v.order_type == "maker"
    t = v.ticket
    for k in ("investment_usdt", "expected_profit_usdt", "max_loss_usdt", "fee_usdt", "p_win",
              "ev_usdt", "breakeven_win_rate", "daily_loss_left_pct", "exposure_room_usdt"):
        assert k in t
    assert len(v.votes) == 13 and v.plan["rr"] >= 1.2          # 12 rol + sleeve öz-oyu


def test_komite_yavas_baglam_yoksa_bekler():
    v = CM.evaluate(_ctx(slow={}), CM.CommitteeParams())
    assert not v.allowed and any("GÜVEN" in x or "OY" in x or "DENETÇİ" in x for x in v.vetoes)


def test_komite_tetikleyici_yoksa_oylama_yapmaz():
    v = CM.evaluate(_ctx(dip=False), CM.CommitteeParams())
    assert not v.allowed and v.result.startswith("BEKLE") and len(v.votes) == 1


def test_komite_pahali_borsada_komisyon_vetosu():
    v = CM.evaluate(_ctx(fees={"maker_bps": 60.0, "taker_bps": 120.0}), CM.CommitteeParams())
    assert not v.allowed and any("KOMİSYON" in x for x in v.vetoes)


def test_komite_ters_orkestrator_oy_esigi():
    s = _slow(100.0, sig_dir="SHORT", sig_conf=0.9)
    s["indicators"]["net"] = -0.6; s["indicators"]["bias"] = "AŞAĞI"
    s["patterns"]["recommendation"]["direction"] = "SHORT"; s["patterns"]["consensus"]["score"] = -0.6
    v = CM.evaluate(_ctx(slow=s), CM.CommitteeParams())
    assert not v.allowed and v.score < 0


def test_komite_bayat_baglam_denetci_vetosu():
    v = CM.evaluate(_ctx(slow=_slow(100.0, age=7200)), CM.CommitteeParams(max_ctx_age_sec=1800))
    assert not v.allowed and any("DENETÇİ" in x for x in v.vetoes)


def test_komite_duraklatilmis_parite():
    v = CM.evaluate(_ctx(paused_reason="4 ardışık zarar"), CM.CommitteeParams())
    assert not v.allowed and any("DURAKLATILDI" in x for x in v.vetoes)


def test_komite_ogrenilmis_guvenilirlik_agirligi_degistirir():
    assert CM._weight("gosterge_konsensusu", {"reliability": {"gosterge_konsensusu": 0.9}}) > \
        CM._weight("gosterge_konsensusu", {})
    assert CM._weight("piyasa_yapisi", {"reliability": {"piyasa_yapisi": 0.1}}) < R.ROLE_BASE_WEIGHT["piyasa_yapisi"]


def test_plan_yapisal_hedef_cok_yakinsa_gecersiz():
    fast = {"sigma_bar_pct": 0.1, "atr_pct": 0.2, "swing_low": 99.0, "swing_high": 100.2}
    plan = CM.build_plan("LONG", 100.0, fast, {"support": 99.5, "resistance": 100.15}, CM.CommitteeParams())
    # 7. tur: yakın yapısal seviye artık VETO değil (canlı gölge 14 hedef / 0 stop) → rr hedefi + kısmi TP + boyut ×0,7
    assert plan["invalid"] is None and plan["size_penalty"] == 0.7 and plan["partial_tp_near"] == pytest.approx(100.15)
    assert plan["target_pct"] >= 1.2 * plan["stop_pct"] and "kısmi TP" in plan["target_source"]
    plan2 = CM.build_plan("LONG", 100.0, fast, {"support": 99.5, "resistance": 103.0}, CM.CommitteeParams())
    assert not plan2["invalid"] and plan2["rr"] >= 1.2 and plan2["partial_tp"] > 100.0


# ═══════════════════════════ ders motoru ═══════════════════════════
def _trade(win, net, reason="TP", bucket="15-60 dk", fees=0.2, gross=None, sym="BTC/USDT", opened=1_000_000.0):
    return {"symbol": sym, "direction": "LONG", "win": win, "net_pnl": net,
            "gross_pnl": (net + fees) if gross is None else gross, "fees": fees, "reason": reason,
            "hold_bucket": bucket, "opened_ts": opened, "entry": 100.0, "exit": 101.0,
            "target": 102.0, "horizon_sec": 3600, "order_type": "maker"}


def _verdict(score_ms=0.6, score_ind=-0.5):
    return {"votes": [
        {"role": "piyasa_yapisi", "score": score_ms, "confidence": 0.8, "data_ok": True, "notes": ["destek"]},
        {"role": "gosterge_konsensusu", "score": score_ind, "confidence": 0.6, "data_ok": True, "notes": ["x"]},
        {"role": "rejim_oynaklik", "score": 0.0, "confidence": 0.7, "data_ok": True, "notes": ["RANGE / YATAY (HMM)"]},
        {"role": "denetci", "score": 0, "confidence": 1, "data_ok": True, "notes": []},
    ], "trigger": "dip", "template": "mean_reversion"}


def test_rol_guvenilirligi_kazanan_yonu_odullendirir(tmp_path):
    e = LessonEngine(tmp_path / "l.json")
    for i in range(5):
        e.on_trade_closed(_trade(True, 1.0), _verdict(), now=1_000_000.0 + i)
    # Beta(1,1) öncülü büzer: 5 uyumlu oy → ~0,77; 5 aykırı oy → ~0,29
    assert e.reliability("piyasa_yapisi") > 0.7 and e.reliability("gosterge_konsensusu") < 0.3
    assert "denetci" not in e.role_stats
    assert e.learned()["reliability"]["piyasa_yapisi"] > 0.7


def test_ders_erken_cikis_asgari_tutmayi_artirir(tmp_path):
    e = LessonEngine(tmp_path / "l.json", journal_md=tmp_path / "G.md", journal_jsonl=tmp_path / "j.jsonl")
    for i in range(8):
        e.on_trade_closed(_trade(False, -1.0, reason="STOP", bucket="0-15 dk"), _verdict(),
                          now=1_000_000.0 + i, current={"min_hold_sec": 600})
    assert e.overrides.get("min_hold_sec") == 900
    md = (tmp_path / "G.md").read_text(encoding="utf-8")
    assert "DERS" in md and "Nasıl kaybedilmezdi" in md
    lines = (tmp_path / "j.jsonl").read_text(encoding="utf-8").splitlines()
    assert any('"type": "lesson"' in l for l in lines)


def test_ders_soguma_ve_sinir(tmp_path):
    e = LessonEngine(tmp_path / "l.json")
    for i in range(8):
        e.on_trade_closed(_trade(False, -1.0, reason="STOP", bucket="0-15 dk"), _verdict(),
                          now=1_000_000.0 + i, current={"min_hold_sec": 3500})
    assert e.overrides["min_hold_sec"] == BOUNDS["min_hold_sec"][1]      # üst sınır
    n = len(e.lessons)
    for i in range(8):                                                      # aynı gün: tekrar yok
        e.on_trade_closed(_trade(False, -1.0, reason="STOP", bucket="0-15 dk"), _verdict(),
                          now=1_000_100.0 + i, current={"min_hold_sec": 3600})
    assert len([l for l in e.lessons if l.get("action")]) == len([l for l in e.lessons[:n] if l.get("action")])


def test_ders_komisyon_payi(tmp_path):
    e = LessonEngine(tmp_path / "l.json")
    for i in range(8):
        e.on_trade_closed(_trade(True, 0.2, fees=0.5, gross=0.7), _verdict(), now=1_000_000.0 + i,
                          current={"min_gross_to_cost": 2.0})
    assert e.overrides.get("min_gross_to_cost") == 2.5


def test_parite_serisi_duraklatir(tmp_path):
    e = LessonEngine(tmp_path / "l.json")
    for i in range(4):
        e.on_trade_closed(_trade(False, -1.0, reason="STOP", sym="DOGE/USDT"), _verdict(), now=1_000_000.0 + i)
    assert e.paused_reason("DOGE/USDT", now=1_000_010.0)
    assert e.paused_reason("DOGE/USDT", now=1_000_000.0 + 25 * 3600) is None


def test_veto_karsi_olgusal_esigi_gevsetir(tmp_path):
    e = LessonEngine(tmp_path / "l.json")
    for i in range(10):
        vd = {"symbol": f"S{i}/USDT", "direction": "LONG", "trigger": "dip",
              "plan": {"entry": 100.0, "target": 102.0, "stop": 99.0},
              "vetoes": [f"OY +0.10 < eşik 0.25"]}
        e.on_candidate_vetoed(vd, 3600, now=1_000_000.0)
    bars = {f"S{i}/USDT": pd.DataFrame({"high": [103.0], "low": [99.5], "close": [102.5]}) for i in range(10)}
    res = e.update_shadows(bars, now=1_000_100.0)
    assert len(res) == 10 and all(r["outcome"] == "TARGET" for r in res)
    assert e.veto_stats["OY"]["would_win"] == 10
    e.derive(now=1_000_200.0, current={"theta": 0.25})
    assert e.overrides.get("theta") == pytest.approx(0.20)
    assert any("gevşetildi" in l["title"] for l in e.lessons)


def test_veto_hakliysa_not_dusulur(tmp_path):
    e = LessonEngine(tmp_path / "l.json")
    for i in range(8):
        e.on_candidate_vetoed({"symbol": f"S{i}/USDT", "direction": "LONG", "trigger": "dip",
                               "plan": {"entry": 100.0, "target": 102.0, "stop": 99.0},
                               "vetoes": ["KONSENSÜS ters SHORT %80"]}, 3600, now=1_000_000.0)
    bars = {f"S{i}/USDT": pd.DataFrame({"high": [100.5], "low": [98.0], "close": [98.5]}) for i in range(8)}
    e.update_shadows(bars, now=1_000_100.0)
    assert any("haklı" in l["title"] for l in e.lessons)


def test_post_stop_golgesi_stopu_genisletir(tmp_path):
    e = LessonEngine(tmp_path / "l.json")
    for i in range(8):
        e.on_trade_closed(_trade(False, -1.0, reason="STOP", sym=f"P{i}/USDT"), _verdict(),
                          now=1_000_000.0 + i, current={"stop_sigma_mult": 1.5})
    bars = {f"P{i}/USDT": pd.DataFrame({"high": [103.0], "low": [100.5], "close": [102.5]}) for i in range(8)}
    e.update_shadows(bars, now=1_000_100.0)
    e.derive(now=1_000_200.0, current={"stop_sigma_mult": 1.5})
    assert e.overrides.get("stop_sigma_mult") == pytest.approx(1.75)


def test_maker_dolum_orani_ogrenilir(tmp_path):
    e = LessonEngine(tmp_path / "l.json")
    for i in range(10):
        e.on_maker_attempt(i < 2)
    assert e.learned()["p_maker_fill"] == pytest.approx(0.2)
    e.derive(now=1_000_000.0, current={"chase_taker_ratio": 3.0})
    assert e.overrides.get("chase_taker_ratio") == 2.5


def test_ders_motoru_diske_yazar_ve_yukler(tmp_path):
    e = LessonEngine(tmp_path / "l.json")
    e.on_trade_closed(_trade(True, 1.0), _verdict(), now=1_000_000.0)
    e2 = LessonEngine(tmp_path / "l.json")
    assert e2.n_trades == 1 and "piyasa_yapisi" in e2.role_stats


# ═══════════════════════════ koşucu (komite) ═══════════════════════════
def _ctx_provider(slow_fn):
    return LR.Context(cm_signal=lambda s: None, qual_cell=lambda *a: {"status": "QUALIFIED", "p_model_live": 0.6},
                      system_health=lambda: {"overall": "GREEN"}, regime=lambda df: None,
                      slow_ctx=slow_fn, candidate_symbols=lambda: ["BTC/USDT", "ETH/USDT"])


def _sim_runner(tmp_path, slow_fn, **over):
    cfg = LR.RunnerConfig.from_dict({**SIM.default_config("mexc").to_dict(),
                                     "symbols": ["BTC/USDT"], "symbols_mode": "fixed", **over})
    reg = LR.RunnerRegistry(output_dir=str(tmp_path), ctx=_ctx_provider(slow_fn), client_factory=factory)
    return reg, reg.create(SIM.SYSTEM_UID, cfg)


def test_komite_kosucu_maker_bekler_dolar_ve_tp_ile_kapatir(tmp_path):
    def slow(sym):
        px = FakeExchange.path[sym][-1]
        return _slow(px, age=30.0)
    # Emir boyutu borsa asgarisinin KATI olmalı: 10 $ emir + 10 $ asgari emir kombinasyonunda
    # hiçbir kısmi/basamaklı kâr alınamaz (canlıda emir REDDEDİLİR, kâğıtta sessizce geçerdi).
    reg, r = _sim_runner(tmp_path, slow, max_order_usdt=200.0, risk_per_trade_pct=2.0,
                         capital_usdt=1000.0, max_exposure_pct=60.0,
                         params={**SIM.default_config("mexc").params, "probe_notional_usdt": 200.0})
    assert r.broker.maker_fee_bps == 0.0 and r.broker.fee_bps == 5.0
    p = _path(dip_last=12, dip_pct=3.0); p[-1] = p[-2] * 1.002       # dip + yeşil bar
    FakeExchange.path["BTC/USDT"] = p
    out = r.run_cycle(now=1_000_000.0)
    assert "BTC/USDT" in r.pending and not r.positions, r.last_decisions
    assert r.last_decisions["BTC/USDT"]["result"].startswith("MAKER BEKLİYOR")
    assert r.last_decisions["BTC/USDT"]["ticket"]["investment_usdt"] > 0
    # sonraki bar limitin içinden geçer → dolum, maker ücreti 0
    FakeExchange.path["BTC/USDT"] = p + [p[-1] * 0.998]
    r.run_cycle(now=1_000_030.0)
    assert "BTC/USDT" in r.positions and not r.pending
    pos = r.positions["BTC/USDT"]
    assert pos.order_type == "maker" and pos.entry_fee == 0.0 and pos.partial_tp > pos.entry
    assert r.lessons.maker["filled"] == 1
    # hedefin üstüne, 20 dk sonra → kısmi TP (1R); kalan KOŞAR (PARTIAL_AND_RUN): stop → giriş, tepe koruması silahlı
    FakeExchange.path["BTC/USDT"] = p + [pos.target * 1.01] * 3
    r.run_cycle(now=1_000_000.0 + 20 * 60)
    assert "BTC/USDT" in r.positions and not r.trades
    pos = r.positions["BTC/USDT"]
    assert pos.partial_done and pos.realized > 0            # kısmi kâr alındı (merdiven varsayılan KAPALI)
    # v2: kilit BAŞABAŞA değil, tepenin retain oranına — net başabaşın ÜSTÜNDE
    assert pos.locked_net_pct > 0 and pos.lock_price > pos.entry * (1 + pos.cost_pct_roundtrip / 100.0)
    assert pos.hard_stop >= pos.entry and pos.armed and pos.peak_net_pct > 1.0
    lvl = pos.track().giveback_level(r.xparams)
    assert pos.entry < lvl < pos.last_price
    # yarı-tepe geri-verme seviyesinin altına → GIVEBACK (NET tepenin ≥ %50'si korunur), hard stop asla kalkmadı
    FakeExchange.path["BTC/USDT"] = p + [pos.target * 1.01] * 3 + [lvl * 0.999] * 2
    r.run_cycle(now=1_000_000.0 + 21 * 60)
    assert not r.positions and len(r.trades) == 1
    t = r.trades[0]
    assert t["order_type"] == "maker" and t["reason"] == "GIVEBACK" and t["net_pnl"] > 0 and t["partial_done"]
    assert t["peak_capture"] is not None and t["peak_capture"] >= 0.3
    assert t["fees"] > 0                      # çıkış taker ücreti
    assert r.lessons.n_trades == 1 and "piyasa_yapisi" in r.lessons.role_stats
    st = r.full_state()
    assert st["strategy"] == "committee" and st["lessons"]["n_trades"] == 1 and st["venue"]["maker_bps"] == 0.0
    assert (Path(tmp_path) / "live" / "GUNLUK_0_mexc.md").exists()


def test_komite_kosucu_maker_dolmazsa_vazgecer(tmp_path):
    def slow(sym):
        return _slow(FakeExchange.path[sym][-1], age=30.0)
    reg, r = _sim_runner(tmp_path, slow, params={**SIM.default_config("mexc").params, "maker_wait_bars": 1,
                                                  "chase_taker_ratio": 50.0})
    p = _path(dip_last=12, dip_pct=3.0); p[-1] = p[-2] * 1.002
    FakeExchange.path["BTC/USDT"] = p
    r.run_cycle(now=1_000_000.0)
    assert "BTC/USDT" in r.pending
    FakeExchange.path["BTC/USDT"] = p + [p[-1] * 1.004] * 2        # fiyat yukarı kaçtı, low limitin üstünde
    r.run_cycle(now=1_000_030.0)
    assert not r.pending and not r.positions
    assert "MAKER DOLMADI" in r.last_decisions["BTC/USDT"]["result"]
    assert r.lessons.maker["attempts"] == 1 and r.lessons.maker["filled"] == 0


def test_komite_kosucu_veto_edilen_adayi_golgeler(tmp_path):
    def slow(sym):
        s = _slow(FakeExchange.path[sym][-1], sig_dir="SHORT", sig_conf=0.9, age=30.0)
        s["indicators"]["net"] = -0.6; s["patterns"]["consensus"]["score"] = -0.6
        s["patterns"]["recommendation"]["direction"] = "SHORT"
        return s
    reg, r = _sim_runner(tmp_path, slow)
    p = _path(dip_last=12, dip_pct=3.0); p[-1] = p[-2] * 1.002
    FakeExchange.path["BTC/USDT"] = p
    r.run_cycle(now=1_000_000.0)
    assert not r.positions and not r.pending
    assert r.last_decisions["BTC/USDT"]["result"].startswith("VETO")
    assert r.lessons.status()["shadows_open"] == 1


def test_komite_kosucu_saglik_bilinmiyorsa_giris_yok(tmp_path):
    ctxp = _ctx_provider(lambda s: _slow(FakeExchange.path[s][-1]))
    ctxp.system_health = lambda: None
    cfg = LR.RunnerConfig.from_dict({**SIM.default_config("mexc").to_dict(), "symbols": ["BTC/USDT"], "symbols_mode": "fixed"})
    reg = LR.RunnerRegistry(output_dir=str(tmp_path), ctx=ctxp, client_factory=factory)
    r = reg.create(0, cfg)
    p = _path(dip_last=12, dip_pct=3.0); p[-1] = p[-2] * 1.002
    FakeExchange.path["BTC/USDT"] = p
    r.run_cycle(now=1_000_000.0)
    assert not r.pending and "SAĞLIK" in r.last_decisions["BTC/USDT"]["result"]


def test_otomatik_parite_listesi_adaylardan(tmp_path):
    reg, r = _sim_runner(tmp_path, lambda s: None, symbols_mode="auto", symbols=["SOL/USDT"])
    r.run_cycle(now=1_000_000.0)
    assert r.cfg.symbols[:2] == ["BTC/USDT", "ETH/USDT"]


# ═══════════════════════════ simülatör ═══════════════════════════
def test_simulator_varsayilan_kurulum_1000_dolar():
    cfg = SIM.default_config("mexc")
    assert cfg.capital_usdt == 1000.0 and cfg.strategy == "committee" and cfg.mode == "paper"
    assert cfg.symbols_mode == "auto" and cfg.max_order_usdt <= 250.0 and cfg.label


def test_simulator_venue_secimi_ve_kurulum(tmp_path):
    reg = LR.RunnerRegistry(output_dir=str(tmp_path), ctx=_ctx_provider(lambda s: None), client_factory=factory)
    v = SIM.pick_venue(reg)
    assert v["exchange_id"] == "mexc" and v["probed"]
    out = SIM.ensure_simulator(reg, start=False)
    assert out["created"] and out["exchange"] == "mexc"
    s = SIM.get_simulator(reg)
    assert s is not None and s.user_id == 0 and s.cfg.capital_usdt == 1000.0
    out2 = SIM.ensure_simulator(reg, start=False)
    assert out2["created"] is False


def test_simulator_public_uclari_anahtar_tasimaz(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    reg = LR.RunnerRegistry(output_dir=str(tmp_path), ctx=_ctx_provider(lambda s: None), client_factory=factory)
    app = FastAPI(); app.include_router(SIM.create_public_router(reg))
    c = TestClient(app)
    d = c.get("/api/simulator").json()
    assert d["configured"] is False and d["venues"][0]["exchange_id"] == "mexc"
    SIM.ensure_simulator(reg, venue="mexc", start=False)
    d = c.get("/api/simulator").json()
    assert d["configured"] and d["public"] and "broker" not in d and d["strategy"] == "committee"
    assert d["config"]["capital_usdt"] == 1000.0 and len(d["strategy_card"]["roles"]) == 13
    j = c.get("/api/simulator/journal").json()
    assert j["configured"] and "lessons" in j and "roles" in j
    ro = c.get("/api/simulator/roles").json()
    assert all("effective_weight" in x for x in ro["roles"])
    assert c.post("/api/simulator").status_code == 405


def test_feed_ilk_islem_bekleniyor_ve_yakinlik(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    reg = LR.RunnerRegistry(output_dir=str(tmp_path), ctx=_ctx_provider(lambda s: _slow(FakeExchange.path[s][-1])),
                            client_factory=factory)
    app = FastAPI(); app.include_router(SIM.create_public_router(reg))
    c = TestClient(app)
    assert c.get("/api/simulator/feed").json()["status"]["state"] == "NOT_READY"
    SIM.ensure_simulator(reg, venue="mexc", start=False)
    s = SIM.get_simulator(reg); s.cfg.symbols = ["BTC/USDT"]; s.cfg.symbols_mode = "fixed"
    s.run_cycle(now=1_000_000.0)
    f = c.get("/api/simulator/feed").json()
    assert f["status"]["label"] == "İLK İŞLEM BEKLENİYOR" and f["status"]["state"] == "WAITING_FIRST"
    assert f["nearest"] and f["nearest"][0]["symbol"] == "BTC/USDT" and "missing" in f["nearest"][0]
    assert f["nearest"][0]["z"] is not None and f["nearest"][0]["rsi"] is not None
    o = f["outlook"]
    assert o["consensus_4h"]["LONG"] == 1 and o["movers"][0]["probability"] == 0.7 and "Garanti yok" in o["note"]
    assert "regimes" in o and o["p_win_committee"] == 0.5
    assert c.post("/api/simulator/feed").status_code == 405
