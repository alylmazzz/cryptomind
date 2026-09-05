# -*- coding: utf-8 -*-
"""FAZ 8 (yürütme) + FAZ 9 (canlı koruma) birim testleri.

Bu iki modül GERÇEK PARA tutacak. Test edilmemiş bir kill-switch, olmayan bir
kill-switch'tir.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agi_trader.execution.twap import (  # noqa: E402
    plan_twap, limit_price, slippage_bps, funding_aware_delay, MIN_NOTIONAL_USDT,
)
from agi_trader.execution.tca import tca_report, record_fill, load_fills  # noqa: E402
from agi_trader.risk.live_guard import (  # noqa: E402
    LiveGuard, live_enabled, reconcile, capital_cap, preflight,
    DAILY_LOSS_LIMIT, MAX_DRAWDOWN,
)


class Cfg:
    """config.get('a.b') arayüzünü taklit eden asgari sahte konfig."""
    def __init__(self, d): self.d = d
    def get(self, path, default=None):
        node = self.d
        for k in path.split("."):
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                return default
        return node


# ─────────────────────────────── TWAP ───────────────────────────────
def test_twap_dilim_sayisi_ve_toplam_korunur():
    p = plan_twap("BTC/USDT", "BUY", qty=1.0, ref_price=60000, slices=6)
    assert len(p.slices) == 6
    assert abs(sum(s.qty for s in p.slices) - 1.0) < 1e-9


def test_twap_kucuk_emri_bolmez():
    p = plan_twap("BTC/USDT", "BUY", qty=0.0001, ref_price=60000, slices=6)
    # 6 $ toplam → minimum altı, hiç işlem yok
    assert p.slices == [] and "minimum" in p.reason


def test_twap_dilim_minimum_altina_dusmez():
    # toplam 90 $, 6 dilim → 15 $/dilim (min 25 altı) → dilim azaltılmalı
    p = plan_twap("BTC/USDT", "BUY", qty=90 / 60000, ref_price=60000, slices=6)
    assert all(s.notional >= MIN_NOTIONAL_USDT - 1e-6 for s in p.slices)
    assert len(p.slices) < 6


def test_twap_hacim_katilimi_dilim_arttirir():
    az = plan_twap("BTC/USDT", "BUY", 10, 60000, slices=6, adv_notional=1e9)
    cok = plan_twap("BTC/USDT", "BUY", 10, 60000, slices=6, adv_notional=1e6)
    assert len(cok.slices) > len(az.slices)


def test_twap_son_dilim_market():
    p = plan_twap("BTC/USDT", "SELL", 1.0, 60000, slices=4)
    assert p.slices[-1].order_type == "market"
    assert all(s.order_type == "post_only_limit" for s in p.slices[:-1])


def test_limit_price_pasif_maker():
    # pasif alış en iyi ALIŞ fiyatında olmalı (maker), asks'e geçmemeli
    assert limit_price("BUY", 100.0, 100.5, 0.0) == 100.0
    assert limit_price("SELL", 100.0, 100.5, 0.0) == 100.5


def test_slippage_isaret_yonu():
    # alışta daha PAHALIYA dolmak aleyhtedir → pozitif
    assert slippage_bps(101, 100, "BUY") > 0
    assert slippage_bps(99, 100, "BUY") < 0
    # satışta daha UCUZA dolmak aleyhtedir → pozitif
    assert slippage_bps(99, 100, "SELL") > 0


def test_funding_aware_delay():
    # long açarken funding pozitif ve uzlaşma yakın → ertele
    r = funding_aware_delay(600, 0.0005, "BUY")
    assert r["delay"] is True and r["wait_sec"] > 600
    # funding lehte → erteleme yok
    assert funding_aware_delay(600, -0.0005, "BUY")["delay"] is False
    # uzlaşma uzak → erteleme yok
    assert funding_aware_delay(7200, 0.0005, "BUY")["delay"] is False


# ─────────────────────────────── TCA ────────────────────────────────
def test_tca_bos_veri():
    assert tca_report([], assumed_cost_bps=6.0)["available"] is False


def test_tca_gercek_maliyet_yuksekse_uyarir(tmp_path):
    fills = [{"qty": 1, "fill_price": 100.5, "ref_price": 100, "order_type": "market",
              "fee": 0.04, "fill_ratio": 1.0, "slippage_bps": 50.0}] * 10
    r = tca_report(fills, assumed_cost_bps=6.0)
    assert r["available"] and r["drift_bps"] > 2
    assert "ÜSTÜNDE" in r["verdict"]


def test_tca_kayit_ve_okuma(tmp_path):
    d = str(tmp_path)
    record_fill("BTC/USDT", "BUY", 0.5, 100.0, 100.1, "post_only_limit",
                fee=0.01, output_dir=d)
    fills = load_fills(d)
    assert len(fills) == 1 and fills[0]["slippage_bps"] > 0


# ────────────────────────── ÜÇLÜ ONAY ───────────────────────────────
def test_canli_mod_varsayilan_kapali():
    cfg = Cfg({"execution": {"mode": "paper", "allow_live": False}})
    assert live_enabled(cfg)["live"] is False


def test_canli_mod_iki_anahtar_yetmez(monkeypatch):
    monkeypatch.delenv("CRYPTOMIND_LIVE_CONFIRM", raising=False)
    cfg = Cfg({"execution": {"mode": "live", "allow_live": True}})
    r = live_enabled(cfg)
    assert r["live"] is False
    assert any("CRYPTOMIND_LIVE_CONFIRM" in m for m in r["missing"])


def test_canli_mod_uc_anahtarla_acilir(monkeypatch):
    monkeypatch.setenv("CRYPTOMIND_LIVE_CONFIRM", "EVET")
    cfg = Cfg({"execution": {"mode": "live", "allow_live": True}})
    assert live_enabled(cfg)["live"] is True


# ───────────────────────── KILL-SWITCH ──────────────────────────────
def test_gunluk_zarar_limiti_durdurur(tmp_path):
    g = LiveGuard(output_dir=str(tmp_path))
    assert g.check(10000.0)["can_trade"] is True
    r = g.check(10000.0 * (1 - DAILY_LOSS_LIMIT - 0.001))
    assert r["can_trade"] is False
    assert any("Günlük zarar" in x for x in r["reasons"])


def test_drawdown_limiti_durdurur(tmp_path):
    g = LiveGuard(output_dir=str(tmp_path))
    g.check(10000.0)
    g.state.day_start_equity = 8000.0          # günlük limiti devre dışı bırak
    r = g.check(10000.0 * (1 - MAX_DRAWDOWN - 0.01))
    assert r["can_trade"] is False
    assert any("Drawdown" in x for x in r["reasons"])


def test_bayat_veri_durdurur(tmp_path):
    g = LiveGuard(output_dir=str(tmp_path))
    r = g.check(10000.0, last_data_ts=time.time() - 10800)   # 3 saat
    assert r["can_trade"] is False
    assert any("bayat" in x.lower() for x in r["reasons"])


def test_ardisik_borsa_hatasi_durdurur(tmp_path):
    g = LiveGuard(output_dir=str(tmp_path))
    for _ in range(3):
        r = g.check(10000.0, exchange_error=True)
    assert r["can_trade"] is False


def test_halt_otomatik_kalkmaz(tmp_path):
    g = LiveGuard(output_dir=str(tmp_path))
    g.check(10000.0)
    g.check(9000.0)                              # HALT
    assert g.check(10000.0)["can_trade"] is False   # düzelse bile açılmaz
    assert g.resume("test")["can_trade"] is True    # yalnız elle


def test_guard_durumu_diske_yazilir(tmp_path):
    d = str(tmp_path)
    g1 = LiveGuard(output_dir=d)
    g1.check(10000.0); g1.check(9000.0)
    g2 = LiveGuard(output_dir=d)                 # yeniden başlatma
    assert g2.state.halted is True               # HALT restart'ta KAYBOLMAZ


# ───────────────────────── MUTABAKAT ────────────────────────────────
def test_mutabakat_eslesir(tmp_path):
    r = reconcile({"BTC": 1.0}, {"BTC": 1.0}, output_dir=str(tmp_path))
    assert r["ok"] is True


def test_mutabakat_sapmayi_yakalar(tmp_path):
    r = reconcile({"BTC": 1.0}, {"BTC": 1.2}, output_dir=str(tmp_path))
    assert r["ok"] is False and "BTC" in r["mismatches"]


def test_mutabakat_eksik_pozisyonu_yakalar(tmp_path):
    r = reconcile({"BTC": 1.0, "ETH": 5.0}, {"BTC": 1.0}, output_dir=str(tmp_path))
    assert r["ok"] is False and "ETH" in r["mismatches"]


# ─────────────────────── KADEMELİ SERMAYE ───────────────────────────
def test_sermaye_baslangicta_sinirli():
    assert capital_cap(0.5)["max_capital_usdt"] == 500.0


def test_sermaye_performanssiz_artmaz():
    r = capital_cap(4.0, live_sharpe=0.3, backtest_sharpe=1.37)   # oran 0.22 < 0.6
    assert r["max_capital_usdt"] == 500.0
    assert "kademe yükseltilmedi" in r["reason"]


def test_sermaye_performansla_artar():
    r = capital_cap(4.0, live_sharpe=1.0, backtest_sharpe=1.37)   # oran 0.73 ≥ 0.6
    assert r["max_capital_usdt"] == 2000.0


# ───────────────────────── PREFLIGHT ────────────────────────────────
def test_preflight_paper_modda_emri_engeller(tmp_path):
    cfg = Cfg({"execution": {"mode": "paper", "allow_live": False}})
    g = LiveGuard(output_dir=str(tmp_path))
    r = preflight(cfg, g, 400.0, {"BTC": 1.0}, {"BTC": 1.0})
    assert r["ok"] is False
    assert any("canlı mod kapalı" in b for b in r["blockers"])


def test_preflight_hepsi_gecerse_izin_verir(tmp_path, monkeypatch):
    monkeypatch.setenv("CRYPTOMIND_LIVE_CONFIRM", "EVET")
    cfg = Cfg({"execution": {"mode": "live", "allow_live": True}})
    g = LiveGuard(output_dir=str(tmp_path))
    r = preflight(cfg, g, 400.0, {"BTC": 1.0}, {"BTC": 1.0}, last_data_ts=time.time())
    assert r["ok"] is True, r["blockers"]


def test_preflight_sermaye_tavanini_uygular(tmp_path, monkeypatch):
    monkeypatch.setenv("CRYPTOMIND_LIVE_CONFIRM", "EVET")
    cfg = Cfg({"execution": {"mode": "live", "allow_live": True}})
    g = LiveGuard(output_dir=str(tmp_path))
    r = preflight(cfg, g, 5000.0, {"BTC": 1.0}, {"BTC": 1.0}, last_data_ts=time.time())
    assert r["ok"] is False
    assert any("sermaye tavanı" in b for b in r["blockers"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
