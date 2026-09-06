# -*- coding: utf-8 -*-
"""
KÂR KORUMA v2 — kâr kilidi (ratchet), kâr merdiveni (T1…T6), asgari tutma
asimetrisi ve yeniden giriş kapısı.

Bu testler 2026-09-06 canlı ölçümünden (200 işlem) çıkan üç kusurun REGRESYONUDUR:
  • BE_LOCK kovası: 38 işlem, tepe net %0,311 → gerçekleşen %0,0115 (PCR 0,042)
  • <15 dk kova: 52 işlem net −4,01 $; koruma kapısı asgari tutmanın ARKASINDAYDI
  • kısmi kâr 200 işlemin yalnız 7'sinde alınabildi
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agi_trader.strategies import exit_engine as XE  # noqa: E402
from agi_trader.strategies import reentry as RE  # noqa: E402


def _t(mode=XE.PARTIAL_AND_RUN, cost=0.2, atr=0.3, stop_pct=1.0, direction="LONG"):
    s = 1.0 if direction == "LONG" else -1.0
    entry = 100.0
    return XE.PositionTrack(direction, entry, entry * (1 - s * stop_pct / 100.0),
                            entry * (1 + s * 1.6 * stop_pct / 100.0), 1000.0,
                            mode, stop_pct, cost, atr)


# ═══════════════════════ 1) KÂR KİLİDİ (ratchet) ═══════════════════════
def test_kilit_tepeyle_birlikte_YUKARI_yurur():
    """Ratchet: tepe büyüdükçe kilit de büyür; ASLA geri gitmez."""
    p = XE.ExitParams(min_hold_sec=0, retain_fraction=0.5).validated()
    t = _t()
    XE.decide_exit(t, 100.6, 100.6, 100.0, p, now=1010.0)      # tepe net %0,40 → kilit %0,20
    lock1 = t.hard_stop
    assert lock1 == pytest.approx(100.40)
    XE.decide_exit(t, 101.2, 101.2, 100.6, p, now=1020.0)      # tepe net %1,00 → kilit %0,50
    lock2 = t.hard_stop
    assert lock2 == pytest.approx(100.70) and lock2 > lock1
    XE.decide_exit(t, 100.8, 100.9, 100.75, p, now=1030.0)     # tepe düşmez → kilit düşmez
    assert t.hard_stop == pytest.approx(lock2)


def test_kilit_net_basabasin_altina_inmez():
    """Tepe küçükken bile kilit en az net başabaş olur — komisyon kadar zarar yazılmaz."""
    p = XE.ExitParams(min_hold_sec=0, retain_fraction=0.35, be_lock_cost_mult=1.0).validated()
    t = _t(cost=0.2)
    XE.decide_exit(t, 100.4, 100.4, 100.0, p, now=1010.0)      # tepe net %0,20 → ham kilit %0,07 → taban %0
    assert t.be_locked and t.hard_stop >= t.breakeven_plus() - 1e-12


def test_kilit_SHORT_yonunde_de_dogru_calisir():
    p = XE.ExitParams(min_hold_sec=0, retain_fraction=0.5).validated()
    t = _t(direction="SHORT")
    XE.decide_exit(t, 99.5, 100.0, 99.5, p, now=1010.0)        # tepe net %0,30 → kilit %0,15
    assert t.be_locked and t.hard_stop == pytest.approx(99.65)
    assert t.hard_stop < t.entry, "SHORT'ta kilit girişin ALTINDA olmalı"
    d = XE.decide_exit(t, 99.60, 99.70, 99.55, p, now=1020.0)
    assert d and d["reason"] == "GIVEBACK" and d["net_pct"] > 0


# ═══════════ 2) ASGARİ TUTMA ASİMETRİSİ (kâr korunamıyordu) ═══════════
def test_koruma_asgari_tutmayi_beklemez_takdir_bekler():
    """Ölçülen kusur: ilk 15 dk'da ZARAR kapanabiliyor, KÂR korunamıyordu."""
    p = XE.ExitParams(min_hold_sec=900, retain_fraction=0.5, ladder_enabled=False).validated()
    t = _t(mode=XE.DYNAMIC_PEAK)
    XE.decide_exit(t, 101.0, 101.0, 100.0, p, now=1060.0)      # yaş 60 sn ≪ 900; tepe net %0,80 → silahlandı
    assert t.armed
    lvl = t.giveback_level(p)
    d = XE.decide_exit(t, lvl - 0.01, lvl + 0.05, lvl - 0.02, p, now=1120.0)
    assert d is not None, "kâr koruması asgari tutmadan ÖNCE de çalışmalı"
    assert d["reason"] in ("GIVEBACK", "TRAIL")

    # buna karşılık TAKDİRE dayalı çıkış (zaman-stopu) asgari tutmadan önce çalışmaz
    t2 = _t(mode=XE.FIXED_TARGET)
    p2 = XE.ExitParams(min_hold_sec=900, time_stop_sec=100, ladder_enabled=False).validated()
    assert XE.decide_exit(t2, 100.05, 100.06, 100.0, p2, now=1200.0) is None


def test_v1_davranisi_parametreyle_geri_gelir():
    """Geri alma yolu: protect_before_min_hold=False + lock_mode=breakeven = v1."""
    p = XE.ExitParams(min_hold_sec=900, protect_before_min_hold=False, lock_mode="breakeven",
                      ladder_enabled=False).validated()
    t = _t(mode=XE.DYNAMIC_PEAK)
    XE.decide_exit(t, 101.0, 101.0, 100.0, p, now=1060.0)
    assert t.be_locked and t.hard_stop == pytest.approx(t.breakeven_plus())
    lvl = t.price_at_net(0.4)
    assert XE.decide_exit(t, lvl - 0.01, lvl + 0.05, lvl - 0.02, p, now=1120.0) is None


# ═══════════════════════ 3) KÂR MERDİVENİ (T1…T6) ═══════════════════════
def test_merdiven_basamaklari_maliyetin_ustunde_kurulur():
    p = XE.ExitParams(ladder_enabled=True, ladder_levels=4, ladder_min_cost_mult=2.0).validated()
    t = _t(stop_pct=1.0, cost=0.2)
    lad = t.build_ladder(p)
    assert 1 <= len(lad) <= 4
    for lv in lad:
        assert lv["net_pct"] >= 2.0 * 0.2 - 1e-9, "her basamak NET kârı maliyetin ≥2 katı olmalı"
    assert [lv["price"] for lv in lad] == sorted(lv["price"] for lv in lad)


def test_merdiven_maliyet_altindaki_basamaklari_ATAR():
    """Stop çok darsa 1R bile komisyonu karşılamaz — o basamak KURULMAZ."""
    p = XE.ExitParams(ladder_enabled=True, ladder_levels=6, ladder_min_cost_mult=2.0).validated()
    t = _t(stop_pct=0.25, cost=0.2)          # 1R brüt %0,25 → net %0,05 < %0,40
    lad = t.build_ladder(p)
    assert all(lv["net_pct"] >= 0.40 - 1e-9 for lv in lad)
    assert not any(lv["source"] == "1R" for lv in lad)


def test_merdiven_kismi_cikis_verir_ve_retaini_yukseltir():
    p = XE.ExitParams(min_hold_sec=900, ladder_enabled=True, ladder_levels=3, retain_fraction=0.5,
                      retain_step_per_level=0.10).validated()
    t = _t(stop_pct=1.0, cost=0.2)
    lad = t.build_ladder(p)
    assert lad
    r0 = t.retain_eff(p)
    d = XE.decide_exit(t, lad[0]["price"] + 0.01, lad[0]["price"] + 0.02, 100.0, p, now=1000.0 + 1000)
    assert d and d["reason"] == "LADDER_TP" and d["partial"] is True
    assert 0 < d["fraction"] <= 0.9 and d["level"] == 1
    assert t.retain_eff(p) == pytest.approx(r0 + 0.10), "basamak sonrası koşucu daha sıkı korunur"
    assert t.ladder[0]["hit"] and not t.ladder[1]["hit"]
    # AYNI basamak iki kez ateşlenmez
    d2 = XE.decide_exit(t, lad[0]["price"] + 0.01, lad[0]["price"] + 0.02, lad[0]["price"], p,
                        now=1000.0 + 1010)
    assert not (d2 and d2.get("level") == 1)


def test_kar_ALMA_asgari_tutmayi_bekler_kar_KORUMA_beklemez():
    """37.905 eşleştirilmiş yolda ölçülen ayrım: koruma erken çalışmalı (+0,0088 puan),
    kâr alma erken çalışmamalı (−0,0015 puan — kazananı kırpıyor)."""
    p = XE.ExitParams(min_hold_sec=900, ladder_enabled=True, ladder_levels=3, retain_fraction=0.5).validated()
    t = _t(stop_pct=1.0, cost=0.2)
    lad = t.build_ladder(p)
    # asgari tutmadan ÖNCE: basamak fiyatı görüldü ama merdiven ATEŞLENMEZ
    d = XE.decide_exit(t, lad[0]["price"] + 0.01, lad[0]["price"] + 0.02, 100.0, p, now=1060.0)
    assert not (d and d.get("reason") == "LADDER_TP")
    assert not t.ladder[0]["hit"]
    # ama aynı anda KÂR KİLİDİ kurulmuştur (koruma çalışır)
    assert t.be_locked and t.hard_stop > t.breakeven_plus()
    # istenirse merdiven de erken çalıştırılabilir (opt-in)
    p2 = XE.ExitParams(min_hold_sec=900, ladder_enabled=True, ladder_levels=3, ladder_before_min_hold=True).validated()
    t2 = _t(stop_pct=1.0, cost=0.2)
    l2 = t2.build_ladder(p2)
    d2 = XE.decide_exit(t2, l2[0]["price"] + 0.01, l2[0]["price"] + 0.02, 100.0, p2, now=1060.0)
    assert d2 and d2["reason"] == "LADDER_TP"


def test_merdiven_kapatilabilir():
    p = XE.ExitParams(min_hold_sec=0, ladder_enabled=False).validated()
    t = _t()
    assert t.build_ladder(p) == []


def test_merdiven_emir_boyutuna_uydurulur():
    """ÖLÇÜLEN KISIT: kâğıt modda asgari emir 10 $, `max_order_usdt` de 10 $ →
    hiçbir dilim asgariyi geçemez. Sessizce çalışmayan özellik yerine, merdiven
    boyuta uydurulur; sığmıyorsa KURULMAZ ve sebebi loglanır."""
    from agi_trader.auto import live_runner as LR

    class _R:                                    # yalnız _fit_ladder_to_size'ı sınamak için
        def __init__(self, mn):
            self.broker = type("B", (), {"market_rules": staticmethod(lambda s: {"min_notional": mn})})()
            self.events = []
        _log = lambda self, *a, **k: None
    lad = [{"price": 101.0, "frac": 0.25}, {"price": 102.0, "frac": 0.20},
           {"price": 103.0, "frac": 0.15}, {"price": 104.0, "frac": 0.15}]
    pos = LR.Position(symbol="BTC/USDT", direction="LONG", entry=100.0, stop=99.0, target=102.0,
                      stop_pct=1.0, target_pct=2.0, amount=0.1, notional=10.0, opened_ts=0.0)
    # 10 $ emir · 10 $ asgari → merdiven yok
    assert LR.LiveRunner._fit_ladder_to_size(_R(10.0), pos, list(lad)) == []
    # 100 $ emir · 10 $ asgari → dilimler sığar
    pos.notional = 100.0
    fit = LR.LiveRunner._fit_ladder_to_size(_R(10.0), pos, list(lad))
    assert len(fit) == 4 and all(f["frac"] * 100.0 >= 10.0 for f in fit)
    # 25 $ emir · 10 $ asgari → dilimler BİRLEŞTİRİLİR, kalan da asgarinin üstünde
    pos.notional = 25.0
    fit2 = LR.LiveRunner._fit_ladder_to_size(_R(10.0), pos, list(lad))
    assert fit2 and all(f["frac"] * 25.0 >= 10.0 - 1e-9 for f in fit2)
    assert (1.0 - sum(f["frac"] for f in fit2)) * 25.0 >= 10.0 - 1e-9, "kalan da kapatılabilir olmalı"


# ═══════════════════════ 4) YENİDEN GİRİŞ KAPISI ═══════════════════════
def _rp(**kw):
    return RE.ReentryParams(**kw).validated()


def test_salinim_maliyetin_altindaysa_giris_yok():
    """'Borsa ücretinin üstünde miktarlarda oyna' kuralı — kodda."""
    st = {}
    d = RE.decide(st, "BTC/USDT", "LONG", 1000.0, expected_swing_pct=0.25, cost_pct=0.20,
                  p=_rp(min_swing_cost_mult=2.0))
    assert not d["allowed"] and d["gate"] == "SALINIM_MALİYET"
    d2 = RE.decide(st, "BTC/USDT", "LONG", 1000.0, expected_swing_pct=0.45, cost_pct=0.20,
                   p=_rp(min_swing_cost_mult=2.0))
    assert d2["allowed"]


def test_zararla_cikilan_fikre_hemen_donulmez():
    st = {}
    p = _rp(loss_cooldown_sec=1800)
    RE.record_exit(st, "ETH/USDT", "LONG", "STOP", -0.35, 0.0, 1000.0, p)
    d = RE.decide(st, "ETH/USDT", "LONG", 1060.0, 1.0, 0.2, p)
    assert not d["allowed"] and d["gate"] == "ZARAR_SOĞUMA"
    d2 = RE.decide(st, "ETH/USDT", "LONG", 1000.0 + 1801, 1.0, 0.2, p)
    assert d2["allowed"]


def test_karla_cikilan_harekete_geri_girilir_ama_tavanla():
    st = {}
    p = _rp(cooldown_sec=300, max_reentries=2, move_window_sec=7200)
    now = 1000.0
    for i in range(2):
        RE.record_exit(st, "SOL/USDT", "LONG", "GIVEBACK", +0.12, 0.5, now + i * 600, p)
    d = RE.decide(st, "SOL/USDT", "LONG", now + 600 + 400, 1.0, 0.2, p)
    assert not d["allowed"] and d["gate"] == "YENİDEN_GİRİŞ_TAVANI"
    # soğuma dolmadan da girilmez
    RE.record_exit(st, "ADA/USDT", "LONG", "TRAIL", +0.2, 0.6, now, p)
    assert not RE.decide(st, "ADA/USDT", "LONG", now + 100, 1.0, 0.2, p)["allowed"]
    d3 = RE.decide(st, "ADA/USDT", "LONG", now + 400, 1.0, 0.2, p)
    assert d3["allowed"] and d3["reentry_count"] == 1


def test_ters_yone_donmek_icin_daha_uzun_beklenir():
    st = {}
    p = _rp(opposite_cooldown_sec=900)
    RE.record_exit(st, "BTC/USDT", "LONG", "GIVEBACK", +0.1, 0.4, 1000.0, p)
    assert not RE.decide(st, "BTC/USDT", "SHORT", 1400.0, 1.0, 0.2, p)["allowed"]
    assert RE.decide(st, "BTC/USDT", "SHORT", 1000.0 + 901, 1.0, 0.2, p)["allowed"]


def test_kapi_kapaliyken_hicbir_sey_engellemez():
    st = {}
    p = _rp(enabled=False)
    RE.record_exit(st, "BTC/USDT", "LONG", "STOP", -1.0, 0.0, 1000.0, p)
    assert RE.decide(st, "BTC/USDT", "LONG", 1001.0, 0.01, 0.5, p)["allowed"]
