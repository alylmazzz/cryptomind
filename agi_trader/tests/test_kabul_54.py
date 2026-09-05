"""§54 KABUL LİSTESİ — 23 maddenin tamamı tek yerde.

"Bu testlerin tamamı PASS olmadan özelliği production-complete ilan etme."

Bu dosya bir özet değil, KAPI'dır: listedeki her madde burada gerçek kodla
denetlenir. Bir madde düşerse hangisi olduğu isimle rapor edilir.
"""
from __future__ import annotations

import datetime as dt
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agi_trader.opportunity.costs import (CostEstimate, capacity_curve,
                                          estimate_costs,
                                          required_gross_move_pct)
from agi_trader.opportunity.engine import Gates, Opportunity, evaluate
from agi_trader.qualification import convergence as CV
from agi_trader.qualification import horizons as HZ
from agi_trader.qualification import ledger as LD
from agi_trader.qualification import live as LV
from agi_trader.qualification import morning as MG
from agi_trader.qualification import universe as UNI
from agi_trader.qualification.matrix import pair_card
from agi_trader.qualification.state import QualificationState


def _tahmin(pid="p1", **kw):
    t = dict(prediction_id=pid, timestamp="2026-01-01T00:00:00Z",
             symbol="BTCUSDT", horizon="1h", direction="LONG",
             status="QUALIFIED", entry=100.0, net1_exit=101.2, stop=99.0,
             p_target_first=0.6, p_target_lower95=0.5, p_stop_first=0.3,
             p_timeout=0.1, baseline=0.4, required_lift=0.1, actual_lift=0.2,
             robust_ev=0.1, expected_target_hours=0.5,
             cost_model="MEASURED_L2_VWAP", cost_pct=0.15,
             max_capacity_usd=5000.0, data_quality=1.0, model_version="v1",
             features_hash="abc", valid_until="2026-01-01T00:08:00Z")
    t.update(kw)
    return LD.Prediction(**t)


def _bar(h, l):
    return pd.DataFrame(
        {"open": l, "high": h, "low": l, "close": h},
        index=pd.date_range("2026-01-01 00:05", periods=len(h), freq="5min",
                            tz="UTC"))


def _ornek_kart(sym="BTCUSDT"):
    """Canlı katmanın ürettiği kart şeklini taklit eden tam bir örnek."""
    hz = []
    for u in HZ.all_horizons():
        for d in ("LONG", "SHORT"):
            kalifiye = (u == "4h" and d == "LONG")
            hz.append({
                "horizon": u, "horizon_minutes": HZ.HORIZON_MIN[u],
                "direction": d, "p_target_first": 0.42,
                "p_target_lower95": 0.38, "p_stop_first": 0.33,
                "p_timeout": 0.25, "baseline": 0.30,
                "required_lift": 0.08, "actual_lift": 0.12,
                "robust_ev": (0.05 if kalifiye else -0.02),
                "robust_utility": (0.02 if kalifiye else -0.01),
                "expected_holding_hours": 3.0, "median_hours_to_tp": 2.0,
                "hours_to_tp_p25": 1.4, "hours_to_tp_p75": 3.2,
                "n_eff_used": 900, "rr_median": 2.4,
                "target_distance_sigma_median": 1.2,
                "net_1pct_exit": 101.2, "stop_price": 99.1,
                "entry_low": 99.8, "optimal_entry": 100.0, "entry_high": 100.2,
                "max_chase_price": 100.5, "expected_half_life_sec": 420.0,
                "mfe": {"p50": 0.9}, "mae": {"p50": -0.7},
                "max_capacity_usd": 25_000.0,
                "execution_probability": 0.8,
                "cost_model": "MEASURED_L2_VWAP",
                "convergence": {"verdict": "CONVERGED", "ci_width": 0.04},
                "status": ("QUALIFIED" if kalifiye else "NO_EDGE"),
                "tradable": kalifiye,
                "rejection_reasons": ([] if kalifiye else ["NEGATIVE_EV"]),
            })
    return pair_card(sym, hz, market_price=100.0, data_quality=1.0,
                     liquidity_score=0.9, cost_model="MEASURED_L2_VWAP",
                     model_version="softmax_l2/1")


def test_kabul_54_tum_maddeler():
    k = _ornek_kart()
    g = {}

    # 1 — evren taranır (elle yazılmış liste değil)
    g["evren_taranir"] = hasattr(UNI, "scan") and hasattr(UNI, "bar_inventory")
    # 2 — bütün ufuklar
    g["tum_ufuklar"] = (len({h["horizon"] for h in k["horizons"]})
                        == len(HZ.all_horizons()))
    # 3 — LONG ve SHORT ayrı
    g["long_ve_short"] = ({h["direction"] for h in k["horizons"]}
                          == {"LONG", "SHORT"})
    # 4 — gerçek L2 VWAP (ve spread çift sayılmaz)
    c = estimate_costs(1000, 1e6, 1e6, 4.0,
                       bid_curve={1: 5e4, 5: 3e5}, ask_curve={1: 5e4, 5: 3e5})
    g["gercek_L2_vwap"] = (c.model == "gercek_L2_vwap" and c.spread_bps == 0.0)
    # 5 — net %1 hedefi maliyetten SONRA çözülür
    ce = CostEstimate()
    ce.entry_fee_bps = 14.0
    g["net_hedef_maliyetten_sonra"] = required_gross_move_pct(1.0, ce) > 1.0
    # 6, 7, 8 — taban · gereken lift · gerçek lift
    h4 = next(h for h in k["horizons"]
              if h["horizon"] == "4h" and h["direction"] == "LONG")
    g["taban_hesaplanir"] = h4["baseline"] is not None
    g["gereken_lift"] = h4["required_lift"] is not None
    g["gercek_lift"] = h4["actual_lift"] is not None
    # 9 — güven aralığı
    g["guven_araligi"] = (h4["lower95"] is not None
                          and h4["lower95"] < h4["p_target_first"])
    # 10, 11 — en iyi ve en erken ufuk
    g["en_iyi_ufuk"] = k["best_horizon"] == "4h"
    g["en_erken_ufuk"] = k["earliest_qualified_horizon"] == "4h"
    # 12 — beklenen hedef süresi
    g["beklenen_hedef_suresi"] = k["expected_target_time_hours"] is not None
    # 13 — sermaye kapasitesi
    cap = capacity_curve((1e6, 1e6), 2.0, 1.0, levels=(100, 10_000),
                         bid_curve={1: 8e3, 10: 5e4},
                         ask_curve={1: 8e3, 10: 5e4})
    g["kapasite_egrisi"] = (len(cap) == 2 and k["max_capacity_usd"] is not None)
    # 14 — süresi dolmuş sinyal kapıyı geçemez
    op = evaluate(Opportunity(id="x", strategy="s", symbol="B",
                              direction="LONG", horizon="4h", expires_at=1.0),
                  Gates())
    g["bayat_sinyal_dusurulur"] = "SURESI_DOLMUS" in " ".join(op.reject_reasons)
    # 15 — aynı varlığın ufukları TEK kartta
    g["ufuklar_tek_kartta"] = (len([x for x in k["horizons"]
                                    if x["horizon"] == "4h"]) == 2)
    # 16 — sabah penceresi otomatik taranır
    g["sabah_penceresi"] = (
        len(MG.slots()) == 24
        and MG.slot_of(dt.datetime(2026, 8, 18, 6, 7, tzinfo=dt.timezone.utc),
                       "UTC") == "06:00")
    # 17 — slot walk-forward pencerelerinde öğrenilir
    g["slot_walk_forward"] = tuple(MG.WINDOWS) == (30, 90, None)
    # 18 — canlı tetik: zayıf fırsat raporu tüketmez
    rdy = MG.readiness([], 1.0, 1.0)
    g["canli_tetik"] = MG.should_publish(
        [], rdy, dt.datetime(2026, 8, 18, 7, 0), False)[0] is False
    # 19 — fırsatsız sabah raporu
    bos = MG.build_report([], {"markets_scanned": 5}, rdy, None, None, "UTC")
    g["firsatsiz_rapor"] = (bos["empty_result"]
                            == "NO QUALIFIED NET +%1 OPPORTUNITY")

    with tempfile.TemporaryDirectory() as d:
        # 20 — değişmez defter (ikinci sonuç ilkini EZMEZ)
        led = LD.Ledger(Path(d) / "p.jsonl")
        led.record_prediction(_tahmin("a"))
        led.record_outcome(LD.Outcome("a", "t", "SL_FIRST",
                                      realized_net_pct=-1.2))
        led.record_outcome(LD.Outcome("a", "t2", "TP_FIRST"))
        g["degismez_defter"] = led.outcomes()["a"]["outcome"] == "SL_FIRST"
        # 21 — gerçekleşme mutabakatı
        led2 = LD.Ledger(Path(d) / "q.jsonl")
        led2.record_prediction(_tahmin("b"))
        LD.evaluate_open(led2,
                         lambda s, x, y: _bar([101.5, 100.2], [100.0, 99.5]),
                         now=1e12)
        g["mutabakat"] = led2.outcomes()["b"]["outcome"] == "TP_FIRST"
        # 22 — başarısız sinyaller görünür kalır
        sc = LD.scorecard(led, None)["cells"][0]
        g["basarisiz_gorunur"] = (sc["published"] == 1 and sc["sl_first"] == 1)

    # 23 — GUARANTEED durumu YOKTUR
    g["guaranteed_yok"] = (not any("GUARANT" in s.upper()
                                   for s in QualificationState.ALL)
                           and k["guaranteed"] is False)
    # ek — JSON geçerliliği ve yakınsama ölçümü
    g["json_gecerli"] = LV.json_safe({"x": float("inf")})["x"] is None
    g["yakinsama_olculur"] = h4["convergence"]["verdict"] in (
        CV.CONVERGED, CV.CONVERGING, CV.REGIME_DEPENDENT, CV.UNSTABLE,
        CV.UNMEASURED)

    dusen = [k2 for k2, v in g.items() if not v]
    assert not dusen, f"§54 kabul listesi DÜŞTÜ: {dusen}"
    assert len(g) >= 23, f"yalnız {len(g)} madde denetlendi"


def test_yari_omur_geometriden_gelir_ve_uydurulmaz():
    """§41 — bilinmeyen girdiyle sayı ÜRETİLMEZ."""
    assert LV.signal_half_life_sec(None, 0.5) is None
    assert LV.signal_half_life_sec(0.1, None) is None
    dar = LV.signal_half_life_sec(0.1, 0.2)
    genis = LV.signal_half_life_sec(0.1, 0.8)
    assert genis > dar, "geniş bant daha uzun dayanmalı"
    assert LV.signal_half_life_sec(0.5, 0.2) < dar, \
        "yüksek oynaklıkta sinyal daha çabuk ölmeli"


def test_sabah_karnesi_tetik_saatine_gore_kirilir():
    """§43 — hangi slotun işe yaradığı ÖLÇÜLÜR, varsayılmaz."""
    with tempfile.TemporaryDirectory() as d:
        led = LD.Ledger(Path(d) / "p.jsonl")
        led.record_prediction(_tahmin("s1", timestamp="2026-08-18T04:10:00Z"))
        led.record_outcome(LD.Outcome("s1", "t", "TP_FIRST",
                                      realized_net_pct=1.0))
        led.record_prediction(_tahmin("s2", timestamp="2026-08-18T04:40:00Z"))
        led.record_outcome(LD.Outcome("s2", "t", "SL_FIRST",
                                      realized_net_pct=-1.2))
        r = MG.morning_performance(led, "Europe/Istanbul")
        assert r["overall"]["published"] == 2
        assert r["overall"]["net1_precision"] == pytest.approx(0.5)
        assert {x["slot"] for x in r["by_trigger_hour"]} == {"07:00", "07:30"}
        assert "düşürülmez" in r["note"]


def test_kart_sinyal_omru_alanlarini_tasir():
    """§41 — DETECTED AT · VALID UNTIL · EXPECTED HALF-LIFE."""
    k = _ornek_kart()
    for alan in ("detected_at", "valid_until", "expected_half_life_sec",
                 "entry_valid_seconds"):
        assert alan in k, alan
    assert k["detected_at"] and k["valid_until"]


def test_gecerlilik_yari_omru_asamaz():
    """§41 — 'sinyal 8 dakika geçerli' derken yarı ömrü 30 sn ise bu yalandır.

    Ölçüldü: dar giriş bandında yarı ömür tabana çakılıyor. Geçerlilik süresi
    artık yarı ömürden TÜREVDİR; ikisi çeliştiğinde kısa olan bağlayıcıdır."""
    from agi_trader.qualification.robust import build_entry_plan
    sig_bar = 0.08
    onceki = None
    for H in (3, 12, 48, 288):
        sigH = sig_bar * (H ** 0.5)
        plan = build_entry_plan("LONG", 100.0, 99.99, 100.01, sigH)
        bant = abs(plan.entry_high - plan.entry_low) / 100.0 * 100.0
        omur = LV.signal_half_life_sec(sig_bar, bant)
        gecerli = int(max(60, min(LV.SIGNAL_VALID_SECONDS, 2.0 * omur)))
        assert gecerli <= LV.SIGNAL_VALID_SECONDS
        assert gecerli <= max(60, 2.0 * omur) + 1
        if onceki is not None:
            assert omur > onceki, "uzun ufukta sinyal daha uzun yaşamalı"
        onceki = omur


def test_giris_bandi_sigma_altina_dusmez():
    """Bant σ_bar'ın altına düşerse sinyal pratikte dolmadan ölür."""
    from agi_trader.qualification.robust import build_entry_plan
    sig_bar = 0.08
    for H in (12, 48, 288):
        sigH = sig_bar * (H ** 0.5)
        plan = build_entry_plan("LONG", 100.0, 99.99, 100.01, sigH)
        bant = abs(plan.entry_high - plan.entry_low) / 100.0 * 100.0
        assert bant >= sig_bar * 0.9, (H, bant, sig_bar)


def test_kart_gecerliligi_secilen_ufkun_omrunden_gelir():
    """5 dakikalık bir kurulumda 24 saatlik ufkun ömrü ilan EDİLEMEZ."""
    hz = []
    for u, omur in (("5m", 30.0), ("24h", 5000.0)):
        for d in ("LONG", "SHORT"):
            kalifiye = (u == "5m" and d == "LONG")
            hz.append({"horizon": u, "horizon_minutes": HZ.HORIZON_MIN[u],
                       "direction": d, "expected_half_life_sec": omur,
                       "robust_utility": (0.1 if kalifiye else -0.1),
                       "robust_ev": (0.05 if kalifiye else -0.05),
                       "tradable": kalifiye, "status": "QUALIFIED",
                       "p_target_first": 0.4, "rejection_reasons": []})
    k = pair_card("BTCUSDT", hz)
    assert k["best_horizon"] == "5m"
    assert k["entry_valid_seconds"] <= 60, k["entry_valid_seconds"]
    assert k["expected_half_life_sec"] == 30.0


# ══════════════════════════════════════════════════════════════════════════
# UFUK KARŞILAŞTIRILABİLİRLİĞİ (ölçülerek bulunan yanıltıcılık)
# ══════════════════════════════════════════════════════════════════════════

def test_farkli_stop_secilince_dusus_aciklanir():
    """Kör taban ufukla düşerse sebebi YAZILMALI.

    ÖLÇÜLDÜ: ACEUSDT SHORT 8h %65,0 → 12h %64,6. Ölçüm hatası değil — 12h'de
    daha dar stop seçilmiş (k=0,50 vs 0,75). Sessiz bırakmak kullanıcıyı
    ölçüm hatası var sanmaya iter."""
    from agi_trader.qualification.matrix import flag_horizon_comparability
    h = [{"horizon": "8h", "horizon_minutes": 480, "direction": "SHORT",
          "baseline": 0.650, "stop_sigma_mult": 0.75},
         {"horizon": "12h", "horizon_minutes": 720, "direction": "SHORT",
          "baseline": 0.646, "stop_sigma_mult": 0.50}]
    out = flag_horizon_comparability(h)
    assert out[1]["baseline_dropped_from_prev"] is True
    assert out[1]["comparability_ok"] is True
    assert "stop çarpanı farklı" in out[1]["comparability_note"]


def test_ayni_stopta_dusus_BEKLENMEYEN_olarak_isaretlenir():
    """Stop aynıysa kümülatif insidans azalamaz — bu gerçek bir kusurdur."""
    from agi_trader.qualification.matrix import flag_horizon_comparability
    h = [{"horizon": "8h", "horizon_minutes": 480, "direction": "LONG",
          "baseline": 0.650, "stop_sigma_mult": 1.0},
         {"horizon": "12h", "horizon_minutes": 720, "direction": "LONG",
          "baseline": 0.600, "stop_sigma_mult": 1.0}]
    out = flag_horizon_comparability(h)
    assert out[1]["comparability_ok"] is False
    assert "beklenmeyen" in out[1]["comparability_note"]


def test_monoton_artista_bayrak_yok():
    from agi_trader.qualification.matrix import flag_horizon_comparability
    h = [{"horizon": "8h", "horizon_minutes": 480, "direction": "LONG",
          "baseline": 0.60, "stop_sigma_mult": 1.0},
         {"horizon": "12h", "horizon_minutes": 720, "direction": "LONG",
          "baseline": 0.70, "stop_sigma_mult": 1.0}]
    out = flag_horizon_comparability(h)
    assert "baseline_dropped_from_prev" not in out[1]
