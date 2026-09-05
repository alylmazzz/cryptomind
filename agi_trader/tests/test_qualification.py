"""NET +%1 NİTELENDİRME MOTORU — KABUL TESTLERİ.

Şartname 118 ve 2. mesaj 54'teki maddelerin HEPSİ burada karşılığını bulur.
"Ekran güzel görünüyor" bu özelliği tamamlanmış saymaz; aşağıdaki testler
geçmeden production-complete denilemez.

Testler DEĞİŞMEZ kilitler: her biri, bozulduğunda sessizce yanlış sayı
üretecek bir davranışı korur.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agi_trader.opportunity.costs import (CostEstimate, estimate_costs,
                                          required_gross_move_pct, vwap_offset_bps)
from agi_trader.qualification import features as FT
from agi_trader.qualification import horizons as HZ
from agi_trader.qualification import ledger as LD
from agi_trader.qualification import morning as MG
from agi_trader.qualification import registry as RG
from agi_trader.qualification import universe as UNI
from agi_trader.qualification.baserate import measure_symbol
from agi_trader.qualification.firstpassage import (LABEL_AMBIGUOUS, LABEL_STOP,
                                                   LABEL_TARGET, cif_curve,
                                                   first_passage_times,
                                                   label_cell,
                                                   monotonic_violation)
from agi_trader.qualification.lift import (Payoff, breakeven_probability,
                                           evaluate_lift, expected_value,
                                           net1_precision)
from agi_trader.qualification.matrix import pair_card, rank_pairs, scanner_summary
from agi_trader.qualification.model import (CLASS_INDEX, decile_table,
                                            fit_softmax, purged_walk_forward)
from agi_trader.qualification.regime import classify
from agi_trader.qualification.robust import (best_horizon,
                                             earliest_qualified_horizon,
                                             horizon_narrative,
                                             robust_expected_value, stress_test)
from agi_trader.qualification.state import (CellEvidence, EvidenceGates,
                                            QualificationState, RejectionCode,
                                            decide_state)
from agi_trader.qualification.stats import (calibration_slope_intercept, ece,
                                            effective_sample_size,
                                            proportion_with_ci, wilson_ci)
from agi_trader.qualification.targets import (CostProfile, gross_target_pct,
                                              sigma_bar_pct, stop_grid)


# ══════════════════════════════════════════════════════════════════════════
# yardımcı: tekrarlanabilir sentetik seri (SABİT tohum — şartname 84)
# ══════════════════════════════════════════════════════════════════════════

def _seri(n=6000, seed=20260818, vol=0.0015, drift=0.0):
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, vol, n)
    c = 100.0 * np.exp(np.cumsum(r))
    h = c * (1 + np.abs(rng.normal(0, vol / 2, n)))
    l = c * (1 - np.abs(rng.normal(0, vol / 2, n)))
    o = np.concatenate([[c[0]], c[:-1]])
    v = np.abs(rng.normal(1000, 200, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"open": o, "high": np.maximum(h, np.maximum(o, c)),
                         "low": np.minimum(l, np.minimum(o, c)), "close": c,
                         "volume": v, "trades": v / 10,
                         "taker_buy_base": v * 0.5,
                         "quote_volume": v * c}, index=idx)


def _profil(model="MEASURED_L2_VWAP"):
    return CostProfile("TESTUSDT", spread_bps=0.0, impact_bps_roundtrip=1.0,
                       fee_bps_roundtrip=8.0, reserve_bps=5.0,
                       funding_rate_8h=0.0001, model=model, source="test")


# ══════════════════════════════════════════════════════════════════════════
# 1. GUARANTEED DURUMU YOKTUR  (şartname 47, 116 · 2. mesaj 22)
# ══════════════════════════════════════════════════════════════════════════

def test_guaranteed_durumu_yoktur():
    """Kod içinde 'garanti' anlamına gelen bir durum OLUŞTURULAMAZ."""
    for s in QualificationState.ALL:
        assert "GUARANT" not in s.upper(), s
    assert not hasattr(QualificationState, "GUARANTEED")
    assert set(QualificationState.TRADABLE) <= set(QualificationState.ALL)


def test_kartta_garanti_satiri_her_zaman_var_ve_false():
    k = pair_card("BTCUSDT", [])
    assert k["guaranteed"] is False
    assert "GARANTİ: YOK" in k["guarantee_line"]
    assert scanner_summary([], 0, 0)["guarantee"] == "YOK"


def test_deftere_guaranteed_yazilamaz():
    with tempfile.TemporaryDirectory() as d:
        led = LD.Ledger(Path(d) / "p.jsonl")
        p = LD.Prediction(
            prediction_id="x", timestamp="2026-01-01T00:00:00Z", symbol="B",
            horizon="4h", direction="LONG", status="QUALIFIED", entry=1.0,
            net1_exit=1.01, stop=0.99, p_target_first=0.5,
            p_target_lower95=0.4, p_stop_first=0.3, p_timeout=0.2,
            baseline=0.4, required_lift=0.1, actual_lift=0.1, robust_ev=0.1,
            expected_target_hours=1.0, cost_model="ESTIMATED", cost_pct=0.1,
            max_capacity_usd=1000.0, data_quality=1.0, model_version="v",
            features_hash="h", valid_until="2026-01-01T00:08:00Z",
            guaranteed=True)
        with pytest.raises(ValueError):
            led.record_prediction(p)


# ══════════════════════════════════════════════════════════════════════════
# 2. LOOK-AHEAD YOK  (şartname 80)
# ══════════════════════════════════════════════════════════════════════════

def test_gelecek_bar_gecmis_ozelligi_degistirmez():
    """Geleceği bozup geçmiş özelliklerin BİT-BİT aynı kaldığını doğrular."""
    d = _seri(3000)
    sb = sigma_bar_pct(d)
    X1, ad, ail, tf = FT.build(d, sb)

    bozuk = d.copy()
    k = 2000
    bozuk.iloc[k:, bozuk.columns.get_loc("close")] *= 3.0
    bozuk.iloc[k:, bozuk.columns.get_loc("high")] *= 3.0
    bozuk.iloc[k:, bozuk.columns.get_loc("low")] *= 3.0
    bozuk.iloc[k:, bozuk.columns.get_loc("volume")] *= 7.0
    X2, _, _, _ = FT.build(bozuk, sigma_bar_pct(bozuk))

    a, b = X1[:k], X2[:k]
    fark = np.nanmax(np.abs(np.nan_to_num(a) - np.nan_to_num(b)))
    assert fark < 1e-9, f"sızıntı: geçmiş özellikler değişti (maks fark {fark})"


def test_giris_barinin_kendisi_taranmaz():
    """i barındaki giriş için tarama i+1'den başlar."""
    h = np.array([99.0, 10.0, 10.0])
    t = first_passage_times(h, np.array([[50.0] * 3]), 2, "up")[0]
    assert t[0] == 0, "giriş barının kendi high'ı bariyer sayılmış"


def test_sigma_ve_atr_nedenseldir():
    d = _seri(2000)
    s1 = sigma_bar_pct(d)
    bozuk = d.copy()
    bozuk.iloc[1500:, bozuk.columns.get_loc("close")] *= 2.0
    s2 = sigma_bar_pct(bozuk)
    assert np.nanmax(np.abs(np.nan_to_num(s1[:1500]) -
                            np.nan_to_num(s2[:1500]))) < 1e-9


# ══════════════════════════════════════════════════════════════════════════
# 3. AYNI-BAR BELİRSİZLİĞİ  (şartname 7, 81)
# ══════════════════════════════════════════════════════════════════════════

def test_ayni_bar_belirsizligi_basari_sayilmaz():
    ls = label_cell(np.array([3]), np.array([3]), 10, n_total=100)
    assert ls.label[0] == LABEL_AMBIGUOUS
    assert ls.label[0] != LABEL_TARGET


def test_belirsiz_ornekler_olcumden_duser():
    tT = np.array([2, 2, 5, 0])
    tS = np.array([2, 4, 3, 0])
    ls = label_cell(tT, tS, 10, n_total=100)
    assert list(ls.label) == [LABEL_AMBIGUOUS, LABEL_TARGET, LABEL_STOP, 0]
    kesin = ls.label != LABEL_AMBIGUOUS
    assert kesin.sum() == 3


def test_ilk_gecis_dogru_sirayi_verir():
    h = np.array([10.0, 11.0, 12.0, 9.0, 20.0])
    l = np.array([10.0, 9.0, 8.0, 5.0, 4.0])
    tu = first_passage_times(h, np.array([[11.0] * 5]), 4, "up")[0]
    td = first_passage_times(l, np.array([[9.0] * 5]), 4, "dn")[0]
    assert list(tu) == [1, 1, 2, 1, 0]
    assert list(td) == [1, 1, 1, 1, 0]


# ══════════════════════════════════════════════════════════════════════════
# 4. MALİYET: L2 VWAP, ÇİFT SAYIM, BİRİM TUTARLILIĞI  (şartname 6, 82, 83)
# ══════════════════════════════════════════════════════════════════════════

def test_gercek_L2_egrisi_varken_spread_cift_sayilmaz():
    egri = {1: 50_000.0, 2: 120_000.0, 5: 400_000.0, 10: 900_000.0}
    c = estimate_costs(10_000, 1e6, 1e6, spread_bps=6.0,
                       bid_curve=egri, ask_curve=egri)
    assert c.model == "gercek_L2_vwap"
    assert c.spread_bps == 0.0, "gerçek eğri modunda spread AYRICA eklenmiş"


def test_egri_yokken_model_tahmini_oldugunu_beyan_eder():
    c = estimate_costs(10_000, 1e6, 1e6, spread_bps=6.0)
    assert c.model == "dogrusal_defter_yaklasimi"
    assert c.spread_bps == 6.0
    assert any("YAKLAŞIM" in w for w in c.warnings)


def test_net_hedef_maliyetten_sonra_cozulur():
    c = CostEstimate(); c.entry_fee_bps = 15.0
    g = required_gross_move_pct(1.0, c)
    assert g > 1.0, "brüt hedef net hedeften büyük olmalı"
    # çarpımsal kimlik: (1+g)(1-c) = 1+n
    assert abs((1 + g / 100) * (1 - 0.0015) - 1.01) < 1e-9


def test_derinlik_birimi_ayri_alanda():
    from agi_trader.data import recorder
    src = Path(recorder.__file__).read_text(encoding="utf-8")
    assert "bid_depth_usd" in src and "ask_depth_usd" in src


def test_emir_defteri_asarsa_fizibil_degil():
    egri = {1: 100.0, 2: 200.0}
    off, uyari = vwap_offset_bps(egri, 10_000)
    assert not math.isfinite(off)
    assert uyari and "aşıyor" in uyari


# ══════════════════════════════════════════════════════════════════════════
# 5. TABAN ORANI, GÜVEN ARALIĞI, ETKİN ÖRNEKLEM  (şartname 8, 14, 15, 16)
# ══════════════════════════════════════════════════════════════════════════

def test_taban_orani_hesaplanir_ve_olasiliklar_toplami_bir():
    d = _seri(20000)
    m = measure_symbol("TESTUSDT", d, _profil(), horizons=["1h", "4h"],
                       stop_mults=(1.0,))
    assert m.cells
    for c in m.cells:
        t = c["p_target_first"] + c["p_stop_first"] + c["p_timeout"]
        # Hücre alanları JSON boyutu için 5 haneye yuvarlanır; tolerans bu
        # yuvarlamayı karşılar, gerçek bir sapmayı DEĞİL (3 × 5e-6 = 1,5e-5).
        assert abs(t - 1.0) < 1e-4, f"olasılık toplamı {t}"
        assert c["n_eff_used"] <= c["n_raw"]


def test_etkin_orneklem_ham_satirdan_kucuk():
    lab = np.repeat([1, 0], 5000)[:10000]
    e = effective_sample_size(lab.astype(float), horizon_bars=288)
    assert e["non_overlap"] < e["raw"]
    assert e["used"] <= e["non_overlap"] + 1e-9


def test_wilson_uc_oranlarda_sifir_genislik_uretmez():
    lo, hi = wilson_ci(3, 3)
    assert hi - lo > 0.2, "3/3 gözlemde 'kesin' aralık üretilmiş"
    lo2, hi2 = wilson_ci(0, 5)
    assert hi2 > 0.1


def test_guven_araligi_etkin_orneklemle_genisler():
    dar = proportion_with_ci(500, 1000, ess=1000)
    genis = proportion_with_ci(500, 1000, ess=50)
    assert (genis["upper95"] - genis["lower95"]) > (dar["upper95"] - dar["lower95"])


def test_alt_sinir_nokta_tahminden_kucuk():
    r = proportion_with_ci(82, 100, ess=100)
    assert r["lower95"] < r["p"] < r["upper95"]


# ══════════════════════════════════════════════════════════════════════════
# 6. BAŞABAŞ / GEREKEN LIFT / GERÇEK LIFT  (şartname 9, 104)
# ══════════════════════════════════════════════════════════════════════════

def test_basabas_olasiligi_maliyetle_yukselir():
    ucuz = Payoff(1.0, 1.0, 0.0)
    pahali = Payoff(1.0, 2.0, 0.0)
    assert breakeven_probability(pahali) > breakeven_probability(ucuz)
    assert abs(breakeven_probability(ucuz) - 0.5) < 1e-9


def test_lift_modelsizken_kenar_iddia_edilemez():
    r = evaluate_lift(0.40, Payoff(1.0, 1.0, 0.0), 0.0, model_rate=None)
    assert r.edge is False and "MODEL_YOK" in r.reason
    assert r.required_lift is not None


def test_lift_yetersizse_kenar_yok():
    r = evaluate_lift(0.40, Payoff(1.0, 1.0, 0.0), 0.0, model_rate=0.45,
                      n_model_eff=1000, n_base_eff=1000)
    assert r.edge is False and "LIFT_YETERSIZ" in r.reason


def test_lift_gecerliyse_kenar_var():
    r = evaluate_lift(0.40, Payoff(1.0, 1.0, 0.0), 0.0, model_rate=0.75,
                      n_model_eff=3000, n_base_eff=30000)
    assert r.edge is True and r.actual_lift > r.required_lift


def test_lift_alt_siniri_orneklemi_tasimazsa_reddedilir():
    r = evaluate_lift(0.40, Payoff(1.0, 1.0, 0.0), 0.0, model_rate=0.62,
                      n_model_eff=25, n_base_eff=1000)
    assert r.edge is False and "ALT_SINIRI" in r.reason


def test_net1_precision_paydasi_yayimlanan_sinyaldir():
    assert net1_precision(3, 10) == pytest.approx(0.3)


# ══════════════════════════════════════════════════════════════════════════
# 7. ZAMAN AŞIMI GETİRİSİ SIFIR VARSAYILMAZ  (şartname 28)
# ══════════════════════════════════════════════════════════════════════════

def test_zaman_asimi_getirisi_olculur():
    d = _seri(20000)
    m = measure_symbol("TESTUSDT", d, _profil(), horizons=["4h"],
                       stop_mults=(1.0,))
    to = [c["timeout_return_pct"] for c in m.cells
          if c["regime"] == "ALL" and c["timeout_events"] > 50]
    assert to and any(x is not None and abs(x) > 1e-9 for x in to)


def test_ev_zaman_asimini_hesaba_katar():
    a = expected_value(0.4, 0.4, 0.2, Payoff(1.0, 1.0, 0.0))
    b = expected_value(0.4, 0.4, 0.2, Payoff(1.0, 1.0, -0.5))
    assert b < a, "zaman aşımı getirisi EV'yi etkilemiyor"


# ══════════════════════════════════════════════════════════════════════════
# 8. ROBUST EV VE UFUK SEÇİMİ  (şartname 12, 13, 48, 94)
# ══════════════════════════════════════════════════════════════════════════

def test_robust_ev_belirsizligi_cezalandirir():
    p = Payoff(1.0, 1.0, 0.0)
    kesin = robust_expected_value(0.6, 0.4, 0.0, p, p_tp_lower=0.6,
                                  psi=0.0, expected_holding_hours=1.0)
    belirsiz = robust_expected_value(0.6, 0.4, 0.0, p, p_tp_lower=0.45,
                                     psi=0.0, expected_holding_hours=1.0)
    assert belirsiz.robust_ev < kesin.robust_ev
    assert belirsiz.uncertainty_penalty > 0


def test_surukleme_olculmediyse_tiras_uygulanir():
    p = Payoff(1.0, 1.0, 0.0)
    r = robust_expected_value(0.6, 0.4, 0.0, p, p_tp_lower=0.6, psi=None,
                              expected_holding_hours=1.0)
    assert r.drift_measured is False and r.drift_penalty > 0


def test_kalifiye_yoksa_en_iyi_ufuk_NONE():
    h = [{"horizon": "4h", "tradable": False, "robust_utility": 5.0,
          "horizon_minutes": 240, "p_target_first": 0.9,
          "status": "NO_EDGE", "rejection_reasons_tr": ["beklenen değer yok"]}]
    assert best_horizon(h) is None
    metin = horizon_narrative(h, None)
    assert "Doğrulanmış ufuk yok" in metin


def test_en_iyi_ufuk_ham_olasilikla_secilmez():
    a = {"horizon": "24h", "tradable": True, "robust_utility": 0.01,
         "horizon_minutes": 1440, "p_target_first": 0.90,
         "expected_holding_hours": 20.0, "status": "QUALIFIED"}
    b = {"horizon": "4h", "tradable": True, "robust_utility": 0.10,
         "horizon_minutes": 240, "p_target_first": 0.60,
         "expected_holding_hours": 3.0, "status": "QUALIFIED"}
    assert best_horizon([a, b])["horizon"] == "4h"
    assert earliest_qualified_horizon([a, b])["horizon"] == "4h"


def test_en_erken_ile_en_iyi_farkli_olabilir():
    a = {"horizon": "2h", "tradable": True, "robust_utility": 0.02,
         "horizon_minutes": 120, "p_target_first": 0.5,
         "expected_holding_hours": 1.5, "status": "QUALIFIED"}
    b = {"horizon": "8h", "tradable": True, "robust_utility": 0.30,
         "horizon_minutes": 480, "p_target_first": 0.8,
         "expected_holding_hours": 6.0, "status": "QUALIFIED"}
    assert best_horizon([a, b])["horizon"] == "8h"
    assert earliest_qualified_horizon([a, b])["horizon"] == "2h"


def test_stres_senaryolari_ev_dusurur():
    s = stress_test(0.2, 15.0, 0.4, 1.0)
    assert len(s) == len(__import__("agi_trader.qualification.robust",
                                    fromlist=["x"]).STRESS_SCENARIOS)
    assert all(r["ev_pct"] <= 0.2 + 1e-9 for r in s)


# ══════════════════════════════════════════════════════════════════════════
# 9. DURUM MAKİNESİ  (şartname 47, 56, 100)
# ══════════════════════════════════════════════════════════════════════════

def _kanit(**kw):
    t = dict(n_effective=1000.0, n_tp_events=200, baseline=0.4,
             p_target_first=0.6, p_lower95=0.55, p_upper95=0.65,
             robust_ev=0.2, edge_proven=True, has_validation_report=True,
             has_model=True, calibration_ece=0.01, calibration_slope=1.0,
             psi=0.05, data_quality=1.0, liquidity_score=1.0,
             cost_model_valid=True, cost_model_measured=True,
             dsr=0.99, pbo=0.1, positive_subperiod_frac=0.8,
             shadow_n=100, shadow_within_bounds=True)
    t.update(kw)
    return CellEvidence(**t)


def test_tam_kanit_high_confidence_verir():
    d = decide_state(_kanit())
    assert d.state == QualificationState.HIGH_CONFIDENCE and d.tradable


def test_tahmini_maliyet_modeli_high_confidence_veremez():
    d = decide_state(_kanit(cost_model_measured=False))
    assert d.state == QualificationState.QUALIFIED
    assert any("HIGH_CONFIDENCE verilemez" in n for n in d.notes)


def test_negatif_ev_reddedilir():
    d = decide_state(_kanit(robust_ev=-0.01))
    assert d.state == QualificationState.NO_EDGE
    assert RejectionCode.NEGATIVE_EV in d.rejection_reasons


def test_model_yoksa_research_only():
    d = decide_state(_kanit(has_model=False, has_validation_report=False))
    assert d.state == QualificationState.RESEARCH_ONLY
    assert RejectionCode.MISSING_MODEL in d.rejection_reasons


def test_dogrulama_raporu_yoksa_reddedilir():
    d = decide_state(_kanit(has_validation_report=False))
    assert RejectionCode.NO_VALIDATION_REPORT in d.rejection_reasons
    assert not d.tradable


def test_bayat_veri_reddedilir():
    d = decide_state(_kanit(data_stale=True))
    assert RejectionCode.DATA_STALE in d.rejection_reasons and not d.tradable


def test_suruklenen_model_degraded():
    d = decide_state(_kanit(psi=0.9, had_edge_before=True))
    assert d.state == QualificationState.DEGRADED
    assert RejectionCode.MODEL_DRIFT in d.rejection_reasons


def test_kalibrasyon_bozuksa_reddedilir():
    d = decide_state(_kanit(calibration_ece=0.4))
    assert RejectionCode.CALIBRATION_FAILED in d.rejection_reasons


def test_coklu_test_kapisi_dsr():
    d = decide_state(_kanit(dsr=0.10))
    assert RejectionCode.MULTIPLE_TESTING in d.rejection_reasons
    assert not d.tradable


def test_pbo_yuksekse_reddedilir():
    d = decide_state(_kanit(pbo=0.9))
    assert RejectionCode.OVERFIT_RISK in d.rejection_reasons


def test_kazanc_tek_doneme_sikismissa_reddedilir():
    d = decide_state(_kanit(positive_subperiod_frac=0.2))
    assert RejectionCode.REGIME_CONCENTRATED in d.rejection_reasons


def test_ornek_azsa_arastirma_modunda_kalir():
    d = decide_state(_kanit(n_effective=10.0, n_tp_events=2))
    assert d.state in (QualificationState.RESEARCH_ONLY,
                       QualificationState.NO_DATA)
    assert RejectionCode.SAMPLE_TOO_SMALL in d.rejection_reasons


def test_ci_cok_genisse_reddedilir():
    d = decide_state(_kanit(p_lower95=0.2, p_upper95=0.9))
    assert RejectionCode.CI_TOO_WIDE in d.rejection_reasons


def test_red_kodlari_turkce_karsiligi_var():
    for k in RejectionCode.ALL:
        assert k in RejectionCode.TR, f"{k} için Türkçe karşılık yok"


# ══════════════════════════════════════════════════════════════════════════
# 10. UFUK IZGARASI VE CIF MONOTONLUĞU  (şartname 1, 11, 38)
# ══════════════════════════════════════════════════════════════════════════

def test_butun_ufuklar_degerlendirilir():
    beklenen = {"5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h",
                "24h", "48h"}
    assert set(HZ.all_horizons()) == beklenen
    assert "48h" not in HZ.primary_horizons(), "48h yalnız referans olmalı"
    for h in HZ.all_horizons():
        assert HZ.horizon_bars(h) * 5 == HZ.HORIZON_MIN[h]


def test_kumulatif_insidans_azalmaz():
    rng = np.random.default_rng(7)
    n = 5000
    tT = rng.integers(0, 300, n)
    tS = rng.integers(0, 300, n)
    egri = cif_curve(tT, tS, [12, 24, 48, 96, 288])
    assert monotonic_violation(egri) is None, monotonic_violation(egri)


def test_uzun_ufukta_hedef_olasiligi_artar():
    d = _seri(30000)
    m = measure_symbol("TESTUSDT", d, _profil(),
                       horizons=["1h", "4h", "24h"], stop_mults=(1.0,))
    p = {c["horizon"]: c["p_target_first"] for c in m.cells
         if c["direction"] == "LONG" and c["regime"] == "ALL"}
    assert p["1h"] < p["4h"] < p["24h"]


def test_long_ve_short_ayri_hesaplanir():
    d = _seri(20000)
    m = measure_symbol("TESTUSDT", d, _profil(), horizons=["4h"],
                       stop_mults=(1.0,))
    yonler = {c["direction"] for c in m.cells}
    assert yonler == {"LONG", "SHORT"}


# ══════════════════════════════════════════════════════════════════════════
# 11. MODEL: PURGED WALK-FORWARD, KALİBRASYON  (şartname 18, 39)
# ══════════════════════════════════════════════════════════════════════════

def test_purged_walk_forward_kronolojik_ve_embargolu():
    kat = purged_walk_forward(10000, horizon_bars=50, n_folds=5)
    assert kat
    for f in kat:
        assert f.train_end < f.test_start, "purge/embargo yok"
        assert f.test_start < f.test_end
        assert f.test_start - f.train_end >= 50


def test_rastgele_bolme_yok_egitim_hep_gecmiste():
    kat = purged_walk_forward(20000, 100, n_folds=4)
    for f in kat:
        assert f.train_start == 0 and f.train_end <= f.test_start


def test_softmax_ogrenebiliyor_ve_kalibre():
    rng = np.random.default_rng(3)
    n = 6000
    x = rng.normal(0, 1, (n, 2))
    z = 1.2 * x[:, 0]
    p = 1 / (1 + np.exp(-z))
    y = np.where(rng.random(n) < p, CLASS_INDEX["TP"], CLASS_INDEX["SL"])
    m = fit_softmax(x, y, l2=0.01, names=["a", "b"])
    P = m.predict_proba(x)
    tahmin = P[:, CLASS_INDEX["TP"]]
    gercek = (y == CLASS_INDEX["TP"]).astype(float)
    k = calibration_slope_intercept(tahmin, gercek)
    assert 0.8 < k["slope"] < 1.25, k
    assert ece(tahmin, gercek) < 0.05


def test_model_ozellik_sayisi_uyusmazsa_none_doner():
    m = fit_softmax(np.random.default_rng(1).normal(0, 1, (500, 3)),
                    np.random.default_rng(2).integers(0, 3, 500),
                    names=["a", "b", "c"])
    assert m.predict_proba(np.zeros((5, 2))) is None


def test_desil_tablosu_sabit_bolme_kullanir():
    rng = np.random.default_rng(11)
    p = rng.random(5000)
    y = (rng.random(5000) < p).astype(float)
    d = decile_table(p, y)
    assert len(d) == 10
    assert d[-1]["actual_tp"] > d[0]["actual_tp"]


# ══════════════════════════════════════════════════════════════════════════
# 12. EVREN TARAMASI  (2. mesaj 1)
# ══════════════════════════════════════════════════════════════════════════

def _market(**kw):
    t = dict(symbol="XUSDT", quote="USDT", active=True, trading=True,
             bars=100_000, trades_per_bar=50.0, has_l2=True, spread_bps=2.0,
             depth_usd=500_000.0, volume_usd_24h=50e6, data_age_sec=10.0,
             book_age_sec=5.0, cost_model="MEASURED_L2_VWAP", rv_24h_pct=0.1)
    t.update(kw)
    return t


def test_uygun_market_eligible():
    s = UNI.evaluate_market(_market())
    assert s.eligible and s.score > 0 and not s.failed_gates


def test_uygun_olmayan_market_nedeniyle_dislanir():
    s = UNI.evaluate_market(_market(volume_usd_24h=1000.0, spread_bps=90.0))
    assert not s.eligible
    assert "volume_sufficient" in s.failed_gates
    assert "spread_acceptable" in s.failed_gates
    assert len(s.reasons) == len(s.failed_gates)


def test_eksik_olcum_kapiyi_dusurur_varsayim_yapmaz():
    s = UNI.evaluate_market(_market(spread_bps=None, depth_usd=None))
    assert not s.eligible
    assert "spread_acceptable" in s.failed_gates


def test_tarama_ozeti_nedenleri_sayar():
    r = UNI.scan([_market(symbol="A"), _market(symbol="B", active=False),
                  _market(symbol="C", quote="BTC")])
    assert r["scanned"] == 2 and r["eligible"] == 1 and r["excluded"] == 1
    kapi = {g["gate"]: g["failed_count"] for g in r["gates"]}
    assert kapi["market_active"] == 1


def test_defter_yasi_esigi_kaydedici_araligindan_kucuk_olamaz():
    """Eşik örnekleme aralığından kısaysa kapı marketi değil KENDİ
    aralığımızı ölçer ve her marketi düşürür (ölçüldü: 865/865)."""
    from agi_trader.data.recorder import INTERVAL_SEC
    assert UNI.EligibilityThresholds().max_book_age_sec >= INTERVAL_SEC


def test_bar_envanteri_yoksa_kapi_duser_tahmin_edilmez():
    with tempfile.TemporaryDirectory() as d:
        assert UNI.bar_inventory(Path(d) / "yok") == {}
    s = UNI.evaluate_market(_market(bars=None))
    assert not s.eligible and "ohlcv_history" in s.failed_gates


def test_bar_envanteri_json_yedeginden_okunur():
    with tempfile.TemporaryDirectory() as d:
        q = Path(d) / "qualification"
        q.mkdir()
        (q / "data_inventory.json").write_text(
            json.dumps({"bars": {"BTCUSDT": 481824}}), encoding="utf-8")
        assert UNI.bar_inventory(Path(d) / "data_5m") == {"BTCUSDT": 481824}


def test_eligibility_skoru_yon_icermez():
    r = UNI.scan([_market()])
    assert "yön" in r["note"] or "yon" in r["note"].lower()
    for m in r["markets"]:
        assert "direction" not in m and "signal" not in m


# ══════════════════════════════════════════════════════════════════════════
# 13. DEFTER: DEĞİŞMEZLİK VE MUTABAKAT  (şartname 44, 45, 106)
# ══════════════════════════════════════════════════════════════════════════

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


def test_basarisiz_sinyaller_silinmez():
    with tempfile.TemporaryDirectory() as d:
        led = LD.Ledger(Path(d) / "p.jsonl")
        led.record_prediction(_tahmin("p1"))
        led.record_prediction(_tahmin("p2"))
        led.record_outcome(LD.Outcome("p1", "2026-01-01T01:00:00Z", "SL_FIRST",
                                      realized_net_pct=-1.2))
        led.record_outcome(LD.Outcome("p2", "2026-01-01T01:00:00Z", "TP_FIRST",
                                      realized_net_pct=1.0))
        assert len(led.predictions()) == 2
        sc = LD.scorecard(led, None)["cells"][0]
        assert sc["published"] == 2 and sc["sl_first"] == 1
        assert sc["net1_precision"] == pytest.approx(0.5)
        assert sc["false_opportunity_rate"] == pytest.approx(0.5)


def test_sonuc_sonradan_degistirilemez():
    with tempfile.TemporaryDirectory() as d:
        led = LD.Ledger(Path(d) / "p.jsonl")
        led.record_prediction(_tahmin("p1"))
        led.record_outcome(LD.Outcome("p1", "t", "SL_FIRST"))
        led.record_outcome(LD.Outcome("p1", "t2", "TP_FIRST"))
        assert led.outcomes()["p1"]["outcome"] == "SL_FIRST", \
            "ikinci sonuç ilkini ezmiş — defter değişmez olmalı"


def test_bilinmeyen_sonuc_reddedilir():
    with tempfile.TemporaryDirectory() as d:
        led = LD.Ledger(Path(d) / "p.jsonl")
        with pytest.raises(ValueError):
            led.record_outcome(LD.Outcome("p1", "t", "KAZANDI"))


def test_islem_olmayan_sonuclar_fiyat_performansina_yazilmaz():
    with tempfile.TemporaryDirectory() as d:
        led = LD.Ledger(Path(d) / "p.jsonl")
        led.record_prediction(_tahmin("p1"))
        led.record_outcome(LD.Outcome("p1", "t", "NOT_FILLED"))
        sc = LD.scorecard(led, None)["cells"][0]
        assert sc["not_traded"] == 1
        assert sc["tp_first_rate_of_traded"] is None


def _bar(h, l, c=None, n=None):
    n = n or len(h)
    return pd.DataFrame({"open": l, "high": h, "low": l,
                         "close": c if c is not None else h},
                        index=pd.date_range("2026-01-01 00:05", periods=n,
                                            freq="5min", tz="UTC"))


def test_gerceklesme_ilk_bari_atlamaz():
    """Girişten SONRAKİ İLK bar hedefi vuruyorsa TP_FIRST sayılmalı.
    (Ölçülen birim kayması: bu bar hiç taranmıyordu.)"""
    with tempfile.TemporaryDirectory() as d:
        led = LD.Ledger(Path(d) / "p.jsonl")
        led.record_prediction(_tahmin("p1"))
        LD.evaluate_open(led, lambda s, a, b: _bar([101.5, 100.2], [100.0, 99.5]),
                         now=1e12)
        o = led.outcomes()["p1"]
        assert o["outcome"] == "TP_FIRST", o
        assert o["bars_to_resolution"] == 1


def test_gerceklesme_stop_once_gelirse_sl_first():
    with tempfile.TemporaryDirectory() as d:
        led = LD.Ledger(Path(d) / "p.jsonl")
        led.record_prediction(_tahmin("p1"))
        LD.evaluate_open(led, lambda s, a, b: _bar([100.3, 101.5], [98.5, 100.0]),
                         now=1e12)
        o = led.outcomes()["p1"]
        assert o["outcome"] == "SL_FIRST"
        assert o["realized_net_pct"] < 0


def test_gerceklesme_bar_yoksa_acik_kalir():
    """Bar yoksa 'herhalde zaman aşımıdır' VARSAYILMAZ."""
    with tempfile.TemporaryDirectory() as d:
        led = LD.Ledger(Path(d) / "p.jsonl")
        led.record_prediction(_tahmin("p1"))
        n = LD.evaluate_open(led, lambda s, a, b: None, now=1e12)
        assert n == 0 and len(led.open_predictions()) == 1


def test_kimlik_deterministik():
    a = LD.make_id("BTCUSDT", "4h", "LONG", "2026-01-01T00:00:00Z")
    b = LD.make_id("BTCUSDT", "4h", "LONG", "2026-01-01T00:00:00Z")
    assert a == b and len(a) == 20


def test_kalibrasyon_tablosu_tahmin_ve_gercegi_ayirir():
    with tempfile.TemporaryDirectory() as d:
        led = LD.Ledger(Path(d) / "p.jsonl")
        for i in range(20):
            pid = f"p{i}"
            led.record_prediction(_tahmin(pid, p_target_first=0.85))
            led.record_outcome(LD.Outcome(
                pid, "t", "TP_FIRST" if i < 10 else "SL_FIRST"))
        b = [x for x in LD.calibration_board(led) if x["n"]]
        assert b and b[0]["predicted"] == pytest.approx(0.85)
        assert b[0]["actual"] == pytest.approx(0.5)


# ══════════════════════════════════════════════════════════════════════════
# 14. SABAH MOTORU  (2. mesaj 24–33, 48, 49, 52)
# ══════════════════════════════════════════════════════════════════════════

def test_sabah_penceresi_ve_slotlari():
    s = MG.slots()
    assert s[0] == "05:00" and s[-1] == "10:45" and len(s) == 24
    ic = dt.datetime(2026, 8, 18, 6, 7, tzinfo=dt.timezone.utc)
    assert MG.slot_of(ic, "UTC") == "06:00"
    dis = dt.datetime(2026, 8, 18, 13, 0, tzinfo=dt.timezone.utc)
    assert MG.slot_of(dis, "UTC") is None and not MG.in_window(dis, "UTC")


def test_zayif_firsat_sabah_raporunu_tuketmez():
    kartlar = [{"best_horizon": "4h", "robust_expected_value": 0.001,
                "p_target_first_lower95": 0.1, "entry_valid_seconds": 60}]
    rdy = MG.readiness(kartlar, data_quality=0.5, liquidity=0.5)
    simdi = dt.datetime(2026, 8, 18, 5, 5)
    yayimla, gerekce = MG.should_publish(kartlar, rdy, simdi, False)
    assert yayimla is False and "güçlüsü bekleniyor" in gerekce


def test_pencere_sonunda_firsat_yoksa_yine_rapor_gider():
    rdy = MG.readiness([], 1.0, 1.0)
    simdi = dt.datetime(2026, 8, 18, 11, 0)
    yayimla, gerekce = MG.should_publish([], rdy, simdi, False)
    assert yayimla is True and "NO QUALIFIED" in gerekce


def test_bos_sabah_raporu_uydurma_sinyal_icermez():
    r = MG.build_report([], {"markets_scanned": 5}, MG.readiness([]), None,
                        None, "UTC")
    assert r["empty_result"] == "NO QUALIFIED NET +%1 OPPORTUNITY"
    assert r["opportunities"] == []
    assert r["guarantee"] == "NONE"


def test_readiness_bir_kar_olasiligi_degildir():
    r = MG.readiness([], 1.0, 1.0)
    assert "READINESS" in r.label
    assert "kâr olasılığı değildir" in r.disclaimer


def test_slot_gurultuyle_degismez():
    ogrenilen = {"best_slot": "07:30", "windows": {"full": [
        {"slot": "07:30", "n": 10, "score": 0.9},
        {"slot": "09:00", "n": 200, "score": 0.5}]}}
    yeni, neden = MG.should_switch_slot("09:00", ogrenilen)
    assert yeni == "09:00" and "örneklemi yetersiz" in neden


def test_slot_anlamli_iyilesmede_devredilir():
    ogrenilen = {"best_slot": "07:30", "windows": {"full": [
        {"slot": "07:30", "n": 500, "score": 0.80},
        {"slot": "09:00", "n": 500, "score": 0.50}]}}
    yeni, neden = MG.should_switch_slot("09:00", ogrenilen)
    assert yeni == "07:30" and "iyileşme" in neden


def test_nitelendirmeye_en_yakin_islem_onerisi_degildir():
    kartlar = [{"symbol": "BTCUSDT", "horizons": [
        {"horizon": "4h", "direction": "LONG", "tradable": False,
         "status": "NO_EDGE", "robust_ev": -0.02, "actual_lift": 0.05,
         "required_lift": 0.12, "rejection_reasons_tr": ["beklenen değer yok"]}]}]
    y = MG.nearest_to_qualification(kartlar)
    assert y and y[0]["not_a_trade"] is True
    assert any("eksik" in m for m in y[0]["missing"])


# ══════════════════════════════════════════════════════════════════════════
# 15. KANIT SİCİLİ: GÖSTERGE VE FORMASYON YÖN AĞIRLIĞI SIFIR
#     (şartname 31, 32 · 2. mesaj 38, 39)
# ══════════════════════════════════════════════════════════════════════════

def test_gosterge_yon_agirligi_sifir():
    assert RG.INDICATOR_DIRECTIONAL_WEIGHT == 0.0


def test_hicbir_aile_yon_agirligi_tasimaz():
    for r in RG.signal_registry():
        assert r["directional_weight"] == 0.0, r["pattern_name"]
        assert r["status"] in (RG.VERIFIED, RG.UNVERIFIED, RG.REFUTED,
                               RG.UNMEASURED)


def test_curutulen_aile_gosterilebilir_ama_agirlik_alamaz():
    curutulen = [r for r in RG.signal_registry() if r["status"] == RG.REFUTED]
    assert curutulen
    for r in curutulen:
        assert r["display"] == "grafikte gösterilebilir"
        assert r["directional_weight"] == 0.0


def test_olculmemis_aile_etkisiz_sayilmaz():
    olcumsuz = [r for r in RG.signal_registry() if r["status"] == RG.UNMEASURED]
    assert olcumsuz, "UNMEASURED durumu kullanılmıyor"
    for r in olcumsuz:
        assert "veri yok" in r["note"] or "yetersiz" in r["note"]


# ══════════════════════════════════════════════════════════════════════════
# 16. KART / MATRİS ŞEMASI  (şartname 88, 89, 93, 115 · 2. mesaj 33, 36)
# ══════════════════════════════════════════════════════════════════════════

def _hucre(hz="4h", d="LONG", tradable=False, **kw):
    t = dict(horizon=hz, horizon_minutes=HZ.HORIZON_MIN[hz], direction=d,
             p_target_first=0.5, p_target_lower95=0.45, p_stop_first=0.3,
             p_timeout=0.2, baseline=0.4, required_lift=0.1, actual_lift=0.1,
             robust_ev=0.05, robust_utility=0.02, status="QUALIFIED",
             tradable=tradable, rejection_reasons=[], n_eff_used=500,
             expected_holding_hours=3.0, median_hours_to_tp=2.0)
    t.update(kw)
    return t


def test_kart_butun_ufuklari_dondurur():
    hucreler = [_hucre(h, d) for h in HZ.all_horizons() for d in ("LONG", "SHORT")]
    k = pair_card("BTCUSDT", hucreler)
    assert len(k["horizons"]) == len(HZ.all_horizons()) * 2
    assert any(h["reference_only"] for h in k["horizons"])


def test_ayni_coin_tek_kartta_birlestirilir():
    hucreler = [_hucre("2h", "LONG", True, robust_utility=0.02),
                _hucre("4h", "LONG", True, robust_utility=0.09),
                _hucre("8h", "LONG", True, robust_utility=0.05)]
    k = pair_card("BTCUSDT", hucreler)
    assert k["best_horizon"] == "4h"
    assert k["earliest_qualified_horizon"] == "2h"
    assert len(k["horizons"]) == 3, "üç ayrı fırsat gibi listelenmemeli"


def test_kart_semasi_zorunlu_alanlari_tasir():
    k = pair_card("BTCUSDT", [_hucre()])
    for alan in ("symbol", "timestamp", "guaranteed", "best_horizon",
                 "direction", "status", "market_price", "entry_low",
                 "optimal_entry", "entry_high", "max_chase_price",
                 "net_1pct_exit", "stop", "time_exit", "p_target_first",
                 "p_target_first_lower95", "p_stop_first", "p_timeout",
                 "baseline_target_rate", "required_probability_lift",
                 "actual_probability_lift", "expected_net_return",
                 "expected_value", "robust_expected_value", "expected_mfe",
                 "expected_mae", "fill_probability", "execution_probability",
                 "max_capacity_usd", "liquidity_score", "data_quality",
                 "risk_score", "effective_sample_size", "brier_score",
                 "calibration_error", "dsr", "pbo", "model_version",
                 "cost_model", "valid_until", "rejection_reasons"):
        assert alan in k, f"şema alanı eksik: {alan}"


def test_kalifiye_yoksa_valid_until_uretilmez():
    k = pair_card("BTCUSDT", [_hucre(tradable=False)])
    assert k["best_horizon"] is None and k["valid_until"] is None
    assert RejectionCode.NO_QUALIFIED_HORIZON in k["rejection_reasons"]


def test_siralama_ham_olasilikla_yapilmaz():
    a = pair_card("A", [_hucre(tradable=True, robust_utility=0.01,
                               p_target_first=0.95)])
    b = pair_card("B", [_hucre(tradable=True, robust_utility=0.5,
                               p_target_first=0.55, robust_ev=0.4)])
    s = rank_pairs([a, b])
    assert s[0]["symbol"] == "B"


def test_korelasyon_kumesi_uyarisi():
    k1 = pair_card("BTCUSDT", [_hucre(tradable=True)])
    k2 = pair_card("ETHUSDT", [_hucre(tradable=True)])
    o = scanner_summary([k1, k2], 2, 0)
    assert o["correlation_warning"], "aynı faktöre maruz fırsatlar uyarısız"
    assert "CRYPTO_BETA" in o["correlation_warning"][0]


def test_firsat_yoksa_mesaj_uydurma_sinyal_icermez():
    o = scanner_summary([], 5, 3)
    assert o["qualified_markets"] == 0
    assert "QUALIFIED FIRSAT YOK" in o["empty_message"]
    assert o["markets_scanned"] == 8


# ══════════════════════════════════════════════════════════════════════════
# 17. REJİM  (şartname 20, 21)
# ══════════════════════════════════════════════════════════════════════════

def test_rejim_etiketleri_nedensel_ve_bilinen_kumede():
    d = _seri(20000)
    r = classify(d)
    izin = {"LOW_VOL", "NORMAL_VOL", "HIGH_VOL", "PANIC", "UNKNOWN"}
    assert set(r["vol_regime"].unique()) <= izin
    assert (r["vol_regime"].iloc[:2000] == "UNKNOWN").all(), \
        "ısınma dönemi etiketlenmiş — eşikler geriye dönük kullanılmış olabilir"


def test_likidite_stresi_gecmiste_atanmaz():
    from agi_trader.qualification.regime import regime_note
    n = regime_note()
    assert "LIQUIDITY_STRESS" in n["live_only"]
    assert "UNMEASURED" in n["liquidity_stress"]


def test_hedef_mesafesi_sigma_olculur():
    d = _seri(20000)
    m = measure_symbol("TESTUSDT", d, _profil(), horizons=["1h", "24h"],
                       stop_mults=(1.0,))
    s = {c["horizon"]: c["target_distance_sigma_median"] for c in m.cells
         if c["direction"] == "LONG" and c["regime"] == "ALL"}
    assert s["1h"] > s["24h"], "uzun ufukta hedef daha yakın olmalı (sigma)"


# ══════════════════════════════════════════════════════════════════════════
# 18. STOP DAVRANIŞI  (şartname 26, 27)
# ══════════════════════════════════════════════════════════════════════════

def test_stop_ufukla_olceklenir():
    d = _seri(10000)
    sb = sigma_bar_pct(d)
    kisa = stop_grid(sb, 12, (1.0,))[1.0]
    uzun = stop_grid(sb, 288, (1.0,))[1.0]
    assert np.nanmedian(uzun) > np.nanmedian(kisa)


def test_genis_stop_hedef_oranini_yukseltir_ama_rr_raporlanir():
    d = _seri(20000)
    m = measure_symbol("TESTUSDT", d, _profil(), horizons=["4h"],
                       stop_mults=(0.5, 2.0))
    h = {c["stop_sigma_mult"]: c for c in m.cells
         if c["direction"] == "LONG" and c["regime"] == "ALL"}
    assert h[2.0]["p_target_first"] > h[0.5]["p_target_first"]
    assert h[2.0]["rr_median"] < h[0.5]["rr_median"], "R/R raporlanmıyor"


# ══════════════════════════════════════════════════════════════════════════
# 19. MFE/MAE VE KAPASİTE  (şartname 22, 64 · 2. mesaj 18, 19)
# ══════════════════════════════════════════════════════════════════════════

def test_mfe_mae_kantilleri_uretilir():
    d = _seri(20000)
    m = measure_symbol("TESTUSDT", d, _profil(), horizons=["4h"],
                       stop_mults=(1.0,))
    c = next(x for x in m.cells if x["regime"] == "ALL")
    for q in ("p10", "p25", "p50", "p75", "p90"):
        assert c["mfe"][q] is not None and c["mae"][q] is not None
    assert c["mfe"]["p90"] >= c["mfe"]["p50"] >= c["mfe"]["p10"]


def test_kapasite_egrisi_sermayeye_gore_bozulur():
    from agi_trader.opportunity.costs import capacity_curve
    # Defter 50 bin $'a kadar kayıtlı → 100 bin $'lık emir KAPSAM DIŞI olmalı
    egri = {1: 8_000.0, 2: 16_000.0, 5: 32_000.0, 10: 50_000.0}
    c = capacity_curve((1e6, 1e6), 2.0, 1.0,
                       levels=(100, 1_000, 10_000, 100_000),
                       bid_curve=egri, ask_curve=egri)
    gerekli = [x["required_gross_pct"] for x in c if x["required_gross_pct"]]
    assert gerekli == sorted(gerekli), "büyük emirde gereken hareket artmıyor"
    assert any(not x["feasible"] for x in c), "defteri aşan emir fizibil sayılmış"


# ══════════════════════════════════════════════════════════════════════════
# 20. DSR ÖLÇEK TUTARLILIĞI  (şartname 40, 41 — ölçülen ve düzeltilen tuzak)
# ══════════════════════════════════════════════════════════════════════════

def test_yillik_sharpe_ufka_gore_olceklenir():
    """Aynı işlem-başı kenar, kısa ufukta daha yüksek YILLIK Sharpe verir."""
    from agi_trader.qualification.research import (_annual_sharpe,
                                                   trades_per_year)
    assert trades_per_year(0.5) > trades_per_year(24.0)
    ortak = dict(p_target_first=0.30, p_stop_first=0.30, p_timeout=0.40,
                 stop_pct_median=1.0, timeout_return_pct=0.0,
                 median_hours_to_tp=None, median_hours_to_sl=None)
    kisa = _annual_sharpe({**ortak, "horizon_hours": 0.5,
                           "direction": "LONG"}, _profil())
    uzun = _annual_sharpe({**ortak, "horizon_hours": 24.0,
                           "direction": "LONG"}, _profil())
    assert abs(kisa) > abs(uzun) * 3, (kisa, uzun)


def test_dsr_dagilimi_ayni_olcekten_alinir():
    """Havuzlanmış dağılım ufuklar arası √(işlem/yıl) farkını içerir; DSR
    bunu kullanırsa her hücrede 0 çıkar. Grup adı verilince gruptan alınır."""
    import agi_trader.research.validation as _V
    from agi_trader.qualification import research as R

    cagri = {}

    def sahte(name=None, output_dir="runs"):
        cagri["name"] = name
        return {"qual_stop_4h_LONG": [1.0, 1.2, 0.9, 1.1]}.get(
            name, [100.0, -100.0, 50.0, -50.0])

    eski = _V.trial_sharpes
    R.V.trial_sharpes = sahte
    try:
        grup = R._trial_dispersion("qual_stop_4h_LONG")
        havuz = R._trial_dispersion()
    finally:
        R.V.trial_sharpes = eski
    assert grup < 1.0 and havuz > 50.0, (grup, havuz)


def test_deflated_sharpe_periyot_birimi_hucreden_gelir():
    """periods_per_year varsayılan 365'te bırakılırsa DSR yanlış ölçeklenir."""
    from agi_trader.research.validation import deflated_sharpe
    rng = np.random.default_rng(5)
    r = rng.normal(0.02, 1.0, 500)
    a = deflated_sharpe(r, 100, sr_std=1.0, periods_per_year=365.0)
    b = deflated_sharpe(r, 100, sr_std=1.0, periods_per_year=17520.0)
    assert a["sr_annual"] != b["sr_annual"]
    assert b["sr_annual"] > a["sr_annual"]


def test_horizon_row_idempotent():
    """`pair_card` kendi çıktısını yeniden işleyebilmeli.

    Canlı katman matrix.json'daki HAZIR satırı alıp tekrar pair_card'a verir.
    İlk geçişte yeniden adlandırılan alanlar ikinci geçişte kaybolursa
    panelde ALT95 / R/R / ETKİN N / HEDEF σ sütunları boşalır (ölçüldü)."""
    h = _hucre(tradable=True)
    h.update({"p_target_lower95": 0.44, "rr_median": 2.5, "n_eff_used": 830.0,
              "target_distance_sigma_median": 1.28, "stop_pct_median": 0.9,
              "stop_price": 99.0,
              "mfe": {"p10": 0.1, "p50": 0.63, "p90": 2.1},
              "mae": {"p10": -0.1, "p50": -0.41, "p90": -1.8}})
    bir = pair_card("BTCUSDT", [h])
    iki = pair_card("BTCUSDT", bir["horizons"])

    # ⚠️ BU TEST ESKİDEN BEŞ ALANI TEK TEK SAYIYORDU ve tam da bu yüzden
    # `mfe_p50`/`mae_p50` gözden kaçtı: listede yoklardı, iç içe adla
    # okunuyorlardı ve ikinci geçişte sessizce NULL'a düşüyorlardı (canlıda
    # 540/540 boş). Artık alan SAYILMAZ — HİÇBİR alan kaybolamaz.
    b0, i0 = bir["horizons"][0], iki["horizons"][0]
    kaybolan = [a for a, v in b0.items()
                if v is not None and i0.get(a) is None]
    assert not kaybolan, f"ikinci geçişte kaybolan alanlar: {kaybolan}"
    degisen = [a for a, v in b0.items() if i0.get(a) != v]
    assert not degisen, f"ikinci geçişte değişen alanlar: {degisen}"

    # Kart düzeyi de aynı kuralla korunur
    kayip_kart = [a for a, v in bir.items()
                  if a != "horizons" and v is not None and iki.get(a) is None]
    assert not kayip_kart, f"kart düzeyinde kaybolan: {kayip_kart}"

    assert b0["mfe_p50"] == 0.63 and b0["mae_p50"] == -0.41
    assert iki["stop"] == bir["stop"] == 99.0


# ══════════════════════════════════════════════════════════════════════════
# 21. YAKINSAMA — "bu sayı kesinleşiyor mu?"
# ══════════════════════════════════════════════════════════════════════════

def test_bagimsiz_gozlemde_aralik_dogru_daralir():
    """4× veri → aralık yarıya inmeli. İnmiyorsa ESS düzeltmesi bozuktur."""
    from agi_trader.qualification.convergence import shrink_curve
    rng = np.random.default_rng(7)
    lab = (rng.random(20000) < 0.3).astype(int)
    egri, oran = shrink_curve(lab, 1, 1)
    assert oran is not None and 0.40 < oran < 0.60, oran
    assert [r["n"] for r in egri] == sorted(r["n"] for r in egri)


def test_ortusen_etiketlerde_ess_duzeltmesi_daralmayi_korur():
    """Her gözlem 10 kez tekrarlansa bile ESS düzeltmesi devredeyse daralma
    normal kalır — bu, düzeltmenin ÇALIŞTIĞININ kanıtıdır."""
    from agi_trader.qualification.convergence import shrink_curve
    rng = np.random.default_rng(11)
    taban = (rng.random(2000) < 0.3).astype(int)
    _, oran = shrink_curve(np.repeat(taban, 10), 1, 10)
    assert oran is not None and oran < 0.60, oran


def test_kayan_tahmin_UNSTABLE():
    """Dönemler arası 30 puan fark → daha çok veri BUNU düzeltmez."""
    from agi_trader.qualification import convergence as CV
    rng = np.random.default_rng(3)
    lab = (rng.random(20000) < 0.3).astype(int)
    r = CV.assess(lab, 1, 1, 20000.0,
                  {"2024": (2000, 10000), "2025": (5000, 10000)}, None)
    assert r.verdict == CV.UNSTABLE
    assert any("KAYIYOR" in x for x in r.reasons)


def test_az_orneklem_CONVERGING_ve_UNSTABLE_degil():
    """Örneklem azlığı ile gerçek kayma AYRI teşhis edilmeli."""
    from agi_trader.qualification import convergence as CV
    rng = np.random.default_rng(5)
    lab = (rng.random(300) < 0.3).astype(int)
    r = CV.assess(lab, 1, 1, 120.0, {"a": (40, 150), "b": (44, 150)}, None)
    assert r.verdict == CV.CONVERGING
    assert r.checks["temporally_stable"] is True


def test_rejim_bagliligi_zamansal_kaymadan_AYRI():
    """İkisi farklı teşhistir ve aynı etiketle işaretlenemez.

    ÖLÇÜLDÜ: BTCUSDT 4h LONG kör tabanı LOW_VOL %8,1 → PANIC %48,4 (40,3 puan).
    Bu ölçüm hatası değil, geometridir: hedef sabit yüzde, stop oynaklıkla
    ölçekleniyor. Veri biriktirmek bunu 'düzeltmez' çünkü bozuk değildir —
    ama zamansal kayma gerçekten bozulmadır."""
    from agi_trader.qualification import convergence as CV
    rng = np.random.default_rng(9)
    lab = (rng.random(20000) < 0.3).astype(int)

    rejim = CV.assess(lab, 1, 1, 20000.0,
                      {"a": (3000, 10000), "b": (3010, 10000)},
                      {"LOW_VOL": (500, 5000), "HIGH_VOL": (2500, 5000)})
    assert rejim.verdict == CV.REGIME_DEPENDENT
    assert any("KOŞULLU" in x for x in rejim.reasons)
    assert rejim.checks["temporally_stable"] is True

    zaman = CV.assess(lab, 1, 1, 20000.0,
                      {"a": (2000, 10000), "b": (5000, 10000)},
                      {"LOW_VOL": (1500, 5000), "HIGH_VOL": (1510, 5000)})
    assert zaman.verdict == CV.UNSTABLE
    assert any("KAYIYOR" in x for x in zaman.reasons)

    # Zamansal kayma rejim bağlılığını EZER — en ağır teşhis kazanır
    ikisi = CV.assess(lab, 1, 1, 20000.0,
                      {"a": (2000, 10000), "b": (5000, 10000)},
                      {"LOW_VOL": (500, 5000), "HIGH_VOL": (2500, 5000)})
    assert ikisi.verdict == CV.UNSTABLE


def test_grup_yoksa_UNMEASURED():
    """Yakınsama ölçülemediyse 'kararlı' DENMEZ."""
    from agi_trader.qualification import convergence as CV
    rng = np.random.default_rng(13)
    lab = (rng.random(5000) < 0.3).astype(int)
    r = CV.assess(lab, 1, 1, 5000.0, None, None)
    assert r.verdict == CV.UNMEASURED


def test_yetersiz_grup_olcume_girmez_ama_gorunur():
    from agi_trader.qualification.convergence import group_spread
    yayilim, liste = group_spread({"buyuk": (300, 1000), "kucuk": (1, 5)})
    assert yayilim is None, "5 gözlemli grup yayılıma katılmamalı"
    assert any(g["group"] == "kucuk" and g["enough"] is False for g in liste)


def test_converged_kesin_demek_degildir():
    from agi_trader.qualification import convergence as CV
    rng = np.random.default_rng(17)
    lab = (rng.random(20000) < 0.3).astype(int)
    r = CV.assess(lab, 1, 1, 20000.0, {"a": (3000, 10000), "b": (3020, 10000)},
                  {"LOW_VOL": (1500, 5000), "HIGH_VOL": (1520, 5000)})
    assert r.verdict == CV.CONVERGED
    assert "kesin" in r.note and "DEĞİLDİR" in r.note


def test_yakinsama_verdiktleri_turkce_karsiligi_var():
    from agi_trader.qualification import convergence as CV
    for v in (CV.CONVERGED, CV.CONVERGING, CV.REGIME_DEPENDENT,
              CV.UNSTABLE, CV.UNMEASURED):
        assert v in CV.VERDICT_TR and v in CV.VERDICT_COLOR


# ══════════════════════════════════════════════════════════════════════════
# 22. JSON GEÇERLİLİĞİ  (ölçülen 500 hatası)
# ══════════════════════════════════════════════════════════════════════════

def test_sonsuz_maliyet_json_bozmaz():
    """İnce defterde emir dolmazsa maliyet sonsuz olur; JSON'da inf GEÇERSİZ.
    Evren 8→27 pariteye çıkınca /api/qualification 500 vermişti."""
    from agi_trader.qualification.live import json_safe
    ham = {"cost": float("inf"), "p": float("nan"), "ok": 1.5,
           "alt": [float("-inf"), 2.0], "np": np.float64("inf"),
           "warnings": ["emir kaydedilen derinliği aşıyor"]}
    temiz = json_safe(ham)
    metin = json.dumps(temiz)          # inf olsaydı burada patlardı
    assert temiz["cost"] is None and temiz["p"] is None
    assert temiz["alt"] == [None, 2.0] and temiz["np"] is None
    assert temiz["ok"] == 1.5
    assert "derinliği aşıyor" in temiz["warnings"][0], "uyarı kaybolmamalı"
    assert "Infinity" not in metin and "NaN" not in metin


def test_json_safe_numpy_tiplerini_cevirir():
    from agi_trader.qualification.live import json_safe
    d = json_safe({"i": np.int64(5), "b": np.bool_(True),
                   "a": np.array([1.0, np.inf])})
    json.dumps(d)
    assert d["i"] == 5 and d["b"] is True and d["a"] == [1.0, None]
