"""FAIL-FAST ŞEMA VE VERİ KALİTESİ TESTLERİ.

Bu projede iki kez "kod çalıştı, sayı çıktı, sayı yanlıştı" oldu:
  • `bid_depth` baz varlık cinsindeyken dolar sanıldı
  • DSR'a işlem-başı Sharpe verildi, fonksiyon yıllık bekliyordu

İkisi de birim hatasıydı ve sessizce geçti. Bu testler aynı sınıfın
çalışma anında yakalanmasını kilitler.
"""
from __future__ import annotations

import pytest

from agi_trader.qualification.schema import (CARD_REQUIRED, FIELD_UNITS,
                                             SchemaError, check_card,
                                             check_probabilities, check_scan,
                                             check_units, data_quality)


def _kart(**kw):
    t = dict(symbol="BTCUSDT", timestamp="2026-08-18T00:00:00Z",
             guaranteed=False, best_horizon=None, direction=None,
             status="NO_EDGE", market_price=100.0, p_target_first=0.4,
             p_target_first_lower95=0.35, p_stop_first=0.35, p_timeout=0.25,
             baseline_target_rate=0.3, required_probability_lift=0.1,
             actual_probability_lift=0.1, robust_expected_value=-0.02,
             cost_model="MEASURED_L2_VWAP", rejection_reasons=[], horizons=[])
    t.update(kw)
    return t


# ── birim ve aralık ────────────────────────────────────────────────────

def test_oran_alani_1_asamaz():
    v = check_units({"p_target_first": 1.5}, "x")
    assert len(v) == 1 and "üst sınır" in v[0].problem


def test_yuzde_alani_negatif_olamaz():
    v = check_units({"stop_pct": -3.0}, "x")
    assert len(v) == 1 and "alt sınır" in v[0].problem


def test_gecerli_degerler_ihlal_uretmez():
    assert check_units({"p_target_first": 0.4, "stop_pct": 1.2,
                        "spread_bps": 2.0, "max_capacity_usd": 5000.0}) == []


def test_sayi_olmayan_deger_yakalanir():
    v = check_units({"p_target_first": "yüksek"}, "x")
    assert len(v) == 1 and "sayı değil" in v[0].problem


def test_sonsuz_deger_sayi_sayilmaz():
    v = check_units({"cost_pct": float("inf")}, "x")
    assert len(v) == 1


def test_None_ihlal_degildir():
    """Eksik veri yanlış veriden iyidir; None sessizce geçer."""
    assert check_units({"p_target_first": None}) == []


# ── olasılık tutarlılığı ───────────────────────────────────────────────

def test_olasilik_toplami_1_olmali():
    v = check_probabilities({"p_target_first": 0.5, "p_stop_first": 0.3,
                             "p_timeout": 0.5}, "x")
    assert any("toplamı 1 değil" in y.problem for y in v)


def test_alt_sinir_nokta_tahmini_asamaz():
    v = check_probabilities({"p_target_first": 0.4,
                             "p_target_lower95": 0.6}, "x")
    assert any("AŞIYOR" in y.problem for y in v)


def test_ust_sinir_nokta_tahminin_altinda_olamaz():
    v = check_probabilities({"p_target_first": 0.6,
                             "p_target_upper95": 0.4}, "x")
    assert any("ALTINDA" in y.problem for y in v)


# ── kart şeması ────────────────────────────────────────────────────────

def test_gecerli_kart_ihlal_uretmez():
    assert check_card(_kart(), strict=True) == []


def test_guaranteed_true_sema_ihlalidir():
    with pytest.raises(SchemaError):
        check_card(_kart(guaranteed=True))


def test_guaranteed_eksikse_sema_ihlalidir():
    k = _kart()
    del k["guaranteed"]
    with pytest.raises(SchemaError):
        check_card(k)


def test_zorunlu_alan_eksikse_yakalanir():
    k = _kart()
    del k["cost_model"]
    v = check_card(k, strict=False)
    assert any(x.field == "cost_model" for x in v)


def test_tradable_ve_red_sebebi_bir_arada_olamaz():
    k = _kart(horizons=[{"horizon": "4h", "horizon_minutes": 240,
                         "direction": "LONG", "status": "QUALIFIED",
                         "tradable": True, "rejection_reasons": ["NEGATIVE_EV"],
                         "robust_ev": 0.1}])
    v = check_card(k, strict=False)
    assert any("red sebebi var" in x.problem for x in v)


def test_QUALIFIED_ama_negatif_EV_yakalanir():
    k = _kart(horizons=[{"horizon": "4h", "horizon_minutes": 240,
                         "direction": "LONG", "status": "QUALIFIED",
                         "tradable": False, "robust_ev": -0.05}])
    v = check_card(k, strict=False)
    assert any("Robust EV ≤ 0" in x.problem for x in v)


def test_strict_mod_hata_firlatir_sessizce_gecmez():
    with pytest.raises(SchemaError):
        check_card(_kart(p_target_first=2.0))


def test_gevsek_mod_ihlalleri_dondurur():
    v = check_card(_kart(p_target_first=2.0), strict=False)
    assert v and not isinstance(v, bool)


def test_tarama_her_karti_dogrular():
    v = check_scan({"cards": [_kart(), _kart(symbol="ETHUSDT",
                                             p_target_first=1.9)]},
                   strict=False)
    assert any(x.where == "ETHUSDT" for x in v)


def test_kart_zorunlu_alan_listesi_semayla_uyumlu():
    """Şema listesi ile gerçek kart üretimi ayrışmamalı."""
    from agi_trader.qualification.matrix import pair_card
    k = pair_card("BTCUSDT", [])
    eksik = [a for a in CARD_REQUIRED if a not in k]
    assert not eksik, f"pair_card zorunlu alanları üretmiyor: {eksik}"


# ── veri kalitesi ──────────────────────────────────────────────────────

def test_tam_saglikli_veri_yuksek_skor():
    d = data_quality(60, 900, 1.0, 1.0, 0, "MEASURED_L2_VWAP")
    assert d.score > 95 and not d.reasons


def test_bayat_veri_skoru_dusurur():
    d = data_quality(5000, 900, 1.0, 1.0, 0, "MEASURED_L2_VWAP")
    assert d.score < 90 and any("veri yaşı" in r for r in d.reasons)


def test_olculemeyen_bilesen_sifir_sayilir():
    """'Ölçemedim' ile 'sorun yok' AYNI ŞEY DEĞİLDİR."""
    d = data_quality(None, 900, None, None, 0, None)
    assert d.components["freshness"] == 0.0
    assert d.components["book_coverage"] == 0.0
    assert d.score < 30


def test_sema_ihlali_kalite_bilesenini_sifirlar():
    d = data_quality(60, 900, 1.0, 1.0, 3, "MEASURED_L2_VWAP")
    assert d.components["schema"] == 0.0
    assert any("şema ihlali" in r for r in d.reasons)


def test_tahmini_maliyet_modeli_kaliteyi_dusurur():
    iyi = data_quality(60, 900, 1.0, 1.0, 0, "MEASURED_L2_VWAP")
    tahmini = data_quality(60, 900, 1.0, 1.0, 0, "ESTIMATED")
    assert tahmini.score < iyi.score
    assert any("gerçek L2 değil" in r for r in tahmini.reasons)


def test_bilesenler_her_zaman_ayri_gorunur():
    """Tek skor hangi boyutun bozuk olduğunu gizler."""
    d = data_quality(60, 900, 1.0, 1.0, 0, "MEASURED_L2_VWAP")
    assert set(d.components) == {"freshness", "completeness", "book_coverage",
                                 "schema", "cost_model"}
    assert "VARSAYILMAZ" in d.note


def test_bilinen_alanlarin_birimi_tanimli():
    for ad, k in FIELD_UNITS.items():
        assert k.get("unit"), ad
        assert "min" in k and "max" in k, ad


# ══════════════════════════════════════════════════════════════════════════
# FUNDING ASİMETRİSİ (şema denetiminin yakaladığı gerçek hata)
# ══════════════════════════════════════════════════════════════════════════

def test_funding_kazanci_maliyeti_negatife_dusuremez():
    """ÖLÇÜLDÜ: canlı negatif funding `cost_pct`'i eksiye düşürüp brüt hedefi
    net hedefin ALTINA indiriyordu ('net %1 için %0,9 hareket yeter').

    Bu ancak funding üç periyot boyunca aynı işarette kalırsa doğrudur —
    ve gelecekteki funding bilinmez."""
    from agi_trader.opportunity.costs import estimate_costs
    c = estimate_costs(1000, 1e6, 1e6, 2.0, holding_hours=24.0,
                       funding_rate_8h=-0.002, direction="LONG")
    assert c.funding_bps == 0.0
    assert c.funding_credit_bps > 0
    assert c.total_bps > 0, "maliyet negatife düşemez"
    assert any("aleyhte yorumlanır" in w for w in c.warnings)


def test_funding_maliyeti_tam_uygulanir():
    """Asimetri tek yönlü: ödenecek funding TAM sayılır."""
    from agi_trader.opportunity.costs import estimate_costs
    az = estimate_costs(1000, 1e6, 1e6, 2.0, holding_hours=24.0,
                        funding_rate_8h=0.0, direction="LONG")
    cok = estimate_costs(1000, 1e6, 1e6, 2.0, holding_hours=24.0,
                         funding_rate_8h=0.002, direction="LONG")
    assert cok.total_bps > az.total_bps
    assert cok.funding_bps > 0 and cok.funding_credit_bps == 0


def test_brut_hedef_net_hedefin_altina_inemez():
    from agi_trader.opportunity.costs import (estimate_costs,
                                              required_gross_move_pct)
    for oran in (-0.005, -0.001, 0.0, 0.001):
        c = estimate_costs(1000, 1e6, 1e6, 2.0, holding_hours=24.0,
                           funding_rate_8h=oran, direction="LONG")
        assert required_gross_move_pct(1.0, c) >= 1.0, oran


def test_cost_profile_ayni_asimetriyi_uygular():
    from agi_trader.qualification.targets import CostProfile, gross_target_pct
    p = CostProfile("X", spread_bps=2.0, impact_bps_roundtrip=2.0,
                    fee_bps_roundtrip=8.0, reserve_bps=5.0,
                    funding_rate_8h=-0.003, model="ESTIMATED", source="t")
    assert p.total_bps(24.0, "LONG") >= p.base_bps
    assert p.funding_credit_bps(24.0, "LONG") > 0
    assert gross_target_pct(p, 24.0, "LONG") >= 1.0


# ══════════════════════════════════════════════════════════════════════════
# ALAN KAPSAMI — "hesaplandı ve boş" ile "hiç hesaplanmadı" ayrımı
# ══════════════════════════════════════════════════════════════════════════
# Bu blok elle yapılan bir denetimden doğdu: canlı panelde `mfe_p50`,
# `mae_p50` ve `dsr_program` sütunları 540/540 hücrede BOŞTU. Hiçbir test,
# hiçbir sağlık kapısı bunu görmedi — çünkü boş bir sütun geçerli JSON'dur.

def test_hep_bos_alan_yakalanir():
    from agi_trader.qualification.matrix import field_coverage
    kartlar = [{"symbol": "X", "horizons": [
        {"horizon": "1h", "p_target_first": 0.3, "hayalet": None},
        {"horizon": "4h", "p_target_first": 0.4, "hayalet": None}]}]
    k = field_coverage(kartlar)
    assert "hayalet" in k["unexplained"]
    assert k["n_undiagnosed"] == 1
    kayit = [x for x in k["always_null"] if x["field"] == "hayalet"][0]
    assert kayit["expected"] is False and kayit["diagnosed"] is False


def test_kismen_dolu_alan_kusur_sayilmaz():
    """Bazı hücrelerde dolu olan alan eksik DEĞİLDİR — sinyal gürültüye
    boğulmasın diye yalnız TAMAMEN boş alanlar raporlanır."""
    from agi_trader.qualification.matrix import field_coverage
    kartlar = [{"symbol": "X", "horizons": [
        {"horizon": "1h", "bazen": 0.3},
        {"horizon": "4h", "bazen": None}]}]
    assert field_coverage(kartlar)["unexplained"] == []


def test_gerekceli_yokluk_ariza_sayilmaz():
    from agi_trader.qualification.matrix import field_coverage, KNOWN_ABSENT
    kartlar = [{"symbol": "X", "horizons": [
        {"horizon": "1h", "fill_probability": None}]}]
    k = field_coverage(kartlar)
    assert k["unexplained"] == []
    kayit = k["always_null"][0]
    assert kayit["expected"] is True and "taker" in kayit["reason"]
    # dsr_program teşhis EDİLMİŞ ama beklenen DEĞİL — üçüncü durum
    assert KNOWN_ABSENT["dsr_program"][1] is False


def test_teshissiz_bos_alan_sistemi_bozulmus_yapar():
    """Sessiz boş sütun artık sağlık kapısına takılır."""
    from agi_trader.qualification import safety as SF
    tarama = {"generated_at": "2026-08-19T00:00:00Z",
              "cards": [{"symbol": "X", "data_age_sec": 10, "data_quality": 1.0}],
              "scanner": {"schema_violations": 0,
                          "field_coverage": {"unexplained": ["gizli_alan"]}}}
    c = SF.market_data_health(tarama)
    assert c.state == SF.DEGRADED and "teşhis edilmedi" in c.detail
