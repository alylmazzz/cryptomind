"""
Fırsat motoru testleri — net %1 eşiği, üçlü bariyer, NO_TRADE varsayılanı.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agi_trader.opportunity import (estimate_costs, required_gross_move_pct,
                                    capacity_curve, first_passage, base_rate,
                                    Opportunity, Gates, evaluate, rank,
                                    build_price_opportunity)


def _bars(seq):
    """(high, low) listesinden bar çerçevesi."""
    h = [x[0] for x in seq]
    l = [x[1] for x in seq]
    return pd.DataFrame({"high": h, "low": l, "close": l, "open": h},
                        index=pd.date_range("2024-01-01", periods=len(seq), freq="4h"))


# ------------------------------------------------------------------ maliyet
def test_net_hedef_carpimsal_hesaplanir():
    """Toplama DEĞİL çarpma: net %1 için gereken brüt hareket, maliyetin
    doğrudan toplanmasından büyüktür — maliyet brüt tutar üzerinden alınır."""
    c = estimate_costs(10_000, 5e6, 5e6, spread_bps=1.0)
    g = required_gross_move_pct(1.0, c)
    assert g > 1.0 + c.total_pct - 1e-9, "çarpımsal hesap toplamadan küçük çıkamaz"
    assert 1.0 < g < 1.5


def test_derinlik_birimi_dolar_olmali():
    """REGRESYON — kaydedici `bid_depth`'i BAZ VARLIK cinsinden tutuyor.
    Dolar sanıp doğrudan vermek BTC'de "derinlik 34× aşıldı" gibi saçma sonuç
    üretiyordu; gerçek derinlik ~15 M$."""
    az = estimate_costs(10_000, 245, 294, 0.016)          # yanlış birim (adet)
    dogru = estimate_costs(10_000, 15.5e6, 18.6e6, 0.016)
    assert not np.isfinite(az.total_bps), "birim hatası yakalanmıyor"
    assert np.isfinite(dogru.total_bps) and dogru.total_bps < 30


def test_sermaye_egrisi_artan_maliyet():
    """Fırsat tek bir yüzde değildir: nominal büyüdükçe maliyet artmalı."""
    rows = capacity_curve((5e6, 5e6), 1.0, 1.0)
    uygun = [r for r in rows if r["feasible"]]
    assert len(uygun) >= 5
    maliyetler = [r["cost_bps"] for r in uygun]
    assert maliyetler == sorted(maliyetler), "maliyet nominalle artmıyor"


def test_derinlik_asilirsa_uygulanamaz():
    c = estimate_costs(1_000_000, 100_000, 100_000, 1.0)
    assert not np.isfinite(c.total_bps)
    assert any("derinli" in w for w in c.warnings)


def test_funding_yonu_dogru_ve_asimetrik():
    """LONG pozitif funding ÖDER, SHORT tahsil eder — ama tahsil edilen
    funding MALİYETİ AZALTMAZ.

    Davranış BİLEREK değişti (eski test `s.funding_bps < 0` bekliyordu):
    gelecekteki funding bilinmediği için beklenen kazancı hedefe saymak,
    brüt hedefi net hedefin altına indiriyordu. Kazanç yok sayılmaz —
    `funding_credit_bps` ile ayrı raporlanır, yalnız toplama girmez."""
    l = estimate_costs(1000, 5e6, 5e6, 1.0, funding_rate_8h=0.0001, direction="LONG")
    s = estimate_costs(1000, 5e6, 5e6, 1.0, funding_rate_8h=0.0001, direction="SHORT")
    assert l.funding_bps > 0, "ödenecek funding maliyete girmeli"
    assert s.funding_bps == 0.0, "tahsil edilecek funding maliyeti AZALTMAZ"
    assert s.funding_credit_bps > 0, "kazanç kaybolmamalı, ayrı raporlanmalı"
    assert s.total_bps > 0


# ----------------------------------------------------------------- bariyer
def test_hedef_once_gelirse_arti_bir():
    fut = _bars([(101.5, 99.8), (102.5, 100.0)])
    r = first_passage(100.0, 1.0, 1.0, "LONG", fut)
    assert r.label == 1 and r.bars_to_hit == 1


def test_stop_once_gelirse_eksi_bir():
    fut = _bars([(100.2, 98.5), (101.5, 99.0)])
    r = first_passage(100.0, 1.0, 1.0, "LONG", fut)
    assert r.label == -1


def test_ayni_barda_ikisi_de_varsa_BELIRSIZ():
    """EN KRİTİK: 4H mumda hem +%1 hem −%1 görülmüşse high/low hangisinin ÖNCE
    geldiğini SÖYLEMEZ. "Hedefe ulaştı" demek look-ahead hatasıdır."""
    fut = _bars([(101.5, 98.5)])
    r = first_passage(100.0, 1.0, 1.0, "LONG", fut)
    assert r.label is None, "belirsiz örnek etiketlenmiş — look-ahead riski"
    assert r.resolved_by == "belirsiz"


def test_alt_zaman_dilimi_belirsizligi_cozer():
    fut = _bars([(101.5, 98.5)])
    sub = _bars([(100.1, 98.5), (101.5, 99.0)])     # önce aşağı gitmiş
    r = first_passage(100.0, 1.0, 1.0, "LONG", fut, resolve_with=sub)
    assert r.label == -1 and r.resolved_by == "alt_bar"


def test_short_yonunde_bariyerler_aynalanir():
    fut = _bars([(100.2, 98.5)])
    r = first_passage(100.0, 1.0, 1.0, "SHORT", fut)
    assert r.label == 1, "SHORT'ta aşağı hareket HEDEF olmalı"


def test_taban_orani_toplami_yuz():
    rng = np.random.default_rng(3)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    df = pd.DataFrame({"high": c * 1.004, "low": c * 0.996, "close": c, "open": c},
                      index=pd.date_range("2024-01-01", periods=300, freq="4h"))
    r = base_rate(df, 1.0, 1.0, horizon_bars=6)
    assert r["n_evaluated"] > 100
    toplam = r["hit_first_pct"] + r["stop_first_pct"] + r["timeout_pct"]
    assert abs(toplam - 100.0) < 0.5, "oranlar 100'e toplanmıyor"


# ------------------------------------------------------------------ motor
def test_varsayilan_NO_TRADE():
    """Boş bir fırsat kapılardan geçmemeli — varsayılan REDDİR."""
    op = Opportunity(id="x", strategy="t", symbol="BTC/USDT",
                     direction="LONG", horizon="4h")
    evaluate(op, Gates())
    assert op.action == "NO_TRADE" and op.reject_reasons


def test_negatif_EV_reddedilir():
    """Net getiri %1'i geçse bile EV negatifse yayımlanmaz."""
    op = Opportunity(id="x", strategy="t", symbol="BTC/USDT", direction="LONG",
                     horizon="4h", net_return_pct=1.5,
                     p_target_first=0.30, p_stop_first=0.65, p_timeout=0.05,
                     execution_probability=0.9, liquidity_score=0.9,
                     data_quality=1.0, confidence=0.9, risk_score=10)
    op.expected_value_pct = 0.30 * 1.5 - 0.65 * 1.5
    evaluate(op, Gates())
    assert op.action == "NO_TRADE"
    assert any("BEKLENEN_DEGER" in x for x in op.reject_reasons)


def test_tum_kapilar_gecilirse_yayimlanir():
    op = Opportunity(id="x", strategy="t", symbol="BTC/USDT", direction="LONG",
                     horizon="4h", net_return_pct=1.3,
                     p_target_first=0.62, p_stop_first=0.30, p_timeout=0.08,
                     expected_value_pct=0.42,
                     execution_probability=0.85, liquidity_score=0.9,
                     data_quality=0.95, confidence=0.72, risk_score=25)
    evaluate(op, Gates())
    assert op.action == "BUY" and not op.reject_reasons
    assert op.score > 0


def test_kalibre_edilmemis_esikler_beyan_edilir():
    op = Opportunity(id="x", strategy="t", symbol="BTC/USDT", direction="LONG",
                     horizon="4h", net_return_pct=1.3, expected_value_pct=0.4,
                     p_target_first=0.6, p_stop_first=0.3, p_timeout=0.1,
                     execution_probability=0.9, liquidity_score=0.9,
                     data_quality=1.0, confidence=0.8, risk_score=20)
    evaluate(op, Gates(calibrated=False))
    assert any("KALİBRE EDİLMEDİ" in n for n in op.notes)


def test_siralama_yuzdeye_gore_degil_skora_gore():
    a = Opportunity(id="a", strategy="t", symbol="A", direction="LONG", horizon="4h",
                    net_return_pct=3.0, expected_value_pct=0.05,
                    p_target_first=0.4, p_stop_first=0.35, p_timeout=0.25,
                    execution_probability=0.62, liquidity_score=0.62,
                    data_quality=0.85, confidence=0.62, risk_score=65)
    b = Opportunity(id="b", strategy="t", symbol="B", direction="LONG", horizon="4h",
                    net_return_pct=1.2, expected_value_pct=0.55,
                    p_target_first=0.7, p_stop_first=0.22, p_timeout=0.08,
                    execution_probability=0.95, liquidity_score=0.95,
                    data_quality=0.99, confidence=0.85, risk_score=12)
    for o in (a, b):
        evaluate(o, Gates())
    sirali = rank([a, b])
    assert sirali and sirali[0].id == "b", "yüksek yüzde düşük EV'yi geçmiş"


def test_build_price_opportunity_hedefi_maliyet_sonrasi_koyar():
    op = build_price_opportunity(
        "BTC/USDT", "LONG", entry=100.0, target_net_pct=1.0, stop_pct=1.0,
        bid_depth=5e6, ask_depth=5e6, spread_bps=1.0, notional=10_000,
        p_target=0.6, p_stop=0.32, p_timeout=0.08, confidence=0.7)
    assert op.target > 101.0, "hedef maliyet eklenmeden konmuş"
    assert op.stop < 100.0
    assert op.gross_return_pct > 1.0


# ------------------------------------------------------- L2 gerçek VWAP
def _egri(**kw):
    return {float(k.replace("b", "").replace("_", ".")): v for k, v in kw.items()}


def test_gercek_vwap_egriden_yurur():
    """Kümülatif eğri: 0-1 bps'te 100k, 1-2 bps'te 100k daha.
    150k'lık emir → 100k @ ~0,5 bps + 50k @ ~1,5 bps = 0,833 bps."""
    from agi_trader.opportunity.costs import vwap_offset_bps
    curve = {1.0: 100_000.0, 2.0: 200_000.0, 5.0: 500_000.0}
    v, u = vwap_offset_bps(curve, 150_000)
    assert u is None
    assert abs(v - (100_000 * 0.5 + 50_000 * 1.5) / 150_000) < 1e-6


def test_gercek_vwap_derinligi_asarsa_sonsuz():
    from agi_trader.opportunity.costs import vwap_offset_bps
    v, u = vwap_offset_bps({1.0: 10_000.0, 2.0: 20_000.0}, 500_000)
    assert not np.isfinite(v) and "aşıyor" in u


def test_sansurlu_kova_dolu_sayilmaz():
    """None = defter o uzaklığa ulaşmıyor. Sıfır sayılırsa likidite olduğundan
    ÇOK görünür; atlanırsa eğri erken biter ve emir 'aşıyor' der — doğrusu bu."""
    from agi_trader.opportunity.costs import vwap_offset_bps
    curve = {1.0: 50_000.0, 2.0: 90_000.0, 50.0: None, 100.0: None}
    v, u = vwap_offset_bps(curve, 500_000)
    assert not np.isfinite(v), "sansürlü kova dolu sayılmış"


def test_yarim_spread_tabani_kucuk_emirde_dogru():
    """Tamamen best ask'te dolan emrin mid'e uzaklığı yarım spread'dir;
    taban verilmezse etki olduğundan büyük çıkar."""
    from agi_trader.opportunity.costs import vwap_offset_bps
    curve = {1.0: 100_000.0}
    tabansiz, _ = vwap_offset_bps(curve, 1_000)
    tabanli, _ = vwap_offset_bps(curve, 1_000, min_offset_bps=0.2)
    assert tabanli > tabansiz - 1e-9
    assert abs(tabanli - 0.6) < 1e-6          # (0,2+1,0)/2


def test_gercek_egri_kullanilinca_spread_cift_sayilmaz():
    """REGRESYON — eğri mid'den ölçülür, giriş+çıkış TAM spread'i içerir.
    Spread'i ayrıca eklemek maliyeti iki kez saymaktır."""
    curve = {1.0: 1e6, 2.0: 2e6, 5.0: 5e6}
    ile = estimate_costs(10_000, 5e6, 5e6, spread_bps=6.0,
                         bid_curve=curve, ask_curve=curve)
    assert ile.model == "gercek_L2_vwap"
    assert ile.spread_bps == 0.0, "gerçek eğri modunda spread çift sayılıyor"
    yaklasik = estimate_costs(10_000, 5e6, 5e6, spread_bps=6.0)
    assert yaklasik.spread_bps == 6.0
    assert yaklasik.model == "dogrusal_defter_yaklasimi"


def test_ladder_from_row_sansurluyu_atlar():
    from agi_trader.opportunity.costs import ladder_from_row
    row = {"ask_cum_1bps": 1000.0, "ask_cum_2bps": 2000.0,
           "ask_cum_50bps": None, "ask_cum_100bps": float("nan")}
    c = ladder_from_row(row, "ask")
    assert set(c) == {1.0, 2.0}, "sansürlü/NaN kova eğriye girmiş"
