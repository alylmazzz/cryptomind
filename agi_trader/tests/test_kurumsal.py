"""KURUMSAL KATMAN TESTLERİ — koşu kimliği, mutabakat, güvenlik kapısı,
model kayıt defteri.

Her test, bozulduğunda sessizce yanlış bir güven üretecek bir davranışı korur.
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import pytest

from agi_trader.qualification import attribution as AT
from agi_trader.qualification import ledger as LD
from agi_trader.qualification import model_registry as MR
from agi_trader.qualification import provenance as PV
from agi_trader.qualification import safety as SF


# ══════════════════════════════════════════════════════════════════════════
# KOŞU KİMLİĞİ (§XCI, CI)
# ══════════════════════════════════════════════════════════════════════════

def test_ayni_girdi_ayni_ozet_uretir():
    """Yeniden üretilebilirlik iddiası ancak özet DETERMİNİSTİKSE geçerlidir."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.parquet"
        p.write_bytes(b"x" * 100)
        pkg = Path(d) / "pkg"
        pkg.mkdir()
        (pkg / "m.py").write_text("print(1)", encoding="utf-8")
        a = PV.build(7, [p], pkg, {"k": 1})
        b = PV.build(7, [p], pkg, {"k": 1})
        assert a.dataset["hash"] == b.dataset["hash"]
        assert a.code["source_hash"] == b.code["source_hash"]
        assert a.config_hash == b.config_hash
        assert PV.same_inputs(a.to_dict(), b.to_dict())


def test_veri_degisince_ozet_degisir():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.parquet"
        pkg = Path(d) / "pkg"
        pkg.mkdir()
        (pkg / "m.py").write_text("print(1)", encoding="utf-8")
        p.write_bytes(b"x" * 100)
        a = PV.build(7, [p], pkg, {"k": 1})
        time.sleep(1.1)                      # mtime saniye çözünürlüklü
        p.write_bytes(b"y" * 250)
        b = PV.build(7, [p], pkg, {"k": 1})
        assert a.dataset["hash"] != b.dataset["hash"]
        assert not PV.same_inputs(a.to_dict(), b.to_dict())


def test_kod_degisince_ozet_degisir():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.parquet"
        p.write_bytes(b"x")
        pkg = Path(d) / "pkg"
        pkg.mkdir()
        (pkg / "m.py").write_text("print(1)", encoding="utf-8")
        a = PV.build(7, [p], pkg, {"k": 1})
        (pkg / "m.py").write_text("print(2)", encoding="utf-8")
        b = PV.build(7, [p], pkg, {"k": 1})
        assert a.code["source_hash"] != b.code["source_hash"], \
            "git yoksa bile kod değişimi görünmeli"


def test_config_degisince_ozet_degisir():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.parquet"
        p.write_bytes(b"x")
        pkg = Path(d) / "pkg"
        pkg.mkdir()
        (pkg / "m.py").write_text("x=1", encoding="utf-8")
        a = PV.build(7, [p], pkg, {"stop": 1.0})
        b = PV.build(7, [p], pkg, {"stop": 2.0})
        assert a.config_hash != b.config_hash


def test_seed_kimlige_dahil():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.parquet"
        p.write_bytes(b"x")
        pkg = Path(d) / "pkg"
        pkg.mkdir()
        (pkg / "m.py").write_text("x=1", encoding="utf-8")
        a = PV.build(1, [p], pkg, {})
        b = PV.build(2, [p], pkg, {})
        assert not PV.same_inputs(a.to_dict(), b.to_dict())


# ══════════════════════════════════════════════════════════════════════════
# ANA ÜRÜN METRİĞİ (§CVI)
# ══════════════════════════════════════════════════════════════════════════

def _tahmin(pid, **kw):
    t = dict(prediction_id=pid, timestamp="2026-08-01T06:00:00Z",
             symbol="BTCUSDT", horizon="4h", direction="LONG",
             status="QUALIFIED", entry=100.0, net1_exit=101.2, stop=99.0,
             p_target_first=0.6, p_target_lower95=0.5, p_stop_first=0.3,
             p_timeout=0.1, baseline=0.4, required_lift=0.1, actual_lift=0.2,
             robust_ev=0.20, expected_target_hours=2.0,
             cost_model="MEASURED_L2_VWAP", cost_pct=0.15,
             max_capacity_usd=5000.0, data_quality=1.0, model_version="v1",
             features_hash="h", valid_until="2026-08-01T06:08:00Z")
    t.update(kw)
    return LD.Prediction(**t)


def _defter(d, n=30, tp_orani=0.6, net=0.20, maliyet=0.15):
    led = LD.Ledger(Path(d) / "p.jsonl")
    for i in range(n):
        pid = f"p{i}"
        led.record_prediction(_tahmin(pid))
        tp = i < int(n * tp_orani)
        led.record_outcome(LD.Outcome(
            pid, "t", "TP_FIRST" if tp else "SL_FIRST",
            realized_net_pct=net, realized_cost_pct=maliyet,
            entry_vwap=100.0, predicted_p=0.6))
    return led


def test_tahmin_gerceklesenle_uyumluysa_ALIGNED():
    with tempfile.TemporaryDirectory() as d:
        led = _defter(d, 30, 0.6, net=0.20)
        r = AT.overall(led)
        assert r["verdict"] == "ALIGNED", r
        assert 0.7 <= r["ratio"] <= 1.3


def test_gerceklesen_cok_dusukse_OPTIMISTIC():
    with tempfile.TemporaryDirectory() as d:
        led = _defter(d, 30, 0.3, net=-0.10)
        r = AT.overall(led)
        assert r["verdict"] == "OPTIMISTIC"
        assert any("iyimser" in x for x in r["reasons"])


def test_az_ornekte_oran_uretilmez():
    """5 çözülmüş tahminden 'sistem kendini tanıyor' sonucu ÇIKARILAMAZ."""
    with tempfile.TemporaryDirectory() as d:
        led = _defter(d, 5)
        r = AT.overall(led)
        assert r["verdict"] == "INSUFFICIENT_SAMPLE" and r["ratio"] is None


def test_acik_pozisyon_olcume_girmez():
    """'Henüz kaybetmedi' kazanç sayılmaz."""
    with tempfile.TemporaryDirectory() as d:
        led = LD.Ledger(Path(d) / "p.jsonl")
        for i in range(40):
            led.record_prediction(_tahmin(f"p{i}"))
        r = AT.overall(led)
        assert r["n"] == 0 and r["verdict"] == "INSUFFICIENT_SAMPLE"


def test_sapma_katmanlara_ayrisir():
    """'Model bozuldu' ile 'borsa pahalılaştı' ayırt edilebilmeli."""
    with tempfile.TemporaryDirectory() as d:
        led = _defter(d, 40, tp_orani=0.6, net=0.05, maliyet=0.45)
        r = AT.overall(led)
        assert r["cost_gap"] is not None and r["cost_gap"] < 0, \
            "maliyet tahminden yüksekse cost_gap negatif olmalı"
        assert any("maliyet sapması" in x for x in r["reasons"])


def test_hucre_kirilimi_uretilir():
    with tempfile.TemporaryDirectory() as d:
        led = _defter(d, 30)
        h = AT.by_cell(led)
        assert h and h[0]["symbol"] == "BTCUSDT" and h[0]["horizon"] == "4h"


# ══════════════════════════════════════════════════════════════════════════
# GÜVENLİK KAPISI (§LXXX)
# ══════════════════════════════════════════════════════════════════════════

def test_bilinmeyen_bilesen_saglikli_sayilmaz():
    """Ölçülmemiş bir sağlık, sağlık değildir."""
    h = SF.assess([SF.risk_health(), SF.security_health()])
    assert h.overall == SF.UNKNOWN
    assert h.autopilot is False
    assert "MARKET_DATA" in h.blocking


def test_genel_durum_en_kotu_bilesenden_gelir():
    """Ortalama alınsaydı bir bileşen kırmızıyken sistem 'çoğunlukla iyi'
    görünürdü — güvenlik kapısının tam tersi."""
    h = SF.assess([
        SF.ComponentHealth("MARKET_DATA", SF.GREEN),
        SF.ComponentHealth("MODELS", SF.GREEN),
        SF.ComponentHealth("EXECUTION", SF.GREEN),
        SF.ComponentHealth("RISK", SF.RED, "limit yok"),
        SF.ComponentHealth("RECONCILIATION", SF.GREEN),
        SF.ComponentHealth("SECURITY", SF.GREEN)])
    assert h.overall == SF.RED and h.autopilot is False
    assert h.blocking == ["RISK"]


def test_hepsi_yesilse_otopilot_acilabilir():
    h = SF.assess([SF.ComponentHealth(a, SF.GREEN) for a in SF.COMPONENTS])
    assert h.overall == SF.GREEN and h.autopilot is True


def test_canli_mod_EMS_olmadan_KIRMIZI():
    """Canlı açık ama idempotency yoksa çift emir riski vardır."""
    c = SF.execution_health(live_enabled=True, ems_ready=False)
    assert c.state == SF.RED and "çift emir" in c.detail


# ── "KURULMADI" ≠ "BOZULDU" ────────────────────────────────────────────────
# Bu blok bir kullanıcı şikâyetinden doğdu: panel "SİSTEM bozulmuş · OTOPİLOT
# KAPALI" yazıyordu. Altı bileşenden beşi GREEN'di; altıncısı (EXECUTION)
# hiç kurulmamıştı, çünkü bu sistem tasarımı gereği yalnız ÖLÇER. Yanlış
# alarm, gerçek alarmı değersizleştirir.

def _olcum_modu_bilesenleri(exec_state=None):
    yurutme = (SF.execution_health(live_enabled=False, ems_ready=False)
               if exec_state is None
               else SF.ComponentHealth("EXECUTION", exec_state))
    return [SF.ComponentHealth(a, SF.GREEN)
            for a in SF.COMPONENTS if a != "EXECUTION"] + [yurutme]


def test_kurulmamis_yurutme_sistemi_BOZUK_gostermez():
    """Asıl kusur buydu: kurulmamış bir katman sistemi arızalı gösteriyordu."""
    h = SF.assess(_olcum_modu_bilesenleri())
    assert h.overall == SF.GREEN, "kurulmamış katman genel sağlığı düşürmemeli"
    assert h.mode == SF.MEASUREMENT_ONLY
    assert h.not_configured == ["EXECUTION"]
    assert SF.STATE_TR[SF.NOT_CONFIGURED] == "kurulmadı"


def test_kurulmamis_yurutme_otopiloti_YINE_DE_kapatir():
    """Etiket düzeltildi, kapı DEĞİL: emir gönderilemiyorsa otopilot açılamaz."""
    h = SF.assess(_olcum_modu_bilesenleri())
    assert h.autopilot is False
    assert h.blocking == ["EXECUTION"]
    assert "ölçüm modunda" in h.autopilot_reason
    assert "arıza değil" in h.autopilot_reason


def test_gercek_ariza_kurulmamisin_arkasina_SAKLANMAZ():
    """İstisna yalnız beklenen yokluğu kapsar; RISK kırmızıysa sistem kırmızıdır."""
    b = _olcum_modu_bilesenleri()
    b = [SF.ComponentHealth("RISK", SF.RED, "limit yok") if c.name == "RISK" else c
         for c in b]
    h = SF.assess(b)
    assert h.overall == SF.RED
    assert "RISK" in h.autopilot_reason and "EXECUTION" not in h.autopilot_reason


def test_islem_modunda_kurulmamis_bilesen_genel_duruma_GIRER():
    """İşlem modunda yürütmenin olmaması bir mod değil, gerçek bir eksikliktir."""
    h = SF.assess(_olcum_modu_bilesenleri(), mode=SF.TRADING_READY)
    assert h.overall == SF.NOT_CONFIGURED
    assert h.not_configured == []


def test_canli_mod_EMS_olmadan_sistem_KIRMIZI_kalir():
    """Mod istisnası tehlikeli hâli gizleyemez."""
    b = [SF.ComponentHealth(a, SF.GREEN)
         for a in SF.COMPONENTS if a != "EXECUTION"]
    b.append(SF.execution_health(live_enabled=True, ems_ready=False))
    h = SF.assess(b)
    assert h.overall == SF.RED and h.autopilot is False
    assert h.mode == SF.TRADING_READY


def test_kurulmus_yurutme_islem_moduna_gecirir():
    b = [SF.ComponentHealth(a, SF.GREEN)
         for a in SF.COMPONENTS if a != "EXECUTION"]
    b.append(SF.execution_health(live_enabled=False, ems_ready=True))
    h = SF.assess(b)
    assert h.mode == SF.TRADING_READY and h.autopilot is True


def test_deneme_kaydi_yoksa_modeller_bozulmus():
    c = SF.models_health(True, {"models": {"a": {"ok": True}},
                                "n_trials_registry": 0})
    assert c.state == SF.DEGRADED and "DSR anlamsız" in c.detail


def test_bayat_veri_kirmizi():
    scan = {"generated_at": "2026-01-01T00:00:00Z",
            "cards": [{"symbol": f"S{i}", "data_age_sec": 5000}
                      for i in range(10)]}
    assert SF.market_data_health(scan).state == SF.RED


def test_yazma_uclari_acikkasa_KIRMIZI():
    c = SF.security_health(write_endpoints_closed=False)
    assert c.state == SF.RED


# ══════════════════════════════════════════════════════════════════════════
# MODEL KAYIT DEFTERİ (§XVI, LVI, CII)
# ══════════════════════════════════════════════════════════════════════════

def _kart(**kw):
    t = dict(model_id="m1", purpose="test", built_by="ali",
             asset_universe=["BTCUSDT"], horizon="4h", direction="LONG",
             inputs=["technical"], training_period="a", validation_period="b",
             locked_test_period="c", version="1", risk_tier=1)
    t.update(kw)
    return MR.ModelCard(**t)


def test_gelistirici_kendi_modelini_dogrulayamaz():
    """§LVI — dört göz prensibi KODLA zorlanır."""
    with pytest.raises(MR.ApprovalError, match="DOĞRULAYAMAZ"):
        MR.approve(_kart(), validated_by="ali", approved_by="veli",
                   target_status=MR.PRODUCTION)


def test_gelistirici_kendi_modelini_onaylayamaz():
    with pytest.raises(MR.ApprovalError, match="ONAYLAYAMAZ"):
        MR.approve(_kart(), validated_by="veli", approved_by="ali",
                   target_status=MR.PRODUCTION)


def test_bagimsiz_dogrulama_ile_onay_gecerli():
    k = MR.approve(_kart(), validated_by="veli", approved_by="ayse",
                   target_status=MR.PRODUCTION)
    assert k.status == MR.PRODUCTION and k.approved_at


def test_kayit_defteri_durum_gecmisini_tutar():
    with tempfile.TemporaryDirectory() as d:
        r = MR.Registry(Path(d) / "reg.json")
        r.register(_kart(status=MR.RESEARCH))
        r.register(_kart(status=MR.OOS_VALIDATED))
        assert r.get("m1")["status"] == MR.OOS_VALIDATED
        h = r.history()
        assert h and h[0]["from"] == MR.RESEARCH and h[0]["to"] == MR.OOS_VALIDATED


def test_onaysiz_model_production_ready_degil():
    with tempfile.TemporaryDirectory() as d:
        r = MR.Registry(Path(d) / "reg.json")
        r.register(_kart(status=MR.PRODUCTION))       # approved_by YOK
        assert r.production_ready() == []


def test_softmax_karti_olculmus_sinirlari_tasir():
    k = MR.card_for_softmax("4h", "LONG", ["BTCUSDT"], {}, {},
                            "2025-01-01", "2026-01-01")
    assert k.risk_tier == MR.RISK_TIER["DIRECTIONAL_ML"]
    assert k.status == MR.RESEARCH and k.approved_by is None
    assert any("YÖNÜ değil" in x for x in k.limitations)
    assert any("tek başına emir" in x for x in k.prohibited_uses)
    assert any("gösterge konsensüsü" in x for x in k.prohibited_uses)
    assert k.retirement_criteria


def test_yasam_dongusu_durumlari_tanimli():
    for s in MR.LIFECYCLE:
        assert s in MR.STATUS_TR
    assert set(MR.REQUIRES_APPROVAL) <= set(MR.LIFECYCLE)


def test_sema_ihlali_market_data_bozulmus_yapar():
    """Şema ihlali VERİ sorunu değil KOD sorunudur — sessizce geçmemeli."""
    scan = {"generated_at": "2026-01-01T00:00:00Z",
            "cards": [{"symbol": "A", "data_age_sec": 30, "data_quality": 1.0}],
            "scanner": {"schema_violations": 4}}
    c = SF.market_data_health(scan)
    assert c.state == SF.DEGRADED and "şema" in c.detail


def test_dusuk_veri_kalitesi_izlemeye_alir():
    scan = {"generated_at": "2026-01-01T00:00:00Z",
            "cards": [{"symbol": "A", "data_age_sec": 30, "data_quality": 0.3}],
            "scanner": {"schema_violations": 0}}
    c = SF.market_data_health(scan)
    assert c.state == SF.WATCH and "veri kalitesi" in c.detail


def test_temiz_veri_yesil():
    scan = {"generated_at": "2026-01-01T00:00:00Z",
            "cards": [{"symbol": "A", "data_age_sec": 30, "data_quality": 0.98}],
            "scanner": {"schema_violations": 0}}
    assert SF.market_data_health(scan).state == SF.GREEN


# ══════════════════════════════════════════════════════════════════════════
# DSR / PBO — KAPI GERÇEKTEN ÇALIŞIYOR MU
# (ölçüldü: DSR 540/540 hücrede tam 0, PBO 540/540'ta tam 1,0 idi)
# ══════════════════════════════════════════════════════════════════════════

def test_dsr_islem_basi_olcekte_ayirt_eder():
    """DSR iyi ve kötü seriyi AYIRT ETMELİ; etmezse kapı çalışıyor görünüp
    hiçbir şey ölçmüyor demektir (ölçüldü: 540/540 hücrede tam 0 idi).

    `sr_std` = denenen adayların Sharpe YAYILIMI. İşlem-başı ölçekte beş stop
    adayının Sharpe'ları birbirine yakındır (ör. −0,25 … −0,19), yani yayılım
    0,05 mertebesindedir. Yıllık ölçekte aynı yayılım 100'e çıkıyordu."""
    import numpy as np
    from agi_trader.research.validation import deflated_sharpe
    rng = np.random.default_rng(11)
    iyi = rng.normal(0.35, 1.0, 800)
    kotu = rng.normal(-0.10, 1.0, 800)
    d_iyi = deflated_sharpe(iyi, 135, sr_std=0.05, periods_per_year=1.0)["dsr"]
    d_kotu = deflated_sharpe(kotu, 135, sr_std=0.05, periods_per_year=1.0)["dsr"]
    assert d_iyi > d_kotu, (d_iyi, d_kotu)
    assert d_iyi > 0.5, f"iyi seri kapıyı geçebilmeli, DSR={d_iyi}"
    assert d_kotu < 0.05, f"kötü seri kapıda kalmalı, DSR={d_kotu}"


def test_yillik_olcek_dsr_yi_oldururdu():
    """Regresyon: eski davranışın neden yanlış olduğunu kilitler.

    Aynı getiri serisi, yalnız ölçek farkı — yıllıkta DSR ölür."""
    import numpy as np
    from agi_trader.research.validation import deflated_sharpe
    rng = np.random.default_rng(11)
    r = rng.normal(0.35, 1.0, 800)
    islem_basi = deflated_sharpe(r, 135, sr_std=0.05,
                                 periods_per_year=1.0)["dsr"]
    yillik = deflated_sharpe(r, 135, sr_std=100.0,
                             periods_per_year=17520.0)["dsr"]
    assert yillik == 0.0, "yıllık ölçekte DSR daima 0 — kapı ölür"
    assert islem_basi > 0.5, "işlem-başı ölçekte DSR ayırt eder"


def test_dsr_deneme_sayisiyla_zorlasir():
    """Çoklu test düzeltmesi gerçekten çalışmalı: daha çok deneme → daha
    yüksek eşik → daha düşük DSR."""
    import numpy as np
    from agi_trader.research.validation import deflated_sharpe
    rng = np.random.default_rng(5)
    r = rng.normal(0.20, 1.0, 800)
    az = deflated_sharpe(r, 10, sr_std=0.05, periods_per_year=1.0)["dsr"]
    cok = deflated_sharpe(r, 3000, sr_std=0.05, periods_per_year=1.0)["dsr"]
    assert az >= cok, (az, cok)


def test_deneme_kaydi_islem_basi_sharpe_tasir():
    """`sharpe` alanı DSR'ın okuduğu alandır ve İŞLEM-BAŞI olmalı."""
    from agi_trader.qualification.research import (_annual_sharpe,
                                                   _per_trade_sharpe)
    from agi_trader.qualification.targets import CostProfile
    prof = CostProfile("X", spread_bps=0.0, impact_bps_roundtrip=1.0,
                       fee_bps_roundtrip=8.0, reserve_bps=5.0,
                       funding_rate_8h=0.0, model="ESTIMATED", source="t")
    hucre = dict(p_target_first=0.35, p_stop_first=0.35, p_timeout=0.30,
                 stop_pct_median=1.0, timeout_return_pct=0.0,
                 median_hours_to_tp=None, median_hours_to_sl=None,
                 horizon_hours=0.5, direction="LONG")
    ps = _per_trade_sharpe(hucre, prof)
    yil = _annual_sharpe(hucre, prof)
    assert abs(ps) < 5, "işlem-başı Sharpe makul aralıkta olmalı"
    assert abs(yil) > abs(ps) * 10, "yıllık karşılık çok daha büyük olmalı"


def test_pbo_tek_sutunlu_matriste_hesaplanmaz():
    """Tek aday varsa PBO tanımsızdır; 1,0 döndürmek 'her seçim aşırı uyum'
    demek olur ve hiçbir şey ayırt etmez."""
    import numpy as np
    import pandas as pd
    from agi_trader.qualification.research import _pbo_from_stop_grid
    from agi_trader.qualification.targets import CostProfile
    prof = CostProfile("X", spread_bps=0.0, impact_bps_roundtrip=1.0,
                       fee_bps_roundtrip=8.0, reserve_bps=5.0,
                       funding_rate_8h=0.0, model="ESTIMATED", source="t")
    satir = dict(symbol="X", horizon="4h", direction="LONG", regime="ALL",
                 stop_sigma_mult=1.0, p_target_first=0.4, p_stop_first=0.35,
                 p_timeout=0.25, p_target_lower95=0.36, stop_pct_median=1.0,
                 timeout_return_pct=0.0, median_hours_to_tp=2.0,
                 median_hours_to_sl=2.0, horizon_hours=4.0,
                 stop_gap_excess_pct=0.0)
    cells = pd.DataFrame([{**satir, "period": p} for p in
                          ("train", "validation", "test")])
    assert _pbo_from_stop_grid(cells, "X", "4h", "LONG", prof) is None


def test_pbo_yetersiz_veride_None_doner_1_degil():
    """'Ölçülemedi' ile 'kesin aşırı uyum' AYNI ŞEY DEĞİLDİR.

    ÖLÇÜLDÜ: yetersiz dilimde 1,0 dönüyordu ve çağıran bunu "seçim tamamen
    aşırı uyum" diye okuyordu — 594/594 hücrede tam 1,0 çıkmasının sebebi
    buydu."""
    import numpy as np
    from agi_trader.research.validation import pbo
    r = pbo(np.array([[0.1, 0.2], [0.15, 0.25]]), n_splits=8)
    assert r["pbo"] is None and "ölçülemedi" in r["verdict"]
    tek = pbo(np.random.default_rng(1).normal(size=(20, 1)))
    assert tek["pbo"] is None


def test_pbo_yeterli_veride_ayirt_eder():
    """En iyi aday OOS'ta da en iyiyse PBO DÜŞÜK olmalı; rastgeleyse yüksek."""
    import numpy as np
    from agi_trader.research.validation import pbo
    rng = np.random.default_rng(7)
    # 1. sütun sistematik olarak daha iyi → aşırı uyum YOK
    iyi = rng.normal(0.0, 1.0, (24, 4))
    iyi[:, 0] += 2.0
    r_iyi = pbo(iyi, n_splits=8)
    # tamamen gürültü → seçim aşırı uyum
    gurultu = rng.normal(0.0, 1.0, (24, 4))
    r_gurultu = pbo(gurultu, n_splits=8)
    assert r_iyi["pbo"] is not None and r_gurultu["pbo"] is not None
    assert r_iyi["pbo"] < r_gurultu["pbo"], (r_iyi["pbo"], r_gurultu["pbo"])
    assert r_iyi["pbo"] < 0.3, "gerçek üstünlük düşük PBO vermeli"


def test_pbo_n_splits_veriye_uydurulur():
    import numpy as np
    from agi_trader.research.validation import pbo
    r = pbo(np.random.default_rng(3).normal(size=(12, 4)), n_splits=8)
    assert r["pbo"] is not None, "12 dilim ölçülebilmeli"
    assert r["n_splits"] <= 6 and r["n_splits"] % 2 == 0


# ══════════════════════════════════════════════════════════════════════════
# GÖLGE TAHMİN KAYDI — ölçmeyen ölçüm sistemi kendi hatasını göremez
# ══════════════════════════════════════════════════════════════════════════
# Canlıda ölçüldü: 540 kombinasyonda sıfır QUALIFIED → defter BOŞ → ana metrik
# (Gerçekleşen ÷ Tahmin Edilen Net EV) ve kalibrasyon tablosu sonsuza dek
# "örneklem yetersiz". Gölge kayıt bu döngüyü canlandırır ama sinyal DEĞİLDİR.

def _golge_kart(sym="BTCUSDT", rev=-0.02, hz="1h"):
    return {
        "symbol": sym, "timestamp": "2026-08-19T10:00:00Z",
        "best_horizon": None, "direction": None,
        "market_price": 100.0, "data_quality": 0.9,
        "horizons": [
            {"horizon": hz, "horizon_minutes": 60, "direction": "LONG",
             "reference_only": False, "robust_ev": rev, "status": "NO_EDGE",
             "optimal_entry": 100.0, "net_1pct_exit": 101.2, "stop_price": 99.0,
             "p_target_first": 0.31, "lower95": 0.28, "p_stop_first": 0.5,
             "p_timeout": 0.19, "cost_pct": 0.14,
             "time_exit": "2026-08-19T11:00:00Z"},
            {"horizon": "48h", "horizon_minutes": 2880, "direction": "LONG",
             "reference_only": True, "robust_ev": 0.9, "status": "NO_EDGE",
             "optimal_entry": 100.0, "net_1pct_exit": 101.2, "stop_price": 99.0},
        ]}


def _golge_defteri(tmp_path):
    from agi_trader.qualification.ledger import Ledger
    return Ledger(tmp_path / "defter.jsonl")


def test_golge_kayit_karne_paydasina_GIRMEZ(tmp_path):
    """§106: payda YAYIMLANAN sinyaldir. Gölge sulandırırsa precision sahte."""
    from agi_trader.qualification import ledger as L
    from agi_trader.qualification.live import _kaydet_golge
    led = _golge_defteri(tmp_path)
    _kaydet_golge(led, _golge_kart(), set())
    assert len(led.predictions()) == 1
    assert len(led.predictions(source=L.SOURCE_SHADOW)) == 1
    assert led.predictions(source=L.SOURCE_PUBLISHED) == []
    assert L.scorecard(led, None)["cells"] == []


def test_golge_referans_ufku_SECMEZ(tmp_path):
    """48h referans-yalnızdır; robust_ev'i en yüksek olsa bile seçilemez."""
    from agi_trader.qualification.live import _kaydet_golge
    led = _golge_defteri(tmp_path)
    _kaydet_golge(led, _golge_kart(), set())
    assert led.predictions()[0]["horizon"] == "1h"


def test_golge_acikken_ikincisi_yazilmaz(tmp_path):
    """Hız sınırı: örtüşen, bağımsız olmayan gözlemler birikmemeli."""
    from agi_trader.qualification.live import _kaydet_golge
    led = _golge_defteri(tmp_path)
    acik = set()
    _kaydet_golge(led, _golge_kart(), acik)
    _kaydet_golge(led, _golge_kart(), acik)
    assert len(led.predictions()) == 1


def test_kalifiye_kartta_golge_yazilmaz(tmp_path):
    from agi_trader.qualification.live import _kaydet_golge
    led = _golge_defteri(tmp_path)
    k = _golge_kart()
    k["best_horizon"] = "1h"
    _kaydet_golge(led, k, set())
    assert led.predictions() == []


def test_golge_nitelendirme_iddiasi_TASIMAZ(tmp_path):
    from agi_trader.qualification.live import _kaydet_golge
    from agi_trader.qualification.state import QualificationState
    led = _golge_defteri(tmp_path)
    _kaydet_golge(led, _golge_kart(), set())
    p = led.predictions()[0]
    assert p["guaranteed"] is False
    assert p["source"] == "shadow"
    assert p["status"] not in QualificationState.TRADABLE


def test_atribusyon_golge_ile_yayimlanani_TOPLAMAZ(tmp_path):
    from agi_trader.qualification import attribution as AT, ledger as L
    from agi_trader.qualification.live import _kaydet_golge
    led = _golge_defteri(tmp_path)
    _kaydet_golge(led, _golge_kart(), set())
    assert AT.overall(led, source=L.SOURCE_SHADOW)["n"] == 0   # henüz çözülmedi
    assert AT.overall(led, source=L.SOURCE_PUBLISHED)["n"] == 0
    assert AT.overall(led, source=L.SOURCE_SHADOW)["source"] == "shadow"


def test_golge_fiyat_duzeyi_eksikse_yazilmaz(tmp_path):
    """Değerlendirilemeyecek bir kayıt yazmak defteri kirletir."""
    from agi_trader.qualification.live import _kaydet_golge
    led = _golge_defteri(tmp_path)
    k = _golge_kart()
    k["horizons"][0]["stop_price"] = None
    k["horizons"][1]["reference_only"] = True
    _kaydet_golge(led, k, set())
    assert led.predictions() == []


def test_sifir_kalifiyede_golge_YAZILIR(tmp_path):
    """⚠️ GERÇEK HATA: ilk sürüm gölge döngüsünü `rank_pairs` çıktısı üzerinde
    koşuyordu; o liste yalnız KALİFİYE kartları içeriyor. Sıfır kalifiye olunca
    hiç gölge yazılmadı — gölge kaydın var olma sebebi tam olarak o durumken.
    Sunucuda ölçüldü: 27 parite tarandı, 0 gölge."""
    from agi_trader.qualification.live import _deftere_yaz
    from agi_trader.qualification.matrix import rank_pairs
    led = _golge_defteri(tmp_path)
    kartlar = [_golge_kart("BTCUSDT"), _golge_kart("ETHUSDT")]
    sirali = rank_pairs(kartlar)
    assert sirali == [], "kurgu geçersiz: kalifiye kart olmamalı"
    _deftere_yaz(led, kartlar, sirali)
    assert len(led.predictions()) == 2
    assert {p["symbol"] for p in led.predictions()} == {"BTCUSDT", "ETHUSDT"}
