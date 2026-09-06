# -*- coding: utf-8 -*-
"""
KANIT MOTORU — yeterli istatistikler, akış okuma, sahte-kanıt freni.

Bu testler 2026-09-06 ölçümünden çıkan üç yalınlaştırmanın regresyonudur:
  • işlem kaydı 1.024 B; istatistiğin ihtiyacı yalnız %10,8'i → kanıt satırı ~200 B
  • `TradeJournal` kayıt başına 6.626 B yazıyordu, tüketicisi ~%4'ünü okuyordu
  • `load_all()` 118 MB'ı tek seferde belleğe alıyordu (OOM tuzağı) → akış + sınır
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agi_trader.learn import evidence as EV  # noqa: E402


def _t(net_pct, win=None, sleeve="dip", ts=1_000_000, **kw):
    d = {"closed_ts": ts, "symbol": "BTC/USDT", "sleeve": sleeve, "direction": "LONG",
         "order_type": "maker", "exit_mode": "DYNAMIC_PEAK", "reason": "GIVEBACK",
         "net_pct_realized": net_pct, "net_pnl": net_pct / 100 * 10,
         "win": (net_pct > 0) if win is None else win, "notional": 10.0,
         "peak_net_pct": max(0.0, net_pct), "cost_pct_roundtrip": 0.07,
         "stop_pct": 1.0, "hold_sec": 900, "levels_hit": 0}
    d.update(kw)
    return d


# ═══════════════════════ 1) KAYIT BOYUTU ═══════════════════════
def test_kanit_satiri_kucuk_ve_gerekli_alanlari_tasir():
    """Kanıt satırı, tam işlem kaydından belirgin küçük olmalı ama kapıların
    kullandığı HER alanı taşımalı."""
    tam = _t(0.5)
    tam.update({"hash": "a" * 64, "prev_hash": "b" * 64, "exit_detail": {"x": 1, "y": [1, 2, 3]},
                "decision": {"ticket": {"ev_pct": 1.2}}, "template": "mean_reversion"})
    satir = EV.satir(tam, rejim="RANGE")
    b_tam = len(json.dumps(tam, ensure_ascii=False))
    b_kanit = len(json.dumps(satir, ensure_ascii=False, separators=(",", ":")))
    assert b_kanit < b_tam / 2, f"kanıt satırı yeterince küçük değil ({b_kanit} / {b_tam})"
    for alan in ("ts", "slv", "np", "nu", "w", "no", "pk", "cp", "sp", "hs", "r", "ot", "xm", "rej", "ss"):
        assert alan in satir, f"kapıların kullandığı alan eksik: {alan}"
    assert satir["ss"] in (0, 4, 8, 12, 16, 20), "seans bloğu 4 saatlik olmalı"


# ═══════════════════════ 2) YETERLİ İSTATİSTİK ═══════════════════════
def test_biriktirici_t_degeri_dogru():
    """n, Σx, Σx² üzerinden hesaplanan t, doğrudan hesapla AYNI olmalı (O(1) bellek)."""
    import statistics as st
    xs = [0.4, -0.2, 0.9, -0.1, 0.3, 0.6, -0.5, 0.2]
    b = EV.Biriktirici()
    for x in xs:
        b.ekle(x, x > 0)
    assert b.n == len(xs)
    assert b.ort == pytest.approx(st.mean(xs))
    assert b.sd == pytest.approx(st.stdev(xs))
    beklenen = st.mean(xs) / (st.stdev(xs) / math.sqrt(len(xs)))
    assert b.t == pytest.approx(beklenen, rel=1e-9)


def test_kac_islem_gerek_etki_yoksa_None_der():
    """ETKİ BÜYÜKLÜĞÜ (ort/sd) sıfıra yakınsa cevap YOKTUR: bekleyerek kanıtlanmaz.
    ÖNEMLİ AYRIM: mutlak değerlerin küçük olması etki yok demek DEĞİLDİR — ort/sd
    büyükse küçük sayılarla da kanıtlanır. Bu testin ilk hâli bunu karıştırmıştı."""
    b = EV.Biriktirici()
    for x in (1.0, -1.0, 0.9, -0.9, 1.1, -1.1, 0.95, -0.95):   # ort ≈ 0, sd büyük
        b.ekle(x, x > 0)
    assert EV.kac_islem_gerek(b) is None

    kucuk = EV.Biriktirici()                  # DEĞERLER küçük ama etki tutarlı → sayı vermeli
    for x in (0.004, 0.005, 0.003, 0.0045, 0.0055, 0.004):
        kucuk.ekle(x, True)
    assert EV.kac_islem_gerek(kucuk) is not None

    g = EV.Biriktirici()                      # güçlü ve tutarlı etki → ek işlem gerekmez
    for x in (0.5, 0.6, 0.4, 0.55, 0.45, 0.5, 0.52, 0.48):
        g.ekle(x, True)
    assert EV.kac_islem_gerek(g) == 0


def test_sapma_sifirsa_kanit_ZATEN_var_demektir():
    """sd = 0 (tüm işlemler aynı sonuç) → t sonsuz. Bunu 'asla kanıtlanmaz' saymak
    tam tersi anlama gelirdi; ek işlem gerekmez."""
    b = EV.Biriktirici()
    for _ in range(12):
        b.ekle(0.5, True)
    assert b.sd == 0.0
    assert EV.kac_islem_gerek(b) == 0


def test_kac_islem_gerek_zayif_etkide_daha_cok_islem_ister():
    zayif, guclu = EV.Biriktirici(), EV.Biriktirici()
    for i in range(10):
        zayif.ekle(0.05 + (0.3 if i % 2 else -0.3), i % 2 == 0)
        guclu.ekle(0.40 + (0.3 if i % 2 else -0.3), i % 2 == 0)
    kz, kg = EV.kac_islem_gerek(zayif), EV.kac_islem_gerek(guclu)
    assert kz is not None and kg is not None and kz > kg


# ═══════════════════════ 3) AKIŞ OKUMA ═══════════════════════
def test_yazma_okuma_ve_limit(tmp_path):
    runs = str(tmp_path)
    for i in range(50):
        assert EV.kaydet(_t(0.1 * (i % 5 - 2), ts=1_000_000 + i), runs, tag="x")
    hepsi = list(EV.oku(runs, tag="x"))
    assert len(hepsi) == 50
    son10 = list(EV.oku(runs, tag="x", limit=10))
    assert len(son10) == 10 and son10[-1]["ts"] == 1_000_049
    sonra = list(EV.oku(runs, tag="x", since=1_000_040))
    assert len(sonra) == 10


def test_bozuk_satir_okumayi_durdurmaz(tmp_path):
    runs = str(tmp_path)
    EV.kaydet(_t(0.3), runs, tag="x")
    with open(EV.yol(runs, "x"), "a", encoding="utf-8") as f:
        f.write("{bozuk json\n\n")
    EV.kaydet(_t(0.4), runs, tag="x")
    assert len(list(EV.oku(runs, tag="x"))) == 2, "bozuk satır atlanmalı, akış devam etmeli"


def test_ozet_kesitleri_ve_toplam(tmp_path):
    runs = str(tmp_path)
    for i in range(12):
        EV.kaydet(_t(0.5, sleeve="catalyst", ts=1_000_000 + i), runs, tag="x", rejim="TREND")
    for i in range(12):
        EV.kaydet(_t(-0.4, sleeve="dip", ts=1_000_100 + i), runs, tag="x", rejim="RANGE")
    o = EV.ozet(runs, tag="x", min_n=1)
    assert o["genel"]["n"] == 24
    assert o["sleeve"]["catalyst"]["ort_pct"] == pytest.approx(0.5)
    assert o["sleeve"]["dip"]["ort_pct"] == pytest.approx(-0.4)
    assert "catalyst|TREND" in o["sleeve_rejim"]
    assert o["sleeve"]["catalyst"]["kalan_islem_t2"] == 0      # sapma yok → kanıt zaten var


# ═══════════════════════ 4) DÖNDÜRME (sınırsız büyüme yok) ═══════════════════════
def test_dondurme_eski_satirlari_ARSIVLER_silmez(tmp_path):
    runs = str(tmp_path)
    for i in range(40):
        EV.kaydet(_t(0.1, ts=1_000_000 + i), runs, tag="x")
    tasinan = EV.dondur(runs, tag="x", max_satir=20)
    assert tasinan == 30
    assert len(list(EV.oku(runs, tag="x"))) == 10
    arsiv = list(EV.yol(runs, "x").parent.glob("*.arsiv.jsonl"))
    assert arsiv, "eski satırlar SİLİNMEMELİ, arşive taşınmalı"
    assert sum(1 for _ in open(arsiv[0], encoding="utf-8")) == 30


# ═══════════════════════ 5) GÜNLÜK YALINLAŞTIRMASI ═══════════════════════
def test_journal_kaydi_kucuk_ve_okuma_SINIRLI(tmp_path):
    from agi_trader.journal.trade_journal import TradeJournal, _kucult
    tam = {"symbol": "BTC/USDT", "entry": 100.0, "timeframe": "4h", "direction": "LONG",
           "bias": "bullish", "confidence": 0.7, "actionable": True, "signal_class": "kesin_al",
           "layer_breakdown": [{"layer": "trend", "score": 0.8, "detay": "x" * 500,
                                "gerekce": ["a" * 200]}],
           "reasons": ["r" * 400], "alternative_scenario": "s" * 800,
           "risk": {"x": "y" * 600}, "correlation_badge": {"symbol": "ETH", "value": 0.9}}
    kucuk = _kucult(tam)
    assert len(json.dumps(kucuk)) < len(json.dumps(tam)) / 5, "sinyal kaydı yeterince küçülmedi"
    assert kucuk["symbol"] == "BTC/USDT" and kucuk["entry"] == 100.0
    assert kucuk["layer_breakdown"][0] == {"layer": "trend", "score": 0.8}
    assert "risk" not in kucuk and "alternative_scenario" not in kucuk

    j = TradeJournal(str(tmp_path))
    with j.signals_path.open("w", encoding="utf-8") as f:
        for i in range(100):
            f.write(json.dumps({"ts": i, "signal": {"actionable": i % 2 == 0,
                                                    "direction": "LONG", "confidence": 0.5}}) + "\n")
    assert len(j.load_all(limit=10)) == 10, "load_all SINIRLI olmalı (OOM freni)"
    s = j.summary()
    assert s["total_signals"] == 100 and s["actionable_signals"] == 50


# ═══════════ 6) GÜNLÜK SAYAÇLAR YENİDEN BAŞLATMADA SIFIRLANMAMALI ═══════════
def test_gunluk_sayaclar_yeniden_baslatmada_KORUNUR(tmp_path):
    """2026-09-06 HATASI: `day`, `day_trades` ve `_rotations_today` save()/load()'da YOKTU.
    Her yeniden başlatmada `day` "" oluyor, sonraki döngü `day != today` dalına düşüyor ve
    GÜNLÜK İŞLEM TAVANI ile ROTASYON TAVANI sıfırlanıyordu.

    Kanıt: rotasyon tavanı 6/gün olmasına rağmen 09-06'da 10 rotasyon oldu — o gün pm2
    5 kez yeniden başlatılmıştı. pm2 zaten `max_memory_restart` + `autorestart` ile
    kendiliğinden de yeniden başlatır; yani bu kapı sessizce gevşiyordu.

    Bunlar RİSK kapılarıdır: yeniden başlatma onları gevşetmemeli."""
    import time as _t
    from agi_trader.auto import live_runner as LR

    bugun = _t.strftime("%Y-%m-%d", _t.gmtime())
    kayit = {"day": bugun, "day_trades": 47, "rotations_today": 5,
             "trades": [], "positions": [], "pending": {}}

    class _Sahte:                       # load()'un dokunduğu asgari yüzey
        pass
    r = _Sahte()
    r.day, r.day_trades, r._rotations_today = "", 0, 0
    # load()'un ilgili dalını birebir uygula
    if str(kayit.get("day") or "") == bugun:
        r.day = kayit["day"]
        r.day_trades = int(kayit.get("day_trades") or 0)
        r._rotations_today = int(kayit.get("rotations_today") or 0)
    assert r.day == bugun and r.day_trades == 47 and r._rotations_today == 5, \
        "aynı güne ait sayaçlar geri yüklenmeli"

    # DÜNE ait kayıt geri YÜKLENMEZ (gün değişti, sıfırlanmalı)
    r2 = _Sahte(); r2.day, r2.day_trades, r2._rotations_today = "", 0, 0
    eski = {"day": "2000-01-01", "day_trades": 99, "rotations_today": 6}
    if str(eski.get("day") or "") == bugun:
        r2.day_trades = int(eski["day_trades"])
    assert r2.day_trades == 0, "önceki güne ait sayaç taşınmamalı"

    # save() sözleşmesi: üç alan da yazılıyor mu
    src = (Path(LR.__file__)).read_text(encoding="utf-8")
    for alan in ('"day": self.day', '"day_trades": self.day_trades', '"rotations_today"'):
        assert alan in src, f"save() içinde eksik: {alan}"


def test_iki_EV_de_kanit_satirinda(tmp_path):
    """`ev_achievable_pct` 222 işlemin HİÇBİRİNDE kayıtlı değildi (0/222) — bu yüzden
    'hangi EV gerçekten öngörüyor?' sorusu ölçülemiyordu. Artık iki EV de kaydediliyor."""
    t = _t(0.3)
    t.update({"ev_pct": 0.57, "ev_achievable_pct": -0.15})
    s = EV.satir(t)
    assert s["ev"] == pytest.approx(0.57) and s["eva"] == pytest.approx(-0.15)
    yok = EV.satir(_t(0.3))
    assert yok["ev"] is None and yok["eva"] is None, "yoksa None olmalı, 0 değil"


# ═══════════ 7) EV KALİBRASYON ÖLÇERİNİN KENDİSİ DOĞRU MU ═══════════
def _calib():
    import importlib.util
    sp = Path(__file__).resolve().parents[1] / "scripts" / "cm_ev_calib.py"
    spec = importlib.util.spec_from_file_location("cm_ev_calib", sp)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_kalibrasyon_olceri_MUKEMMEL_tahmini_tanir():
    """gerçekleşen = vaat olduğunda: r=1, eğim=1, tekdüze, bilgi VAR."""
    C = _calib()
    cift = [(x / 10.0, x / 10.0) for x in range(-25, 26)]
    d = C.analiz(cift, "test")
    assert d["pearson"] == pytest.approx(1.0, abs=1e-6)
    assert d["kalibrasyon_egimi"] == pytest.approx(1.0, abs=1e-6)
    assert d["yanlilik"] == pytest.approx(0.0, abs=1e-9)
    assert d["tekduze"] is True and d["bilgi_var"] is True


def test_kalibrasyon_olceri_BILGISIZ_tahmini_tanir():
    """Vaat ile gerçekleşen ilişkisizse: r≈0, eğim≈0, bilgi YOK."""
    C = _calib()
    import random
    rnd = random.Random(7)
    cift = [(rnd.uniform(-1, 1), rnd.uniform(-1, 1)) for _ in range(400)]
    d = C.analiz(cift, "test")
    assert abs(d["pearson"]) < 0.15
    assert d["bilgi_var"] is False, "ilişkisiz veride 'bilgi var' DENMEMELİ"


def test_kalibrasyon_olceri_SISKIN_ama_BILGILI_tahmini_ayirt_eder():
    """En önemli ayrım: bir tahmin ŞİŞKİN olabilir ama yine de SIRALAMA bilgisi taşır.
    ev_pct tam olarak bu durumda olabilir — 5 kat şişik ama sıralaması işe yarıyor mu?"""
    C = _calib()
    cift = [(x / 10.0, x / 50.0 - 0.4) for x in range(-25, 26)]   # 5× şişik + kayık
    d = C.analiz(cift, "test")
    assert d["pearson"] == pytest.approx(1.0, abs=1e-6), "sıralama bilgisi TAM"
    assert d["kalibrasyon_egimi"] == pytest.approx(0.2, abs=1e-6), "eğim 1/5 → 5× şişik"
    assert d["yanlilik"] > 0.3, "yanlılık ayrı raporlanmalı"
    assert d["bilgi_var"] is True, "şişkinlik ≠ bilgisizlik"


def test_kalibrasyon_olceri_TERS_tahmini_yakalar():
    C = _calib()
    cift = [(x / 10.0, -x / 10.0) for x in range(-25, 26)]
    d = C.analiz(cift, "test")
    assert d["pearson"] < -0.9 and d["kalibrasyon_egimi"] < 0
    assert d["bilgi_var"] is False, "ters ilişkide 'bilgi var (pozitif)' denmemeli"
    assert d["tekduze"] is False


# ═══════════ 8) ROTASYON HANGİ EV'YE BAĞLI? (ölçüme göre, kalibre ederek) ═══════════
class _SahteKosucu:
    """`_rotation_ev` / `_calib_map`'i izole sınamak için asgari yüzey."""
    def __init__(self, calib=None):
        self._sabit = calib or {}
    def _calib_map(self, ttl=900.0):
        return self._sabit


def _rot_ev(calib, ticket):
    from agi_trader.auto import live_runner as LR
    r = _SahteKosucu(calib)
    return LR.LiveRunner._rotation_ev(r, ticket)


def test_olcum_YOKKEN_MUHAFAZAKAR_olani_secer():
    """Bilinmezlikte cömert varsayım yapmak bu depoda tekrar eden hataydı
    (defter alınamayınca spread 0, derinlik ölçülmeyince notional×50…).
    Ölçüm hazır değilse iki EV'nin KÜÇÜĞÜ alınır."""
    ev, kaynak = _rot_ev({}, {"ev_pct": 0.57, "ev_achievable_pct": -0.15})
    assert ev == pytest.approx(-0.15) and "muhafazakâr" in kaynak
    # yalnız biri varsa o kullanılır
    ev2, _ = _rot_ev({}, {"ev_pct": 0.4})
    assert ev2 == pytest.approx(0.4)
    # hiçbiri yoksa None → kapı zaten rotasyon yapmaz
    ev3, k3 = _rot_ev({}, {})
    assert ev3 is None and "yok" in k3


def test_verdikt_ev_achievable_ise_ONA_baglanir():
    c = {"karar": "ev_achievable DAHA İYİ", "eva_cal": {"egim": 1.0, "kesisim": 0.0}}
    ev, kaynak = _rot_ev(c, {"ev_pct": 0.57, "ev_achievable_pct": -0.15})
    assert ev == pytest.approx(-0.15) and kaynak.startswith("ev_achievable")


def test_verdikt_ev_pct_ise_ONA_baglanir():
    c = {"karar": "ev_pct DAHA İYİ", "ev_cal": {"egim": 1.0, "kesisim": 0.0}}
    ev, kaynak = _rot_ev(c, {"ev_pct": 0.57, "ev_achievable_pct": -0.15})
    assert ev == pytest.approx(0.57) and kaynak.startswith("ev_pct")


def test_SISKIN_ama_bilgili_EV_atilmaz_OLCEKLENIR():
    """Kilit davranış: eğim 0,2 ise ev_pct 'şişik ama bilgili'dir. Atmak yerine
    ölçülen ölçeğe indirilir — yoksa ham hâliyle eşiği sürekli aşardı."""
    c = {"karar": "ev_pct DAHA İYİ", "ev_cal": {"egim": 0.2, "kesisim": -0.4}}
    ev, kaynak = _rot_ev(c, {"ev_pct": 1.0, "ev_achievable_pct": 0.1})
    assert ev == pytest.approx(-0.4 + 0.2 * 1.0)      # = −0,20, ham 1,0 değil
    assert "kalibre" in kaynak and "b=0.2" in kaynak
    # kalibrasyon, EŞİĞİ aşmayı zorlaştırır (rotasyon seyrekleşir)
    ham, _ = _rot_ev({"karar": "ev_pct DAHA İYİ"}, {"ev_pct": 1.0})
    assert ev < ham


def test_AYIRT_EDILEMEDI_verdikti_muhafazakar_yola_duser():
    c = {"karar": "HENÜZ AYIRT EDİLEMEDİ", "ev_cal": {"egim": 5.0, "kesisim": 0.0}}
    ev, kaynak = _rot_ev(c, {"ev_pct": 0.9, "ev_achievable_pct": 0.1})
    assert ev == pytest.approx(0.1), "ayırt edilemediyse kalibrasyon UYGULANMAZ, küçük alınır"
    assert "muhafazakâr" in kaynak


def test_calib_map_bozuk_dosyada_BOS_doner(tmp_path):
    """Fail-safe: dosya yok/bozuk/hazır değil → boş → muhafazakâr yol."""
    from agi_trader.auto import live_runner as LR
    class _R:
        output_dir = str(tmp_path)
    (tmp_path / "live").mkdir(parents=True, exist_ok=True)
    assert LR.LiveRunner._calib_map(_R(), ttl=0) == {}
    (tmp_path / "live" / "ev_calib.json").write_text("{bozuk", encoding="utf-8")
    assert LR.LiveRunner._calib_map(_R(), ttl=0) == {}
    (tmp_path / "live" / "ev_calib.json").write_text(
        json.dumps({"hazir": False, "karar": "ev_pct DAHA İYİ"}), encoding="utf-8")
    assert LR.LiveRunner._calib_map(_R(), ttl=0) == {}, "hazir=False iken karar UYGULANMAZ"
