# -*- coding: utf-8 -*-
"""Topluluk katkısı kapısı — 2026-09-05.

Depo herkese açık ve katkıya açık. Tek kural: **katkı SHADOW doğar.** Kod incelemesinden
geçmiş olmak kenar olduğunu göstermez; bu depodaki ölçümler bunu defalarca gösterdi
(21 videodan çıkarılan 10 kurulum, 7 günlük gerçek veride t = −20,4).

Bu testler yükleyicinin PARANOYAK kalmasını kilitler:
  * Geçersiz META / imza YÜKLENMEZ ve sebebi KAYDEDİLİR (sessizce yutulmaz).
  * Bir katkı mevcut bir sleeve'in yerine geçemez (ad çakışması reddedilir).
  * Bir katkının patlaması diğerlerini ya da koşucuyu düşüremez.
  * Spot'ta SHORT üretilemez; `size` kelepçelenir.
  * Statik denetim ölçümü geçersiz kılacak şeyleri ölçümden ÖNCE yakalar.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agi_trader.strategies import contrib as CB  # noqa: E402

GECERLI_META = '''
META = {
    "name": "%s",
    "author": "@biri",
    "source": "kendi fikrim",
    "claim": "bir sey iddia ediyorum",
    "claim_evidence": "YOK",
    "mechanism": "rsi < 40 ve trend yukari",
    "exit_mode": "FIXED_TARGET",
    "time_stop_min": 60,
    "urgency": 0,
    "regimes": ["TREND YUKARI"],
}
def fire(f, p, price, atr_abs):
    return %s
'''


@pytest.fixture
def contrib_dizini(tmp_path, monkeypatch):
    """Katkı paketini geçici bir dizine yönlendir; test sonrası gerçek paket geri yüklenir."""
    monkeypatch.setattr(CB, "__path__", [str(tmp_path)])
    onceki = {k for k in sys.modules if k.startswith(CB.__name__ + ".")}
    yield tmp_path
    for k in [k for k in sys.modules if k.startswith(CB.__name__ + ".") and k not in onceki]:
        del sys.modules[k]  # geçici test modülleri gerçek paketi kirletmesin
    CB.load(set())          # gerçek (boş) durumu geri yükle


def yaz(dizin: Path, ad: str, govde: str) -> None:
    (dizin / f"{ad}.py").write_text(govde, encoding="utf-8")


# ───────────────────────────────────────────────────────────── yükleme
def test_gecerli_katki_yuklenir(contrib_dizini):
    yaz(contrib_dizini, "iyi_kurulum", GECERLI_META % ("iyi_kurulum", "None"))
    CB.load(set())
    assert "iyi_kurulum" in CB.CONTRIB and not CB.LOAD_ERRORS


def test_sablon_atlanir(contrib_dizini):
    yaz(contrib_dizini, "SABLON", GECERLI_META % ("sablon_kurulum", "None"))
    CB.load(set())
    assert CB.CONTRIB == {} and CB.LOAD_ERRORS == []


@pytest.mark.parametrize("alan", ["author", "source", "claim", "claim_evidence", "mechanism"])
def test_bos_kunye_alani_reddedilir(contrib_dizini, alan):
    """Kaynağını ve iddiasını yazmak ZORUNLU — kanıt yoksa 'YOK' yazılır, boş bırakılmaz."""
    import re as _re
    # Anahtar KALIR, yalnız değeri boşalır: test edilen şey "eksik anahtar" değil "BOŞ değer".
    govde = _re.sub(rf'("{alan}":\s*)"[^"]*"', lambda m: m.group(1) + '""',
                    GECERLI_META % ("kunyesiz", "None"))
    assert f'"{alan}": ""' in govde, "test girdisi anahtarı silmemeli, değerini boşaltmalı"
    yaz(contrib_dizini, f"kunyesiz_{alan}", govde)
    CB.load(set())
    assert "kunyesiz" not in CB.CONTRIB
    assert any(alan in h and "boş" in h for e in CB.LOAD_ERRORS for h in e["hatalar"])


def test_ad_cakismasi_reddedilir(contrib_dizini):
    """Bir katkı MEVCUT bir sleeve'in yerine geçemez — sessiz ele geçirme olurdu."""
    yaz(contrib_dizini, "sahtekar", GECERLI_META % ("dip", "None"))
    CB.load({"dip", "breakout"})
    assert "dip" not in CB.CONTRIB
    assert any("çakış" in h for e in CB.LOAD_ERRORS for h in e["hatalar"])


@pytest.mark.parametrize("kotu,beklenen", [
    ('"exit_mode": "FIXED_TARGET"', "exit_mode"),
    ('"time_stop_min": 60', "time_stop_min"),
    ('"urgency": 0', "urgency"),
    ('"regimes": ["TREND YUKARI"]', "rejim"),
])
def test_gecersiz_meta_alanlari(contrib_dizini, kotu, beklenen):
    bozuk = {'"exit_mode": "FIXED_TARGET"': '"exit_mode": "SIHIRLI"',
             '"time_stop_min": 60': '"time_stop_min": 99999',
             '"urgency": 0': '"urgency": 7',
             '"regimes": ["TREND YUKARI"]': '"regimes": ["UZAY"]'}[kotu]
    yaz(contrib_dizini, f"bozuk_{beklenen}",
        (GECERLI_META % ("bozuk_kurulum", "None")).replace(kotu, bozuk))
    CB.load(set())
    assert "bozuk_kurulum" not in CB.CONTRIB
    assert any(beklenen in h for e in CB.LOAD_ERRORS for h in e["hatalar"])


def test_yanlis_fire_imzasi_reddedilir(contrib_dizini):
    govde = (GECERLI_META % ("imzasiz", "None")).replace(
        "def fire(f, p, price, atr_abs):", "def fire(veri):")
    yaz(contrib_dizini, "imzasiz", govde)
    CB.load(set())
    assert "imzasiz" not in CB.CONTRIB
    assert any("imza" in h for e in CB.LOAD_ERRORS for h in e["hatalar"])


def test_import_hatasi_sessizce_yutulmaz(contrib_dizini):
    yaz(contrib_dizini, "patlak", "raise RuntimeError('acildi')\n")
    CB.load(set())
    assert any("import edilemedi" in h for e in CB.LOAD_ERRORS for h in e["hatalar"])


# ───────────────────────────────────────────────────────────── tetikleme
def _f():
    return {"ok": True, "price": 100.0, "atr_pct": 0.5, "rsi": 45.0, "trend_up": True}


def test_size_kelepcelenir_ve_kunye_eklenir(contrib_dizini):
    yaz(contrib_dizini, "buyuk", GECERLI_META % (
        "buyuk", '{"direction": "LONG", "size": 9.9, "note": "test"}'))
    CB.load(set())
    out = CB.fire_contrib_sleeves(_f(), ["buyuk"], object())
    assert len(out) == 1 and out[0]["size"] == 1.0
    assert out[0]["note"].startswith("[katkı · @biri]")
    assert out[0]["exit_mode"] == "FIXED_TARGET"


def test_spotta_short_uretilemez(contrib_dizini):
    yaz(contrib_dizini, "kisa", GECERLI_META % (
        "kisa", '{"direction": "SHORT", "size": 0.5}'))
    CB.load(set())
    assert CB.fire_contrib_sleeves(_f(), ["kisa"], object(), allow_short=False) == []
    assert len(CB.fire_contrib_sleeves(_f(), ["kisa"], object(), allow_short=True)) == 1


def test_izin_listesi_disindaki_katki_calismaz(contrib_dizini):
    yaz(contrib_dizini, "izinsiz", GECERLI_META % ("izinsiz", '{"direction": "LONG", "size": 0.5}'))
    CB.load(set())
    assert CB.fire_contrib_sleeves(_f(), [], object()) == []


def test_patlayan_katki_digerlerini_dusurmez(contrib_dizini):
    """Bir katkının hatası koşucuyu durduramaz — ama SESSİZCE de yutulmaz."""
    yaz(contrib_dizini, "patlar", (GECERLI_META % ("patlar", "None")).replace(
        "    return None", "    raise ZeroDivisionError('bom')"))
    yaz(contrib_dizini, "saglam", GECERLI_META % ("saglam", '{"direction": "LONG", "size": 0.4}'))
    CB.load(set())
    CB.FIRE_ERRORS.clear()
    out = CB.fire_contrib_sleeves(_f(), ["patlar", "saglam"], object())
    assert [o["kind"] for o in out] == ["saglam"]
    assert CB.FIRE_ERRORS and CB.FIRE_ERRORS[-1]["sleeve"] == "patlar"
    assert "ZeroDivisionError" in CB.FIRE_ERRORS[-1]["hata"]


def test_ozellikler_hazir_degilse_tetiklenmez(contrib_dizini):
    yaz(contrib_dizini, "hep", GECERLI_META % ("hep", '{"direction": "LONG", "size": 0.5}'))
    CB.load(set())
    assert CB.fire_contrib_sleeves({"ok": False}, ["hep"], object()) == []


# ───────────────────────────────────────────────────────── statik denetim
def _statik(kaynak: str):
    sys.path.insert(0, str(ROOT / "scripts"))
    from cm_verify_contribution import statik_denetim
    return statik_denetim(kaynak, "t")


@pytest.mark.parametrize("kod,beklenen", [
    ("import requests\n", "yasak import"),
    ("from urllib import request\n", "yasak import"),
    ("def f():\n    open('/etc/passwd')\n", "yasak çağrı"),
    ("def f(df):\n    return df.shift(-1)\n", "ileriye bakış"),
    ("import numpy\ndef f():\n    return numpy.random.rand()\n", "rastgelelik"),
    ("x = 1\ndef f():\n    global x\n    x = 2\n", "global"),
])
def test_statik_denetim_yakalar(kod, beklenen):
    bulgular = _statik(kod)
    assert any(beklenen in b for b in bulgular), f"yakalanmadı: {beklenen} → {bulgular}"


def test_statik_denetim_temiz_kodu_gecirir():
    temiz = ("def fire(f, p, price, atr_abs):\n"
             "    rsi = f.get('rsi')\n"
             "    if rsi is None or rsi > 40:\n"
             "        return None\n"
             "    return {'direction': 'LONG', 'size': 0.5}\n")
    assert _statik(temiz) == []


# ─────────────────────────────────────────────────────────── yaşam döngüsü
def test_katkilar_shadow_dogar():
    """Terfi bir kod incelemesi kararı değil, ÖLÇÜM kararıdır."""
    from agi_trader.strategies import lifecycle as LC
    for s in LC.CONTRIB_SLEEVES:
        assert s in LC.SHADOW_SLEEVES
    lc = LC.Lifecycle(path=Path(__file__).parent / "_gecici_lifecycle.json")
    try:
        for s in LC.CONTRIB_SLEEVES:
            assert lc.stage(s) == "SHADOW"
            assert lc.can_trade(s, "paper") is False, "SHADOW katkı paper'da bile emir veremez"
    finally:
        (Path(__file__).parent / "_gecici_lifecycle.json").unlink(missing_ok=True)
