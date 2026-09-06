# -*- coding: utf-8 -*-
"""
HABER ETKİ MOTORU — gözle → ufuk dolunca çöz → öğren.

Kilitlenen davranışlar:
  • Aynı olayın tekrar başlıklarıyla ÇİFT sayılmaması (t'yi sahte büyütürdü)
  • Ufuk dolmadan çözüm YOK (ileriye bakış olurdu)
  • BÜYÜKLÜK ile YÖN'ün AYRI raporlanması (birini diğerinin kanıtı saymak en sık hata)
  • İstatistik kapısı: n < MIN_GOZLEM ya da |t| < 2 → "ölçülmedi" (varsayım kanıt değil)
  • Bekleyen kuyruğun süreç yeniden başlasa da kaybolmaması
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agi_trader.sentiment.news_impact import (  # noqa: E402
    HaberEtkiMotoru, UFUKLAR, MIN_GOZLEM, T_ESIK)

T0 = 1_700_000_000.0


def _m(tmp_path):
    return HaberEtkiMotoru(str(tmp_path), "t")


# ═══════════════════════ 1) GÖZLEM ═══════════════════════
def test_ayni_olay_tekrar_basliklarla_CIFT_sayilmaz(tmp_path):
    m = _m(tmp_path)
    assert m.gozle("BTC/USDT", "LISTING", 1, 0.6, 100.0, 0.3, T0)
    # aynı kategori + aynı parite, 30 dk içinde → SAYILMAZ
    assert not m.gozle("BTC/USDT", "LISTING", 2, 0.5, 100.5, 0.3, T0 + 600)
    # farklı kategori sayılır
    assert m.gozle("BTC/USDT", "HACK", 1, -0.9, 100.0, 0.3, T0 + 600)
    # pencere dolunca aynı kategori tekrar sayılır
    assert m.gozle("BTC/USDT", "LISTING", 1, 0.6, 101.0, 0.3, T0 + 2000)
    assert len(m.bekleyen) == 3


def test_kategorisiz_ve_gecersiz_fiyat_kaydedilmez(tmp_path):
    m = _m(tmp_path)
    assert not m.gozle("BTC/USDT", "OTHER", 1, 0.5, 100.0, 0.3, T0)
    assert not m.gozle("BTC/USDT", "", 1, 0.5, 100.0, 0.3, T0)
    assert not m.gozle("BTC/USDT", "LISTING", 1, 0.5, 0.0, 0.3, T0)
    assert len(m.bekleyen) == 0


# ═══════════════════════ 2) ÇÖZÜM ═══════════════════════
def test_ufuk_dolmadan_cozum_YOK(tmp_path):
    m = _m(tmp_path)
    m.gozle("BTC/USDT", "LISTING", 1, 0.6, 100.0, 0.3, T0)
    assert m.coz({"BTC/USDT": 105.0}, T0 + 60) == [], "5 dk dolmadan çözülmemeli"
    r = m.coz({"BTC/USDT": 105.0}, T0 + UFUKLAR["5m"] + 1)
    assert len(r) == 1 and r[0]["ufuk"] == "5m"
    assert r[0]["r"] == pytest.approx(5.0)


def test_her_ufuk_AYRI_satir_uretir_ve_kuyruk_temizlenir(tmp_path):
    m = _m(tmp_path)
    m.gozle("ETH/USDT", "HACK", 1, -0.9, 200.0, 0.5, T0)
    m.coz({"ETH/USDT": 190.0}, T0 + UFUKLAR["5m"] + 1)
    m.coz({"ETH/USDT": 180.0}, T0 + UFUKLAR["1h"] + 1)
    m.coz({"ETH/USDT": 170.0}, T0 + UFUKLAR["4h"] + 1)
    son = m.coz({"ETH/USDT": 160.0}, T0 + UFUKLAR["24h"] + 1)
    assert son and son[0]["ufuk"] == "24h" and son[0]["r"] == pytest.approx(-20.0)
    assert m.bekleyen == [], "tüm ufuklar dolunca gözlem kuyruktan düşmeli"
    assert len(list(m.satirlar())) == 4


def test_ATR_normalize_buyukluk(tmp_path):
    """Aynı yüzde hareket, DÜŞÜK oynaklıklı pariteyle YÜKSEK oynaklıklıda aynı
    'anormallik' anlamına gelmez — z bunu düzeltir."""
    m = _m(tmp_path)
    m.gozle("A/USDT", "ETF", 1, 0.5, 100.0, 0.1, T0)      # sakin
    m.gozle("B/USDT", "ETF", 1, 0.5, 100.0, 2.0, T0)      # oynak
    r = m.coz({"A/USDT": 103.0, "B/USDT": 103.0}, T0 + UFUKLAR["1h"] + 1)
    z = {x["sym"]: x["z"] for x in r if x["ufuk"] == "1h"}
    assert z["A/USDT"] > z["B/USDT"], "sakin paritede aynı hareket DAHA anormaldir"


def test_bekleyen_kuyruk_diskte_KALICI(tmp_path):
    m = _m(tmp_path)
    m.gozle("BTC/USDT", "LISTING", 1, 0.6, 100.0, 0.3, T0)
    m2 = HaberEtkiMotoru(str(tmp_path), "t")              # süreç yeniden başladı
    assert len(m2.bekleyen) == 1 and m2.bekleyen[0].sym == "BTC/USDT"
    r = m2.coz({"BTC/USDT": 102.0}, T0 + UFUKLAR["5m"] + 1)
    assert len(r) == 1


# ═══════════════════════ 3) ÖĞRENME ═══════════════════════
def _doldur(m, kat, n, getiriler, sym="X/USDT", atr=0.1):
    """atr = 1 dakikalık ATR%. 4 saatlik ölçek = atr × √240 ≈ atr × 15,5.
    atr=0,1 → 4 saatte 'normal' hareket ≈ %1,55; %6'lık hareket ≈ 3,9× normal."""
    for i, g in enumerate(getiriler[:n]):
        ts = T0 + i * 100000
        m.gozle(sym, kat, 1, 0.5, 100.0, atr, ts)
        m.coz({sym: 100.0 * (1 + g / 100)}, ts + UFUKLAR["4h"] + 1)


def test_yon_olculmesi_icin_n_VE_t_sarti(tmp_path):
    m = _m(tmp_path)
    _doldur(m, "LISTING", 5, [3.0] * 5)                   # güçlü ama n az
    o = m.ozet("4h", min_n=MIN_GOZLEM)
    assert o["kategori"]["LISTING"]["yon_olculdu"] is False, "n < MIN_GOZLEM → ölçülmedi"

    m2 = _m(tmp_path / "b")
    _doldur(m2, "HACK", 16, [-2.0, -2.4, -1.8, -2.2, -2.1, -1.9, -2.3, -2.0,
                             -2.5, -1.7, -2.1, -2.2, -1.9, -2.4, -2.0, -2.1])
    v = m2.ozet("4h")["kategori"]["HACK"]
    assert v["n"] == 16 and v["yon_ort_pct"] < 0 and abs(v["yon_t"]) >= T_ESIK
    assert v["yon_olculdu"] is True


def test_BUYUK_ama_YONSUZ_kategori_yon_olculmus_sayilmaz(tmp_path):
    """En sık hata: 'büyük hareket yapıyor' ile 'yönü öngörülüyor' aynı şey değil."""
    m = _m(tmp_path)
    _doldur(m, "REGULATORY", 16, [6.0, -6.0, 5.5, -5.5, 6.2, -6.1, 5.8, -5.9,
                                  6.0, -6.0, 5.7, -5.6, 6.1, -6.2, 5.9, -5.8])
    v = m.ozet("4h")["kategori"]["REGULATORY"]
    # %6'lık hareketler, 1 dk ATR %0,1 olan bir paritede 4 saatlik normalin ~3,9 katı
    assert v["buyukluk_z"] > 1.3 and v["buyukluk_olculdu"] is True,         f"hareket büyük olmalı (z={v['buyukluk_z']})"
    assert v["yon_olculdu"] is False, "yön öngörülemiyor — ölçüldü sayılmamalı"
    assert abs(v["yukari_orani"] - 0.5) < 0.2


def test_prior_karsilastirma_yanlis_varsayimi_yakalar(tmp_path):
    """EVENT_PRIOR'da TOKEN_UNLOCK −0,3 yazıyor. Ölçüm tersini gösterirse rapor bunu söyler."""
    m = _m(tmp_path)
    _doldur(m, "TOKEN_UNLOCK", 16, [2.0, 2.4, 1.8, 2.2, 2.1, 1.9, 2.3, 2.0,
                                    2.5, 1.7, 2.1, 2.2, 1.9, 2.4, 2.0, 2.1])
    rows = {r["kategori"]: r for r in m.prior_karsilastir("4h")}
    r = rows["TOKEN_UNLOCK"]
    assert r["varsayim_prior"] < 0 and r["olculen_pct"] > 0
    assert r["olculdu"] is True and r["uyumlu"] is False, "varsayım YANLIŞ olarak işaretlenmeli"


def test_kategori_parite_kesiti(tmp_path):
    m = _m(tmp_path)
    _doldur(m, "LISTING", 14, [1.0] * 14, sym="AAA/USDT")
    _doldur(m, "LISTING", 14, [-1.0] * 14, sym="BBB/USDT")
    o = m.ozet("4h")
    kp = o["kategori_parite"]
    assert kp["LISTING|AAA/USDT"]["yon_ort_pct"] > 0
    assert kp["LISTING|BBB/USDT"]["yon_ort_pct"] < 0
    # kategori geneli ikisinin ortalaması → yön yok
    assert abs(o["kategori"]["LISTING"]["yon_ort_pct"]) < 0.2
