# -*- coding: utf-8 -*-
"""Bar içi stop ihlali — 2026-09-05 ölçümünün regresyonu.

ARIZA: `decide_exit` sert stopu yalnız `price` (bar kapanışı) ile karşılaştırıyordu.
Oysa aynı fonksiyon `bar_high`/`bar_low`'u zaten alıyor ve bunları MAE ile chandelier
için KULLANIYORDU. Sonuç asimetrikti:

    ZARAR  bar uçlarıyla ölçülüyor  (mae_pct → lowest_low → bar_low)
    STOP   kapanışla kontrol ediliyor (price)

Bar içinde stop delinip kapanış stopun üstünde biterse pozisyon yaşamaya devam
ediyordu; risk modeli gerçeğin ALTINI gösteriyordu.

CANLI KANIT (116 işlem, 2026-09-05): 11 EARLY_ABORT işleminin HEPSİNDE tepe %0 ve
MAE %0,47–0,94 iken stop mesafesi ~%0,52 idi — yani hepsi stopunu aşmıştı ve
ortalama −%0,77'de kapandı; gerçek STOP çıkışlarının ortalaması −%0,25'ti. Erken
iptal "stopun %60'ına gitti" kuralıyla ateşlendiği için, stop kaçırıldığında
zararı stopun ötesinde kapatan ikinci bir kapı hâline gelmişti.

KİLİTLENEN DAVRANIŞ:
  1. Bar dibi (LONG) / tepesi (SHORT) stopu deldiyse ÇIKIŞ olur, kapanış nerede olursa olsun.
  2. Dolum stop SEVİYESİNDEDİR (kapanışta değil) — zarar olduğundan küçük yazılamaz.
  3. Bar tamamen seviyenin ötesindeyse (boşluk) dolum bardaki EN KÖTÜ fiyattır.
  4. Trailing için bar içi test YALNIZ bardan önce var olan seviyeye uygulanır
     (aynı barın tepesinden türetilmiş seviyeyi aynı barın dibiyle test etmek,
     gözlenemeyen bir bar içi sıralama varsaymak olurdu).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agi_trader.strategies import exit_engine as XE  # noqa: E402


def _long(stop=99.0, mode=None, cost=0.2, atr=0.3):
    return XE.PositionTrack("LONG", 100.0, stop, 101.6, 1000.0,
                            mode or XE.PARTIAL_AND_RUN, 1.0, cost, atr)


def _short(stop=101.0, mode=None, cost=0.2, atr=0.3):
    return XE.PositionTrack("SHORT", 100.0, stop, 98.4, 1000.0,
                            mode or XE.PARTIAL_AND_RUN, 1.0, cost, atr)


# --------------------------------------------------------------- 1) bar içi
def test_long_bar_dibi_stopu_delerse_kapanis_ustunde_olsa_bile_cikar():
    """ASIL REGRESYON: kapanış 99,8 (stopun ÜSTÜNDE) ama bar dibi 98,7 (stopun ALTINDA)."""
    p = XE.ExitParams(min_hold_sec=0)
    t = _long(stop=99.0)
    d = XE.decide_exit(t, 99.8, 100.0, 98.7, p, now=1010.0)
    assert d is not None, "bar içinde stop delindi — pozisyon yaşamaya devam edemez"
    assert d["reason"] == "STOP"
    assert d["intrabar"] is True
    assert abs(d["exit_price"] - 99.0) < 1e-9, "dolum stop seviyesinde olmalı"


def test_short_bar_tepesi_stopu_delerse_cikar():
    p = XE.ExitParams(min_hold_sec=0)
    t = _short(stop=101.0)
    d = XE.decide_exit(t, 100.2, 101.4, 100.0, p, now=1010.0)
    assert d and d["reason"] == "STOP" and d["intrabar"] is True
    assert abs(d["exit_price"] - 101.0) < 1e-9


def test_dokunulmayan_stop_cikis_uretmez():
    """Yanlış pozitif olmamalı: bar stopa değmediyse çıkış yok."""
    p = XE.ExitParams(min_hold_sec=0)
    t = _long(stop=99.0)
    assert XE.decide_exit(t, 99.8, 100.0, 99.4, p, now=1010.0) is None


# ------------------------------------------------------------- 2) dolum fiyatı
def test_zarar_stop_seviyesinden_yazilir_kapanistan_degil():
    """Bar içi ihlalde net zarar, kapanışın değil STOP'un zararı olmalı."""
    p = XE.ExitParams(min_hold_sec=0)
    t = _long(stop=99.0, cost=0.2)
    d = XE.decide_exit(t, 99.9, 100.0, 98.5, p, now=1010.0)
    # stop 99,0 → brüt −%1,0 ; net = brüt − maliyet %0,2 = −%1,2
    assert abs(d["gross_pct"] - (-1.0)) < 1e-6
    assert abs(d["net_pct"] - (-1.2)) < 1e-6


def test_bosluk_durumunda_dolum_bardaki_en_kotu_fiyat():
    """Bar TAMAMEN stopun altındaysa stop seviyesinde dolum varsaymak iyimserdir."""
    p = XE.ExitParams(min_hold_sec=0)
    t = _long(stop=99.0)
    d = XE.decide_exit(t, 98.0, 98.4, 97.6, p, now=1010.0)      # bar tümü < 99
    assert d and d["reason"] == "STOP"
    assert abs(d["exit_price"] - 97.6) < 1e-9, "boşlukta bardaki en kötü fiyat"


# ---------------------------------------------------- 3) erken iptal artık öne geçemez
def test_stop_erken_iptalden_once_ateslenir():
    """EARLY_ABORT, stop mesafesinin %60'ında ateşlenir; stop %100'de. Bar dibi stopu
    deldiyse sonuç STOP olmalı — erken iptal stopun ÖTESİNDE kapatan bir kapı olamaz."""
    p = XE.ExitParams(min_hold_sec=0, time_stop_sec=3600)
    t = _long(stop=99.0)
    t.stop_pct = 1.0
    d = XE.decide_exit(t, 99.5, 100.0, 98.6, p, now=1100.0)      # MAE %1,4 > stop %1,0
    assert d["reason"] == "STOP", f"stop öncelikli olmalı, gelen: {d['reason']}"


# ------------------------------------------------------------- 4) trailing sırası
def test_ayni_barda_yukselen_trailing_o_barin_dibiyle_test_edilmez():
    """Chandelier bu barın TEPESİNDEN türetilir; onu aynı barın DİBİYLE test etmek
    'önce tepe, sonra dip' sırasını varsaymak olurdu — gözlenemez."""
    p = XE.ExitParams(min_hold_sec=0, retain_fraction=0.5)
    t = XE.PositionTrack("LONG", 100.0, 99.0, 101.6, 1000.0, XE.DYNAMIC_PEAK, 1.0, 0.2, 0.3)
    assert XE.decide_exit(t, 100.4, 100.4, 100.0, p, now=1001.0) is None and not t.armed
    # bu barda silahlanır ve trailing kurulur; bar dibi 101,0 yeni seviyenin altında olsa
    # bile ÇIKIŞ ÜRETİLMEZ (seviye bu bardan önce yoktu)
    d = XE.decide_exit(t, 102.0, 102.0, 101.0, p, now=1002.0)
    assert d is None and t.armed and t.trail_stop is not None


def test_onceki_bardan_kalan_trailing_bar_ici_test_edilir():
    p = XE.ExitParams(min_hold_sec=0, retain_fraction=0.5)
    t = XE.PositionTrack("LONG", 100.0, 99.0, 101.6, 1000.0, XE.DYNAMIC_PEAK, 1.0, 0.2, 0.3)
    XE.decide_exit(t, 100.4, 100.4, 100.0, p, now=1001.0)
    XE.decide_exit(t, 102.0, 102.0, 101.0, p, now=1002.0)        # trailing kuruldu
    lvl = t.trail_stop
    assert lvl is not None
    d = XE.decide_exit(t, lvl + 0.3, lvl + 0.4, lvl - 0.2, p, now=1003.0)   # dip seviyeyi deldi
    assert d is not None and d["reason"] in ("TRAIL", "GIVEBACK", "STOP")
    if d["reason"] == "TRAIL":
        assert d["intrabar"] is True and abs(d["exit_price"] - lvl) < 1e-9


# ----------------------------------------------------------------- 5) yardımcılar
def test_stop_touched_ve_fill_yardimcilari():
    assert XE._stop_touched(99.0, 100.0, 98.5, 1.0) is True      # LONG: dip deldi
    assert XE._stop_touched(99.0, 100.0, 99.2, 1.0) is False
    assert XE._stop_touched(101.0, 101.5, 100.0, -1.0) is True   # SHORT: tepe deldi
    assert XE._stop_fill(99.0, 100.0, 98.5, 1.0) == 99.0         # seviye bar içinde
    assert XE._stop_fill(99.0, 98.4, 97.6, 1.0) == 97.6          # boşluk → en kötü
    assert XE._stop_fill(101.0, 101.5, 100.0, -1.0) == 101.0
    assert XE._stop_fill(101.0, 103.0, 102.0, -1.0) == 103.0     # boşluk (SHORT)
