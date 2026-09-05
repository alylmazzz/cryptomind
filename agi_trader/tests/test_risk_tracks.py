# -*- coding: utf-8 -*-
"""Risk rayları (kaldıraçlı trend takip) — 2026-09-05.

NEDEN VAR: "daha riskli, kazanımı yüksek yöntem" isteği, bu sistemde tek bir
ölçülmüş yere işaret ediyor — trend katmanı (48 gerçek paper günü, Sharpe 2,61).
Scalp katmanı ÖLÇÜLDÜ ve negatif (116 işlem, net −%7,3; en iyi hedef politikası
bile −%5,96), dolayısıyla oraya kaldıraç koymak zararı çarpar. Kaldıraç yalnız
kenarı ölçülmüş katmana uygulanır.

KİLİTLENEN DAVRANIŞ:
  1. Kaldıraç ölçeği büyütür, kenarı DEĞİL — aynı veride ağırlıklar oransal artar.
  2. Düşüş kısıcısı: dd_soft'a kadar tam, dd_hard'da yarı, dd_kill'de sıfır.
  3. Kilit histerezisi — dd_kill'i aşan ray dd_hard'ın ALTINA dönene dek nakitte.
  4. Tepe özsermaye kalıcıdır; eski (tepesiz) durum dosyaları geçmişten kurtarılır.
  5. Her rayın kendi durum dosyası vardır; base eski adı korur (geriye uyum).
  6. NaN koruması kaldıraç eklendikten sonra da geçerlidir.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agi_trader.auto.trend_engine import RISK_TRACKS, TrendTrader  # noqa: E402
from agi_trader.config import load_config  # noqa: E402

PAIRS = ["BTC/USDT", "ETH/USDT", "GLD", "SPY"]


def _series(n=300, start=100.0, drift=0.0006, seed=1):
    rng = np.random.default_rng(seed)
    px = start * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.DataFrame({"open": px, "high": px * 1.01, "low": px * 0.99,
                         "close": px, "volume": np.full(n, 1000.0)}, index=idx)


def _data(pairs=PAIRS, seed0=1):
    return {p: _series(seed=seed0 + i) for i, p in enumerate(pairs)}


def _tt(track="base", **kw):
    return TrendTrader(load_config(), pairs=list(PAIRS), initial=10_000.0, track=track, **kw)


# ---------------------------------------------------------------- 1) raylar
def test_raylar_artan_risk_sirasinda():
    """base < aggressive < extreme — hem hedef vol hem kaldıraç tavanı."""
    b, a, e = RISK_TRACKS["base"], RISK_TRACKS["aggressive"], RISK_TRACKS["extreme"]
    assert b["target_vol"] < a["target_vol"] < e["target_vol"]
    assert b["max_lev"] < a["max_lev"] < e["max_lev"]
    # Kill eşiği hard'ın, hard soft'un üstünde olmalı — aksi hâlde kısıcı ters çalışır.
    for p in (b, a, e):
        assert p["dd_soft"] < p["dd_hard"] < p["dd_kill"]


def test_kaldirac_olcegi_buyutur_kenari_degil():
    """Aynı veride agresif rayın ağırlıkları base'in oransal katıdır."""
    d = _data()
    wb = _tt("base").compute_targets(d)
    wa = _tt("aggressive").compute_targets(d)
    sb, sa = sum(wb.values()), sum(wa.values())
    assert sb > 0 and sa > sb, "agresif ray daha yüksek yatırım oranı üretmeli"
    # Oran her varlıkta AYNI olmalı: kaldıraç seçici değildir, tek çarpandır.
    oranlar = [wa[k] / wb[k] for k in wb if wb[k] > 1e-9]
    assert max(oranlar) - min(oranlar) < 1e-6, "kaldıraç varlıklar arasında ayrım yapamaz"


def test_kaldirac_tavani_baglayici():
    """max_lev aşılamaz: çok düşük oynaklıkta bile tavan tutar."""
    tt = _tt("aggressive", target_vol=50.0, max_lev=2.0)   # absürt hedef → tavan bağlar
    w = tt.compute_targets(_data())
    assert sum(w.values()) <= 2.0 + 1e-6


# ------------------------------------------------------ 2) düşüş kısıcısı
def test_dusus_kisicisi_kademeleri():
    tt = _tt("aggressive")           # soft 0,10 · hard 0,20 · kill 0,35
    tt.peak_equity = 10_000.0
    tt.equity = 10_000.0
    assert tt.dd_multiplier() == 1.0                      # düşüş yok
    tt.equity = 9_500.0                                   # %5 < soft
    assert tt.dd_multiplier() == 1.0
    tt.equity = 8_000.0                                   # tam hard (%20) → yarı
    assert abs(tt.dd_multiplier() - 0.5) < 1e-9
    tt.equity = 9_000.0                                   # soft/hard arası (%10) → tam
    assert abs(tt.dd_multiplier() - 1.0) < 1e-9
    tt.equity = 8_500.0                                   # %15 → 0,75
    assert abs(tt.dd_multiplier() - 0.75) < 1e-9


def test_kill_esigi_nakite_ceker_ve_histerezisle_kilitlenir():
    tt = _tt("aggressive")
    tt.peak_equity, tt.equity = 10_000.0, 6_400.0         # %36 > kill 0,35
    assert tt.dd_multiplier() == 0.0 and tt.dd_locked is True
    tt.equity = 7_900.0                                   # %21 — hard'ın (%20) ÜSTÜNDE
    assert tt.dd_multiplier() == 0.0, "kilit hard altına dönmeden açılamaz"
    tt.equity = 8_100.0                                   # %19 < hard %20 → kilit açılır
    assert tt.dd_multiplier() > 0.0 and tt.dd_locked is False


def test_kisici_hedef_agirliklara_uygulanir():
    """Kısıcı yalnız rapor değil: gerçekten pozisyon boyutunu düşürür."""
    d = _data()
    tam = _tt("aggressive")
    kisik = _tt("aggressive")
    kisik.peak_equity, kisik.equity = 10_000.0, 8_000.0   # %20 → çarpan 0,5
    s_tam = sum(tam.compute_targets(d).values())
    s_kisik = sum(kisik.compute_targets(d).values())
    assert abs(s_kisik - 0.5 * s_tam) < 1e-6


def test_kilitli_ray_sifir_pozisyon_acar():
    d = _data()
    tt = _tt("extreme")
    tt.peak_equity, tt.equity = 10_000.0, 5_000.0         # %50 > kill 0,45
    assert sum(tt.compute_targets(d).values()) == 0.0


# --------------------------------------------------------- 3) tepe & kalıcılık
def test_tepe_ozsermaye_mark_ile_guncellenir():
    tt = _tt("aggressive")
    tt.weights = {p: 0.25 for p in PAIRS}
    tt.mark({p: 0.02 for p in PAIRS})                     # +%2
    assert tt.peak_equity >= tt.equity > 10_000.0
    tepe = tt.peak_equity
    tt.mark({p: -0.05 for p in PAIRS})                    # düşüş — tepe GERİLEMEZ
    assert tt.peak_equity == tepe and tt.equity < tepe


def test_tepe_diske_yazilir_ve_geri_yuklenir(tmp_path):
    tt = _tt("aggressive")
    tt.weights = {p: 0.25 for p in PAIRS}
    tt.mark({p: 0.03 for p in PAIRS})
    tt.mark({p: -0.01 for p in PAIRS})
    f = tmp_path / "trend_state_aggressive.json"
    tt.save_state(str(f))
    d = json.loads(f.read_text(encoding="utf-8"))
    assert d["track"] == "aggressive" and d["risk"]["max_lev"] == RISK_TRACKS["aggressive"]["max_lev"]
    yeni = _tt("aggressive")
    assert yeni.load_state(str(f))
    assert abs(yeni.peak_equity - tt.peak_equity) < 1e-6


def test_eski_durum_dosyasinda_tepe_gecmisten_kurtarilir(tmp_path):
    """Tepesiz (eski sürüm) durum dosyası: tepe geçmişten yeniden kurulur.

    Aksi hâlde tepe = güncel özsermaye sanılır, düşüş 0 görünür ve kaldıraç
    kısıcısı en çok gerektiği anda (zaten düşüşteyken) uyumaya devam eder."""
    f = tmp_path / "eski.json"
    f.write_text(json.dumps({
        "pairs": PAIRS, "initial": 10_000.0, "equity": 9_000.0, "weights": {},
        "last_close": {}, "last_signals": {}, "last_rebalance": "2026-09-01",
        "history": [{"date": "2026-08-01", "equity": 10_000.0},
                    {"date": "2026-08-15", "equity": 11_000.0},   # gerçek tepe
                    {"date": "2026-09-01", "equity": 9_000.0}],
    }), encoding="utf-8")
    tt = _tt("aggressive")
    assert tt.load_state(str(f))
    assert abs(tt.peak_equity - 11_000.0) < 1e-6
    assert abs(tt.drawdown_pct() - 18.18) < 0.05          # 1 − 9000/11000


# ------------------------------------------------------------ 4) durum yolu
def test_ray_basina_durum_dosyasi():
    import trend_daemon as TD
    assert TD.state_path("base").name == "trend_state.json"          # geriye uyum
    assert TD.state_path("aggressive").name == "trend_state_aggressive.json"
    assert TD.state_path("extreme").name == "trend_state_extreme.json"
    assert TD.state_path("max").name == "trend_state_max.json"
    assert set(TD.TRACKS) == {"base", "aggressive", "extreme", "max"}


# --------------------------------------------------------------- 5) NaN kalır
def test_nan_korumasi_kaldiracla_birlikte_gecerli():
    """Kaldıraç eklenmesi NaN korumasını bozmamalı (2026-09-04 arızasının regresyonu)."""
    tt = _tt("extreme")
    tt.step(_data(), date_str="2026-09-01")
    kismi = {k: v for k, v in _data().items() if k not in ("GLD", "SPY")}
    ev = tt.step(kismi, date_str="2026-09-02")
    assert math.isfinite(tt.equity) and math.isfinite(tt.peak_equity)
    assert set(ev["missing_prices"]) == {"GLD", "SPY"}


def test_bozuk_tepe_sistemi_kilitlemez():
    """peak NaN/0 gelirse kısıcı 1,0 döner — bozuk metrik pozisyonu sıfırlayamaz."""
    tt = _tt("aggressive")
    tt.peak_equity = float("nan")
    assert tt.dd_multiplier() == 1.0
    tt.peak_equity = 0.0
    assert tt.dd_multiplier() == 1.0


# ------------------------------------------------------------ 6) durum raporu
def test_status_risk_alanlarini_yayimlar():
    tt = _tt("aggressive")
    st = tt.status()
    for k in ("track", "drawdown_pct", "dd_multiplier", "dd_locked", "peak_equity", "risk"):
        assert k in st, f"status alanı eksik: {k}"
    assert st["risk"]["target_vol_pct"] == 45.0
    assert st["risk"]["max_exposure"] == RISK_TRACKS["aggressive"]["max_exposure"]


# ------------------------------------------------------- 7) aynı gün koruması
def test_ayni_gun_iki_kez_adim_atilmaz(tmp_path, capsys, monkeypatch):
    """`step()` gün başına BİR kez çağrılmak üzere yazıldı: ikinci çağrı gün içi
    farkı 'günlük getiri' sanır, bir kez daha yeniden dengeleme maliyeti düşer ve
    geçmişe aynı tarihten iki kayıt girer. 49 günlük gerçek kayıt bu yüzden korunur."""
    import trend_daemon as TD
    monkeypatch.setattr(TD, "STATE", tmp_path / "trend_state.json")
    tt = _tt("aggressive")
    d = _data()
    TD.one_step(tt, cfg=None, data=d)
    assert tt.last_rebalance is not None
    n1, eq1 = len(tt.history), tt.equity
    TD.one_step(tt, cfg=None, data=d)                     # aynı gün ikinci çağrı
    assert len(tt.history) == n1 and tt.equity == eq1, "aynı gün ikinci adım kaydı değiştiremez"
    assert "ATLANDI" in capsys.readouterr().out
    TD.one_step(tt, cfg=None, data=d, force=True)         # --force ile bilinçli
    assert len(tt.history) == n1 + 1


# ------------------------------------------------- 8) kaldıraç finansman gideri
def test_kaldirac_finansman_gideri_dusulur():
    """1×'in üstü BORÇTUR ve bedava değildir. Saymazsak agresif raylar kayırılır:
    4,2× kaldıraçta yılda ~%32'lik gerçek bir gider görünmez olur."""
    tt = _tt("aggressive")
    tt.finance_rate = 0.10
    tt.weights = {p: 1.0 for p in PAIRS}            # toplam 4,0× kaldıraç
    r = tt.mark({p: 0.0 for p in PAIRS})            # piyasa hareketi YOK
    assert abs(r["leverage"] - 4.0) < 1e-9
    beklenen = 3.0 * 0.10 / 365.0                   # borç 3,0× × %10/yıl
    assert abs(r["finance_pct"] - beklenen * 100) < 1e-4   # finance_pct 5 haneye yuvarlanır
    assert tt.equity < 10_000.0, "hareketsiz günde bile finansman gideri düşülmeli"


def test_kaldiracsiz_portfoyde_finansman_yok():
    tt = _tt("base")
    tt.weights = {p: 0.2 for p in PAIRS}            # toplam 0,8× — borç yok
    r = tt.mark({p: 0.0 for p in PAIRS})
    assert r["finance_pct"] == 0.0 and tt.equity == 10_000.0


def test_finansman_nan_uretemez():
    tt = _tt("extreme")
    tt.weights = {PAIRS[0]: float("nan"), PAIRS[1]: 0.5}
    r = tt.mark({p: 0.01 for p in PAIRS})
    assert math.isfinite(tt.equity) and math.isfinite(r["finance_pct"])


# ============================================================ MAKSİMUM rayı
# %1/gün hedefinin ölçülmüş fiyatı: 15,9× kaldıraç. Bu seviyede DÜŞÜK kaldıraçta
# hiç ortaya çıkmayan bir durum gerçek bir olasılık hâline gelir — TASFİYE.
def test_maksimum_rayi_tanimlari():
    m = RISK_TRACKS["max"]
    e = RISK_TRACKS["extreme"]
    assert m["target_vol"] > e["target_vol"] and m["max_lev"] > e["max_lev"]
    assert m["dd_soft"] < m["dd_hard"] < m["dd_kill"]
    # DD kapıları kaldıraçla birlikte GENİŞLEMELİ: 15,9×'te günlük sapma ~%10, dar kapı
    # ilk günde tetiklenir ve ray hiç ölçülemez.
    assert m["dd_soft"] > e["dd_soft"] and m["dd_kill"] > e["dd_kill"]


def test_maksimum_rayi_maruziyet_tavani():
    """Hedef vol %225 düşük oynaklıkta tavanı aşardı; bağlayıcı olan MARUZİYET tavanıdır.

    (2026-09-05 canlı bulgusu: `max_lev` ham hedeflerin çarpanı olduğu için gerçek
    maruziyeti sınırlamıyordu — `max_lev=16` sahada ~6,7× maruziyet veriyordu.)"""
    tt = _tt("max")
    assert tt.max_exposure == 17.0
    w = tt.compute_targets(_data())
    assert sum(w.values()) <= 17.0 + 1e-6


# ------------------------------------------------------------------ tasfiye
def test_tasfiye_ozsermayeyi_negatife_dusurmez():
    """16× kaldıraçta dayanakta −%6,3 sermayeyi bitirir. Model bunu yakalamazsa equity
    NEGATİFE düşer ve sistem 'eksi sermayeyle' işlem yapmaya devam eder."""
    tt = _tt("max")
    tt.weights = {p: 4.0 for p in PAIRS}                  # 16× kaldıraç
    r = tt.mark({p: -0.07 for p in PAIRS})                # dayanak −%7
    assert r.get("ruined") is True
    assert tt.equity == 0.0 and tt.ruined is True
    assert tt.equity >= 0.0, "özsermaye negatife düşemez"


def test_tasfiye_edilmis_ray_pozisyon_acamaz():
    tt = _tt("max")
    tt.ruined = True
    assert tt.dd_multiplier() == 0.0
    assert sum(tt.compute_targets(_data()).values()) == 0.0


def test_tasfiye_diske_yazilir_ve_geri_yuklenir(tmp_path):
    tt = _tt("max")
    tt.weights = {p: 4.0 for p in PAIRS}
    tt.mark({p: -0.07 for p in PAIRS})
    f = tmp_path / "trend_state_max.json"
    tt.save_state(str(f))
    assert json.loads(f.read_text(encoding="utf-8"))["ruined"] is True
    yeni = _tt("max")
    assert yeni.load_state(str(f))
    assert yeni.ruined is True and yeni.dd_multiplier() == 0.0


def test_sifir_ozsermaye_eski_dosyadan_da_tasfiye_sayilir(tmp_path):
    """`ruined` alanı olmayan eski/elle yazılmış dosyada equity 0 ise ray tasfiyedir."""
    f = tmp_path / "eski_max.json"
    f.write_text(json.dumps({"pairs": PAIRS, "initial": 10_000.0, "equity": 0.0,
                             "weights": {}, "last_close": {}, "last_signals": {},
                             "last_rebalance": "2026-09-05", "history": []}), encoding="utf-8")
    tt = _tt("max")
    assert tt.load_state(str(f)) and tt.ruined is True


def test_dusuk_kaldiracta_tasfiye_tetiklenmez():
    """Aynı kod düşük kaldıraçta yanlış pozitif üretmemeli."""
    tt = _tt("base")
    tt.weights = {p: 0.25 for p in PAIRS}                 # 1× kaldıraç
    r = tt.mark({p: -0.07 for p in PAIRS})
    assert not r.get("ruined") and tt.ruined is False and tt.equity > 0


# ==================================================== maruziyet tavanı (2026-09-05)
# max_lev HAM hedeflerin çarpanıdır; ham hedefler tipik olarak ~0,42 topladığı için
# `max_lev = 16` gerçekte ~6,7× maruziyet veriyordu. Gerçek risk sum(ağırlıklar)'dır.
def test_maruziyet_tavani_carpandan_bagimsiz_baglar():
    """Ham hedefler ne toplarsa toplasın, maruziyet tavanı aşılamaz."""
    for ray in ("base", "aggressive", "extreme", "max"):
        tt = _tt(ray)
        tot = sum(tt.compute_targets(_data()).values())
        assert tot <= tt.max_exposure + 1e-6, f"{ray}: maruziyet {tot} > tavan {tt.max_exposure}"


def test_maruziyet_tavani_oranlari_bozmaz():
    """Tavan devreye girince ağırlıklar ORANTILI kısılır — varlık seçimi değişmez."""
    d = _data()
    tt = _tt("max", max_exposure=0.5)          # kesin bağlayıcı
    w = tt.compute_targets(d)
    ham = _tt("max", max_exposure=1e9).compute_targets(d)
    assert abs(sum(w.values()) - 0.5) < 1e-9
    oran = [w[k] / ham[k] for k in ham if ham[k] > 1e-12]
    assert max(oran) - min(oran) < 1e-9


def test_bandlar_artan_maruziyet_sirasinda():
    b = RISK_TRACKS["base"]["max_exposure"]
    a = RISK_TRACKS["aggressive"]["max_exposure"]
    e = RISK_TRACKS["extreme"]["max_exposure"]
    m = RISK_TRACKS["max"]["max_exposure"]
    assert b < a < e < m
    # MAKSİMUM bandı, süpürmedeki 15,94× çarpanına (temel maruziyet ~1,065) denk gelmeli
    assert 16.0 <= m <= 18.0


def test_dd_kisicisi_maruziyet_tavanindan_SONRA_da_etkili():
    """Tavan, kısıcıyı geçersiz kılmamalı: kısılmış ray tavana dayanamaz."""
    d = _data()
    tam = sum(_tt("max").compute_targets(d).values())
    kisik_tt = _tt("max")
    kisik_tt.peak_equity, kisik_tt.equity = 10_000.0, 7_000.0   # %30 düşüş = dd_hard → 0,5
    kisik = sum(kisik_tt.compute_targets(d).values())
    assert kisik < tam, "düşüş kısıcısı maruziyet tavanı varken de boyutu düşürmeli"
