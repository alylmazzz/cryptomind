"""
KAPSAM TESTLERİ — testi hiç olmayan fazlar.

Mevcut paket FAZ 1 / 6 / 7 / 8 / 9 + mover'ı kapsıyordu; FAZ 2 (sleeve'ler),
FAZ 3 (kaydedici), FAZ 4 (etiketleme/meta), FAZ 5 (formasyon/harmonik),
FAZ 10 (evren) ve yeni gösterge tablosu TESTSİZDİ. Bu dosya o boşluğu kapatır.

Testler DEĞİŞMEZ'lere odaklanır (ileriye bakış yok, işaret tutarlılığı,
aritmetik kimlikler) — "şu sayı şu çıkmalı" gibi kırılgan iddialara değil.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ===========================================================================
# yardımcılar
# ===========================================================================
def _prices(n=600, k=4, seed=0, drift=0.0004, vol=0.02) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    out = {}
    for j in range(k):
        r = rng.normal(drift, vol, n)
        out[f"A{j}"] = 100 * np.exp(np.cumsum(r))
    return pd.DataFrame(out, index=idx)


def _ohlcv(n=500, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, n)))
    o = np.concatenate([[c[0]], c[:-1]])
    w = c * 0.01
    return pd.DataFrame({
        "open": o,
        "high": np.maximum(o, c) + np.abs(rng.normal(0, 1, n)) * w,
        "low": np.minimum(o, c) - np.abs(rng.normal(0, 1, n)) * w,
        "close": c,
        "volume": rng.uniform(500, 1500, n),
    }, index=pd.date_range("2022-01-01", periods=n, freq="D"))


# ===========================================================================
# FAZ 2 — sleeve'ler
# ===========================================================================
def test_sleeve_getirisi_ileriye_bakmaz():
    """EN KRİTİK DEĞİŞMEZ: bugünün pozisyonu bugünün getirisini kullanamaz.

    Son barın fiyatını değiştirmek GEÇMİŞ getirileri etkilememelidir. Bu test
    `.shift(1)` unutulursa kırılır — bu projede tam olarak o hata bir kez
    Sharpe'ı 0,80'den 3,68'e çıkarmıştı."""
    from agi_trader.strategies.sleeves.price_sleeves import TrendSleeve
    px = _prices()
    s = TrendSleeve()
    r1 = s.returns(px)
    px2 = px.copy()
    px2.iloc[-1] *= 1.35                    # yalnız SON barı değiştir
    r2 = s.returns(px2)
    ortak = r1.index.intersection(r2.index)[:-1]     # son bar hariç
    pd.testing.assert_series_equal(r1.loc[ortak], r2.loc[ortak],
                                   check_names=False)


def test_sleeve_maliyet_getiriyi_dusurur():
    from agi_trader.strategies.sleeves.price_sleeves import TrendSleeve
    px = _prices()
    ucretsiz = TrendSleeve(cost=0.0).returns(px).sum()
    ucretli = TrendSleeve(cost=0.005).returns(px).sum()
    assert ucretli < ucretsiz, "maliyet uygulanmıyor"


def test_vol_hedefleme_olceklendirir():
    from agi_trader.strategies.sleeves.base import vol_target_scale
    sakin = _prices(vol=0.005, seed=3)
    calkanti = _prices(vol=0.05, seed=3)
    a = vol_target_scale(sakin).iloc[-1].mean()
    b = vol_target_scale(calkanti).iloc[-1].mean()
    assert a > b, "oynaklık arttıkça kaldıraç azalmalı"


@pytest.mark.parametrize("cls_adi", ["TrendSleeve", "CrossSectionalSleeve",
                                     "ShortReversalSleeve"])
def test_sleeve_pozisyonlari_sonlu_ve_hizali(cls_adi):
    import agi_trader.strategies.sleeves.price_sleeves as ps
    px = _prices()
    pos = getattr(ps, cls_adi)().positions(px)
    assert list(pos.columns) == list(px.columns)
    assert np.isfinite(pos.to_numpy()).all(), "pozisyonda NaN/inf var"


def test_carry_riski_olculemez_diye_isaretlenir():
    """Carry sleeve Sharpe raporlamamalı — gün-içi risk günlük veriyle ölçülemez."""
    from agi_trader.strategies.sleeves.carry import CarrySleeve
    px = _prices(k=2)
    fund = {c: pd.Series(0.0002, index=px.index) for c in px.columns}
    r = CarrySleeve().returns(px, funding=fund)
    assert r.attrs.get("risk_measurable") is False
    assert "Sharpe" in r.attrs.get("risk_note", "")


def test_carry_dataframe_funding_de_kabul_eder():
    """REGRESYON: DataFrame verilince `if not funding` 'truth value is
    ambiguous' ValueError'ı fırlatıyordu; gerisi zaten DataFrame ile çalışıyordu."""
    from agi_trader.strategies.sleeves.carry import CarrySleeve
    px = _prices(k=2)
    fdf = pd.DataFrame({c: np.full(len(px), 0.0002) for c in px.columns},
                       index=px.index)
    r = CarrySleeve().returns(px, funding=fdf)      # patlamamalı
    assert isinstance(r, pd.Series)
    assert CarrySleeve().returns(px, funding=None).empty
    assert CarrySleeve().returns(px, funding={}).empty


# ===========================================================================
# FAZ 2b — dağıtıcı
# ===========================================================================
def test_erc_agirliklari_toplami_bir_ve_sinirli():
    from agi_trader.auto.sleeve_allocator import erc_weights, MAX_SLEEVE_RISK
    rng = np.random.default_rng(0)
    X = rng.normal(0, 0.01, (400, 4))
    w = erc_weights(np.cov(X.T))
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= -1e-9).all(), "negatif ağırlık"
    assert w.max() <= MAX_SLEEVE_RISK + 1e-6, "sleeve bütçe tavanı aşıldı"


def test_max_sharpe_kaybedene_agirlik_vermez():
    """ERC beklenen getiriyi yok sayıp negatif Sharpe'lı sleeve'e %41 vermişti.
    max_sharpe bunu yapmamalı."""
    from agi_trader.auto.sleeve_allocator import max_sharpe_weights
    rng = np.random.default_rng(1)
    X = rng.normal(0, 0.01, (500, 3))
    mu = np.array([0.001, 0.0008, -0.002])      # 3. sleeve kaybediyor
    w = max_sharpe_weights(np.cov(X.T), mu)
    assert w[2] < w[0] and w[2] < w[1], "kaybeden sleeve en düşük ağırlığı almalı"


# ===========================================================================
# FAZ 3 — kaydedici
# ===========================================================================
def test_kaydedici_ay_yolu_ve_birlestirme(tmp_path, monkeypatch):
    from agi_trader.data import recorder
    monkeypatch.setattr(recorder, "_out_dir", lambda: tmp_path)
    ts = pd.Timestamp("2026-03-15 12:00", tz="UTC")
    df = pd.DataFrame([{"ts": ts, "symbol": "BTCUSDT", "funding": 0.0001}])
    p1 = recorder.append(df)
    assert p1.exists() and "2026-03" in p1.name
    recorder.append(df)                    # aynı kayıt tekrar
    okunan = pd.read_parquet(p1)
    assert len(okunan) >= 1


# ===========================================================================
# FAZ 4 — etiketleme / meta
# ===========================================================================
def test_ucul_bariyer_etiketleri_gecerli():
    from agi_trader.research.labeling import triple_barrier_labels
    px = _ohlcv()["close"]
    lab = triple_barrier_labels(px)
    assert len(lab) > 0
    assert set(np.unique(lab["bin"])) <= {-1.0, 0.0, 1.0}
    assert set(lab.columns) >= {"t1", "ret", "bin", "barrier"}
    t1 = lab["t1"].to_numpy()
    i = np.arange(len(lab))
    # -1 = nöbetçi: ufuk veri sonunu aşıyor, etiket değerlendirilemez
    gecerli = t1 >= 0
    assert (t1[gecerli] > i[gecerli]).all(), "t1 olaydan sonra bitmiyor (ileriye bakış)"
    # nöbetçiler yalnız serinin SONUNDA olabilir
    if (~gecerli).any():
        assert i[~gecerli].min() >= len(lab) - 12, "nöbetçi seri ortasında"
    # nöbetçi satırlar bahis üretmemeli
    assert (lab.loc[~gecerli, "bin"] == 0).all()


def test_ucul_bariyer_meta_etiket_ikili():
    """side verilirse çıktı META-etikettir: bin ∈ {0,1}."""
    from agi_trader.research.labeling import triple_barrier_labels
    px = _ohlcv()["close"]
    side = pd.Series(1.0, index=px.index)
    lab = triple_barrier_labels(px, side=side)
    assert set(np.unique(lab["bin"])) <= {0.0, 1.0}


def test_benzersizlik_agirliklari_ortalamasi_bir():
    """`uniqueness_weights` ham 1/eşzamanlılık'ı ORTALAMASI 1 olacak şekilde
    yeniden ölçekler (sklearn sample_weight uyumu). Değerlerin (0,1] olması
    beklenmez; beklenen şey ortalamanın 1 ve çok örtüşenin daha DÜŞÜK olmasıdır."""
    from agi_trader.research.labeling import uniqueness_weights
    w = uniqueness_weights(np.arange(10) + 3)
    assert len(w) == 10
    assert (w > 0).all(), "sıfır/negatif ağırlık"
    assert abs(w.mean() - 1.0) < 1e-6, "ortalama 1 değil"
    # az örtüşen (baştaki, komşusu az) çok örtüşenden yüksek ağırlık almalı
    az = uniqueness_weights(np.arange(20))          # her etiket 1 bar → örtüşme yok
    cok = uniqueness_weights(np.arange(20) + 10)    # her etiket 11 bar → çok örtüşme
    assert az.std() <= cok.std() + 1e-9


def test_kelly_kesri_tavani_asmaz():
    from agi_trader.ai.meta_label import kelly_size
    assert kelly_size(0.5) == 0.0, "P=0,5'te bahis olmamalı"
    assert kelly_size(0.2) == 0.0, "eşik altında bahis olmamalı"
    assert 0 < kelly_size(0.75) <= 0.5
    assert kelly_size(1.0) <= 0.5, "tam Kelly asla"


# ===========================================================================
# FAZ 5 — formasyonlar
# ===========================================================================
def test_formasyon_aritmetigi_tutarli():
    """Geçerli her formasyonda hedef/stop yönü ve yüzdeler kendi içinde tutmalı."""
    from agi_trader.analysis.chart_patterns import detect_chart_patterns
    for seed in range(12):
        df = _ohlcv(seed=seed)
        price = float(df["close"].iloc[-1])
        for p in detect_chart_patterns(df, top_n=5):
            if not p["valid"]:
                continue
            t, s, d = p["target"], p["stop"], p["direction"]
            assert d in ("LONG", "SHORT")
            if d == "LONG":
                assert t > price and s < price, f"{p['name']}: LONG kurulumu ters"
            else:
                assert t < price and s > price, f"{p['name']}: SHORT kurulumu ters"
            assert abs((t - price) / price * 100 - p["target_pct"]) < 0.05
            assert abs((s - price) / price * 100 - p["stop_pct"]) < 0.05
            assert p["rr"] > 0
            assert p["clears_min_move"] == (abs(p["target_pct"]) >= 1.0)


def test_gecersiz_formasyon_islenebilir_sayilmaz():
    """Fiyat stop'u ihlal ettiyse formasyon ≥%1 rozeti ALMAMALI."""
    from agi_trader.analysis.chart_patterns import ChartPattern, _apply_validity
    p = ChartPattern(key="k", name="Test", family="dönüş", direction="LONG",
                     status="oluşuyor", completion=0.8, quality=0.8, score=0.5,
                     color="#fff", start_i=0, end_i=10, breakout=100.0,
                     target=110.0, stop=95.0, target_pct=10.0, stop_pct=-5.0,
                     rr=2.0, clears_min_move=True)
    _apply_validity(p, price=94.0)            # stop'un ALTINDA
    assert p.valid is False and p.validity == "stop_ihlal"
    p2 = ChartPattern(**{**p.__dict__, "valid": True, "validity": "işlenebilir"})
    _apply_validity(p2, price=115.0)          # hedefin ÜSTÜNDE
    assert p2.valid is False and p2.validity == "hedef_aşıldı"


def test_dusen_kama_tespit_edilir():
    """REGRESYON: yakınsama koşulu `up_n > dn_n` (ıraksama) yazıldığı için
    düşen kama 20 sentetik denemede 0 kez tespit edilmişti."""
    from agi_trader.analysis.chart_patterns import detect_wedges, find_pivots
    rng = np.random.default_rng(11)
    hit = 0
    for _ in range(10):
        n = 90
        x = np.arange(n)
        c = 120 - 0.28 * x + np.linspace(7.0, 0.7, n) * np.sin(2 * np.pi * x / 13)
        c = c * (1 + rng.normal(0, 0.0015, n))
        o = np.concatenate([[c[0]], c[:-1]])
        df = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.002,
                           "low": np.minimum(o, c) * 0.998, "close": c,
                           "volume": np.full(n, 1000.0)},
                          index=pd.date_range("2024-01-01", periods=n, freq="4h"))
        sh, sl = find_pivots(df, 3, 3)
        if any(p.key == "falling_wedge" for p in detect_wedges(df, sh, sl)):
            hit += 1
    assert hit >= 7, f"düşen kama 10 denemede yalnız {hit} kez bulundu"


def _tri_ohlc(close, rng, noise=0.004):
    close = np.asarray(close, float); n = len(close)
    op = np.concatenate([[close[0]], close[:-1]]); w = close * noise
    return pd.DataFrame({
        "open": op,
        "high": np.maximum(op, close) + np.abs(rng.normal(0, 1, n)) * w,
        "low": np.minimum(op, close) - np.abs(rng.normal(0, 1, n)) * w,
        "close": close, "volume": rng.uniform(800, 1200, n)},
        index=pd.date_range("2024-01-01", periods=n, freq="4h"))


def test_boundary_fit_dokunmayan_pivotu_ayiklar():
    """REGRESYON — üçgen dedektörünün asıl kusuru buydu.

    Ölçülen gerçek vaka: yatay direnç 110'da, ama son 5 tepe pivotunun ikisi
    dirence DEĞMEYEN ara salınım tepeleri (109,8 · 109,7). Düz en-küçük-kareler
    bunlara da uyup eğimi %+3,72/pencere gösteriyor, dolayısıyla "yatay direnç"
    koşulu sağlanmıyor ve yükselen üçgen kaçırılıyordu."""
    from agi_trader.analysis.chart_patterns import _boundary_fit, _fit_line, _norm_slope
    xs = [110, 116, 125, 129, 136]
    ys = [109.8, 109.7, 111.3, 111.0, 111.3]
    duz_m, _, _ = _fit_line(xs, ys)
    m, b, r2, kept = _boundary_fit(xs, ys, "upper", tol=0.4)
    assert abs(m) < abs(duz_m), "zarf uydurması düz uydurmadan daha yatay olmalı"
    assert 110 not in kept and 116 not in kept, "dokunmayan pivotlar ayıklanmadı"
    assert len(kept) >= 3
    # eğim artık 'yatay' eşiğinin (%1,5/pencere) altında olmalı
    assert abs(_norm_slope(m, 110.0, 100)) < 1.5


def test_boundary_fit_alt_sinirda_ustte_kalani_atar():
    from agi_trader.analysis.chart_patterns import _boundary_fit
    xs = [10, 20, 30, 40, 50]
    ys = [90.0, 90.2, 95.0, 90.1, 89.9]        # 95 destek çizgisine ait değil
    m, b, r2, kept = _boundary_fit(xs, ys, "lower", tol=0.5)
    assert 30 not in kept, "destekten uzak pivot ayıklanmadı"


def _asc_tri(rng, tail=2):
    y = []
    for lo in (96, 100, 103, 106, 108):
        y += list(np.linspace(lo, 110, 14)) + list(np.linspace(110, lo + 2.5, 14))
    if tail:
        y += list(np.linspace(109, 109 + tail * 0.6, tail))
    y = np.array(y)
    return y * (1 + rng.normal(0, 0.002, len(y)))


def _desc_tri(rng, tail=2):
    y = []
    for hi in (104, 100, 97, 94, 92):
        y += list(np.linspace(hi, 90, 14)) + list(np.linspace(90, hi - 2.5, 14))
    if tail:
        y += list(np.linspace(91, 91 - tail * 0.6, tail))
    y = np.array(y)
    return y * (1 + rng.normal(0, 0.002, len(y)))


def _tri_hit(gen, key, seed, n=15):
    from agi_trader.analysis.chart_patterns import detect_triangles, find_pivots
    rng = np.random.default_rng(seed)
    hit = 0
    for _ in range(n):
        df = _tri_ohlc(gen(rng), rng)
        sh, sl = find_pivots(df, 3, 3)
        if any(p.key == key for p in detect_triangles(df, sh, sl)):
            hit += 1
    return hit


@pytest.mark.parametrize("gen,key,ad", [
    (_asc_tri, "ascending_triangle", "yükselen"),
    (_desc_tri, "descending_triangle", "alçalan"),
])
def test_ucgen_tespit_orani(gen, key, ad):
    """Zarf uydurması öncesi: yükselen 6/20, alçalan 4/20. Sonrası: 20/20, 17/20."""
    hit = _tri_hit(gen, key, seed=1001)
    assert hit >= 11, f"{ad} üçgen 15 denemede yalnız {hit} kez bulundu"


def test_ucgen_rastgele_seride_nadir():
    """Düzeltme yanlış pozitifle satın alınmamalı."""
    from agi_trader.analysis.chart_patterns import detect_triangles, find_pivots
    rng = np.random.default_rng(9090)
    hit = 0
    for _ in range(25):
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, 190)))
        df = _tri_ohlc(c, rng)
        sh, sl = find_pivots(df, 3, 3)
        if any("triangle" in p.key for p in detect_triangles(df, sh, sl)):
            hit += 1
    assert hit <= 6, f"rastgele yürüyüşte {hit}/25 üçgen — ayırt edici değil"


def test_ucgen_bayat_apeks_reddedilir():
    """Apeks çok geride kaldıysa üçgen bayattır; ama apeks CIVARINDA olan
    (kırılım anı) reddedilmemeli — eski `top_now <= bot_now` kapısı tam o anda
    körleşiyordu."""
    from agi_trader.analysis.chart_patterns import detect_triangles, find_pivots
    rng = np.random.default_rng(5)
    # apeksten çok sonra: daralma bitmiş, uzun süre yatay sürüklenme
    y = []
    for hi in (104, 100, 97, 94, 92):
        y += list(np.linspace(hi, 90, 14)) + list(np.linspace(90, hi - 2.5, 14))
    y += list(np.linspace(91, 91, 90))          # apeksten 90 bar sonra
    y = np.array(y) * (1 + rng.normal(0, 0.002, len(y)))
    df = _tri_ohlc(y, rng)
    sh, sl = find_pivots(df, 3, 3)
    keys = [p.key for p in detect_triangles(df, sh, sl)]
    assert "descending_triangle" not in keys, "bayat üçgen hâlâ raporlanıyor"


def test_ucgen_stop_dogru_tarafta():
    """Apekste çizgiler ters sıraya geçebilir; stop gerçek uç noktayla
    kelepçelendiği için kurulum yine de geçerli kalmalı."""
    from agi_trader.analysis.chart_patterns import detect_triangles, find_pivots
    for gen, seed in ((_asc_tri, 1001), (_desc_tri, 1001)):
        rng = np.random.default_rng(seed)
        for _ in range(15):
            df = _tri_ohlc(gen(rng), rng)
            price = float(df["close"].iloc[-1])
            sh, sl = find_pivots(df, 3, 3)
            for p in detect_triangles(df, sh, sl):
                if p.direction == "LONG":
                    assert p.stop < p.breakout, f"{p.name}: LONG stop kırılımın üstünde"
                else:
                    assert p.stop > p.breakout, f"{p.name}: SHORT stop kırılımın altında"


# --- kama / kanal / dikdörtgen: değişken derinlikli salınım üreteci ----------
# Saf sinüs kullanmak YANILTIR: orada her salınım sınıra değer, yani
# "dokunmayan pivot" hiç oluşmaz ve zarf uydurmasının etkisi ölçülemez
# (ölçüldü: saf sinüste beş dedektör de 25/25 → tavan etkisi).
_DERINLIK = [1.0, 0.45, 1.0, 0.55, 1.0, 0.40, 1.0, 0.60, 1.0]


def _swings(ust_fn, alt_fn, n_swing, bar):
    """Her iki sınırda da değişken derinlikli salınım — gerçek grafik gibi."""
    y, t = [], 0.0
    for k in range(n_swing):
        u, a = ust_fn(t), alt_fn(t)
        orta = (u + a) / 2.0
        tepe = orta + (u - orta) * _DERINLIK[k % len(_DERINLIK)]
        dip = orta - (orta - a) * _DERINLIK[(k + 3) % len(_DERINLIK)]
        y += list(np.linspace(dip, tepe, bar // 2, endpoint=False))
        t += bar / 2
        u, a = ust_fn(t), alt_fn(t)
        orta = (u + a) / 2.0
        dip2 = orta - (orta - a) * _DERINLIK[(k + 1) % len(_DERINLIK)]
        y += list(np.linspace(tepe, dip2, bar // 2, endpoint=False))
        t += bar / 2
    return y


def _wcr_hit(gen, det_adi, key, seed, n=12):
    from agi_trader.analysis import chart_patterns as CP
    from agi_trader.analysis.chart_patterns import find_pivots
    det = getattr(CP, det_adi)
    rng = np.random.default_rng(seed)
    hit = 0
    for _ in range(n):
        df = _tri_ohlc(gen(rng), rng)
        sh, sl = find_pivots(df, 3, 3)
        if any(p.key == key for p in det(df, sh, sl)):
            hit += 1
    return hit


def _rising_wedge(rng):
    y = _swings(lambda t: 102 + 0.22 * t, lambda t: 88 + 0.36 * t, 8, 13)
    y += list(np.linspace(y[-1], y[-1] * 0.97, 3))
    y = np.array(y)
    return y * (1 + rng.normal(0, 0.0015, len(y)))


def _asc_channel(rng):
    y = np.array(_swings(lambda t: 104 + 0.20 * t, lambda t: 94 + 0.20 * t, 8, 14))
    return y * (1 + rng.normal(0, 0.0018, len(y)))


def _desc_channel(rng):
    y = np.array(_swings(lambda t: 118 - 0.20 * t, lambda t: 108 - 0.20 * t, 8, 14))
    return y * (1 + rng.normal(0, 0.0018, len(y)))


def _rect(rng):
    y = np.array(_swings(lambda t: 106.0, lambda t: 94.0, 8, 14))
    return y * (1 + rng.normal(0, 0.002, len(y)))


@pytest.mark.parametrize("gen,det,key,ad", [
    (_rising_wedge, "detect_wedges", "rising_wedge", "yükselen kama"),
    (_asc_channel, "detect_channel", "ascending_channel", "yükselen kanal"),
    (_desc_channel, "detect_channel", "descending_channel", "alçalan kanal"),
    (_rect, "detect_rectangle", "rectangle", "dikdörtgen"),
])
def test_kama_kanal_dikdortgen_kirlenmeye_dayanikli(gen, det, key, ad):
    """Zarf uydurması öncesi (iki taraflı kirlenmede): yükselen kama 23/25,
    yükselen kanal 18/25. Sonrası: hepsi 24-25/25."""
    hit = _wcr_hit(gen, det, key, seed=3131)
    assert hit >= 9, f"{ad} 12 denemede yalnız {hit} kez bulundu"


def test_dikdortgen_kapsama_sarti_yanlis_pozitifi_tutar():
    """REGRESYON — dikdörtgene `_boundary_fit` KAPSAMA ŞARTI OLMADAN uygulanınca
    rastgele yürüyüşteki yanlış pozitif 0/25'ten 9/25'e fırlamıştı: "dokunanları
    seç" dendiğinde gürültüde bile düz sınır uydurulabiliyor. Dikdörtgenin tanımı
    fiyatın iki sınıra da TEKRAR TEKRAR gitmesidir."""
    from agi_trader.analysis.chart_patterns import detect_rectangle, find_pivots
    rng = np.random.default_rng(8484)
    hit = 0
    for _ in range(25):
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, 200)))
        df = _tri_ohlc(c, rng)
        sh, sl = find_pivots(df, 3, 3)
        if any(p.key == "rectangle" for p in detect_rectangle(df, sh, sl)):
            hit += 1
    assert hit <= 4, f"rastgele yürüyüşte {hit}/25 dikdörtgen — kapsama şartı çalışmıyor"


def test_kama_kanal_rastgelede_nadir():
    from agi_trader.analysis.chart_patterns import detect_wedges, detect_channel, find_pivots
    rng = np.random.default_rng(8484)
    kama = kanal = 0
    for _ in range(25):
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, 200)))
        df = _tri_ohlc(c, rng)
        sh, sl = find_pivots(df, 3, 3)
        if detect_wedges(df, sh, sl):
            kama += 1
        if detect_channel(df, sh, sl):
            kanal += 1
    assert kama <= 12, f"rastgelede {kama}/25 kama"
    assert kanal <= 12, f"rastgelede {kanal}/25 kanal"


# --- fincan-kulp ve bayrak: kırılım sonrası da görünmeli --------------------
def _cup_shape(rng, tail):
    x = np.linspace(-1, 1, 80)
    y = list(110 - 22 * (1 - x ** 2))          # U fincan
    y += list(np.linspace(109, 103, 16))       # kulp iniş
    y += list(np.linspace(103, 108.5, 10))     # kulp dönüş
    if tail:
        y += list(np.linspace(109.5, 109.5 + tail * 0.55, tail))   # kırılım
    y = np.array(y)
    return y * (1 + rng.normal(0, 0.0015, len(y)))


def _flag_shape(rng, tail, pole_bars=14, pole_pct=0.32):
    y = list(80 * np.exp(np.cumsum(rng.normal(0, 0.004, 40))))
    base = y[-1]
    y += list(np.linspace(base, base * (1 + pole_pct), pole_bars))
    top = y[-1]
    xx = np.arange(16)
    y += list(top * (1 - 0.035 * xx / 16) + top * 0.012 * np.sin(2 * np.pi * xx / 8))
    if tail:
        y += list(np.linspace(y[-1], y[-1] * (1 + 0.006 * tail), tail))
    y = np.array(y)
    return y * (1 + rng.normal(0, 0.0015, len(y)))


def _cf_hit(gen, det_adi, keys, tail, seed, n=10, **kw):
    from agi_trader.analysis import chart_patterns as CP
    det = getattr(CP, det_adi)
    rng = np.random.default_rng(seed)
    hit = 0
    for _ in range(n):
        df = _tri_ohlc(gen(rng, tail, **kw), rng)
        if any(p.key in keys for p in det(df)):
            hit += 1
    return hit


@pytest.mark.parametrize("tail", [0, 4, 8, 14])
def test_fincan_kirilimdan_sonra_da_gorunur(tail):
    """REGRESYON — iki kusur birden.

    1) Fincanın SON BARA kadar sürmesi şart koşuluyordu; kırılım gerçekleşince
       kırılım zirvesi `right_rim` oluyor ve ±%6 kenar simetrisi düşüyordu.
    2) Parabol uyumu KULBU DA içine alarak ölçülüyordu; kulp tanımı gereği
       U'dan sapar, bu yüzden gerçek fincanlar bile eşiği geçemiyordu
       (kanonik denetimde 0/20)."""
    hit = _cf_hit(_cup_shape, "detect_cup_handle", ("cup_handle",), tail, seed=5150)
    assert hit >= 7, f"kuyruk={tail}: fincan 10 denemede yalnız {hit} kez bulundu"


@pytest.mark.parametrize("tail", [0, 8, 20])
def test_bayrak_kirilimdan_sonra_da_gorunur(tail):
    """REGRESYON — sıkışma penceresi kırılım barlarını içine alınca `cons_move`
    direğin yönüne dönüyor ve "sıkışma ters yönde olmalı" kapısı reddediyordu."""
    hit = _cf_hit(_flag_shape, "detect_flags", ("bull_flag", "bull_pennant"),
                  tail, seed=5150)
    assert hit >= 7, f"kuyruk={tail}: bayrak 10 denemede yalnız {hit} kez bulundu"


def test_bayrak_tedrici_direkte_de_bulunur():
    """Eski `pole_len` ızgarası (10,15,20) dardı: 30 barda +%22 gibi tedrici
    bir direk hiç yakalanmıyordu."""
    hit = _cf_hit(_flag_shape, "detect_flags", ("bull_flag", "bull_pennant"),
                  2, seed=5150, pole_bars=30, pole_pct=0.22)
    assert hit >= 7, f"tedrici direkli bayrak yalnız {hit}/10"


def test_fincan_bayrak_rastgelede_nadir():
    """Arama uzayı 9'dan ~200 kombinasyona çıkarıldı; kabul eşikleri bununla
    ORANTILI sıkılaştırılmazsa çoklu deneme yanlılığı devreye girer. Ölçüldü:
    eşiksiz hâlde bayrak rastgele yürüyüşün 25/25'inde, fincan 19/25'inde
    ateşliyordu."""
    from agi_trader.analysis.chart_patterns import detect_cup_handle, detect_flags
    rng = np.random.default_rng(8484)
    fincan = bayrak = 0
    for _ in range(25):
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, 200)))
        df = _tri_ohlc(c, rng)
        if detect_cup_handle(df):
            fincan += 1
        if detect_flags(df):
            bayrak += 1
    assert fincan <= 8, f"rastgelede {fincan}/25 fincan"
    assert bayrak <= 8, f"rastgelede {bayrak}/25 bayrak"


def test_direk_anormallik_kapisi_calisir():
    """Direk, varlığın kendi oynaklığına göre anormal olmalı: sabit %3 eşiği tek
    başına yetmez (sakin varlıkta %3 devasa, çalkantılıda sıradan)."""
    from agi_trader.analysis.chart_patterns import detect_flags, POLE_ATR_MULT
    assert POLE_ATR_MULT >= 1.5, "direk anormallik çarpanı gevşetilmiş"
    # çok çalkantılı seri: %3'lük hareketler sıradan → bayrak sayılmamalı
    rng = np.random.default_rng(31)
    cok = 0
    for _ in range(15):
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.030, 160)))
        if detect_flags(_tri_ohlc(c, rng)):
            cok += 1
    assert cok <= 6, f"çalkantılı seride {cok}/15 bayrak — anormallik kapısı zayıf"


# --- çift tepe/dip: ayırt edicilik --------------------------------------------
def _zig(pts, rng, noise=0.002):
    out, cur = [], pts[0][1]
    for ln, tgt in pts[1:]:
        out.append(np.linspace(cur, tgt, int(ln), endpoint=False)); cur = tgt
    y = np.concatenate(out)
    return y * (1 + rng.normal(0, noise, len(y)))


def _dbl_hit(gen, key, seed, n=12):
    from agi_trader.analysis.chart_patterns import detect_hs_and_doubles, find_pivots
    rng = np.random.default_rng(seed)
    hit = 0
    for _ in range(n):
        df = _tri_ohlc(gen(rng), rng)
        sh, sl = find_pivots(df, 3, 3)
        if any(p.key == key for p in detect_hs_and_doubles(df, sh, sl)):
            hit += 1
    return hit


def _ders_dbl_top(rng):
    """Ders kitabı: ÖNCE YÜKSELİŞ, iki eşit tepe, derin boğaz."""
    return _zig([(0, 80), (34, 80), (26, 110), (20, 92), (26, 109), (26, 84)], rng)


def _ders_dbl_bottom(rng):
    return _zig([(0, 120), (34, 120), (26, 90), (20, 108), (26, 91), (26, 116)], rng)


def _tuzak_sig_bogaz(rng):
    """İki eşit tepe ama boğaz yalnız %2 — dönüş değil, yatay gürültü."""
    return _zig([(0, 80), (34, 80), (26, 110), (20, 107.8), (26, 109.5), (26, 106)], rng)


def _tuzak_trendsiz(rng):
    """Derin boğazlı iki eşit tepe ama ÖNCESİNDE yükseliş yok — dönecek trend yok.

    NOT: ilk sürümde bant ±2 birim genişti ve tepe 110'a çıkıyordu; bu aslında
    ~%4'lük bir öncül yükseliş demekti, yani şekil "trendsiz" değildi. Bant
    daraltıldı ve tepe bandın hemen üstüne alındı (öncül yükseliş ~%1,5)."""
    y = list(108 + 0.8 * np.sin(2 * np.pi * np.arange(45) / 13))
    return np.concatenate([np.array(y) * (1 + rng.normal(0, 0.0015, 45)),
                           _zig([(0, 108.3), (6, 108.3), (20, 108.8), (20, 96),
                                 (26, 108.5), (20, 101)], rng)])


def test_cift_tepe_dip_ders_kitabi_bulunur():
    assert _dbl_hit(_ders_dbl_top, "double_top", 2727) >= 10
    assert _dbl_hit(_ders_dbl_bottom, "double_bottom", 2727) >= 10


def test_cift_tepe_sig_bogazi_reddeder():
    """Boğaz derinliği kapısı: %2'lik geri çekilme dönüş formasyonu değildir.
    Kapı öncesi bu tuzak 16/25 geçiyordu, kapıyla 6/25'e indi."""
    assert _dbl_hit(_tuzak_sig_bogaz, "double_top", 2727) <= 5


def test_oncul_trend_kapisi_calisiyor():
    """Öncül trend kapısının MEKANİZMASINI sınar (yayımlanan eşiği değil).

    Yayımlanan kalibrasyon (%2) bilinçli olarak gevşektir: sıkı ayarda
    (%8) gerçek piyasa verisinde çift tepe oranı %41'den %0'a düşüyordu —
    hiç ateşlemeyen dedektör işe yaramaz. Bu test kapının eşiği yükseltilince
    gerçekten devreye girdiğini doğrular."""
    import agi_trader.analysis.chart_patterns as CP
    eski = CP.DBL_MIN_PRIOR_ADV
    try:
        CP.DBL_MIN_PRIOR_ADV = 0.0
        gevsek = _dbl_hit(_tuzak_trendsiz, "double_top", 2727)
        CP.DBL_MIN_PRIOR_ADV = 8.0
        siki = _dbl_hit(_tuzak_trendsiz, "double_top", 2727)
    finally:
        CP.DBL_MIN_PRIOR_ADV = eski
    assert gevsek >= 8, "tuzak şekli zaten bulunamıyor, test anlamsız"
    assert siki <= 2, f"öncül trend kapısı iş görmüyor ({gevsek} → {siki})"


@pytest.mark.parametrize("seed", [8484, 1234])
def test_cift_tepe_dip_rastgelede_azaldi(seed):
    """Kapılar öncesi rastgele yürüyüşte çift tepe 13-15/25, çift dip 8-12/25
    ateşliyordu. Kapılarla belirgin şekilde azaldı — ama SIFIRLANMADI ve
    sıfırlanamaz da: bkz. `test_cift_tepe_dip_olculdu_kanit_yok`."""
    from agi_trader.analysis.chart_patterns import detect_hs_and_doubles, find_pivots
    rng = np.random.default_rng(seed)
    tepe = dip = 0
    for _ in range(25):
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, 200)))
        df = _tri_ohlc(c, rng)
        sh, sl = find_pivots(df, 3, 3)
        keys = {p.key for p in detect_hs_and_doubles(df, sh, sl)}
        tepe += "double_top" in keys
        dip += "double_bottom" in keys
    assert tepe <= 13, f"rastgelede {tepe}/25 çift tepe"
    assert dip <= 13, f"rastgelede {dip}/25 çift dip"


def test_cift_tepe_dip_olculdu_kanit_yok():
    """ASIL BULGU — ayırt edicilik "düzeltilemedi", çünkü ortada bir şey yok.

    Dört ayrı ölçüm aynı yeri gösterdi:
      • frekans: HİÇBİR kapı ayarında gerçek veri oranı, eşleştirilmiş
        oynaklıktaki rastgele yürüyüşü geçmiyor (kapısız %41 vs %40);
      • yön: 20-bar olay çalışmasında çift tepe −%0,34 (kontrol −%0,92) →
        düşüş beklenirken DAHA AZ düştü; çift dip −%1,38 → yükseliş
        beklenirken DAHA ÇOK düştü. İkisi de anlamsız (|t| < 1,3).
    Bu sabit, bulgunun panelde gösterilmesini garanti eder."""
    from agi_trader.analysis.chart_patterns import PATTERN_EVIDENCE
    for k in ("double_top", "double_bottom"):
        ev = PATTERN_EVIDENCE[k]
        assert ev["tested"] is True and ev["edge"] == "yok"
        assert ev["n_events"] >= 100
        assert abs(ev["t"]) < 2.0, "kanıt 'yok' derken t anlamlı görünüyor"


def test_kanitsiz_aile_yon_oyuna_girmez():
    """Ölçülüp ÇÜRÜTÜLMÜŞ bir aile, ölçülmemiş bir aileyle aynı ağırlığı almaz:
    çizilmeye devam eder ama işlem önerisinin yönünü belirlemez."""
    from agi_trader.analysis.chart_patterns import trade_recommendation
    pats = [
        {"key": "double_top", "name": "Çift Tepe", "direction": "SHORT",
         "score": 9.0, "target_pct": -9.0, "stop_pct": 3.0, "rr": 3.0,
         "completion": 0.9, "quality": 0.9, "status": "kırılım", "valid": True},
        {"key": "ascending_triangle", "name": "Yükselen Üçgen", "direction": "LONG",
         "score": 1.0, "target_pct": 5.0, "stop_pct": -2.0, "rr": 2.5,
         "completion": 0.8, "quality": 0.8, "status": "oluşuyor", "valid": True},
    ]
    rec = trade_recommendation(pats, price=100.0, atr_pct=1.0)
    assert rec["available"] and rec["direction"] == "LONG",         "kanıtsız aile yönü belirlemiş"
    assert "Çift Tepe" in rec["excluded_no_edge"]
    # yalnız kanıtsız formasyon varsa öneri ÜRETİLMEZ ve sebebi yazılır
    tek = trade_recommendation([pats[0]], price=100.0, atr_pct=1.0)
    assert tek["available"] is False
    assert "Çift Tepe" in tek["excluded_no_edge"]


def test_boyun_en_dusuk_dip_olmali():
    """Boyun, tepeler arasındaki EN DÜŞÜK dip olmalı; eskiden İLK dip alınıyordu,
    birden fazla dip varsa boyun yanlış yere konup derinlik küçük çıkıyordu."""
    from agi_trader.analysis.chart_patterns import detect_hs_and_doubles, find_pivots
    rng = np.random.default_rng(3)
    # tepeler arasında İKİ dip: önce sığ (104), sonra DERİN (92)
    y = _zig([(0, 80), (34, 80), (24, 110), (10, 104), (8, 107), (12, 92),
              (24, 109), (26, 86)], rng)
    df = _tri_ohlc(y, rng)
    sh, sl = find_pivots(df, 3, 3)
    ps = [p for p in detect_hs_and_doubles(df, sh, sl) if p.key == "double_top"]
    if ps:                              # şekil bulunduysa boyun derin dipte olmalı
        assert ps[0].breakout < 100, \
            f"boyun sığ dibe kondu ({ps[0].breakout:.1f}) — en düşük dip alınmıyor"


def test_obo_ayirt_ediciligi_bozulmadi():
    """Kapılar yalnız çift tepe/dibi etkilemeli; OBO aynı fonksiyonda."""
    def hs(rng):
        return _zig([(0, 85), (22, 85), (18, 100), (16, 90), (18, 112), (16, 89),
                     (18, 101), (24, 80)], rng)
    assert _dbl_hit(hs, "head_shoulders", 2727) >= 10


def test_isleme_onerisi_yalniz_gecerlilerden_hesaplar():
    from agi_trader.analysis.chart_patterns import trade_recommendation
    pats = [
        {"name": "A", "direction": "LONG", "score": 1.0, "target_pct": 5.0,
         "stop_pct": -2.0, "rr": 2.5, "completion": 0.8, "quality": 0.8,
         "status": "oluşuyor", "valid": True},
        {"name": "B", "direction": "SHORT", "score": 9.0, "target_pct": -9.0,
         "stop_pct": 3.0, "rr": 3.0, "completion": 0.9, "quality": 0.9,
         "status": "kırılım", "valid": False},        # GEÇERSİZ, ağır skorlu
    ]
    rec = trade_recommendation(pats, price=100.0, atr_pct=1.0)
    assert rec["available"] and rec["direction"] == "LONG", \
        "geçersiz formasyon yönü belirlemiş"
    assert rec["n_invalid"] == 1


def test_oneri_hepsi_gecersizse_uretilmez():
    from agi_trader.analysis.chart_patterns import trade_recommendation
    pats = [{"name": "A", "direction": "LONG", "score": 1.0, "target_pct": 5.0,
             "stop_pct": -2.0, "rr": 2.5, "completion": 0.8, "quality": 0.8,
             "status": "oluşuyor", "valid": False}]
    assert trade_recommendation(pats, 100.0, 1.0)["available"] is False


def test_basabas_kazanma_orani_dogru():
    """R/R 2 ise başabaş kazanma oranı %33,3 olmalı (1/(1+R))."""
    from agi_trader.analysis.chart_patterns import trade_recommendation
    pats = [{"name": "A", "direction": "LONG", "score": 1.0, "target_pct": 4.0,
             "stop_pct": -2.0, "rr": 2.0, "completion": 0.9, "quality": 0.9,
             "status": "kırılım", "valid": True}]
    rec = trade_recommendation(pats, 100.0, 1.0)
    assert abs(rec["breakeven_winrate"] - 33.3) < 0.5


# ===========================================================================
# FAZ 5b — harmonikler
# ===========================================================================
def _harm_ohlc(close, rng, noise=0.0025):
    close = np.asarray(close, float); n = len(close)
    op = np.concatenate([[close[0]], close[:-1]]); w = close * noise
    return pd.DataFrame({"open": op,
        "high": np.maximum(op, close) + np.abs(rng.normal(0, 1, n)) * w,
        "low": np.minimum(op, close) - np.abs(rng.normal(0, 1, n)) * w,
        "close": close, "volume": rng.uniform(800, 1200, n)},
        index=pd.date_range("2024-01-01", periods=n, freq="4h"))


def _xabcd_seri(name, rng, bullish=False, bar=13):
    """Kuralın TAM ORTASINDAN XABCD üret.

    Ön tarih DÜZ GÜRÜLTÜ değil, X'e giden temiz bir eğim olmalı: düz gürültüde
    sahte pivot çıkıp ARDIŞIK İKİ TEPE oluşuyor, dedektör de (haklı olarak)
    almaşık pivot istediği için formasyonu eliyor. İlk denetim koşumumda boğa
    yönü 10/15 çıkmasının sebebi buydu — kodun değil, tezgâhın kusuru."""
    from agi_trader.analysis.patterns import HARMONIC_RULES
    rule = HARMONIC_RULES[name]
    mid = lambda r: (float(r[0]) + float(r[1])) / 2 if isinstance(r, (tuple, list)) else float(r)
    X = 100.0
    A = X + 30.0
    B = A - mid(rule["AB_XA"]) * (A - X)
    C = B + mid(rule["BC_AB"]) * (A - B)
    D = A - mid(rule["AD_XA"]) * (A - X)
    pts = [X, A, B, C, D]
    if not bullish:
        pts = [200.0 - p for p in pts]
    y = list(np.linspace(pts[0] * (0.97 if bullish else 1.03), pts[0], 40))
    cur = pts[0]
    for t in pts[1:]:
        y += list(np.linspace(cur, t, bar, endpoint=False)); cur = t
    y += list(np.linspace(cur, cur * (1.02 if bullish else 0.98), 6))
    y = np.array(y, dtype=float)
    return y * (1 + rng.normal(0, 0.0012, len(y)))


@pytest.mark.parametrize("name,key", [
    ("Gartley", "gartley"), ("Butterfly", "butterfly"), ("Bat", "bat"),
    ("Crab", "crab"), ("Cypher", "cypher"), ("Shark", "shark"),
])
def test_harmonik_ders_kitabi_bulunur(name, key):
    """Kuralın tam ortasından çizilen XABCD tanınmalı — her iki yönde."""
    from agi_trader.analysis.harmonics import detect_harmonics_rich
    for bullish in (True, False):
        rng = np.random.default_rng(4242)
        hit = 0
        for _ in range(8):
            df = _harm_ohlc(_xabcd_seri(name, rng, bullish), rng)
            if any(p["key"] == key for p in detect_harmonics_rich(df).get("patterns", [])):
                hit += 1
        assert hit >= 5, f"{name} ({'boğa' if bullish else 'ayı'}) 8 denemede {hit}"


def test_bir_pencere_tek_harmonik():
    """REGRESYON — tekilleştirme (anahtar, durum) üzerindendi, dolayısıyla AYNI
    beş nokta gartley+crab+cypher+shark diye DÖRT formasyon olarak
    raporlanabiliyordu. Bir XABCD penceresi ancak TEK bir harmonik olabilir."""
    from agi_trader.analysis.harmonics import detect_harmonics_rich
    rng = np.random.default_rng(4242)
    for _ in range(8):
        df = _harm_ohlc(_xabcd_seri("Gartley", rng, False), rng)
        ps = [p for p in detect_harmonics_rich(df).get("patterns", [])
              if p["status"] == "tamamlandı"]
        imza = {}
        for p in ps:
            k = tuple(sorted((pt["i"], round(pt["price"], 4)) for pt in p["points"]))
            imza.setdefault(k, []).append(p["key"])
        for k, keys in imza.items():
            assert len(keys) == 1, f"aynı beş nokta {len(keys)} formasyon: {keys}"


def test_harmonik_gecerlilik_kapisi():
    """REGRESYON — grafik formasyonlarındaki hatanın aynısı harmoniklerde de
    vardı: denetimde 69 kurulumun 10'unda fiyat STOP'u ihlal etmiş, 29'unda ilk
    hedefi geçmişti; hepsi hâlâ 'işlem planı' diye sunuluyordu."""
    from agi_trader.analysis.harmonics import _apply_h_validity, HarmonicPattern
    p = HarmonicPattern(key="bat", name="Bat", direction="LONG", status="tamamlandı",
                        quality=0.9, completion=1.0, score=0.8, color="#fff",
                        entry=100.0, stop=95.0, targets=[105.0, 110.0, 120.0])
    _apply_h_validity(p, price=94.0)
    assert p.valid is False and p.validity == "stop_ihlal"
    p2 = HarmonicPattern(key="bat", name="Bat", direction="LONG", status="tamamlandı",
                         quality=0.9, completion=1.0, score=0.8, color="#fff",
                         entry=100.0, stop=95.0, targets=[105.0, 110.0, 120.0])
    _apply_h_validity(p2, price=125.0)
    assert p2.valid is False and p2.validity == "hedef_aşıldı"
    p3 = HarmonicPattern(key="bat", name="Bat", direction="SHORT", status="tamamlandı",
                         quality=0.9, completion=1.0, score=0.8, color="#fff",
                         entry=100.0, stop=105.0, targets=[95.0, 90.0, 80.0])
    _apply_h_validity(p3, price=106.0)
    assert p3.valid is False and p3.validity == "stop_ihlal"


def test_harmonik_rr_sabit_degil():
    """REGRESYON — ödül 0,382×AD, risk 0,15×AD olduğu için R/R her formasyonda
    TAM 2,55 çıkıyordu (69 kurulumun 69'unda). Böyle bir sayı formasyon hakkında
    hiçbir şey söylemez. Stop tabanı ATR olunca R/R piyasaya göre değişir."""
    from agi_trader.analysis.harmonics import _trade_plan
    # AD = 30 → geometrik taban 0,15×30 = 4,5. ATR ancak bunu AŞARSA devreye
    # girer, o yüzden test değerleri tabanı straddle etmeli.
    rr = set()
    for atr in (0.0, 5.0, 8.0, 15.0):
        rr.add(_trade_plan(100.0, 130.0, "LONG", 100.0, atr)["rr"])
    assert len(rr) >= 3, f"R/R ATR'ye göre değişmiyor: {rr}"
    # hangi tabanın kullanıldığı raporlanmalı
    assert _trade_plan(100.0, 130.0, "LONG", 100.0, 15.0)["stop_source"] == "ATR"
    assert _trade_plan(100.0, 130.0, "LONG", 100.0, 0.0)["stop_source"] == "AD%15"
    # ATR sıfırken eski davranış korunur (geometrik plan)
    assert abs(_trade_plan(100.0, 130.0, "LONG", 100.0, 0.0)["rr"] - 2.55) < 0.02


def test_oluşuyor_kalite_esigi_gosterilene_uygulanir():
    """REGRESYON — kapı ceza ÖNCESİ kaliteye bakıyordu; iki D projeksiyonu
    örtüşmediğinde ×0,7 cezası devreye girip panelde eşiğin ALTINDA kalite
    görünüyordu (canlıda 0,47 · 0,55 — başlıkta 'uyum eşiği %65' yazarken)."""
    from agi_trader.analysis.harmonics import (detect_harmonics_rich,
                                               MIN_QUALITY_FORMING)
    rng = np.random.default_rng(31)
    bakilan = 0
    for _ in range(25):
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.016, 240)))
        df = _harm_ohlc(c, rng)
        for p in detect_harmonics_rich(df).get("patterns", []):
            bakilan += 1
            assert p["quality"] >= MIN_QUALITY_FORMING - 1e-9, (
                f"{p['name']} kalite {p['quality']} < eşik {MIN_QUALITY_FORMING}")
    assert bakilan > 0, "hiç formasyon üretilmedi, test anlamsız"


def test_harmonik_kanit_notu():
    """Harmonikler ŞEKİL sınavını geçti, YÖN sınavını geçemedi; ikisi de panelde."""
    from agi_trader.analysis.harmonics import HARMONIC_EVIDENCE as E
    assert E["tested"] is True
    assert E["shape_real"] is True and E["edge"] == "yok"
    assert E["frequency"]["real_pct"] > E["frequency"]["random_pct"],         "şekil sınavı geçilmemişse 'gerçek' denemez"
    assert abs(E["direction"]["long"]["t"]) < 2 and abs(E["direction"]["short"]["t"]) < 2,         "kanıt 'yok' derken t anlamlı görünüyor"


# --- mum formasyonları ---------------------------------------------------
def _mum_df(bars, pre=0, atr=1.0, n_pre=20):
    """Öncül trend + formasyon mumları. En az 20 ön bar: detect_candles
    len(df) >= 16 istiyor ve ATR'nin oturması gerekiyor."""
    o, h, l, c = [], [], [], []
    base = 100.0
    for k in range(n_pre):
        px = base + pre * 0.35 * atr * k
        o.append(px); c.append(px + pre * 0.30 * atr)
        h.append(max(o[-1], c[-1]) + 0.1 * atr)
        l.append(min(o[-1], c[-1]) - 0.1 * atr)
    base = c[-1]
    for (bo, bh, bl, bc) in bars:
        o.append(base + bo * atr); h.append(base + bh * atr)
        l.append(base + bl * atr); c.append(base + bc * atr)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c,
                         "volume": np.full(len(o), 1000.0)},
                        index=pd.date_range("2024-01-01", periods=len(o), freq="4h"))


_MUM = {
    "hammer": (-1, [(0.0, 0.1, -1.6, 0.05)]),
    "shooting_star": (+1, [(0.0, 1.6, -0.1, -0.05)]),
    "marubozu": (0, [(0.0, 1.0, 0.0, 1.0)]),
    "bullish_engulfing": (-1, [(0.0, 0.05, -0.5, -0.45), (-0.5, 0.5, -0.55, 0.4)]),
    "bearish_engulfing": (+1, [(0.0, 0.5, -0.05, 0.45), (0.5, 0.55, -0.5, -0.4)]),
    "morning_star": (-1, [(0.0, 0.05, -1.0, -0.95), (-1.1, -1.0, -1.25, -1.05),
                          (-1.0, -0.2, -1.05, -0.25)]),
    "evening_star": (+1, [(0.0, 1.0, -0.05, 0.95), (1.1, 1.25, 1.0, 1.05),
                          (1.0, 1.05, 0.2, 0.25)]),
    "three_white_soldiers": (0, [(0.0, 0.5, -0.05, 0.45), (0.15, 0.95, 0.10, 0.90),
                                 (0.60, 1.40, 0.55, 1.35)]),
}


@pytest.mark.parametrize("key", sorted(_MUM))
def test_mum_formasyonu_tanınır(key):
    from agi_trader.analysis.candles import detect_candles
    pre, bars = _MUM[key]
    ps = detect_candles(_mum_df(bars, pre), lookback=1)
    assert key in {p["key"] for p in ps}, f"{key} tanınmadı: {[p['key'] for p in ps]}"


def test_mum_baglam_penceresi_dongusel_degil():
    """REGRESYON — EN ÖNEMLİSİ. `_trend` penceresi formasyonun KENDİ mumlarını
    içeriyordu: "düşüş sonrası boğa yutan" derken kastedilen düşüş, yutan
    formasyonun kendi ilk kırmızı mumuydu. Döngüsel bir bağlam şartı, şart
    değildir. Ölçüldü: bu hatayla trendsiz ortamda 16 dönüş formasyonunun
    12'si yine de raporlanıyordu; düzeltmeden sonra 0'ı."""
    from agi_trader.analysis.candles import detect_candles
    kacan = []
    for key, (pre, bars) in _MUM.items():
        if pre == 0:
            continue                      # devam formasyonu, bağlam istemiyor
        ps = detect_candles(_mum_df(bars, pre=0), lookback=1)   # TRENDSİZ
        if key in {p["key"] for p in ps}:
            kacan.append(key)
    assert not kacan, f"trendsiz ortamda raporlanan dönüş formasyonları: {kacan}"


def test_mum_kararsizlik_ailesi_yon_vermez():
    """Doji ve topaç KARARSIZLIKTIR; LONG/SHORT'a zorlamak uydurma bilgidir."""
    from agi_trader.analysis.candles import detect_candles
    for bars in ([(0.0, 0.05, -1.2, 0.01)], [(0.0, 1.2, -0.05, 0.01)],
                 [(0.0, 0.8, -0.8, 0.01)]):
        for p in detect_candles(_mum_df(bars, pre=0), lookback=1):
            if p["family"] == "kararsızlık":
                assert p["direction"] == "NÖTR", f"{p['key']} yön veriyor"


def test_mum_kanit_notu():
    """9 formasyon ölçüldü; anlamlı çıkan İKİSİ de TERS yönde. Panel bunu yazar."""
    from agi_trader.analysis.candles import CANDLE_MEASURED, CANDLE_EVIDENCE
    m = CANDLE_MEASURED
    assert m["n_measurable"] >= 5
    assert m["significant_wrong_way"] == m["significant"],         "anlamlı ama DOĞRU yönde bir formasyon çıkmışsa metin güncellenmeli"
    for k, e in CANDLE_EVIDENCE.items():
        assert e["edge"] in ("yok", "ters")
        if e["edge"] == "ters":
            assert abs(e["t"]) >= 2, f"{k} 'ters' deniyor ama t anlamsız"


def test_mum_atr_olcekli():
    """Eşikler ATR cinsinden olmalı: aynı şekil, 100× fiyat ölçeğinde de bulunmalı."""
    from agi_trader.analysis.candles import detect_candles
    a = detect_candles(_mum_df(_MUM["hammer"][1], pre=-1, atr=1.0), lookback=1)
    b = detect_candles(_mum_df(_MUM["hammer"][1], pre=-1, atr=100.0), lookback=1)
    assert {p["key"] for p in a} == {p["key"] for p in b}, "ölçek değişince sonuç değişti"


def test_harmonik_ciktisi_tutarli():
    from agi_trader.analysis.harmonics import detect_harmonics_rich
    for seed in range(6):
        out = detect_harmonics_rich(_ohlcv(seed=seed))
        for p in out.get("patterns", []):
            assert p["direction"] in ("LONG", "SHORT")
            assert len(p["points"]) >= 4, "XABCD en az 4 nokta ister"
            if p.get("stop") and p.get("entry"):
                if p["direction"] == "LONG":
                    assert p["stop"] < p["entry"], "LONG'da stop girişin altında olmalı"
                else:
                    assert p["stop"] > p["entry"], "SHORT'ta stop girişin üstünde olmalı"


# ===========================================================================
# FAZ 10 — evren
# ===========================================================================
def test_evren_skorlamasi_calisir():
    from agi_trader.research.universe import score_asset, trend_returns
    px = _prices(n=800, k=1, seed=5)["A0"]
    r = trend_returns(px)
    assert len(r) > 0 and np.isfinite(r.to_numpy()).all()
    sc = score_asset("A0", px)
    assert hasattr(sc, "to_dict")
    assert np.isfinite(sc.to_dict().get("sharpe", 0.0))


# ===========================================================================
# GÖSTERGE TABLOSU (yeni)
# ===========================================================================
def test_mikroyapi_yoksa_atlanir_uydurulmaz():
    """Mikroyapı göstergeleri fiyattan TÜRETİLEMEZ. Kaydedici verisi yoksa
    bölüm atlanmalı — sıfır/varsayılan değerle doldurulmamalı."""
    from agi_trader.analysis.indicator_board import build_board
    b = build_board(_ohlcv(n=420, seed=5), micro=None)
    aileler = {i["family"] for i in b["indicators"]}
    assert b["microstructure"] is False, "mikroyapı yokluğu beyan edilmiyor"
    assert "ATLANDI" in b["microstructure_note"]
    for yasak in ("funding", "book_slope", "kyle", "depth_multi"):
        assert yasak not in aileler, f"{yasak} veri olmadan üretilmiş"


def _ornek_micro():
    return pd.Series({
        "funding_rate": 0.00012, "open_interest": 5_000.0,
        "mark_price": 100.5, "index_price": 100.0, "taker_buy_ratio": 0.53,
        "ls_account_ratio": 1.1, "top_trader_ratio": 1.05, "spread_bps": 1.2,
        "bid_cum_1bps": 100_000.0, "ask_cum_1bps": 90_000.0,
        "bid_cum_5bps": 400_000.0, "ask_cum_5bps": 380_000.0,
        "bid_cum_10bps": 800_000.0, "ask_cum_10bps": 750_000.0,
        "bid_cum_20bps": 1_500_000.0, "ask_cum_20bps": 1_400_000.0,
        "bid_truncated": False, "ask_truncated": False})


def test_mikroyapi_varsa_eklenir():
    from agi_trader.analysis.indicator_board import build_board
    b = build_board(_ohlcv(n=420, seed=5), micro=_ornek_micro())
    aileler = {i["family"] for i in b["indicators"]}
    for beklenen in ("funding", "book_imb_multi", "book_slope", "kyle"):
        assert beklenen in aileler, f"{beklenen} eklenmemiş"
    assert b["microstructure"] is True
    # Kendi KATEGORİSİNDE olmalı: "Hacim"in içine gömülünce panelde ayırt
    # edilemiyor ve kullanıcı hangi göstergenin gerçekten yeni bilgi taşıdığını
    # göremiyordu (20 gösterge 19'u volume, 1'i volatility altındaydı).
    assert "microstructure" in b["by_category"], "mikroyapı kendi kategorisinde değil"
    assert b["by_category"]["microstructure"]["n"] >= 12
    kats = {i["category"] for i in b["indicators"] if i["family"] in
            ("funding", "book_slope", "kyle", "depth_multi", "positioning")}
    assert kats == {"microstructure"}, f"mikroyapı başka kategoriye sızmış: {kats}"


def test_mikroyapi_kaniti_maliyet_sinavini_tasir():
    """Mikroyapı AYRICA ölçüldü ve fiyat türevlerinden FARKLI çıktı: ölçülebilir
    yön bilgisi VAR ama maliyetin altında. Panel iki sayıyı da göstermeli —
    yalnız brüt getiriyi göstermek okuyucuyu yanıltır."""
    from agi_trader.analysis.indicator_board import MICRO_EVIDENCE as E
    assert E["measured"] is True and E["signals"]
    for s in E["signals"]:
        # brüt POZİTİF ama net NEGATİF — iddianın tamamı bu
        assert s["gross_pct"] > 0, f"{s['name']} brüt pozitif değil"
        assert s["net_pct"] < 0, f"{s['name']} net negatif değil — karar metni yanlış"
        assert s["cost_multiple"] > 1.0
        # aritmetik tutmalı
        assert abs((s["gross_pct"] - s["cost_pct"]) - s["net_pct"]) < 0.002
    assert "MALİYETİ KARŞILAMIYOR" in E["verdict"]
    assert E["noise"], "gürültü çıkan göstergeler listelenmemiş"
    assert "TEK REJİM" in E["power_warning"], "güç sınırı beyan edilmemiş"


def test_mikroyapi_kaniti_yalniz_veri_varken_verilir():
    """Kaydedici verisi yoksa kanıt da verilmemeli."""
    from agi_trader.analysis.indicator_board import build_board
    b = build_board(_ohlcv(n=420, seed=5), micro=None)
    assert b["micro_evidence"] is None
    b2 = build_board(_ohlcv(n=420, seed=5), micro=_ornek_micro())
    assert b2["micro_evidence"] is not None


def test_gosterge_tablosu_300_ve_nan_yok():
    from agi_trader.analysis.indicator_board import build_board
    b = build_board(_ohlcv(n=420, seed=7))
    assert b["available"]
    assert b["total"] >= 350, f"yalnız {b['total']} gösterge"
    nan = [i["name"] for i in b["indicators"] if i["value"] != i["value"]]
    assert not nan, f"NaN değerli gösterge: {nan[:5]}"
    # genişletme paketi yüklenememişse tablo bunu SESSİZCE geçmemeli
    assert not any(i["family"] == "pack_error" for i in b["indicators"]),         "indicator_pack yüklenemedi"


def test_ayni_gostergenin_periyotlari_tek_AILE():
    """Aile indirgemesinin tüm amacı budur: RSI'ın altı periyodu altı bağımsız
    kanıt DEĞİLDİR. İlk yazımda aile adlarına periyot eklenmiş ve bağımsız oy
    sayısı 144 yerine 205 görünmüştü — uyarı işlevsizleşiyordu."""
    from agi_trader.analysis.indicator_board import build_board
    b = build_board(_ohlcv(n=420, seed=11))
    aileler = {}
    for i in b["indicators"]:
        aileler.setdefault(i["family"], []).append(i["name"])
    rsi_aile = {f for f, ns in aileler.items() if any(n.startswith("RSI ") for n in ns)}
    assert len(rsi_aile) <= 2, f"RSI {len(rsi_aile)} ayrı aileye bölünmüş: {rsi_aile}"
    assert b["family"]["total"] < b["total"] * 0.75,         "aile indirgemesi neredeyse hiç indirgemiyor"


def test_gosterge_sayimlari_tutarli():
    from agi_trader.analysis.indicator_board import build_board
    b = build_board(_ohlcv(n=420, seed=8))
    r = b["raw"]
    assert r["al"] + r["sat"] + r["notr"] == b["total"]
    f = b["family"]
    assert f["al"] + f["sat"] + f["notr"] == f["total"]
    assert f["total"] < b["total"], "aile indirgemesi çalışmıyor"
    for v in b["by_category"].values():
        assert v["al"] + v["sat"] + v["notr"] == v["n"]


def test_gosterge_tablosu_kisa_veride_reddeder():
    from agi_trader.analysis.indicator_board import build_board
    assert build_board(_ohlcv(n=50))["available"] is False


def test_gosterge_kanit_notu_var():
    """Panel ölçüm sonucunu göstermeli — süsleme değil, kanıtlı olmalı."""
    from agi_trader.analysis.indicator_board import BOARD_EVIDENCE
    assert BOARD_EVIDENCE["measured"] is True
    assert BOARD_EVIDENCE["rows"], "ölçüm satırı yok"
    for r in BOARD_EVIDENCE["rows"]:
        assert r["corr_fwd1"] < 0, "kanıt negatif korelasyonu göstermeli"
    assert "DEĞİL" in BOARD_EVIDENCE["verdict"]
    # ters-konsensüs DSR kapısında elenmiş olmalı — geçtiği iddia edilirse
    # sinyal olarak kullanılmama gerekçesi çürür
    dsr = BOARD_EVIDENCE["dsr"]
    assert dsr["4h"] < dsr["threshold"] and dsr["1d"] < dsr["threshold"], \
        "ters-konsensüs DSR kapısını geçiyor görünüyor — kanıt metni güncellenmeli"
    # Kanıt, YAYIMLANAN gösterge sayısıyla ölçülmüş olmalı. Tablo büyütülüp
    # ölçüm yenilenmezse panel eski bir koşumun sonucunu gösterir.
    # Kanıt MİKROYAPILI yapılandırmada ölçüldü; karşılaştırma da öyle olmalı.
    from agi_trader.analysis.indicator_board import build_board
    n = build_board(_ohlcv(n=420, seed=7), micro=_ornek_micro())["total"]
    assert abs(BOARD_EVIDENCE["n_indicators"] - n) <= 8, (
        f"kanıt {BOARD_EVIDENCE['n_indicators']} göstergeyle ölçülmüş ama tablo "
        f"{n} gösterge üretiyor — ölçüm yeniden yapılmalı")


def test_gosterge_sayisi_kaliteyi_artirmadi_kaniti():
    """129 → 300 karşılaştırması panelde durmalı: gösterge sayısını 2,3 katına
    çıkarmak korelasyonu, takip getirisini ve kazanma oranını DEĞİŞTİRMEDİ.
    "Daha çok gösterge = daha iyi karar" sezgisinin ölçülmüş reddi."""
    from agi_trader.analysis.indicator_board import BOARD_EVIDENCE
    cc = BOARD_EVIDENCE["count_comparison"]
    assert cc["after"]["n"] > cc["before"]["n"]
    assert cc["before"]["corr"] < 0 and cc["after"]["corr"] < 0
    assert cc["before"]["follow_ret_pct"] < 0 and cc["after"]["follow_ret_pct"] < 0
    assert abs(cc["after"]["winrate"] - cc["before"]["winrate"]) < 5.0


def test_gosterge_tablosu_ileriye_bakmaz():
    """Son barı değiştirmek ÖNCEKİ barın tablosunu değiştirmemeli."""
    from agi_trader.analysis.indicator_board import build_board
    df = _ohlcv(n=420, seed=9)
    a = build_board(df.iloc[:-1])
    df2 = df.copy()
    df2.iloc[-1, df2.columns.get_loc("close")] *= 1.5
    b = build_board(df2.iloc[:-1])
    assert a["raw"] == b["raw"] and a["net"] == b["net"]
