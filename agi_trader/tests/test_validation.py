# -*- coding: utf-8 -*-
"""agi_trader.research.validation birim testleri.

Doğrulama aracının kendisi doğrulanmadan hiçbir stratejiye kapı olamaz.
Çalıştır:  python -m pytest tests/ -q     (veya)  python tests/test_validation.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agi_trader.research.validation import (  # noqa: E402
    norm_cdf, norm_ppf, sharpe, psr, expected_max_sharpe, deflated_sharpe,
    min_backtest_length, purged_kfold_splits, combinatorial_purged_splits,
    pbo, trial_log, trial_count, acceptance_gate, shuffle_test,
)


def make_returns(n: int, annual_sharpe: float, vol_daily: float = 0.01,
                 seed: int = 0, ppy: float = 365.0) -> np.ndarray:
    """Gerçekleşen yıllık Sharpe'ı TAM olarak `annual_sharpe` olan günlük getiri
    serisi üretir. Testlerin rastgeleliğe değil, kastedilen büyüklüğe bakması için.

    NOT: günlük ortalama 0.004 / std 0.01 gibi değerler yıllık Sharpe 7-8 demektir —
    gerçek dünyada görülmez ve "önce bug varsayılır" kapısına takılır."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    x = (x - x.mean()) / x.std(ddof=1)
    return x * vol_daily + (annual_sharpe / math.sqrt(ppy)) * vol_daily


# --------------------------------------------------------------- normal dağılım
def test_norm_ppf_bilinen_degerler():
    assert abs(norm_ppf(0.5)) < 1e-9
    assert abs(norm_ppf(0.975) - 1.959964) < 1e-5
    assert abs(norm_ppf(0.95) - 1.644854) < 1e-5
    assert abs(norm_ppf(0.005) + 2.575829) < 1e-5


def test_norm_ppf_cdf_tersi():
    for p in (0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
        assert abs(norm_cdf(norm_ppf(p)) - p) < 1e-9


# ------------------------------------------------------------------- Sharpe/PSR
def test_sharpe_yilliklandirma():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 2000)
    s_daily = sharpe(r)
    s_ann = sharpe(r, 365)
    assert abs(s_ann - s_daily * math.sqrt(365)) < 1e-9


def test_sharpe_sifir_varyans():
    assert sharpe([0.01] * 50) == 0.0
    assert sharpe([0.01]) == 0.0


def test_psr_guclu_edge_yuksek():
    rng = np.random.default_rng(1)
    r = rng.normal(0.002, 0.01, 1000)          # günlük Sharpe ~0.2, n büyük
    assert psr(r) > 0.99


def test_psr_edge_yoksa_dusuk():
    rng = np.random.default_rng(2)
    r = rng.normal(0.0, 0.01, 300)
    assert psr(r) < 0.9


def test_psr_ornek_sayisi_arttikca_artar():
    rng = np.random.default_rng(3)
    small = rng.normal(0.001, 0.01, 40)
    big = np.concatenate([small] * 25)          # aynı dağılım, 25× örnek
    assert psr(big) > psr(small)


# ------------------------------------------------------------------------ DSR
def test_expected_max_sharpe_deneme_ile_artar():
    a = expected_max_sharpe(10, 1.0)
    b = expected_max_sharpe(1000, 1.0)
    c = expected_max_sharpe(100000, 1.0)
    assert a < b < c
    assert a > 0


def test_dsr_deneme_arttikca_duser():
    """Aynı getiri serisi, daha çok deneme → DSR düşmeli.
    DSR'ın var oluş nedeni tam olarak budur."""
    rng = np.random.default_rng(4)
    r = rng.normal(0.0015, 0.01, 1000)
    az = deflated_sharpe(r, n_trials=2, sr_std=0.5)["dsr"]
    cok = deflated_sharpe(r, n_trials=5000, sr_std=0.5)["dsr"]
    assert az > cok


def test_dsr_gercek_guclu_edge_gecer():
    r = make_returns(1500, annual_sharpe=1.9, seed=5)   # gerçekçi güçlü strateji
    d = deflated_sharpe(r, n_trials=50, sr_std=0.3)
    assert d["dsr"] >= 0.95 and d["verdict"] == "GEÇTİ", d


def test_dsr_gurultu_kalir():
    r = make_returns(400, annual_sharpe=0.38, seed=6)   # zayıf, çok deneme
    d = deflated_sharpe(r, n_trials=2000, sr_std=1.0)
    assert d["dsr"] < 0.95 and d["verdict"] == "KALDI", d


def test_dsr_birim_uyumu_regresyon():
    """REGRESYON: sr0 yıllık, gözlenen Sharpe dönem-başıydı → DSR hep 0 çıkıyordu.
    Aynı ekonomik strateji, farklı periyotta raporlansa da benzer DSR vermeli."""
    gunluk = make_returns(1500, annual_sharpe=1.9, seed=7, ppy=365)
    d_gun = deflated_sharpe(gunluk, n_trials=50, sr_std=0.3, periods_per_year=365)
    # aynı yıllık Sharpe, saatlik raporlanmış (aynı gözlem sayısı)
    saatlik = make_returns(1500, annual_sharpe=1.9, seed=7, ppy=8760)
    d_saat = deflated_sharpe(saatlik, n_trials=50, sr_std=0.3, periods_per_year=8760)
    assert abs(d_gun["sr_annual"] - 1.9) < 0.02
    assert abs(d_saat["sr_annual"] - 1.9) < 0.02
    assert d_gun["dsr"] > 0.5 and d_saat["dsr"] > 0.5


def test_min_backtest_length():
    """Daha çok deneme veya daha düşük Sharpe → daha uzun veri gerekir."""
    az = min_backtest_length(1.0, n_trials=10)
    cok = min_backtest_length(1.0, n_trials=10000)
    assert cok > az
    assert min_backtest_length(2.0, 100) < min_backtest_length(1.0, 100)
    assert min_backtest_length(0.0, 100) == math.inf


# -------------------------------------------------------------- purged CV
def test_purged_kfold_train_test_kesismez():
    n, horizon = 300, 10
    t1 = np.arange(n) + horizon
    for tr, te in purged_kfold_splits(t1, n_splits=5, embargo_pct=0.01):
        assert len(np.intersect1d(tr, te)) == 0


def test_purged_kfold_ortusmeyi_temizler():
    """Test aralığına SARKAN etikete sahip eğitim örneği kalmamalı."""
    n, horizon = 200, 15
    t1 = np.arange(n) + horizon
    for tr, te in purged_kfold_splits(t1, n_splits=4, embargo_pct=0.0):
        start = int(te.min())
        oncekiler = tr[tr < start]
        assert (t1[oncekiler] < start).all(), "purge başarısız: sızıntı var"


def test_purged_kfold_embargo_uygular():
    n, horizon = 400, 5
    t1 = np.arange(n) + horizon
    hic_embargo = list(purged_kfold_splits(t1, 4, embargo_pct=0.0))
    genis = list(purged_kfold_splits(t1, 4, embargo_pct=0.10))
    assert len(genis[0][0]) < len(hic_embargo[0][0])


def test_combinatorial_purged_yol_sayisi():
    n = 240
    t1 = np.arange(n) + 5
    splits = combinatorial_purged_splits(t1, n_groups=6, n_test_groups=2)
    assert len(splits) == 15                    # C(6,2)
    for tr, te in splits:
        assert len(np.intersect1d(tr, te)) == 0


# ------------------------------------------------------------------------ PBO
def test_pbo_saf_gurultude_yuksek():
    """Tamamen rastgele adaylar: en iyi IS konfig OOS'ta şansa kalır → PBO ~0.5"""
    rng = np.random.default_rng(7)
    M = rng.normal(0, 0.01, size=(600, 25))
    r = pbo(M, n_splits=8)
    assert r["pbo"] > 0.3, f"gürültüde PBO düşük çıktı: {r}"


def test_pbo_gercek_edge_dusuk():
    """Bir aday gerçekten üstün: PBO düşük olmalı."""
    rng = np.random.default_rng(8)
    M = rng.normal(0, 0.01, size=(600, 10))
    M[:, 3] += 0.004                            # 4. aday gerçek edge taşıyor
    r = pbo(M, n_splits=8)
    assert r["pbo"] < 0.3, f"gerçek edge'de PBO yüksek çıktı: {r}"


def test_pbo_yetersiz_veri():
    """Yetersiz veride PBO **None** döner, 1,0 DEĞİL.

    Davranış BİLEREK değişti. Eski hâl 1,0 döndürüyordu ve çağıran bunu
    "seçim tamamen aşırı uyum" diye okuyordu; ölçüldü ki nitelendirme
    katmanında 594/594 hücrenin PBO'su tam 1,0 çıkıyor ve kapı hiçbir şey
    ayırt etmiyordu. 'Ölçülemedi' ile 'kesin aşırı uyum' aynı şey değildir."""
    az = pbo(np.zeros((5, 3)))
    assert az["pbo"] is None and "ölçülemedi" in az["verdict"]
    tek_aday = pbo(np.zeros((100, 1)))
    assert tek_aday["pbo"] is None and "yetersiz aday" in tek_aday["verdict"]


# ------------------------------------------------------- deneme günlüğü
def test_trial_log_ve_sayim(tmp_path):
    d = str(tmp_path)
    assert trial_count("x", d) == 0
    trial_log("x", {"a": 1}, {"sharpe": 1.2}, d)
    trial_log("x", {"a": 2}, {"sharpe": 0.8}, d)
    trial_log("y", {"a": 3}, {"sharpe": 2.0}, d)
    assert trial_count("x", d) == 2
    assert trial_count("y", d) == 1
    assert trial_count(None, d) == 3


# ------------------------------------------------------------ kabul kapısı
def test_kapi_kilitli_test_olmadan_reddeder():
    rng = np.random.default_rng(9)
    r = rng.normal(0.004, 0.01, 800)
    res = acceptance_gate("t1", r, n_trials=5)
    assert res.passed is False
    assert any("Kilitli test" in x for x in res.reasons)


def test_kapi_supheli_yuksek_sharpe_reddeder():
    rng = np.random.default_rng(10)
    r = rng.normal(0.02, 0.01, 500)             # yıllık Sharpe ~38 → bug şüphesi
    res = acceptance_gate("t2", r, n_trials=3,
                          locked_test_returns=r, baseline_test_returns=r * 0.1)
    assert res.passed is False
    assert any("BUG VARSAYILIR" in x for x in res.reasons)


def test_kapi_yuksek_korelasyonu_reddeder():
    rng = np.random.default_rng(11)
    kitap = rng.normal(0.001, 0.01, 600)
    aday = kitap * 0.98 + rng.normal(0, 0.0005, 600)   # neredeyse aynı bahis
    res = acceptance_gate("t3", aday, n_trials=2, book_returns=kitap,
                          locked_test_returns=aday, baseline_test_returns=kitap)
    assert res.passed is False
    assert any("korelasyon" in x for x in res.reasons)


def test_kapi_iyi_aday_gecer():
    """Gerçekçi aday: yıllık Sharpe 2,2 · 2000 gün (~5,5 yıl) · kitapla bağımsız.

    2000 gün tesadüf değil: 8 denemede DSR 0,95'i geçmek için gereken asgari
    veri uzunluğu bu (bkz. min_backtest_length). Daha kısa veriyle Sharpe'ı
    yükselterek geçmeye çalışmak 2,5 şüphe tavanına çarpar — çerçeve tutarlı."""
    kitap = make_returns(2000, annual_sharpe=1.3, seed=100)
    aday = make_returns(2000, annual_sharpe=2.2, seed=101)
    rng = np.random.default_rng(12)
    M = np.column_stack([aday, rng.normal(0, 0.01, size=(2000, 6))])
    res = acceptance_gate("t4", aday, n_trials=8, book_returns=kitap,
                          perf_matrix=M,
                          locked_test_returns=aday,
                          baseline_test_returns=kitap)
    assert res.passed is True, str(res)


def test_kapi_az_ornek_reddeder():
    assert acceptance_gate("t5", [0.01] * 10).passed is False


# ------------------------------------------------------------- sızıntı testi
def test_shuffle_test_sizintiyi_yakalar():
    """Geleceği gören strateji: karıştırılmış veride bile yüksek Sharpe."""
    import pandas as pd

    def hileli(df):                              # look-ahead: yarını bugün bilir
        return (df["ret"].shift(-1).fillna(0) * np.sign(df["ret"].shift(-1).fillna(0))).values

    rng = np.random.default_rng(13)
    df = pd.DataFrame({"ret": rng.normal(0, 0.01, 400)})
    r = shuffle_test(hileli, df, n_shuffles=10)
    assert r["leak"] is True


def test_shuffle_test_temiz_stratejide_sizinti_yok():
    import pandas as pd

    def durust(df):                              # yalnız geçmişi kullanır
        sig = np.sign(df["ret"].shift(1).fillna(0))
        return (sig * df["ret"]).values

    rng = np.random.default_rng(14)
    df = pd.DataFrame({"ret": rng.normal(0, 0.01, 400)})
    r = shuffle_test(durust, df, n_shuffles=10)
    assert r["leak"] is False


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
