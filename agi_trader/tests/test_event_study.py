# -*- coding: utf-8 -*-
"""FAZ 6 — olay çalışması ve toplayıcı testleri.

En kritik davranış: YETERSİZ VERİDE SKOR ÜRETMEMEK. Bir hesabın etkisini
ölçemiyorsak "etkisi yok" değil "ölçülmedi" demeliyiz ve o hesap ağırlık
almamalıdır — elle yazılmış varsayımlar modele sızmamalı.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agi_trader.sentiment.event_study import (  # noqa: E402
    abnormal_returns, run_event_study, effective_weights, HORIZONS,
    MIN_EVENTS, T_THRESHOLD,
)
from agi_trader.sentiment.collector import (  # noqa: E402
    extract_assets, score_text, append, load_events,
)


def price_series(n=4000, start="2026-01-01", freq="1min", drift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    r = rng.normal(drift, 0.0004, n)
    return pd.Series(100 * np.exp(np.cumsum(r)), index=idx)


def make_events(handle, times, asset="BTC", sentiment=None):
    return pd.DataFrame({
        "ts": times, "handle": handle, "asset": asset,
        "sentiment": ([sentiment] * len(times) if sentiment is not None
                      else [np.nan] * len(times)),
    })


# ───────────────────────── anormal getiri ───────────────────────────
def test_abnormal_return_hesaplanir():
    px = price_series()
    ts = px.index[1000]
    r = abnormal_returns(px, ts, 60)
    assert r is not None and np.isfinite(r[0])


def test_pencere_disinda_none():
    px = price_series(n=200)
    assert abnormal_returns(px, px.index[-1], 60) is None      # ufuk veri dışı
    assert abnormal_returns(px, px.index[0], 60) is None       # baz penceresi yok


def test_siçrama_yakalanir():
    """Olay anında yapay %5 sıçrama → anormal getiri belirgin pozitif olmalı."""
    px = price_series(seed=1)
    i = 2000
    px.iloc[i:] = px.iloc[i:] * 1.05
    r = abnormal_returns(px, px.index[i - 1], 60)
    assert r is not None and r[0] > 3.0


# ─────────────────────── istatistiksel kapı ─────────────────────────
def test_az_olayda_olculmedi_denir():
    px = price_series()
    times = list(px.index[500::300])[:5]                       # yalnız 5 olay
    st = run_event_study(make_events("azhesap", times), {"BTC": px})
    a = st["accounts"][0]
    assert a["measured"] is False
    assert "ÖLÇÜLMEDİ" in a["note"] and a["impact_score"] == 0.0


def test_olculmeyen_hesap_agirlik_almaz():
    px = price_series()
    times = list(px.index[500::300])[:5]
    st = run_event_study(make_events("azhesap", times), {"BTC": px})
    assert effective_weights(st) == {}                          # sıfır ağırlık


def test_oncul_yalniz_acikca_istenirse_kullanilir():
    px = price_series()
    times = list(px.index[500::300])[:5]
    st = run_event_study(make_events("azhesap", times), {"BTC": px},
                         priors={"azhesap": 9.6})
    assert effective_weights(st) == {}                                   # varsayılan
    assert effective_weights(st, fallback_to_prior=True) == {"azhesap": 9.6}


def test_etkisiz_hesap_olculur_ama_agirlik_almaz():
    """Bol veri + rastgele fiyat → anlamlı etki YOK, ağırlık 0."""
    px = price_series(n=20000, seed=7)
    times = list(px.index[500::300])[:60]
    st = run_event_study(make_events("etkisiz", times), {"BTC": px})
    a = st["accounts"][0]
    assert a["n_events"] >= MIN_EVENTS
    assert a["impact_score"] == 0.0 or a["measured"] is False
    assert effective_weights(st).get("etkisiz", 0.0) == 0.0


def test_gercek_etkili_hesap_olculur():
    """Her gönderiden sonra tutarlı sıçrama → ölçülmeli ve ağırlık almalı."""
    px = price_series(n=30000, seed=3)
    times = list(px.index[600::400])[:60]
    v = px.values.copy()
    for t in times:
        i = px.index.searchsorted(t)
        v[i:] = v[i:] * 1.01                                    # her olayda +%1
    px = pd.Series(v, index=px.index)
    st = run_event_study(make_events("etkili", times), {"BTC": px})
    a = st["accounts"][0]
    assert a["measured"] is True, a["note"]
    assert a["impact_score"] > 0
    assert effective_weights(st)["etkili"] > 0


def test_piyasa_hareketi_cikarilir():
    """Tüm piyasa yükseliyorsa hesaba özgü etki SIFIRA yakın olmalı."""
    base = price_series(n=20000, seed=5, drift=0.00005)
    times = list(base.index[600::400])[:50]
    st_raw = run_event_study(make_events("beta", times), {"BTC": base})
    st_adj = run_event_study(make_events("beta", times), {"BTC": base}, market=base)
    h_raw = [h for h in st_raw["accounts"][0]["horizons"] if h["horizon"] == "1h"][0]
    h_adj = [h for h in st_adj["accounts"][0]["horizons"] if h["horizon"] == "1h"][0]
    assert abs(h_adj["mean_abnormal_pct"]) < abs(h_raw["mean_abnormal_pct"]) + 1e-9
    assert abs(h_adj["mean_abnormal_pct"]) < 1e-6      # kendisiyle fark = 0


# ─────────────────────── çıktı sözleşmesi ───────────────────────────
def test_bos_veride_uydurmaz():
    st = run_event_study(pd.DataFrame(columns=["ts", "handle", "asset", "sentiment"]), {})
    assert st["accounts"] == [] and st["n_events"] == 0
    assert "collector" in st["note"]


def test_yon_uyarisi_her_zaman_var():
    px = price_series()
    st = run_event_study(make_events("h", list(px.index[500::300])[:5]), {"BTC": px})
    assert "AYRI" in st["direction_warning"]


def test_tum_ufuklar_raporlanir():
    px = price_series(n=20000, seed=9)
    times = list(px.index[600::400])[:40]
    st = run_event_study(make_events("h", times), {"BTC": px})
    hs = {h["horizon"] for h in st["accounts"][0]["horizons"]}
    assert hs == set(HORIZONS)


# ───────────────────────── toplayıcı ────────────────────────────────
def test_varlik_cikarimi():
    assert extract_assets("Bitcoin pumping, ETH next") == ["BTC", "ETH"]
    assert extract_assets("hiçbir şey") == []


def test_duygu_skoru_yonu():
    assert score_text("bullish breakout rally") > 0
    assert score_text("crash hack lawsuit dump") < 0
    assert score_text("nötr bir cümle") == 0.0


def test_kayit_ve_okuma(tmp_path, monkeypatch):
    import agi_trader.sentiment.collector as col
    monkeypatch.setattr(col, "_out_dir", lambda: tmp_path)
    rows = [{"ts": pd.Timestamp.now("UTC"), "source": "rss", "handle": "coindesk",
             "text": "Bitcoin rally", "sentiment": 0.5, "assets": "BTC", "url": ""}]
    assert col.append(rows) is not None
    df = col.load_events()
    assert len(df) == 1 and df.iloc[0]["handle"] == "coindesk"


def test_tekrar_eden_kayit_ikilenmez(tmp_path, monkeypatch):
    import agi_trader.sentiment.collector as col
    monkeypatch.setattr(col, "_out_dir", lambda: tmp_path)
    row = {"ts": pd.Timestamp.now("UTC"), "source": "rss", "handle": "x",
           "text": "aynı başlık", "sentiment": 0.0, "assets": "", "url": ""}
    col.append([row]); col.append([row])
    assert len(col.load_events()) == 1        # RSS tekrarları şişirmemeli


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def test_pyarrow_yokken_de_tekrar_ayiklanir(tmp_path, monkeypatch):
    """YEDEK YOL REGRESYONU (CI bulgusu, 2026-09-05).

    `append` parquet yazamazsa .csv.gz'ye düşüyordu ve o yolda tekrar ayıklama
    ATLANIYORDU: `gzip.open(..., "at")` ile körlemesine ekliyordu. Sonuç, pyarrow
    kurulu OLMAYAN her makinede aynı haberin defalarca sayılması — haber yoğunluğuna
    bakan sinyaller sahte bir yoğunluk görürdü. Yerelde pyarrow kurulu olduğu için bu
    yol hiç çalışmamış, hata yalnız temiz kurulumda (CI) ortaya çıkmıştı.

    Kilitlenen davranış: yedek yol ASIL yolla AYNI semantiği uygular."""
    import sys
    import agi_trader.sentiment.collector as col
    monkeypatch.setattr(col, "_out_dir", lambda: tmp_path)
    monkeypatch.setitem(sys.modules, "pyarrow", None)      # import pyarrow → ImportError

    row = {"ts": pd.Timestamp.now("UTC"), "source": "rss", "handle": "x",
           "text": "aynı başlık", "sentiment": 0.0, "assets": "", "url": ""}
    p1 = col.append([row])
    p2 = col.append([row])
    assert p1 is not None and p1.name.endswith(".csv.gz"), "yedek yola düşmeliydi"
    assert p2 == p1
    assert len(col.load_events()) == 1, "yedek yolda da tekrar ayıklanmalı"

    # Farklı metin AYRI kayıttır — tekilleştirme fazla agresif olmamalı.
    col.append([{**row, "text": "başka başlık"}])
    assert len(col.load_events()) == 2
    # Parquet'ten neden düşüldüğü kayıtlı olmalı (sessizce yutulmaz).
    assert col._SON_YEDEK_SEBEP, "yedek yola düşme sebebi kaydedilmeli"
