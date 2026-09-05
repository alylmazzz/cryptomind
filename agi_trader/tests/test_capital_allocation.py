# -*- coding: utf-8 -*-
"""SERMAYE TAHSİSİ — risk payı ölçülmüş sonuca göre verilir (2026-09-05).

Bağlam: trend katmanı 48 günde +%4,29 (Sharpe 2,68 · DD %2,0) kazanırken scalping katmanı
105 işlemde −%0,62 kaybediyordu, ama ikisi de aynı risk limitleriyle çalışıyordu. Bu modül
payı ölçüme bağlar. Kilitlenen davranışlar:
  · kazanan katman payı alır, eşik altı katman yalnız ÖLÇÜM BÜTÇESİ (kapatılmaz — kapatılan
    katman bir daha ölçülemez ve rejim değişince fark edilmez)
  · İKİSİ DE kaybediyorsa sermaye NAKİTTE kalır (normalize edilmez!)
  · scalping Sharpe'ı GÜNLÜK ölçekte hesaplanır (işlem başına değil; 65 işlem/gün yapan
    katman aksi hâlde yanıltıcı görünür)
  · koşucu limitleri paydan TÜREtilir (pay %2 iken 200 $ emir açmak payı anlamsız kılar)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agi_trader.auto import capital_allocator as CA  # noqa: E402


def _layer(name, sharpe, measured=True, n=48, equity=100.0, initial=100.0, reason=None):
    d = {"layer": name, "sharpe": sharpe, "measured": measured, "n": n,
         "equity": equity, "initial": initial}
    if reason:
        d["not_measured_reason"] = reason
    return d


def _trend_state(equities, initial=10000.0):
    return {"initial": initial,
            "history": [{"date": f"2026-07-{(i % 28) + 1:02d}", "equity": e} for i, e in enumerate(equities)]}


def _trade(net, ts, notional=50.0, gross=None, fees=0.05):
    return {"net_pnl": net, "gross_pnl": (net + fees) if gross is None else gross,
            "fees": fees, "notional": notional, "closed_ts": ts}


# ═══════════════════════ tahsis kuralı ═══════════════════════
def test_kazanan_katman_payi_alir_kaybeden_olcum_butcesi():
    a = CA.allocate([_layer("trend", 2.68), _layer("scalp", -1.5)])
    w = {r["layer"]: r["weight_pct"] for r in a["layers"]}
    assert w["trend"] > 90 and w["scalp"] == CA.MEASURE_BUDGET_PCT
    assert a["cash_pct"] >= 0 and round(sum(w.values()) + a["cash_pct"], 2) == 100.0
    scalp = next(r for r in a["layers"] if r["layer"] == "scalp")
    assert scalp["eligible"] is False and "risk sermayesi YOK" in scalp["reason"]


def test_ikisi_de_kaybederse_sermaye_nakitte_kalir_regresyon():
    """REGRESYON: ağırlıklar normalize edilirse %2+%2 → %50/%50 olur, yani 'ikisi de kaybediyor'
    durumu 'sermayeyi ikiye böl'e dönüşürdü. Kalan pay NAKİTTE durmalı."""
    a = CA.allocate([_layer("trend", -1.0), _layer("scalp", -2.0)])
    w = {r["layer"]: r["weight_pct"] for r in a["layers"]}
    assert w["trend"] == CA.MEASURE_BUDGET_PCT and w["scalp"] == CA.MEASURE_BUDGET_PCT
    assert a["cash_pct"] == pytest.approx(100.0 - 2 * CA.MEASURE_BUDGET_PCT)
    assert a["invested_pct"] == pytest.approx(2 * CA.MEASURE_BUDGET_PCT)


def test_ikisi_de_kazanirsa_sharpe_oraniyla_paylasilir():
    a = CA.allocate([_layer("trend", 3.0), _layer("scalp", 1.0)])
    w = {r["layer"]: r["weight_pct"] for r in a["layers"]}
    assert w["trend"] > w["scalp"] > 0
    assert w["trend"] / w["scalp"] == pytest.approx(3.0, rel=0.05)      # Sharpe oranı
    assert a["cash_pct"] == pytest.approx(0.0, abs=0.05)


def test_olculmemis_katman_pay_alamaz():
    a = CA.allocate([_layer("trend", 2.0), _layer("scalp", 9.9, measured=False, reason="2 gün < 5 gün")])
    scalp = next(r for r in a["layers"] if r["layer"] == "scalp")
    assert scalp["weight_pct"] == CA.MEASURE_BUDGET_PCT, "yüksek Sharpe bile olsa ölçülmeden pay yok"
    assert "ölçülmedi" in scalp["reason"] and "2 gün" in scalp["reason"]


def test_tek_katman_tavani():
    a = CA.allocate([_layer("trend", 5.0), _layer("scalp", -1.0)])
    w = {r["layer"]: r["weight_pct"] for r in a["layers"]}
    assert w["trend"] <= CA.MAX_WEIGHT, "yoğunlaşma tavanı aşılamaz"


# ═══════════════════════ metrikler ═══════════════════════
def test_trend_metrikleri_nan_gunleri_atlar():
    eq = [10000.0, 10050.0, float("nan"), 10120.0, 10080.0] + [10100.0 + i * 5 for i in range(20)]
    m = CA.trend_metrics(_trend_state(eq))
    assert m["n"] == len(eq) - 1                      # NaN gün sayılmaz
    assert math.isfinite(m["return_pct"]) and math.isfinite(m["max_dd_pct"])
    assert m["sharpe"] is None or math.isfinite(m["sharpe"])


def test_trend_metrikleri_yetersiz_gecmis():
    m = CA.trend_metrics(_trend_state([10000.0]))
    assert m["measured"] is False and m["sharpe"] is None


def test_scalp_sharpe_gunluk_olcekte_hesaplanir():
    """65 işlem/gün yapan katman, işlem-başı ölçekte yanıltıcı görünür; gün bazında toplanmalı."""
    day = 86400
    rnd = __import__("random").Random(4)
    trades = []
    for d in range(8):                                 # 8 gün × 10 işlem (günler FARKLI sonuçlansın:
        for k in range(10):                            # sabit günlük getiri σ=0 verir, Sharpe tanımsız kalır)
            trades.append(_trade(round(rnd.gauss(0.005, 0.06), 4), 1_788_000_000 + d * day + k * 60))
    m = CA.scalp_metrics(trades, 1000.0)
    assert m["days"] == 8 and m["n"] == 80 and m["trades_per_day"] == 10.0
    assert m["measured"] is True and m["sharpe"] is not None
    assert m["fee_share_of_gross_pct"] is not None


def test_scalp_az_gunde_olculmez():
    day = 86400
    trades = [_trade(-0.05, 1_788_000_000 + d * day) for d in range(3)]
    m = CA.scalp_metrics(trades, 1000.0)
    assert m["measured"] is False and "gün" in (m["not_measured_reason"] or "")


def test_scalp_islem_yoksa_bos_metrik():
    m = CA.scalp_metrics([], 1000.0)
    assert m["measured"] is False and m["n"] == 0


# ═══════════════════════ limit türetme ═══════════════════════
def test_limitler_paydan_turetilir():
    small = CA.scalp_risk_budget(2.0, 1000.0)
    big = CA.scalp_risk_budget(40.0, 1000.0)
    assert small["max_order_usdt"] < big["max_order_usdt"]
    assert small["max_trades_per_day"] < big["max_trades_per_day"]
    assert small["risk_per_trade_pct"] <= big["risk_per_trade_pct"]
    assert small["max_open"] >= 2 and big["max_open"] >= 2        # çeşitlendirme tabanı
    assert small["capital_at_risk_usdt"] == pytest.approx(20.0)
    # borsa asgarisinin altına inilmez
    tiny = CA.scalp_risk_budget(0.0, 1000.0)
    assert tiny["max_order_usdt"] >= 10.0 and tiny["max_trades_per_day"] >= 10


def test_simulator_limitleri_olcum_butcesinden_gelir():
    from agi_trader.auto import simulator as SIM
    cfg = SIM.default_config("mexc")
    lim = SIM.scalp_limits()
    assert cfg.max_order_usdt == lim["max_order_usdt"] == 10.0
    assert cfg.max_trades_per_day == lim["max_trades_per_day"] == 10
    assert cfg.max_open == lim["max_open"] >= 2
    assert cfg.capital_usdt == SIM.SIM_CAPITAL          # muhasebe tabanı değişmez
    # pay büyürse limitler de büyür (kanıt gelirse otomatik)
    big = SIM.scalp_limits(40.0)
    assert big["max_order_usdt"] > lim["max_order_usdt"]


# ═══════════════════════ uçtan uca rapor ═══════════════════════
def test_rapor_iki_katmani_birlestirir():
    rnd = __import__("random").Random(7)
    eq, cur = [], 10000.0
    for _ in range(48):                                # gerçekçi: pozitif sürüklenme + gürültü
        cur *= (1 + rnd.gauss(0.0012, 0.004))
        eq.append(cur)
    day = 86400
    trades = [_trade(round(rnd.gauss(-0.02, 0.05), 4), 1_788_000_000 + d * day + k * 60)
              for d in range(6) for k in range(5)]
    rep = CA.report(_trend_state(eq), trades, 1000.0)
    layers = {r["layer"]: r for r in rep["layers"]}
    assert set(layers) == {"trend", "scalp"}
    assert layers["trend"]["weight_pct"] > layers["scalp"]["weight_pct"]
    assert "scalp_budget" in rep and rep["scalp_budget"]["weight_pct"] == layers["scalp"]["weight_pct"]
    assert rep["combined"]["initial"] == pytest.approx(11000.0)
    assert round(sum(r["weight_pct"] for r in rep["layers"]) + rep["cash_pct"], 2) == 100.0


def test_bozuk_trend_state_patlamaz():
    for bad in ({}, {"history": []}, {"history": [{"date": "x"}]}, {"initial": 0, "history": []}):
        m = CA.trend_metrics(bad)
        assert m["measured"] is False
    rep = CA.report(None, None, 1000.0)
    assert rep["layers"] and rep["cash_pct"] >= 0


def test_sync_alanlari_risk_yuzdesini_icerir():
    """REGRESYON: `risk_per_trade_pct` senkron listesinde yoktu → sermaye tahsisi diskteki eski
    koşucuya hiç uygulanmıyordu (limitler kodda 10 $, koşucuda 200 $ kaldı)."""
    import inspect
    from agi_trader.auto import simulator as SIM
    src = inspect.getsource(SIM.ensure_simulator)
    for f in ("max_order_usdt", "max_open", "max_trades_per_day", "risk_per_trade_pct",
              "max_exposure_pct", "params"):
        assert f in src, f"{f} sync_config alanlarında olmalı"
