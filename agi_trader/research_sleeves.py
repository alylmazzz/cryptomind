#!/usr/bin/env python3
"""
FAZ 2 ölçümü — çoklu sleeve mimarisi gerçekten Sharpe kazandırıyor mu?

Yalnız TRAIN+VALIDATION (2022-01-01 → 2025-12-31) üzerinde çalışır.
KİLİTLİ TEST (2026) burada AÇILMAZ — runs/SPLIT.md kuralı.

  python research_sleeves.py
  python research_sleeves.py --no-tail-shock    # carry kuyruk cezasını kapat
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent))

from agi_trader.research import dataset as ds
from agi_trader.research.validation import (
    acceptance_gate, deflated_sharpe, trial_log, sharpe,
)
from agi_trader.strategies.sleeves import build_sleeve
from agi_trader.auto.sleeve_allocator import allocate, diversification_report

CRYPTO = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "AVAXUSDT"]
NONCRYPTO = ["GLD", "SPY", "QQQ", "TLT", "UUP", "USO", "SLV", "DBC",
             "HYG", "FXB", "FXF", "FXE"]
UNIVERSE = CRYPTO + NONCRYPTO
OUT = Path(__file__).parent / "runs" / "sleeve_research.json"


def fmt(m: dict) -> str:
    return (f"Sharpe {m['sharpe']:+.2f} | CAGR {m['cagr']:+6.1f}% | "
            f"DD {m['dd']:5.1f}% | vol {m['vol']:5.1f}% | Calmar {m['calmar']:5.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-tail-shock", action="store_true",
                    help="carry sleeve'inde modellenmemiş kuyruk cezasını kapat")
    args = ap.parse_args()

    print("=" * 78)
    print("FAZ 2 — ÇOKLU SLEEVE ÖLÇÜMÜ  (train+validation 2022-2025, test KİLİTLİ)")
    print("=" * 78)

    print(f"\n[1/5] Evren yükleniyor ({len(UNIVERSE)} varlık, yerel önbellek)…")
    series = ds.load_universe(UNIVERSE)
    prices_all = ds.align(series, start="2021-01-01")
    prices = ds.train_val(prices_all)
    print(f"      {prices.shape[1]} varlık × {len(prices)} gün "
          f"({prices.index[0].date()} → {prices.index[-1].date()})")

    print("\n[2/5] Funding + baz verisi (carry sleeve için)…")
    # Carry delta-nötrdür: yönlü fiyat serisine ihtiyacı yok, bu yüzden fiyat
    # evreniyle sınırlı değil — mevcut TÜM funding geçmişini kullanır.
    funding, basis = {}, {}
    for p in sorted((Path(__file__).parent / "runs" / "data_funding").glob("*_funding.csv")):
        coin = p.stem.replace("_funding", "")
        f = ds.load_funding(coin)
        if f is None or len(f) < 1000:
            continue
        funding[coin] = f
        b = ds.load_basis(coin)
        if b is not None:
            basis[coin] = b
    print(f"      {len(funding)} paritede funding · {len(basis)} paritede baz")

    print("\n[3/5] Sleeve getirileri hesaplanıyor…")
    specs = [
        ("trend", {}),
        ("xsec_momentum", {}),
        ("xsec_momentum", {"neutral": True}),     # dolar-nötr varyant
        ("short_reversal", {}),
        ("term_structure", {}),
        ("carry", {"tail_shock_prob": 0.0 if args.no_tail_shock else 0.0005}),
    ]
    rets: dict[str, pd.Series] = {}
    for name, kw in specs:
        sl = build_sleeve(name, **kw)
        key = sl.name
        r = (sl.returns(prices, funding=funding, basis=basis) if name == "carry"
             else sl.returns(prices))
        r = r.reindex(prices.index).fillna(0.0)
        if r.abs().sum() < 1e-9:
            print(f"      ⚠️ {key:16s} sinyal üretmedi — atlanıyor")
            continue
        rets[key] = r
        m = ds.annualized(r)
        flag = "" if r.attrs.get("risk_measurable", True) else "  ⚠️ RİSK ÖLÇÜLEMEDİ → Sharpe GEÇERSİZ"
        print(f"      {key:16s} {fmt(m)}{flag}")
        trial_log(key, {"sleeve": key, **kw, "period": "train_val"},
                  {"sharpe": m["sharpe"], "cagr": m["cagr"], "dd": m["dd"]})

    R = pd.DataFrame(rets)

    print("\n[4/5] Çeşitlendirme ölçümü…")
    div = diversification_report(R)
    print(f"      ortalama sleeve Sharpe : {div['mean_sleeve_sharpe']:+.3f}")
    print(f"      ortalama korelasyon    : {div['mean_correlation']:+.3f}")
    print(f"      en yüksek çift kor.    : {div['max_pair_correlation']:+.3f}")
    print(f"      → teorik portföy Sharpe: {div['expected_portfolio_sharpe']:+.3f}")
    print("\n      korelasyon matrisi:")
    print(R.corr().round(2).to_string().replace("\n", "\n      "))

    print("\n[5/5] ERC + hedef-vol birleştirme…")
    m_trend = ds.annualized(rets["trend"])
    book = rets["trend"]

    # ERC bir RİSK tahsisçisidir; girdilerin pozitif beklenen getirisi olduğunu
    # VARSAYAR. Doğrulanmamış sleeve'i içine koymak, düşük volatiliteli bir
    # kaybedene büyük ağırlık verdirir. Bu yüzden iki ayrı ölçüm:
    def combine(names, label, method="max_sharpe"):
        sub = R[names]
        port, W, lev = allocate(sub, method=method)
        m = ds.annualized(port)
        print(f"      {label:30s} {fmt(m)}")
        if (lev > 0).any():
            print(f"      {'':30s} ort. kaldıraç {lev[lev > 0].mean():.2f}× | ağırlık "
                  f"{ {c: round(float(W[c][W[c] != 0].mean() or 0), 3) for c in W.columns} }")
        return m, port

    print(f"      {'BASELINE (yalnız trend)':30s} {fmt(m_trend)}")

    # kapıyı geçenler: pozitif Sharpe + kitapla düşük korelasyon
    eligible = ["trend"] + [
        c for c in R.columns if c != "trend"
        and ds.annualized(R[c])["sharpe"] > 0
        and abs(float(R[c].corr(book))) < 0.40
    ]
    print(f"      uygun sleeve'ler: {eligible}\n")

    m_erc, _ = combine(eligible, "ERC (yalnız risk paritesi)", method="erc")
    m_port, port = combine(eligible, "MAX-SHARPE (büzültmeli)", method="max_sharpe")
    m_all, _ = combine(list(R.columns), "MAX-SHARPE · filtresiz", method="max_sharpe")

    # sadece trend+carry — teorik optimumla karşılaştırma
    if "carry" in R.columns:
        s1 = ds.annualized(R["trend"])["sharpe"]
        s2 = ds.annualized(R["carry"])["sharpe"]
        rho = float(R["trend"].corr(R["carry"]))
        teorik = np.sqrt(max((s1**2 + s2**2 - 2*rho*s1*s2) / (1 - rho**2), 0))
        m_tc, _ = combine(["trend", "carry"], "trend + carry")
        print(f"      {'':30s} teorik optimum Sharpe = {teorik:.2f} "
              f"(S₁={s1:.2f}, S₂={s2:.2f}, ρ={rho:+.2f})")

    print(f"\n      Δ Sharpe (max-sharpe − baseline): {m_port['sharpe'] - m_trend['sharpe']:+.3f}")
    print(f"      Δ Sharpe (ERC − baseline)       : {m_erc['sharpe'] - m_trend['sharpe']:+.3f}")
    print("      → ERC ile max-sharpe farkı, beklenen getiriyi yok saymanın bedelidir.")

    # --- kabul kapısı: her yeni sleeve mevcut kitaba (trend) karşı
    print("\n" + "=" * 78)
    print("KABUL KAPISI (kilitli test HARİÇ — o ancak konfig dondurulunca açılır)")
    print("=" * 78)
    gate_out = {}
    for name, r in rets.items():
        if name == "trend":
            continue
        res = acceptance_gate(name, r.values, book_returns=book.values,
                              periods_per_year=365.0)
        gate_out[name] = res.to_dict()
        print(f"\n{res}")

    # --- PORTFÖY DÜZEYİ KARAR: eklemenin ARTAN değeri anlamlı mı?
    # Sleeve kapısı tek başına yetmez; asıl soru "kitaba eklemek portföyü
    # gerçekten iyileştiriyor mu?". Bunun doğru testi FARK serisidir:
    # (aday portföy) − (baseline), ikisi de aynı volatiliteye ölçeklenerek.
    if "carry" in R.columns:
        _, tc_port = combine(["trend", "carry"], "(portföy testi)")
        base = book.reindex(tc_port.index).fillna(0.0)
        s_base, s_cand = base.std(), tc_port.std()
        if s_base > 1e-12 and s_cand > 1e-12:
            diff = (tc_port / s_cand - base / s_base) * s_base   # eş-vol fark
            m_diff = ds.annualized(diff)
            print("\n" + "-" * 78)
            print("PORTFÖY DÜZEYİ: (trend+carry) − trend, eş-volatilitede")
            print(f"  fark serisi Sharpe: {m_diff['sharpe']:+.3f} "
                  f"(pozitif = ekleme gerçekten katkı sağlıyor)")
            d_inc = deflated_sharpe(diff.values, n_trials=max(1, len(R.columns)),
                                    periods_per_year=365.0)
            print(f"  DSR: {d_inc['dsr']:.3f} → {d_inc['verdict']}")
            gate_out["_incremental_trend_plus_carry"] = {
                "diff_metrics": m_diff, "dsr": d_inc}

    payload = {
        "period": {"start": str(prices.index[0].date()), "end": str(prices.index[-1].date())},
        "universe": list(prices.columns),
        "sleeve_metrics": {k: ds.annualized(v) for k, v in rets.items()},
        "diversification": div,
        "baseline_trend": m_trend,
        "multi_sleeve_filtered": m_port,
        "multi_sleeve_unfiltered": m_all,
        "eligible_sleeves": eligible,
        "delta_sharpe": round(m_port["sharpe"] - m_trend["sharpe"], 3),
        "gates": gate_out,
        "note": "KİLİTLİ TEST (2026) açılmadı; bu sonuçlar train+validation'dır.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
