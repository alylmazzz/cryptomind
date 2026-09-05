#!/usr/bin/env python3
"""
Carry sleeve'i — genişletilmiş örneklemle yeniden ölçüm.

İlk ölçüm (5 parite × 4 yıl): Sharpe 0,80 · kitapla korelasyon +0,01 · DSR 0,66.
Ekonomik profil tam istenen ama istatistiksel güç yetersizdi. Bu betik aynı
HİPOTEZİ (parametreler DEĞİŞMEDEN) 23 parite × ~6 yılda sınar.

Parametre değiştirmek aşırı uyumdur; örneklem büyütmek kanıt toplamaktır.

KİLİTLİ TEST (2026) burada AÇILMAZ.
  python research_carry.py
"""
from __future__ import annotations

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
from agi_trader.strategies.sleeves import build_sleeve, TrendSleeve
from agi_trader.auto.sleeve_allocator import allocate

FUND_DIR = Path(__file__).parent / "runs" / "data_funding"
ORIGINAL_5 = ["BTC", "ETH", "SOL", "DOGE", "AVAX"]
TRAIN_VAL_END = "2025-12-31"
OUT = Path(__file__).parent / "runs" / "carry_research.json"


def load_all_funding() -> dict:
    out = {}
    for p in sorted(FUND_DIR.glob("*_funding.csv")):
        coin = p.stem.replace("_funding", "")
        s = ds.load_funding(coin)
        if s is not None and len(s) > 1000:
            out[coin] = s[s.index <= pd.Timestamp(TRAIN_VAL_END)]
    return out


def load_all_basis(coins) -> dict:
    out = {}
    for c in coins:
        b = ds.load_basis(c)
        if b is not None and len(b) > 300:
            out[c] = b[b.index <= pd.Timestamp(TRAIN_VAL_END)]
    return out


def carry_returns(funding: dict, index: pd.DatetimeIndex,
                  basis: dict | None = None, **kw) -> pd.Series:
    """Carry sleeve'i yönlü fiyat kullanmaz (delta-nötr); indeks + baz yeter."""
    dummy = pd.DataFrame(index=index, columns=list(funding), data=1.0)
    return build_sleeve("carry", **kw).returns(dummy, funding=funding, basis=basis)


def fmt(m: dict) -> str:
    return (f"Sharpe {m['sharpe']:+.2f} | CAGR {m['cagr']:+6.2f}% | "
            f"DD {m['dd']:5.2f}% | vol {m['vol']:5.2f}%")


def main() -> None:
    print("=" * 78)
    print("CARRY — GENİŞLETİLMİŞ ÖRNEKLEM (parametreler DEĞİŞMEDİ)")
    print("=" * 78)

    funding = load_all_funding()
    if not funding:
        print("funding verisi yok — önce: python fetch_funding_ext.py")
        return
    spans = {k: (v.index[0].date(), v.index[-1].date()) for k, v in funding.items()}
    first = min(v[0] for v in spans.values())
    print(f"\n{len(funding)} parite · en erken {first} · bitiş {TRAIN_VAL_END} (test KİLİTLİ)")

    basis = load_all_basis(list(funding))
    print(f"{len(basis)} paritede baz (perp/spot) serisi yüklendi")

    idx = pd.date_range(str(first), TRAIN_VAL_END, freq="D")

    print("\n[1] Örneklem ve model gerçekçiliğinin etkisi")
    results = {}
    all_coins = list(funding)
    for label, coins, start, use_basis in (
        ("5 parite × 2022+ (ilk ölçüm)", ORIGINAL_5, "2022-01-01", True),
        ("5 parite × tam geçmiş", ORIGINAL_5, str(first), True),
        (f"{len(funding)} parite · BAZ YOK (iyimser)", all_coins, str(first), False),
        (f"{len(funding)} parite · baz dahil", all_coins, str(first), True),
    ):
        sub = {c: funding[c] for c in coins if c in funding}
        if not sub:
            continue
        ix = pd.date_range(start, TRAIN_VAL_END, freq="D")
        b = {k: v for k, v in basis.items() if k in sub} if use_basis else None
        r = carry_returns(sub, ix, basis=b).reindex(ix).fillna(0.0)
        m = ds.annualized(r)
        n_years = len(ix) / 365.25
        print(f"    {label:34s} {fmt(m)} | {n_years:.1f} yıl")
        results[label] = {"metrics": m, "n_assets": len(sub), "years": round(n_years, 1),
                          "basis_modeled": use_basis}
        if use_basis and len(sub) > 5:
            best = r

    print("\n[2] REJİM AYRIMI — hangi sayı ileriye dönük kullanılmalı?")
    early = best[best.index <= pd.Timestamp("2021-12-31")]
    mature = best[best.index >= pd.Timestamp("2022-01-01")]
    m_early, m_mature = ds.annualized(early), ds.annualized(mature)
    print(f"    2019-2021 (aşırı kaldıraçlı boğa) {fmt(m_early)}")
    print(f"    2022-2025 (olgun piyasa)          {fmt(m_mature)}")
    print("    → 2021'de funding yıllık %37'ye çıkmıştı; o rejim tekrarlamadı.")
    print("    → İLERİYE DÖNÜK BEKLENTİ = olgun piyasa dilimi.")

    print("\n[3] Deflated Sharpe (olgun piyasa dilimi, aynı hipotez)")
    trial_log("carry", {"sleeve": "carry", "n_assets": len(funding),
                        "period": "mature_2022_2025"},
              {"sharpe": m_mature["sharpe"]})
    d = deflated_sharpe(mature.values, n_trials=2, periods_per_year=365.0)
    print(f"    yıllık SR {d['sr_annual']:+.3f} · şansla beklenen {d['sr0_annual']:.3f} "
          f"· {d['n_obs']} gözlem")
    print(f"    DSR = {d['dsr']:.3f} → {d['verdict']}")
    d_full = deflated_sharpe(best.values, n_trials=2, periods_per_year=365.0)
    print(f"    (tam geçmiş DSR {d_full['dsr']:.3f} · SR {d_full['sr_annual']:.2f} "
          f"— REJİM ARTEFAKTI, kullanma)")

    print("\n[4] Trend kitabıyla korelasyon ve birleşim (ortak dönem 2022-2025)")
    CRYPTO = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "AVAXUSDT"]
    NONCRYPTO = ["GLD", "SPY", "QQQ", "TLT", "UUP", "USO", "SLV", "DBC",
                 "HYG", "FXB", "FXF", "FXE"]
    prices = ds.train_val(ds.align(ds.load_universe(CRYPTO + NONCRYPTO),
                                   start="2021-01-01"))
    trend = TrendSleeve().returns(prices).reindex(prices.index).fillna(0.0)
    carry_common = best.reindex(prices.index).fillna(0.0)

    rho = float(trend.corr(carry_common))
    s1 = ds.annualized(trend)["sharpe"]
    s2 = ds.annualized(carry_common)["sharpe"]
    teorik = float(np.sqrt(max((s1**2 + s2**2 - 2*rho*s1*s2) / (1 - rho**2), 0)))
    print(f"    trend Sharpe {s1:+.2f} · carry Sharpe {s2:+.2f} · ρ = {rho:+.3f}")
    print(f"    iki-varlık teorik optimum Sharpe = {teorik:.2f}")

    R = pd.DataFrame({"trend": trend, "carry": carry_common})
    port, W, lev = allocate(R, method="max_sharpe")
    m_port = ds.annualized(port)
    m_trend = ds.annualized(trend)
    print(f"\n    BASELINE (trend)     {fmt(m_trend)}")
    print(f"    TREND + CARRY        {fmt(m_port)}")
    print(f"    Δ Sharpe {m_port['sharpe'] - m_trend['sharpe']:+.3f} · "
          f"ort. kaldıraç {lev[lev > 0].mean():.2f}×")

    print("\n[5] Artan değerin istatistiksel testi (eş-vol fark serisi)")
    base = trend.reindex(port.index).fillna(0.0)
    sb, sc = base.std(), port.std()
    diff = (port / sc - base / sb) * sb
    m_diff = ds.annualized(diff)
    d_inc = deflated_sharpe(diff.values, n_trials=2, periods_per_year=365.0)
    print(f"    fark Sharpe {m_diff['sharpe']:+.3f} · DSR {d_inc['dsr']:.3f} "
          f"→ {d_inc['verdict']}")

    print("\n" + "=" * 78)
    print("HÜKÜM — carry")
    print("=" * 78)
    res = acceptance_gate("carry", best.values, book_returns=trend.values,
                          periods_per_year=365.0)
    print(res)

    if not bool(best.attrs.get("risk_measurable", True)):
        print("\n  ⛔ RİSK ÖLÇÜLEMEDİ — Sharpe/DSR sayıları GEÇERSİZDİR.")
        print("     " + str(best.attrs.get("risk_note", "")))
        print()
        print("  GÜVENİLİR OLAN (getiri tarafı, 23 parite, maliyet+baz+kuyruk dahil):")
        print(f"     2019-2021 (aşırı kaldıraçlı boğa) : net %{ds.annualized(early)['cagr']:+.2f}/yıl")
        print(f"     2022-2025 (olgun piyasa)          : net %{ds.annualized(mature)['cagr']:+.2f}/yıl  ← ileriye dönük")
        print(f"     trend kitabıyla korelasyon        : {rho:+.3f} (gerçekten bağımsız)")
        print()
        print("  KARAR: carry DAĞITILMAZ. Getirisi gerçek ve bağımsız, ama risk-ayarlı")
        print("  büyüklüğü bilinmeden pozisyon boyutu belirlenemez. Gerekli veri")
        print("  (gün-içi baz + tasfiye) data/recorder.py ile bugünden toplanıyor;")
        print("  ~6 ay sonra gerçek risk ölçümüyle yeniden değerlendirilecek.")

    OUT.write_text(json.dumps({
        "n_assets": len(funding), "earliest": str(first),
        "sample_expansion": results,
        "dsr_full": d, "correlation_to_trend": round(rho, 4),
        "theoretical_two_asset_sharpe": round(teorik, 3),
        "baseline_trend": m_trend, "trend_plus_carry": m_port,
        "incremental": {"metrics": m_diff, "dsr": d_inc},
        "gate": res.to_dict(),
        "note": "KİLİTLİ TEST (2026) açılmadı.",
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
