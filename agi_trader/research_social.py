#!/usr/bin/env python3
"""
FAZ 6 — sosyal olay çalışması: hangi hesap gerçekten fiyat hareket ettiriyor?

Toplanan olayları (`runs/social/*.parquet`) fiyat verisiyle eşleştirir, hesap
başına ANORMAL getiri ölçer ve `runs/account_scores.json` üretir. Panel bu
dosyayı okur.

DÜRÜSTLÜK KURALI: yeterli gözlemi olmayan hesap "ölçülmedi" damgası alır ve
ağırlık ALMAZ. `accounts.py`'deki elle yazılmış `weight` değerleri modele
GİRMEZ — yalnız karşılaştırma için raporlanır.

  python research_social.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).parent))

from agi_trader.sentiment.collector import load_events
from agi_trader.sentiment.event_study import (
    run_event_study, save_scores, effective_weights, MIN_EVENTS,
)

ASSET_TO_SYMBOL = {"BTC": "BTC/USDT", "ETH": "ETH/USDT", "SOL": "SOL/USDT",
                   "DOGE": "DOGE/USDT", "AVAX": "AVAX/USDT"}


def priors() -> dict:
    """accounts.py'deki elle yazılmış ağırlıklar — YALNIZ karşılaştırma için."""
    try:
        from agi_trader.sentiment.accounts import CRITICAL_ACCOUNTS
    except Exception:
        return {}
    out = {}
    for cat in CRITICAL_ACCOUNTS.values():
        for h, meta in cat.items():
            if isinstance(meta, dict) and "weight" in meta:
                out[h] = float(meta["weight"])
    return out


def price_panel(tf: str = "5m") -> dict:
    """Olay ufukları dakikalık olduğu için ince zaman dilimi gerekir."""
    from agi_trader.config import load_config
    from agi_trader.agents import Orchestrator
    o = Orchestrator(load_config("config.yaml"))
    panel = {}
    for asset, sym in ASSET_TO_SYMBOL.items():
        try:
            df = o.data.fetch_ohlcv(sym, tf)
            if df is not None and len(df) > 50:
                panel[asset] = df["close"].astype(float)
        except Exception:
            continue
    return panel


def expand_events(df: pd.DataFrame) -> pd.DataFrame:
    """Bir gönderi birden çok varlığı etiketleyebilir → varlık başına satır."""
    rows = []
    for _, r in df.iterrows():
        assets = [a for a in str(r.get("assets") or "").split(",") if a]
        for a in assets:
            rows.append({"ts": r["ts"], "handle": r.get("handle"),
                         "asset": a, "sentiment": r.get("sentiment")})
    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 74)
    print("FAZ 6 — SOSYAL OLAY ÇALIŞMASI")
    print("=" * 74)

    raw = load_events()
    print(f"\ntoplanan ham kayıt: {len(raw)}")
    if len(raw) == 0:
        print("Henüz veri yok — cryptomind-social servisi topluyor.")
        save_scores({"accounts": [], "n_events": 0,
                     "note": "toplayıcı henüz veri üretmedi"})
        return

    ev = expand_events(raw)
    print(f"varlık etiketli olay : {len(ev)}")
    if len(ev) == 0:
        print("Hiçbir kayıt bir varlıkla eşleşmedi — ölçüm yapılamaz.")
        save_scores({"accounts": [], "n_events": 0,
                     "note": "olaylar varlıkla eşleşmedi"})
        return

    print("fiyat paneli çekiliyor (5m)…")
    panel = price_panel()
    print(f"  {len(panel)} varlık: {sorted(panel)}")
    if not panel:
        print("Fiyat verisi alınamadı.")
        return

    study = run_event_study(ev, panel, market=panel.get("BTC"), priors=priors())
    path = save_scores(study)

    print(f"\n{study['note']}")
    print(f"eşik: {MIN_EVENTS} olay · |t| ≥ 2")
    print("\nEN ÇOK OLAYA SAHİP 12 HESAP:")
    print(f"  {'hesap':22s} {'olay':>5s} {'ölçüldü':>8s} {'skor':>6s} {'öncül':>6s}")
    for a in sorted(study["accounts"], key=lambda x: -x["n_events"])[:12]:
        pw = f"{a['prior_weight']:.1f}" if a.get("prior_weight") is not None else "—"
        print(f"  {a['handle'][:22]:22s} {a['n_events']:>5d} "
              f"{('EVET' if a['measured'] else 'hayır'):>8s} "
              f"{a['impact_score']:>6.2f} {pw:>6s}")

    w = effective_weights(study)
    print(f"\nmotorun kullanacağı ağırlık sayısı: {len(w)}")
    if not w:
        print("  (hiçbiri kapıyı geçmedi — sentiment katmanı ağırlıksız çalışır)")
    print(f"\n→ {path}")


if __name__ == "__main__":
    main()
