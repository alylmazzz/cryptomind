#!/usr/bin/env python3
"""Nitelendirme evrenini SEÇ — hacim değil, ÖLÇÜLEBİLİRLİK ölçütüyle.

NEDEN LİSTE ELLE YAZILMAZ
"En likit 8 parite" gibi bir liste iki hatayı birden yapar: fırsatın olabileceği
marketleri dışarıda bırakır ve seçimin gerekçesini gizler. Bu betik evreni
Binance'in kendi verisinden türetir ve HER dışlamayı nedeniyle yazar.

İKİ KAPI
  1. LİSTELENME TARİHİ — 4,5 yıllık ölçüm penceresinin anlamlı bir kısmını
     kapsamayan market, taban oranını dört rejimde ölçemez. 2022'de listelenmiş
     bir coin 2022 ayı piyasasını hiç görmemiştir.
  2. HACİM — ince marketlerde net %1 hedefi maliyet yer; ölçüm yapılabilse bile
     sonuç uygulanabilir olmaz.

Çıktı: `runs/qualification/universe_5m.json` — seçilenler ve NEDEN seçilmedikleri.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

FAPI = "https://fapi.binance.com"
OUT = Path(__file__).parent / "runs" / "qualification"
QUOTES = ("USDT", "USDC")
MIN_VOLUME_USD = 20_000_000.0
MAX_ONBOARD = "2024-01-01"          # bu tarihten SONRA listelenen dışarıda
MAX_SYMBOLS = 60                    # disk/süre bütçesi — hacme göre kırpılır


def _get(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main() -> int:
    info = _get(f"{FAPI}/fapi/v1/exchangeInfo")
    tick = {t["symbol"]: t for t in _get(f"{FAPI}/fapi/v1/ticker/24hr")}
    esik_ms = int(time.mktime(time.strptime(MAX_ONBOARD, "%Y-%m-%d")) * 1000)

    secilen, dislanan = [], []
    for s in info.get("symbols", []):
        sym = s["symbol"]
        neden = []
        if s.get("quoteAsset") not in QUOTES:
            continue                                  # kapsam dışı, dışlama değil
        if s.get("status") != "TRADING":
            neden.append("işlem görmüyor")
        if s.get("contractType") != "PERPETUAL":
            neden.append("perpetual değil")
        ob = int(s.get("onboardDate") or 0)
        if ob > esik_ms:
            neden.append(f"listelenme {time.strftime('%Y-%m', time.gmtime(ob/1000))} "
                         f"> {MAX_ONBOARD[:7]}")
        hacim = float(tick.get(sym, {}).get("quoteVolume", 0.0) or 0.0)
        if hacim < MIN_VOLUME_USD:
            neden.append(f"24s hacim {hacim/1e6:.1f}M$ < {MIN_VOLUME_USD/1e6:.0f}M$")
        kayit = {"symbol": sym, "quote": s.get("quoteAsset"),
                 "volume_usd_24h": round(hacim),
                 "onboard": time.strftime("%Y-%m-%d", time.gmtime(ob / 1000)) if ob else None}
        if neden:
            dislanan.append({**kayit, "reasons": neden})
        else:
            secilen.append(kayit)

    secilen.sort(key=lambda x: -x["volume_usd_24h"])
    kirpilan = secilen[MAX_SYMBOLS:]
    secilen = secilen[:MAX_SYMBOLS]
    for k in kirpilan:
        dislanan.append({**k, "reasons": [f"hacim sırasında ilk {MAX_SYMBOLS} dışında"]})

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "universe_5m.json").write_text(json.dumps({
        "selected": secilen, "excluded": dislanan,
        "n_selected": len(secilen), "n_excluded": len(dislanan),
        "gates": {"min_volume_usd_24h": MIN_VOLUME_USD,
                  "max_onboard_date": MAX_ONBOARD,
                  "max_symbols": MAX_SYMBOLS,
                  "quotes": list(QUOTES)},
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": ("Kapılar ÖLÇÜLEBİLİRLİK içindir, kârlılık değil. Dışlanan her "
                 "market nedeniyle kayıtlıdır."),
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"seçilen {len(secilen)} · dışlanan {len(dislanan)}")
    for k in secilen:
        print(f"  {k['symbol']:16s} {k['volume_usd_24h']/1e6:9.1f}M$  {k['onboard']}")
    print(",".join(k["symbol"] for k in secilen))
    return 0


if __name__ == "__main__":
    sys.exit(main())
