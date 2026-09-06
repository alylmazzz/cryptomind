#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HABER ETKİ RAPORU — hangi tür haber, hangi paritede, hangi yönde, yüzde kaç?

  python scripts/cm_news_impact.py                 # 4 saatlik ufuk
  python scripts/cm_news_impact.py --ufuk 1h
  python scripts/cm_news_impact.py --prior         # elle yazılmış varsayımlar tutuyor mu
  python scripts/cm_news_impact.py --json

İKİ SÜTUNU KARIŞTIRMA:
  büyüklük_z : haber sonrası hareketin NORMALE oranı (1,0 = normal oynaklık).
               Yüksek olması "kâr edilir" demek DEĞİL — yalnız "hareket var" demek.
  yön_t      : hareketin YÖNÜ öngörülebilir mi. Asıl para bu sütunda.
Bir kategori büyük hareket yapıp yönü öngörülemez olabilir; o durumda ona göre
AL/SAT yapmak kayıptır — yalnız pozisyon küçültülür / girilmez.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agi_trader.sentiment.news_impact import HaberEtkiMotoru, MIN_GOZLEM, T_ESIK  # noqa: E402


def _utf8():
    if hasattr(sys.stdout, "buffer") and getattr(sys.stdout, "encoding", "").lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--tag", default="0_mexc")
    ap.add_argument("--ufuk", default="4h", choices=("5m", "1h", "4h", "24h"))
    ap.add_argument("--prior", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    m = HaberEtkiMotoru(a.runs, a.tag)
    o = m.ozet(a.ufuk)
    if a.json:
        o["prior_karsilastirma"] = m.prior_karsilastir(a.ufuk)
        print(json.dumps(o, ensure_ascii=False, indent=2))
        return 0

    print("=" * 100)
    print(f"HABER ETKİSİ · ufuk {a.ufuk} · {o['toplam_gozlem']} çözülmüş gözlem · "
          f"{o['bekleyen']} bekleyen")
    print(f"kapı: n ≥ {MIN_GOZLEM} VE |t| ≥ {T_ESIK} → 'ölçüldü'. Altındakiler VARSAYIM olarak kalır.")
    print("=" * 100)
    if not o["kategori"]:
        print("\nHenüz yeterli gözlem yok. Motor canlıda çalışıyor; gözlemler ufuk dolunca çözülür")
        print("(5 dk / 1 sa / 4 sa / 24 sa). İlk anlamlı tablo için ~kategori başına 12 gözlem gerekir.")
        return 0

    print(f"\n### KATEGORİ (haber TÜRÜ)")
    print(f"{'kategori':18s} {'n':>4s} {'yön %':>8s} {'t':>7s} {'yukarı':>7s} "
          f"{'medyan%':>8s} {'p90%':>8s} {'büyüklük':>9s}  verdikt")
    for k, v in sorted(o["kategori"].items(), key=lambda kv: -abs(kv[1]["yon_ort_pct"])):
        vd = []
        if v["yon_olculdu"]:
            vd.append("✅ YÖN ÖLÇÜLDÜ")
        elif v["n"] >= MIN_GOZLEM:
            vd.append("yön öngörülemiyor")
        else:
            vd.append(f"n<{MIN_GOZLEM} — veri yok")
        if v["buyukluk_olculdu"]:
            vd.append("⚡ hareket normalden büyük")
        print(f"{k:18s} {v['n']:4d} {v['yon_ort_pct']:+8.3f} {v['yon_t']:+7.2f} "
              f"{v['yukari_orani']:7.2f} {v['medyan_pct']:+8.3f} {v['p90_pct']:+8.3f} "
              f"{v['buyukluk_z']:9.2f}  {' · '.join(vd)}")

    kp = o.get("kategori_parite") or {}
    if kp:
        print(f"\n### KATEGORİ × PARİTE (en çok gözlemli 20)")
        print(f"{'kategori | parite':32s} {'n':>4s} {'yön %':>8s} {'t':>7s} {'büyüklük':>9s}")
        for k, v in list(kp.items())[:20]:
            print(f"{k:32s} {v['n']:4d} {v['yon_ort_pct']:+8.3f} {v['yon_t']:+7.2f} {v['buyukluk_z']:9.2f}"
                  + ("  ✅" if v["yon_olculdu"] else ""))

    if a.prior:
        print(f"\n### ELLE YAZILMIŞ VARSAYIM (EVENT_PRIOR) vs ÖLÇÜM")
        print(f"{'kategori':18s} {'varsayım':>9s} {'ölçülen %':>10s} {'t':>7s} {'n':>4s}  sonuç")
        for r in m.prior_karsilastir(a.ufuk):
            if r["olculdu"]:
                s = "UYUMLU ✅" if r["uyumlu"] else "⛔ VARSAYIM YANLIŞ"
            else:
                s = "ölçülmedi — varsayım KANIT DEĞİL"
            print(f"{r['kategori']:18s} {r['varsayim_prior']:+9.2f} {r['olculen_pct']:+10.3f} "
                  f"{r['t']:+7.2f} {r['n']:4d}  {s}")

    olculen = [k for k, v in o["kategori"].items() if v["yon_olculdu"]]
    buyuk = [k for k, v in o["kategori"].items() if v["buyukluk_olculdu"] and not v["yon_olculdu"]]
    print("\n" + "=" * 100)
    print("KARAR ÖZETİ")
    print("=" * 100)
    print(f"  YÖNÜ ölçülmüş (işlem açılabilir)      : {olculen if olculen else 'YOK'}")
    print(f"  Yalnız BÜYÜKLÜK ölçülmüş (yön yok)    : {buyuk if buyuk else 'YOK'}")
    print("  → ikinci gruba göre AL/SAT YAPILMAZ; yalnız oynaklık beklenir (boyut küçültülür).")

    if a.out:
        o["prior_karsilastirma"] = m.prior_karsilastir(a.ufuk)
        Path(a.out).write_text(json.dumps(o, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nkayıt: {a.out}")
    return 0


if __name__ == "__main__":
    _utf8()
    sys.exit(main())
