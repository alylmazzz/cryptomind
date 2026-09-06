#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KANIT DURUMU — sürekli, otomatik, yalın.

  python scripts/cm_evidence.py                    # tablo
  python scripts/cm_evidence.py --gun 7            # son 7 gün
  python scripts/cm_evidence.py --json             # makine okunur
  python scripts/cm_evidence.py --migrate          # eski defteri kanıt defterine aktar (bir kez)

Cevapladığı soru: **"hangi sleeve kanıtlanmaya ne kadar yakın, kaç işlem daha lazım?"**

Neden bu sayı önemli: sistemin kanıtlanmış kenarı olan bir sleeve'i yok ve kanıt toplamak
komisyon maliyeti taşıyor (ölçüm: ~0,5 $/gün). "Ne kadar daha ödeyeceğim?" sorusunun
cevabı `kalan_islem_t2` sütunudur. Cevap YOKSA (—) etki büyüklüğü sıfıra çok yakındır:
o sleeve hiçbir örneklemde kanıtlanmaz, beklemek para yakmaktır.

Bellek: kanıt defteri AKIŞLA okunur; dosya ne kadar büyürse büyüsün sabit yer tutar.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agi_trader.learn import evidence as EV  # noqa: E402


def _utf8():
    if hasattr(sys.stdout, "buffer") and getattr(sys.stdout, "encoding", "").lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def _tablo(baslik: str, d: dict, en_az: int = 3, tr: dict | None = None) -> None:
    satir = [(k, v) for k, v in d.items() if v["n"] >= en_az]
    if not satir:
        return
    satir.sort(key=lambda kv: -(kv[1]["ort_pct"] or 0))
    print(f"\n### {baslik}")
    print(f"{'':24s} {'n':>4s} {'ort %':>9s} {'t':>7s} {'kazanma':>8s} {'net $':>8s} {'t>2 için':>9s}")
    for k, v in satir:
        ad = (tr or {}).get(k, k)
        kalan = v.get("kalan_islem_t2")
        kalan_s = "—" if kalan is None else (f"+{kalan}" if kalan > 0 else "ULAŞTI")
        isaret = ""
        if v["n"] < 8:
            isaret = "  (n<8 — verdikt yok)"
        elif v["t"] >= 2.0:
            isaret = "  ✅ KANIT"
        elif v["t"] <= -2.0:
            isaret = "  ⛔ KANITLI ZARAR"
        print(f"{str(ad)[:24]:24s} {v['n']:4d} {v['ort_pct']:+9.4f} {v['t']:+7.2f} "
              f"{(v['kazanma'] if v['kazanma'] is not None else 0):8.2f} {v['net_usd']:+8.3f} {kalan_s:>9s}{isaret}")


SEANS_TR = {"0": "00-04 UTC", "4": "04-08 UTC", "8": "08-12 UTC",
            "12": "12-16 UTC", "16": "16-20 UTC", "20": "20-24 UTC"}
EMIR_TR = {"m": "maker", "t": "taker"}
MOD_TR = {"F": "FIXED_TARGET", "P": "PARTIAL_AND_RUN", "D": "DYNAMIC_PEAK"}


def migrate(runs: str, tag: str) -> int:
    """Eski `runner_*.json` defterini kanıt defterine bir kez aktar (idempotent değil —
    yalnız kanıt defteri BOŞSA çalışır, yoksa çift sayım olurdu)."""
    p = EV.yol(runs, tag)
    if p.exists() and p.stat().st_size > 0:
        print(f"kanıt defteri zaten var ({p}) — aktarım atlandı (çift sayım olmasın)")
        return 0
    st = Path(runs)
    if not st.is_absolute():
        st = ROOT / runs
    src = st / "live" / f"runner_{tag}.json"
    if not src.exists():
        print(f"kaynak yok: {src}")
        return 0
    d = json.loads(src.read_text(encoding="utf-8"))
    n = 0
    for t in d.get("trades") or []:
        rej = ((t.get("decision") or {}).get("regime")) if isinstance(t.get("decision"), dict) else None
        if EV.kaydet(t, runs, tag=tag, rejim=rej):
            n += 1
    print(f"aktarıldı: {n} işlem → {p}")
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--tag", default="0_mexc")
    ap.add_argument("--gun", type=float, default=0.0, help="yalnız son N gün (0 = tümü)")
    ap.add_argument("--en-az", type=int, default=3, help="tabloya girmek için asgari n")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--migrate", action="store_true")
    ap.add_argument("--out", default="", help="raporu bu JSON dosyasına da yaz")
    a = ap.parse_args()

    if a.migrate:
        migrate(a.runs, a.tag)

    since = (time.time() - a.gun * 86400) if a.gun else None
    o = EV.ozet(a.runs, tag=a.tag, since=since, min_n=1)

    if a.json:
        print(json.dumps(o, ensure_ascii=False, indent=2))
        return 0

    g = o["genel"]
    if not g["n"]:
        print("kanıt defteri boş. İlk kez kuruyorsan: python scripts/cm_evidence.py --migrate")
        return 0
    ilk, son = o.get("ilk_ts"), o.get("son_ts")
    gun = (son - ilk) / 86400 if (ilk and son and son > ilk) else 0
    print("=" * 92)
    print(f"KANIT DURUMU · {g['n']} işlem · {gun:.1f} gün · net {g['net_usd']:+.3f} $ · "
          f"beklenti {g['ort_pct']:+.4f} %/işlem (t {g['t']:+.2f})")
    if gun > 0:
        print(f"kanıt toplama hızı: {g['n']/gun:.1f} işlem/gün · maliyet ≈ "
              f"{abs(g['net_usd'])/gun:.2f} $/gün")
    print("=" * 92)

    _tablo("SLEEVE", o["sleeve"], a.en_az)
    _tablo("SEANS (açılış UTC bloğu)", o["seans"], a.en_az, SEANS_TR)
    _tablo("EMİR TİPİ", o["emir"], a.en_az, EMIR_TR)
    _tablo("ÇIKIŞ MODU", o["cikis_modu"], a.en_az, MOD_TR)
    _tablo("PARİTE (n≥5)", o["parite"], max(5, a.en_az))
    print("  NOT: ~30 parite taranıyor — çoklu test yüzünden birinin t>2 çıkması ŞANSLA da\n"
          "       beklenir. Parite tablosu KEŞİF içindir, kanıt kapısı DEĞİLDİR.")

    # karar özeti
    # ASGARİ ÖRNEKLEM ŞART: t, n=2'de anlamsızdır — tek kazanan işlem |t|'yi uçurur.
    # Bu kapı olmadan rapor SAHTE KANIT üretir (ilk sürümde `squeeze_breakout` n=1 ile
    # "KANITLANMIŞ kâr" göründü). Kanıt verdikti için en az MIN_N işlem gerekir.
    MIN_N = 8
    kanit = [k for k, v in o["sleeve"].items() if v["n"] >= MIN_N and v["t"] >= 2.0 and v["ort_pct"] > 0]
    zarar = [k for k, v in o["sleeve"].items() if v["n"] >= MIN_N and v["t"] <= -2.0]
    yakin = sorted(((v.get("kalan_islem_t2"), k) for k, v in o["sleeve"].items()
                    if v["n"] >= 5 and v["ort_pct"] > 0 and v.get("kalan_islem_t2") is not None),
                   key=lambda x: x[0])[:3]
    print("\n" + "=" * 92)
    print("KARAR ÖZETİ")
    print("=" * 92)
    print(f"  KANITLANMIŞ kâr (t ≥ +2) : {kanit if kanit else 'YOK'}")
    print(f"  KANITLANMIŞ zarar (t ≤ −2): {zarar if zarar else 'YOK'}")
    if yakin:
        print("  kanıta en yakın adaylar  :")
        for kalan, k in yakin:
            v = o["sleeve"][k]
            gunluk = (g["n"] / gun) if gun else 0
            pay = (v["n"] / g["n"]) if g["n"] else 0
            sure = (kalan / (gunluk * pay)) if (gunluk and pay) else None
            print(f"     {k:22s} n={v['n']:3d} t={v['t']:+.2f} → +{kalan} işlem"
                  + (f" ≈ {sure:.0f} gün" if sure and sure < 3650 else ""))
    else:
        print("  kanıta yakın aday        : YOK — hiçbir sleeve'in etki büyüklüğü ölçülebilir değil")

    if a.out:
        Path(a.out).write_text(json.dumps(o, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nkayıt: {a.out}")
    return 0


if __name__ == "__main__":
    _utf8()
    sys.exit(main())
