#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EV KALİBRASYONU — hangi EV gerçekten öngörüyor?

  python scripts/cm_ev_calib.py            # tablo
  python scripts/cm_ev_calib.py --json     # makine okunur / cron

SORU (2026-09-06 rotasyon incelemesinden):
`_maybe_rotate` yeni adayın `ticket["ev_pct"]` değerine bakıyor. 182 işlemde ölçüldü:
vaadedilen +0,5745% vs gerçekleşen −0,1152%, korelasyon **+0,089 → öngörü gücü YOK**.
Komite ZATEN `ev_achievable_pct` de hesaplıyor (ölçülmüş sleeve MFE medyanıyla) ve
adayların çoğunda İŞARET TERS. Ama o alan hiç kaydedilmiyordu (0/222).

Artık ikisi de kanıt defterinde. Bu betik şunu ölçer:
  • korelasyon (Pearson + Spearman) — vaat ile gerçekleşen arasında bağ var mı?
  • kalibrasyon eğimi — gerçekleşen = a + b·vaat regresyonunda b.
      b ≈ 1  → kalibre;  b ≈ 0  → BİLGİ YOK;  b < 0  → TERS
  • desil tekdüzeliği — yüksek EV kovası gerçekten daha mı iyi?
  • KARAR TESTİ — "yalnız EV > eşik olanları alsaydık" beklentisi ne olurdu?

KAPI: n ≥ MIN_N ve eşleştirilmiş fark testinin %95 GA'sı 0'ı dışlamalı. Aksi hâlde
"HENÜZ AYIRT EDİLEMEDİ" denir — iki EV'den birini seçmek için kanıt yok demektir.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agi_trader.learn import evidence as EV  # noqa: E402

MIN_N = 50          # bu sayının altında hiçbir hüküm verilmez


def _utf8():
    if hasattr(sys.stdout, "buffer") and getattr(sys.stdout, "encoding", "").lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = st.mean(xs), st.mean(ys)
    sx, sy = st.pstdev(xs), st.pstdev(ys)
    if sx <= 0 or sy <= 0:
        return 0.0
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / n / (sx * sy)


def _spearman(xs, ys):
    def sira(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for k, i in enumerate(s):
            r[i] = k
        return r
    return _pearson(sira(xs), sira(ys))


def _regresyon(xs, ys):
    """gerçekleşen ≈ a + b·vaat → (b, a). Rotasyon kapısı bu HARİTAYI kullanır:
    ham EV'yi ölçülen ölçeğe indirmeden eşikle kıyaslamak elmayla armut kıyaslamaktır."""
    n = len(xs)
    if n < 3:
        return 0.0, 0.0
    mx, my = st.mean(xs), st.mean(ys)
    vx = sum((a - mx) ** 2 for a in xs)
    if vx <= 0:
        return 0.0, my
    b = sum((a - mx) * (c - my) for a, c in zip(xs, ys)) / vx
    return b, my - b * mx


def analiz(cift, ad):
    xs = [a for a, _ in cift]
    ys = [b for _, b in cift]
    n = len(cift)
    r = _pearson(xs, ys)
    # r'nin standart hatası (Fisher z)
    ci = None
    if n > 4:
        if abs(r) >= 0.999:
            # DEJENERE DURUM (testlerin yakaladığı hata): Fisher-z, r=±1'de tanımsızdır.
            # İlk sürüm burada ci=None bırakıyor, bu da `bilgi_var=False` demek oluyordu —
            # yani MÜKEMMEL öngörüye "bilgi yok" diyordu. Tam tersi doğru.
            ci = (round(r, 3), round(r, 3))
        else:
            z = 0.5 * math.log((1 + r) / (1 - r))
            se = 1 / math.sqrt(n - 3)
            ci = (round(math.tanh(z - 1.96 * se), 3), round(math.tanh(z + 1.96 * se), 3))
    d = {
        "ad": ad, "n": n,
        "vaat_ort": round(st.mean(xs), 4),
        "gercek_ort": round(st.mean(ys), 4),
        "yanlilik": round(st.mean(xs) - st.mean(ys), 4),
        "pearson": round(r, 3), "pearson_ci95": ci,
        "spearman": round(_spearman(xs, ys), 3),
        "kalibrasyon_egimi": round(_regresyon(xs, ys)[0], 4),
        "kalibrasyon_kesisim": round(_regresyon(xs, ys)[1], 4),
        "bilgi_var": bool(ci and ci[0] > 0),
    }
    # desiller
    srt = sorted(cift, key=lambda p: p[0])
    k = max(1, n // 5)
    d["besli"] = [round(st.mean([b for _, b in srt[i * k:(i + 1) * k]]), 4)
                  for i in range(5) if srt[i * k:(i + 1) * k]]
    d["tekduze"] = bool(len(d["besli"]) == 5 and
                        all(d["besli"][i] <= d["besli"][i + 1] + 1e-9 for i in range(4)))
    # karar testi: yalnız üst yarıyı alsak
    ust = [b for a, b in cift if a >= st.median(xs)]
    d["ust_yari_beklenti"] = round(st.mean(ust), 4) if ust else None
    d["tum_beklenti"] = round(st.mean(ys), 4)
    d["secim_kazanci"] = (round(d["ust_yari_beklenti"] - d["tum_beklenti"], 4)
                          if ust else None)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--tag", default="0_mexc")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    ev_c, eva_c, ikisi = [], [], []
    for r in EV.oku(a.runs, tag=a.tag):
        y = r.get("np")
        if y is None:
            continue
        if r.get("ev") is not None:
            ev_c.append((float(r["ev"]), float(y)))
        if r.get("eva") is not None:
            eva_c.append((float(r["eva"]), float(y)))
        if r.get("ev") is not None and r.get("eva") is not None:
            ikisi.append((float(r["ev"]), float(r["eva"]), float(y)))

    out = {"n_ev": len(ev_c), "n_eva": len(eva_c), "n_ikisi": len(ikisi), "min_n": MIN_N}
    if ev_c:
        out["ev_pct"] = analiz(ev_c, "ev_pct (PLAN hedefi)")
    if eva_c:
        out["ev_achievable"] = analiz(eva_c, "ev_achievable (ÖLÇÜLMÜŞ hedef)")

    # EŞLEŞTİRİLMİŞ karşılaştırma: aynı işlemlerde hangisi daha az yanılıyor?
    if len(ikisi) >= 10:
        h1 = [abs(x - y) for x, _, y in ikisi]        # ev'nin mutlak hatası
        h2 = [abs(z - y) for _, z, y in ikisi]        # eva'nın mutlak hatası
        d = [p - q for p, q in zip(h1, h2)]           # >0 → eva DAHA AZ yanılıyor
        se = (st.pstdev(d) or 1e-9) / math.sqrt(len(d))
        m = st.mean(d)
        out["eslestirilmis_hata_farki"] = {
            "n": len(d), "ort_fark": round(m, 4), "t": round(m / se, 2),
            "ci95": (round(m - 1.96 * se, 4), round(m + 1.96 * se, 4)),
            "eva_daha_iyi_sayisi": sum(1 for x in d if x > 0),
            "karar": ("ev_achievable DAHA İYİ" if m - 1.96 * se > 0 else
                      "ev_pct DAHA İYİ" if m + 1.96 * se < 0 else "HENÜZ AYIRT EDİLEMEDİ"),
        }

    # Rotasyon kapısının okuyacağı sadeleştirilmiş harita
    for anahtar, kaynak in (("ev_cal", "ev_pct"), ("eva_cal", "ev_achievable")):
        d = out.get(kaynak)
        if d and d["n"] >= MIN_N:
            out[anahtar] = {"egim": d["kalibrasyon_egimi"], "kesisim": d["kalibrasyon_kesisim"],
                            "n": d["n"], "r": d["pearson"]}
    yeterli = len(ikisi) >= MIN_N
    out["hazir"] = yeterli
    out["karar"] = (out.get("eslestirilmis_hata_farki") or {}).get("karar", "HENÜZ AYIRT EDİLEMEDİ")
    out["ozet"] = ("ölçüm hazır" if yeterli else
                   f"yetersiz örneklem: iki EV'si birden kayıtlı {len(ikisi)}/{MIN_N} işlem")

    if a.out:
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    if a.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    print("=" * 92)
    print(f"EV KALİBRASYONU · ev_pct kayıtlı {len(ev_c)} · ev_achievable kayıtlı {len(eva_c)} · "
          f"İKİSİ birden {len(ikisi)}")
    print(f"kapı: iki EV'si birden kayıtlı n ≥ {MIN_N} ve %95 GA 0'ı dışlamalı")
    print("=" * 92)
    for k in ("ev_pct", "ev_achievable"):
        d = out.get(k)
        if not d:
            print(f"\n### {k}: henüz kayıt yok")
            continue
        print(f"\n### {d['ad']}  (n={d['n']})")
        print(f"  vaat ort / gerçek ort : {d['vaat_ort']:+.4f} % / {d['gercek_ort']:+.4f} %  "
              f"(yanlılık {d['yanlilik']:+.4f})")
        print(f"  Pearson r             : {d['pearson']:+.3f}  GA95 {d['pearson_ci95']}")
        print(f"  Spearman              : {d['spearman']:+.3f}")
        print(f"  kalibrasyon eğimi     : {d['kalibrasyon_egimi']:+.3f}   "
              f"(1,0 = kalibre · 0 = bilgi yok · <0 = TERS)")
        print(f"  beşli (düşük→yüksek)  : {d['besli']}  {'tekdüze ✅' if d['tekduze'] else 'tekdüze DEĞİL'}")
        print(f"  üst yarıyı alsak      : {d['ust_yari_beklenti']} vs tümü {d['tum_beklenti']} "
              f"→ seçim kazancı {d['secim_kazanci']}")
        print(f"  BİLGİ VAR MI          : {'EVET' if d['bilgi_var'] else 'HAYIR (GA 0''ı içeriyor)'}")
    p = out.get("eslestirilmis_hata_farki")
    if p:
        print(f"\n### EŞLEŞTİRİLMİŞ HATA KARŞILAŞTIRMASI (aynı işlemler)")
        print(f"  n={p['n']} · |ev hatası| − |eva hatası| = {p['ort_fark']:+.4f} · t={p['t']:+.2f} · GA95 {p['ci95']}")
        print(f"  eva daha az yanılan: {p['eva_daha_iyi_sayisi']}/{p['n']}")
        print(f"  → {p['karar']}")
    print("\n" + "=" * 92)
    print(f"DURUM: {out['ozet']}")
    if not yeterli:
        print("Kanıt defteri her kapanışta doluyor; saatlik cron bu raporu tazeliyor.")
    return 0


if __name__ == "__main__":
    _utf8()
    sys.exit(main())
