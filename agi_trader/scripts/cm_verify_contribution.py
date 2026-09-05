#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KATKI DOĞRULAYICI — bir topluluk kurulumunun sisteme girip giremeyeceğine ÖLÇÜM karar verir.

    python scripts/cm_verify_contribution.py --sleeve benim_kurulumum --days 7
    python scripts/cm_verify_contribution.py --sleeve benim_kurulumum --days 7 --evidence

Dört aşama, hepsi geçilmeden bir üste çıkılmaz:

  1. YÜKLEME     — META şeması + `fire` imzası + ad çakışması.
  2. STATİK      — ağ/dosya erişimi, ileriye bakış (`shift(-n)`), tohumsuz rastgelelik,
                   global durum değişimi. Bunlar ölçümü GEÇERSİZ kılar, o yüzden önce bakılır.
  3. ATEŞLEME    — GERÇEK 1 dk veride tetiklenme oranı. Bu depoda ölçülmüş ders:
                   sıfıra yakın ya da aşırı yüksek oran, mantık hatasının İLK işaretidir
                   (ters-FVG "hiç aşılmamış + şimdi aşıldı" çelişkisiyle 1815 pencerede
                   0 kez ateşledi; gevşetilince %9,5 ateşleyip saf gürültü üretti).
  4. KENAR       — kurulumun KENDİ stop/hedefiyle ileri test, gidiş-dönüş maliyet düşülmüş.
                   Beklenti, t-istatistiği, bootstrap CI, alt-dönem tutarlılığı.

Verdikt:
    REDDEDİLDİ — yükleme/statik başarısız ya da ateşleme oranı imkânsız.
    GÖLGE      — yüklendi ve ölçüldü, kenar POZİTİF DEĞİL. Katkı yine de birleşebilir:
                 sinyal üretir, emir vermez, ölçülmeye devam eder. Bu bir ret değildir.
    KANIT VAR  — kenar pozitif ve maliyete dayanıklı. `--evidence` ile lifecycle kaydına
                 yazılır; PAPER'a terfi yine `lifecycle.gates()` kararıdır (n≥30, DSR>0,
                 PBO<0,5, CI alt sınırı>0, 2× maliyette pozitif, alt-dönem tutarlı).

Bu betik TEK BAŞINA terfi ettirmez. Kanıt üretir; kapıyı kod açar.
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _utf8():
    try:
        if hasattr(sys.stdout, "buffer") and (getattr(sys.stdout, "encoding", "") or "").lower() != "utf-8":
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────── 2) statik denetim
YASAK_MODUL = {"requests", "urllib", "urllib3", "httpx", "aiohttp", "socket", "ccxt",
               "subprocess", "os", "shutil", "pickle", "sqlite3", "threading", "multiprocessing"}
YASAK_CAGRI = {"open", "eval", "exec", "compile", "__import__", "input", "sleep"}


def statik_denetim(kaynak: str, modul_adi: str) -> List[str]:
    """Ölçümü GEÇERSİZ kılacak şeyleri ara. Bulgular ölçümden ÖNCE raporlanır."""
    bulgular: List[str] = []
    try:
        agac = ast.parse(kaynak)
    except SyntaxError as e:
        return [f"sözdizimi hatası: {e}"]

    for d in ast.walk(agac):
        if isinstance(d, ast.Import):
            for a in d.names:
                kok = a.name.split(".")[0]
                if kok in YASAK_MODUL:
                    bulgular.append(f"yasak import: {a.name} — katkı yalnız `f` sözlüğünden okur")
        elif isinstance(d, ast.ImportFrom):
            kok = (d.module or "").split(".")[0]
            if kok in YASAK_MODUL:
                bulgular.append(f"yasak import: {d.module} — katkı yalnız `f` sözlüğünden okur")
        elif isinstance(d, ast.Call):
            ad = getattr(d.func, "id", None) or getattr(d.func, "attr", None)
            if ad in YASAK_CAGRI:
                bulgular.append(f"yasak çağrı: {ad}()")
            # İLERİYE BAKIŞ: negatif shift geleceği bugüne taşır → ölçüm sahte çıkar.
            if ad == "shift":
                for arg in d.args:
                    if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                        bulgular.append("ileriye bakış: shift(-n) geleceği kullanır")
            if ad in ("random", "uniform", "randint", "choice", "normal", "rand"):
                bulgular.append(f"tohumsuz rastgelelik: {ad}() — ölçüm tekrarlanabilir olmalı")
        elif isinstance(d, ast.Global):
            bulgular.append(f"global durum değişimi: {', '.join(d.names)}")
    return bulgular


# ────────────────────────────────────────────────────── 3-4) ateşleme + kenar
def _ozellikler(df, p):
    from agi_trader.strategies import committee as CM, sleeves_fast as SF, sleeves_video as SV
    f = CM.fast_features(df, p)
    if not f.get("ok"):
        return None
    f["price"] = float(df["close"].iloc[-1])
    f = SF.extra_features(df, f, None, {"spread_bps": 2.0, "bid_depth_usd": 0.0, "ask_depth_usd": 0.0}, None)
    return SV.video_features(df, f)


def olc(sleeve: str, hist: Dict, p, maliyet_pct: float, ufuk_bar: int,
        pencere: int = 240, adim: int = 5) -> Dict:
    """Her pencerede kurulumu çalıştır; ateşlerse KENDİ stop/hedefiyle ileri test et."""
    from agi_trader.strategies import contrib as CB
    meta = CB.CONTRIB[sleeve]["meta"]
    pencere_n = 0
    ateslemeler: List[Dict] = []
    for (sym, tf), df in hist.items():
        if tf != "1m" or df is None or len(df) < pencere + ufuk_bar + 5:
            continue
        n = len(df)
        for i in range(pencere, n - ufuk_bar, adim):
            pencere_n += 1
            f = _ozellikler(df.iloc[i - pencere:i], p)
            if not f:
                continue
            trig = CB.fire_contrib_sleeves(f, [sleeve], p, allow_short=False, now_ts=None)
            if not trig:
                continue
            t = trig[0]
            giris = float(df["close"].iloc[i - 1])
            atr_abs = max(1e-12, float(f.get("atr_pct") or 0.3) / 100.0 * giris)
            stop = float(t.get("stop_hint") or (giris - 1.5 * atr_abs))
            hedef = float(t.get("target_hint") or (giris + 3.0 * atr_abs))
            if not (stop < giris < hedef):
                continue                       # tutarsız kurulum: ölçülemez
            ileri = df.iloc[i:i + ufuk_bar]
            cikis, sebep = float(ileri["close"].iloc[-1]), "ZAMAN"
            for j in range(len(ileri)):
                lo, hi = float(ileri["low"].iloc[j]), float(ileri["high"].iloc[j])
                if lo <= stop:                 # ALEYHE ÖNCE: aynı barda ikisi de değerse
                    cikis, sebep = stop, "STOP"; break      # stopun dolduğu varsayılır
                if hi >= hedef:
                    cikis, sebep = hedef, "HEDEF"; break
            brut = (cikis / giris - 1.0) * 100.0
            ateslemeler.append({"symbol": sym, "net_pct": brut - maliyet_pct,
                                "brut_pct": brut, "sebep": sebep,
                                "stop_pct": (giris - stop) / giris * 100.0,
                                "hedef_pct": (hedef / giris - 1.0) * 100.0})
    return {"pencere": pencere_n, "atesleme": len(ateslemeler), "kayitlar": ateslemeler,
            "exit_mode": meta["exit_mode"]}


def _istatistik(x: List[float]) -> Dict:
    n = len(x)
    if n == 0:
        return {"n": 0}
    ort = sum(x) / n
    var = sum((v - ort) ** 2 for v in x) / n
    sd = math.sqrt(var)
    t = (ort / (sd / math.sqrt(n))) if (sd > 0 and n > 1) else 0.0
    # bootstrap CI (tohumlu → tekrarlanabilir)
    import random
    rnd = random.Random(7)
    ortalamalar = sorted(sum(rnd.choice(x) for _ in range(n)) / n for _ in range(2000))
    yari = n // 2
    return {"n": n, "ort": round(ort, 4), "sapma": round(sd, 4), "t": round(t, 2),
            "ci95": [round(ortalamalar[50], 4), round(ortalamalar[1949], 4)],
            "kazanma": round(sum(1 for v in x if v > 0) / n, 3),
            "ilk_yari": round(sum(x[:yari]) / yari, 4) if yari else None,
            "ikinci_yari": round(sum(x[yari:]) / (n - yari), 4) if yari else None}


def main() -> int:
    _utf8()
    ap = argparse.ArgumentParser(description="Topluluk katkısı doğrulayıcı")
    ap.add_argument("--sleeve", required=True, help="doğrulanacak katkı adı (META['name'])")
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--symbols", default="BTC/USDT,ETH/USDT,SOL/USDT,DOGE/USDT,AVAX/USDT")
    ap.add_argument("--venue", default="mexc")
    ap.add_argument("--cost-pct", type=float, default=0.14, help="gidiş-dönüş maliyet (%%)")
    ap.add_argument("--step", type=int, default=5, help="pencere adımı (bar)")
    ap.add_argument("--evidence", action="store_true", help="kanıtı lifecycle kaydına yaz")
    ap.add_argument("--json", default="", help="sonucu bu dosyaya yaz")
    a = ap.parse_args()

    from agi_trader.strategies import contrib as CB, sleeves_fast as SF, committee as CM

    print("═" * 78)
    print(f"KATKI DOĞRULAYICI · {a.sleeve} · {a.days} gün · {a.venue}")
    print("═" * 78)

    # ── 1) yükleme ────────────────────────────────────────────────────────
    print("\n[1/4] YÜKLEME")
    if CB.LOAD_ERRORS:
        for e in CB.LOAD_ERRORS:
            print(f"  ✗ {e['modul']}: " + "; ".join(e["hatalar"]))
    if a.sleeve not in CB.CONTRIB:
        print(f"  ✗ '{a.sleeve}' yüklenmedi. Yüklenenler: {CB.all_sleeves() or '(yok)'}")
        print("\nVERDİKT: REDDEDİLDİ — yükleme aşaması geçilemedi.")
        return 2
    meta = CB.CONTRIB[a.sleeve]["meta"]
    print(f"  ✓ yüklendi · yazar {meta['author']} · çıkış {meta['exit_mode']} · "
          f"zaman-stop {meta['time_stop_min']} dk · rejim {', '.join(meta['regimes'])}")
    print(f"    kaynak : {meta['source']}")
    print(f"    iddia  : {meta['claim']}")
    print(f"    kanıtı : {meta['claim_evidence']}")

    # ── 2) statik ─────────────────────────────────────────────────────────
    print("\n[2/4] STATİK DENETİM")
    src = (Path(CB.__path__[0]) / f"{CB.CONTRIB[a.sleeve]['modul']}.py").read_text(encoding="utf-8")
    bulgular = statik_denetim(src, a.sleeve)
    if bulgular:
        for b in bulgular:
            print(f"  ✗ {b}")
        print("\nVERDİKT: REDDEDİLDİ — statik denetim ölçümü geçersiz kılacak bulgular buldu.")
        return 2
    print("  ✓ ağ/dosya erişimi yok · ileriye bakış yok · tohumsuz rastgelelik yok · global yok")

    # ── 3) ateşleme oranı ────────────────────────────────────────────────
    print(f"\n[3/4] GERÇEK VERİDE ATEŞLEME ({a.days} gün, {a.venue})")
    from agi_trader.auto.replay import HistoryFetcher
    syms = [s.strip() for s in a.symbols.split(",") if s.strip()]
    end_ms = int(time.time() * 1000)
    span = int(a.days * 24 * 3600 * 1000)
    hf = HistoryFetcher(a.venue)
    hist = {}
    for s in syms:
        try:
            hist[(s, "1m")] = hf.fetch(s, "1m", end_ms - span, end_ms)
            print(f"  {s:12s} {len(hist[(s,'1m')])} bar")
        except Exception as e:
            print(f"  {s:12s} veri alınamadı: {type(e).__name__}: {e}")
    if not hist:
        print("\nVERDİKT: ÖLÇÜLEMEDİ — veri yok.")
        return 3

    p = CM.CommitteeParams()
    ufuk_bar = max(5, int(meta["time_stop_min"]))
    r = olc(a.sleeve, hist, p, a.cost_pct, ufuk_bar, adim=a.step)
    oran = r["atesleme"] / max(1, r["pencere"]) * 100.0
    print(f"  pencere {r['pencere']} · ateşleme {r['atesleme']} · oran %{oran:.3f}")
    if r["atesleme"] == 0:
        print("  ✗ HİÇ ateşlemedi — koşullar aynı anda sağlanamıyor olabilir "
              "(bu depoda ters-FVG tam olarak böyle bir mantık çelişkisi taşıyordu).")
        print("\nVERDİKT: REDDEDİLDİ — ölçülemeyen kurulum sisteme giremez.")
        return 2
    if oran > 15.0:
        print(f"  ✗ oran %{oran:.1f} — aşırı yüksek; kurulum seçici değil, gürültü üretiyor.")
        print("\nVERDİKT: REDDEDİLDİ — koşulları sıkılaştırın ve yeniden ölçün.")
        return 2
    print("  ✓ ateşleme oranı makul bandda (%0–15)")

    # ── 4) kenar ─────────────────────────────────────────────────────────
    print(f"\n[4/4] KENAR (maliyet %{a.cost_pct} düşülmüş, kurulumun kendi stop/hedefiyle)")
    net = [k["net_pct"] for k in r["kayitlar"]]
    st = _istatistik(net)
    sebepler: Dict[str, int] = {}
    for k in r["kayitlar"]:
        sebepler[k["sebep"]] = sebepler.get(k["sebep"], 0) + 1
    print(f"  n {st['n']} · ortalama net %{st['ort']} · t {st['t']} · CI95 {st['ci95']} · "
          f"kazanma %{st['kazanma']*100:.1f}")
    print(f"  alt-dönem: ilk yarı %{st['ilk_yari']} · ikinci yarı %{st['ikinci_yari']}")
    print(f"  çıkış sebepleri: {sebepler}")

    ci_alt_poz = st["ci95"][0] > 0
    tutarli = bool(st["ilk_yari"] is not None and st["ilk_yari"] > 0 and st["ikinci_yari"] > 0)
    iki_kat = _istatistik([k["brut_pct"] - 2.0 * a.cost_pct for k in r["kayitlar"]])["ort"]
    print(f"  2× maliyette beklenti: %{iki_kat}")

    kapilar = {"beklenti_pozitif": st["ort"] > 0, "ci_alt_sinir_pozitif": ci_alt_poz,
               "iki_kat_maliyette_pozitif": iki_kat > 0, "alt_donem_tutarli": tutarli,
               "n_yeterli": st["n"] >= 30}
    for k, v in kapilar.items():
        print(f"    {'✓' if v else '✗'} {k}")

    sonuc = {"sleeve": a.sleeve, "meta": meta, "gun": a.days, "venue": a.venue,
             "pencere": r["pencere"], "atesleme": r["atesleme"], "atesleme_orani_pct": round(oran, 3),
             "istatistik": st, "iki_kat_maliyet_beklenti": iki_kat, "kapilar": kapilar,
             "cikis_sebepleri": sebepler, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    print()
    if all(kapilar.values()):
        print("VERDİKT: KANIT VAR — kenar pozitif ve maliyete dayanıklı.")
        print("  Terfi OTOMATİK DEĞİLDİR: lifecycle.gates() ayrıca DSR>0 ve PBO<0,5 ister;")
        print("  bunlar `scripts/cm_replay.py --evidence` ile üretilir.")
        if a.evidence:
            from agi_trader.strategies.lifecycle import Lifecycle
            lc = Lifecycle()
            lc.record_evidence(a.sleeve, {
                "oos_expectancy": st["ort"], "ci_lower": st["ci95"][0],
                "expectancy_cost_x2": iki_kat, "subperiod_consistent": tutarli,
                "n_trades": st["n"], "source": "contrib_verify"})
            g = lc.gates(a.sleeve)
            print(f"  kanıt yazıldı · lifecycle kapıları: {'GEÇTİ' if g['passed'] else 'GEÇMEDİ'} → {g['checks']}")
    else:
        print("VERDİKT: GÖLGE — kenar kanıtlanmadı.")
        print("  Bu bir RET DEĞİLDİR: katkı birleşebilir, sinyal üretir, EMİR VERMEZ ve")
        print("  ölçülmeye devam eder. Kanıt pozitife dönerse terfi yolu açıktır.")

    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(sonuc, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
