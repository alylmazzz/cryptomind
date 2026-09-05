#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KATKI DOĞRULAYICI — bir topluluk kurulumunun sisteme girip giremeyeceğine ÖLÇÜM karar verir.

    python scripts/cm_verify_contribution.py --sleeve benim_kurulumum --days 7
    python scripts/cm_verify_contribution.py --sleeve benim_kurulumum --days 7 --evidence

Dört aşama, hepsi geçilmeden bir üste çıkılmaz:

  1. YÜKLEME     — META şeması + `fire` imzası + ad çakışması. `fire` beşinci parametre
                   `df` tanımlarsa kendi göstergesini hesaplayabilir (DI±, MACD, SAR …).
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
                    bulgular.append(f"yasak import: {a.name} — katkı yalnız `f` ve `df`'den okur, "
                                    f"dış dünyaya erişemez")
        elif isinstance(d, ast.ImportFrom):
            kok = (d.module or "").split(".")[0]
            if kok in YASAK_MODUL:
                bulgular.append(f"yasak import: {d.module} — katkı yalnız `f` ve `df`'den okur, "
                                f"dış dünyaya erişemez")
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
    tutarsiz: List[Dict] = []
    for (sym, tf), df in hist.items():
        if tf != "1m" or df is None or len(df) < pencere + ufuk_bar + 5:
            continue
        n = len(df)
        for i in range(pencere, n - ufuk_bar, adim):
            pencere_n += 1
            pencere_df = df.iloc[i - pencere:i]
            f = _ozellikler(pencere_df, p)
            if not f:
                continue
            trig = CB.fire_contrib_sleeves(f, [sleeve], p, allow_short=False, now_ts=None,
                                           df=pencere_df)
            if not trig:
                continue
            t = trig[0]
            giris = float(df["close"].iloc[i - 1])
            atr_abs = max(1e-12, float(f.get("atr_pct") or 0.3) / 100.0 * giris)
            stop = float(t.get("stop_hint") or (giris - 1.5 * atr_abs))
            hedef = float(t.get("target_hint") or (giris + 3.0 * atr_abs))
            if not (stop < giris < hedef):
                # SESSİZ ATLAMA YASAK: "hiç ateşlemedi" ile "ateşledi ama kurulum
                # tutarsız" AYRI şeylerdir. İlki kuralın nadirliğini, ikincisi KODUN
                # kusurunu gösterir. 2026-09-05'te ClucMay portu tam olarak bu yüzden
                # "hiç ateşlemedi" göründü: 28 ateşlemenin 28'inde stop girişin
                # ÜSTÜNDEYDİ ve hepsi burada sessizce düşüyordu.
                tutarsiz.append({"symbol": sym, "idx": i,
                                 "neden": ("stop ≥ giriş" if stop >= giris else
                                           "hedef ≤ giriş" if hedef <= giris else "?"),
                                 "giris": giris, "stop": stop, "hedef": hedef})
                continue
            ileri = df.iloc[i:i + ufuk_bar]
            cikis, sebep = float(ileri["close"].iloc[-1]), "ZAMAN"
            for j in range(len(ileri)):
                lo, hi = float(ileri["low"].iloc[j]), float(ileri["high"].iloc[j])
                if lo <= stop:                 # ALEYHE ÖNCE: aynı barda ikisi de değerse
                    cikis, sebep = stop, "STOP"; break      # stopun dolduğu varsayılır
                if hi >= hedef:
                    cikis, sebep = hedef, "HEDEF"; break
            brut = (cikis / giris - 1.0) * 100.0
            ateslemeler.append({"symbol": sym, "idx": i, "net_pct": brut - maliyet_pct,
                                "brut_pct": brut, "sebep": sebep,
                                "stop_pct": (giris - stop) / giris * 100.0,
                                "hedef_pct": (hedef / giris - 1.0) * 100.0})
    return {"pencere": pencere_n, "atesleme": len(ateslemeler), "kayitlar": ateslemeler,
            "tutarsiz": tutarsiz, "exit_mode": meta["exit_mode"]}


def _bagimsiz(kayitlar: List[Dict], ufuk_bar: int) -> List[Dict]:
    """Örtüşmeyen alt küme — ETKİN örneklem.

    Doğrulayıcı her `adim` barda bir pencere açar; ateşlerse `ufuk_bar` boyunca ileri
    test eder. adım 5 / ufuk 240 iken ardışık iki ateşleme ileri pencerenin %98'ini
    PAYLAŞIR: aynı ticaret onlarca kez sayılır, nominal n şişer ve |t| olduğundan büyük
    çıkar. Burada bir işlem, aynı paritede bir öncekinin ufku BİTTİKTEN sonra sayılır."""
    out: List[Dict] = []
    son: Dict[str, int] = {}
    for k in sorted(kayitlar, key=lambda x: (x["symbol"], x["idx"])):
        s_ = k["symbol"]
        if s_ not in son or k["idx"] >= son[s_]:
            out.append(k)
            son[s_] = k["idx"] + ufuk_bar
    return out


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


def _yukle_ve_statik(sleeve: str, CB) -> Optional[Dict]:
    """Aşama 1–2. Geçerse META döner, geçmezse None (sebep basılmıştır)."""
    print(f"\n[1/4] YÜKLEME · {sleeve}")
    if sleeve not in CB.CONTRIB:
        print(f"  ✗ '{sleeve}' yüklenmedi. Yüklenenler: {CB.all_sleeves() or '(yok)'}")
        for e in CB.LOAD_ERRORS:
            print(f"    ✗ {e['modul']}: " + "; ".join(e["hatalar"]))
        return None
    meta = CB.CONTRIB[sleeve]["meta"]
    print(f"  ✓ yüklendi · yazar {meta['author']} · çıkış {meta['exit_mode']} · "
          f"zaman-stop {meta['time_stop_min']} dk · rejim {', '.join(meta['regimes'])}"
          + (" · kendi göstergesini hesaplıyor (df)" if CB.CONTRIB[sleeve].get("df_ister") else ""))
    print(f"    kaynak : {meta['source']}")
    print(f"    iddia  : {meta['claim']}")
    print(f"    kanıtı : {meta['claim_evidence']}")

    print(f"\n[2/4] STATİK DENETİM · {sleeve}")
    src = (Path(CB.__path__[0]) / f"{CB.CONTRIB[sleeve]['modul']}.py").read_text(encoding="utf-8")
    bulgular = statik_denetim(src, sleeve)
    if bulgular:
        for b in bulgular:
            print(f"  ✗ {b}")
        return None
    print("  ✓ ağ/dosya erişimi yok · ileriye bakış yok · tohumsuz rastgelelik yok · global yok")
    return meta


def _olc_ve_karar(sleeve: str, meta: Dict, hist: Dict, p_, a) -> Dict:
    """Aşama 3–4 + verdikt. Dönen sözlük özet tabloya girer."""
    from agi_trader.strategies.lifecycle import Lifecycle

    print(f"\n[3/4] GERÇEK VERİDE ATEŞLEME · {sleeve}")
    ufuk_bar = max(5, int(meta["time_stop_min"]))
    r = olc(sleeve, hist, p_, a.cost_pct, ufuk_bar, adim=a.step)
    oran = r["atesleme"] / max(1, r["pencere"]) * 100.0
    print(f"  pencere {r['pencere']} · ateşleme {r['atesleme']} · oran %{oran:.3f}"
          + (f" · TUTARSIZ {len(r['tutarsiz'])}" if r["tutarsiz"] else ""))
    if r["tutarsiz"]:
        nedenler: Dict[str, int] = {}
        for x in r["tutarsiz"]:
            nedenler[x["neden"]] = nedenler.get(x["neden"], 0) + 1
        print(f"  ⚠ {len(r['tutarsiz'])} kurulum TUTARSIZ olduğu için ölçülemedi: {nedenler}")
        print("    Bu kuralın nadirliği DEĞİL, kurulumun kendi stop/hedefinin kusurudur.")

    ortak = {"sleeve": sleeve, "pencere": r["pencere"], "atesleme": r["atesleme"],
             "atesleme_orani_pct": round(oran, 3), "gun": a.days, "venue": a.venue,
             "meta": meta, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    if r["atesleme"] == 0:
        if r["tutarsiz"]:
            print(f"  ✗ {len(r['tutarsiz'])} kez ateşledi ama HİÇBİRİ kurulabilir kurulum "
                  "üretmedi — stop/hedef tutarsız. Kodu düzeltip yeniden ölçün.")
            print(f"\nVERDİKT [{sleeve}]: REDDEDİLDİ — kurulum tutarsız (kural nadirliği değil).")
            return {**ortak, "verdikt": "REDDEDİLDİ", "sebep": "kurulum tutarsız (stop/hedef)",
                    "tutarsiz_n": len(r["tutarsiz"])}
        print("  ✗ HİÇ ateşlemedi — koşullar aynı anda sağlanamıyor olabilir "
              "(bu depoda ters-FVG tam olarak böyle bir mantık çelişkisi taşıyordu).")
        print(f"\nVERDİKT [{sleeve}]: REDDEDİLDİ — ölçülemeyen kurulum sisteme giremez.")
        return {**ortak, "verdikt": "REDDEDİLDİ", "sebep": "hiç ateşlemedi"}
    if oran > 15.0:
        print(f"  ✗ oran %{oran:.1f} — aşırı yüksek; kurulum seçici değil, gürültü üretiyor.")
        print(f"\nVERDİKT [{sleeve}]: REDDEDİLDİ — koşulları sıkılaştırın ve yeniden ölçün.")
        return {**ortak, "verdikt": "REDDEDİLDİ", "sebep": f"ateşleme oranı %{oran:.1f} > %15"}
    print("  ✓ ateşleme oranı makul bandda (%0–15)")

    print(f"\n[4/4] KENAR · {sleeve} (maliyet %{a.cost_pct} düşülmüş, kendi stop/hedefiyle)")
    # ETKİN ÖRNEKLEM: örtüşen ileri pencereler aynı ticareti defalarca sayar ve |t|'yi
    # şişirir. İstatistik ve KAPILAR örtüşmeyen alt kümede hesaplanır.
    bagimsiz = _bagimsiz(r["kayitlar"], ufuk_bar)
    sisme = len(r["kayitlar"]) / max(1, len(bagimsiz))
    print(f"  örneklem: nominal {len(r['kayitlar'])} → ETKİN {len(bagimsiz)} "
          f"(örtüşme şişmesi {sisme:.1f}×) — istatistik etkin küme üzerinden")
    net = [k["net_pct"] for k in bagimsiz]
    st = _istatistik(net)
    sebepler: Dict[str, int] = {}
    for k in bagimsiz:
        sebepler[k["sebep"]] = sebepler.get(k["sebep"], 0) + 1
    print(f"  n {st['n']} · ortalama net %{st['ort']} · t {st['t']} · CI95 {st['ci95']} · "
          f"kazanma %{st['kazanma'] * 100:.1f}")
    print(f"  alt-dönem: ilk yarı %{st['ilk_yari']} · ikinci yarı %{st['ikinci_yari']}")
    print(f"  çıkış sebepleri: {sebepler}")

    tutarli = bool(st["ilk_yari"] is not None and st["ilk_yari"] > 0 and st["ikinci_yari"] > 0)
    iki_kat = _istatistik([k["brut_pct"] - 2.0 * a.cost_pct for k in bagimsiz])["ort"]
    print(f"  2× maliyette beklenti: %{iki_kat}")
    kapilar = {"beklenti_pozitif": st["ort"] > 0, "ci_alt_sinir_pozitif": st["ci95"][0] > 0,
               "iki_kat_maliyette_pozitif": iki_kat > 0, "alt_donem_tutarli": tutarli,
               "n_yeterli": st["n"] >= 30}
    for k, v in kapilar.items():
        print(f"    {'✓' if v else '✗'} {k}")

    sonuc = {**ortak, "istatistik": st, "iki_kat_maliyet_beklenti": iki_kat,
             "kapilar": kapilar, "cikis_sebepleri": sebepler,
             "nominal_n": len(r["kayitlar"]), "etkin_n": len(bagimsiz),
             "ortusme_sismesi": round(sisme, 2)}

    if all(kapilar.values()):
        print(f"\nVERDİKT [{sleeve}]: KANIT VAR — kenar pozitif ve maliyete dayanıklı.")
        print("  Terfi OTOMATİK DEĞİLDİR: lifecycle.gates() ayrıca DSR>0 ve PBO<0,5 ister;")
        print("  bunlar `scripts/cm_replay.py --evidence` ile üretilir.")
        if a.evidence:
            lc = Lifecycle()
            lc.record_evidence(sleeve, {
                "oos_expectancy": st["ort"], "ci_lower": st["ci95"][0],
                "expectancy_cost_x2": iki_kat, "subperiod_consistent": tutarli,
                "n_trades": st["n"], "source": "contrib_verify"})
            g = lc.gates(sleeve)
            print(f"  kanıt yazıldı · lifecycle kapıları: "
                  f"{'GEÇTİ' if g['passed'] else 'GEÇMEDİ'} → {g['checks']}")
        return {**sonuc, "verdikt": "KANIT VAR"}

    print(f"\nVERDİKT [{sleeve}]: GÖLGE — kenar kanıtlanmadı.")
    print("  Bu bir RET DEĞİLDİR: katkı birleşebilir, sinyal üretir, EMİR VERMEZ ve")
    print("  ölçülmeye devam eder. Kanıt pozitife dönerse terfi yolu açıktır.")
    return {**sonuc, "verdikt": "GÖLGE"}


def main() -> int:
    _utf8()
    ap = argparse.ArgumentParser(description="Topluluk katkısı doğrulayıcı")
    ap.add_argument("--sleeve", required=True,
                    help="katkı adı; virgülle birden fazla ('hepsi' = yüklü tüm katkılar)")
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--symbols", default="BTC/USDT,ETH/USDT,SOL/USDT,DOGE/USDT,AVAX/USDT")
    ap.add_argument("--venue", default="mexc")
    ap.add_argument("--cost-pct", type=float, default=0.14, help="gidiş-dönüş maliyet (%%)")
    ap.add_argument("--step", type=int, default=5, help="pencere adımı (bar)")
    ap.add_argument("--evidence", action="store_true", help="kanıtı lifecycle kaydına yaz")
    ap.add_argument("--json", default="", help="sonuçları bu dosyaya yaz")
    a = ap.parse_args()

    from agi_trader.strategies import contrib as CB, sleeves_fast as SF, committee as CM  # noqa: F401

    sleeves = (CB.all_sleeves() if a.sleeve.strip().lower() in ("hepsi", "all", "*")
               else [x.strip() for x in a.sleeve.split(",") if x.strip()])
    print("═" * 78)
    print(f"KATKI DOĞRULAYICI · {len(sleeves)} kurulum · {a.days} gün · {a.venue}")
    print(f"  {', '.join(sleeves) or '(yok)'}")
    print("═" * 78)
    if not sleeves:
        print("\nÖlçülecek katkı yok.")
        return 3

    gecen: List[tuple] = []
    elenen: List[Dict] = []
    for s_ in sleeves:
        meta = _yukle_ve_statik(s_, CB)
        if meta is None:
            print(f"\nVERDİKT [{s_}]: REDDEDİLDİ — yükleme/statik aşaması geçilemedi.")
            elenen.append({"sleeve": s_, "verdikt": "REDDEDİLDİ", "sebep": "yükleme/statik"})
        else:
            gecen.append((s_, meta))
    if not gecen:
        return 2

    # ── VERİ BİR KEZ ─────────────────────────────────────────────────────
    # Aynı pencere bütün kurulumlara uygulanır: hem borsayı gereksiz yormaz hem de
    # kurulumlar BİRBİRİYLE karşılaştırılabilir olur (farklı pencere = farklı piyasa).
    print(f"\n── VERİ ({a.days} gün, {a.venue}) — bir kez çekilir, hepsine uygulanır")
    from agi_trader.auto.replay import HistoryFetcher
    syms = [x.strip() for x in a.symbols.split(",") if x.strip()]
    # Pencereyi SAATE yuvarla: aksi hâlde her koşum yeni bir önbellek anahtarı üretir
    # ve aynı veri tekrar tekrar indirilir (hız sınırına bu yüzden takılmıştık).
    end_ms = int(time.time() // 3600 * 3600 * 1000)
    span = int(a.days * 24 * 3600 * 1000)
    hf = HistoryFetcher(a.venue)
    hist = {}
    for s_ in syms:
        try:
            hist[(s_, "1m")] = hf.fetch(s_, "1m", end_ms - span, end_ms)
            print(f"  {s_:12s} {len(hist[(s_, '1m')])} bar")
        except Exception as e:
            print(f"  {s_:12s} veri alınamadı: {type(e).__name__}: {e}")
    if not hist:
        print("\nVERDİKT: ÖLÇÜLEMEDİ — veri yok. (Ağ/hız sınırı olabilir; tekrar deneyin.)")
        return 3

    p_ = CM.CommitteeParams()
    sonuclar = list(elenen)
    for s_, meta in gecen:
        sonuclar.append(_olc_ve_karar(s_, meta, hist, p_, a))

    # ── ÖZET ─────────────────────────────────────────────────────────────
    print("\n" + "═" * 78)
    print("ÖZET — aynı pencere, aynı maliyet, aynı kapılar")
    print("═" * 78)
    print(f"{'kurulum':<26} {'ateşleme':>9} {'oran %':>8} {'etkin n':>8} "
          f"{'ort net %':>10} {'t':>7} {'verdikt':>12}")
    for r in sonuclar:
        st = r.get("istatistik") or {}
        print(f"{r['sleeve']:<26} {r.get('atesleme', '—'):>9} "
              f"{r.get('atesleme_orani_pct', '—'):>8} {r.get('etkin_n', '—'):>8} "
              f"{st.get('ort', '—'):>10} {st.get('t', '—'):>7} {r['verdikt']:>12}")
    print("\nNOT: 'etkin n' örtüşmeyen işlem sayısıdır. Örtüşen ileri pencereler aynı")
    print("ticareti defalarca sayar ve |t|'yi şişirir; kapılar etkin küme üzerinden geçilir.")

    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(sonuclar, ensure_ascii=False, indent=2),
                                encoding="utf-8")
        print(f"\nJSON: {a.json}")
    return 0 if any(r["verdikt"] != "REDDEDİLDİ" for r in sonuclar) else 2


if __name__ == "__main__":
    raise SystemExit(main())
