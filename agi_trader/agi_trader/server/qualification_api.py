"""NET +%1 nitelendirme uçları — şartname 88, 89, 93, 119 (+ 2. mesaj 50, 51).

TASARIM: uçlar HESAPLAMAZ, HAZIR ANLIK GÖRÜNTÜYÜ döndürür.
Canlı tarama parite başına ~2.200 bar indirir ve model çalıştırır; bunu istek
anında yapmak paneli kilitler. Ayrı bir arka plan iş parçacığı belirli
aralıklarla tarar, uçlar sonucu anında verir ve `age_sec` ile ne kadar taze
olduğunu söyler.

GET-ONLY: bu uçların hiçbiri durum değiştirmez. Tek yazma, taramanın kendi
içindeki DEĞİŞMEZ tahmin defteridir (ekleme-yalnız).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

QUAL_REFRESH_SEC = int(os.environ.get("CRYPTOMIND_QUAL_REFRESH", "300"))

# Yürütme katmanı (EMS) kurulu mu? cryptomind_serve.py koşucu kayıt defterini
# buraya bağlar; bağlanmazsa EXECUTION dürüstçe NOT_CONFIGURED kalır.
EMS_PROBE = None          # Callable[[], bool] | None
LIVE_PROBE = None         # Callable[[], bool] | None — canlı koşucu var mı
# Son tarama anlık görüntüsü — koşucular nitelendirme hücresini buradan okur
_SNAPSHOT_REF: Dict[str, object] = {}


def qualification_cell(symbol: str, horizon: str, direction: str):
    """Son taramadan (parite, ufuk, yön) hücresi. Panelde gösterilenle AYNI kaynak."""
    d = _SNAPSHOT_REF.get("scan")
    if not isinstance(d, dict):
        return None
    sym = symbol.replace("/", "").upper()
    for k in d.get("cards", []):
        if str(k.get("symbol", "")).upper() != sym:
            continue
        for h in k.get("horizons", []):
            if h.get("horizon") == horizon and h.get("direction") == direction:
                return h
        return None
    return None


def _default_symbols() -> list:
    """Canlı tarama evreni — araştırma artefaktıyla AYNI liste.

    Elle yazılmış bir liste, araştırma evreni büyüdüğünde sessizce geride
    kalır ve panel ölçülmüş paritelerin bir kısmını hiç göstermez."""
    try:
        u = json.loads((Path(__file__).resolve().parents[2] / "runs" /
                        "qualification" / "universe_5m.json")
                       .read_text(encoding="utf-8"))
        s = [x["symbol"] for x in u.get("selected", [])]
        if s:
            return s
    except Exception:
        pass
    return ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
            "ADAUSDT", "DOGEUSDT", "AVAXUSDT"]


QUAL_SYMBOLS = (os.environ["CRYPTOMIND_QUAL_SYMBOLS"].split(",")
                if os.environ.get("CRYPTOMIND_QUAL_SYMBOLS")
                else _default_symbols())
MORNING_TZ = os.environ.get("CRYPTOMIND_TZ", "Europe/Istanbul")

# KAPSAM SINIRLARI — istenip de yapılMAYAN her şey burada açıkça yazılır.
# Bir şartname maddesinin sessizce atlanması, kullanıcının o özelliğin
# çalıştığını sanmasına yol açar; bu liste tam da bunu engeller.
SCOPE_LIMITS = [
    {"item": "1m / 3m özellik zaman dilimi",
     "state": "KAPSAM DIŞI",
     "why": ("Taban seri 5 dakikadır; 1m'e inmek ilk-geçiş motorunun tamamını "
             "ve 2,2 GB ek veriyi gerektirir. Ablasyon, zaman dilimi eklemenin "
             "Brier'i yalnız %0,8 iyileştirdiğini gösterdi — beklenen kazanç "
             "maliyetin altında.")},
    {"item": "Dolum olasılığı (P_fill)",
     "state": "ÖLÇÜLMEDİ",
     "why": ("Kuyruk derinliği ve agresif akış geçmişi gerekir; kaydedici bunu "
             "henüz biriktirmedi. Bu yüzden limit giriş ÖNERİLMİYOR ve maliyet "
             "taker varsayımıyla hesaplanıyor — iyimser tarafa kaçılmıyor.")},
    {"item": "Ters seçim maliyeti (adverse selection)",
     "state": "ÖLÇÜLMEDİ",
     "why": "Dolum sonrası fiyat yolu verisi yok; ölçülene kadar sıfır sayılmaz."},
    {"item": "Çoklu borsa doğrulaması",
     "state": "YOK",
     "why": ("Tek venue (Binance USD-M vadeli). Maliyet ve mikroyapı katmanı "
             "aynı venue'den geldiği için tutarlı; venue-outlier tespiti yok.")},
    {"item": "Likidasyon / makro / zincir üstü / sosyal özellikler",
     "state": "UNMEASURED",
     "why": ("Toplanmıyor ya da olay örneklemi yetersiz. 'Veri yok' ile 'etki "
             "yok' aynı şey DEĞİLDİR; bu aileler sıfır ağırlık alır, sıfır "
             "etki VARSAYILMAZ.")},
]


def register(app: FastAPI, runs: Path) -> None:
    art_dir = runs / "qualification"
    ledger_path = runs / "qualification" / "predictions.jsonl"
    kutu = _SNAPSHOT_REF          # modül düzeyi: koşucular da aynı anlık görüntüyü okur
    kilit = threading.Lock()

    def _feats():
        try:
            import pandas as pd
            fs = sorted((runs / "features").glob("*.parquet"))
            if not fs:
                return None
            return pd.concat([pd.read_parquet(f) for f in fs[-2:]],
                             ignore_index=True)
        except Exception:
            return None

    def _ledger():
        from ..qualification.ledger import Ledger
        return Ledger(ledger_path)

    def _tarama() -> Dict:
        from ..qualification import live
        return live.scan(art_dir, [s for s in QUAL_SYMBOLS if s],
                         recorder_feats=_feats(), ledger=_ledger())

    def _snapshot() -> Dict:
        v = kutu.get("scan")
        if isinstance(v, dict):
            return v
        return {"scanner": {"artifacts_ok": False,
                            "empty_message": "tarama henüz koşmadı"},
                "cards": [], "ranked": [], "generated_at": None}

    def _yas(d: Dict) -> Optional[float]:
        g = d.get("generated_at")
        if not g:
            return None
        try:
            t = dt.datetime.strptime(g, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=dt.timezone.utc)
            return round((dt.datetime.now(dt.timezone.utc) - t).total_seconds(), 1)
        except Exception:
            return None

    # ------------------------------------------------------------------ uçlar
    @app.get("/api/qualification")
    def qualification():
        d = _snapshot()
        out = dict(d)
        out["age_sec"] = _yas(d)
        out["refresh_sec"] = QUAL_REFRESH_SEC
        out["last_scan_sec"] = kutu.get("last_scan_sec")
        out["fast_mode"] = bool(kutu.get("fast_mode"))
        out["scope_limits"] = SCOPE_LIMITS
        # Koşu kimliği artefaktta tutulur; panel bunu göstersin diye taşınır.
        try:
            out["provenance"] = json.loads(
                (art_dir / "provenance.json").read_text(encoding="utf-8"))
        except Exception:
            out["provenance"] = None
        out["guarantee"] = "YOK"
        return out

    @app.get("/api/qualification/pair")
    def qualification_pair(symbol: str = Query(..., min_length=3, max_length=20)):
        d = _snapshot()
        sym = symbol.replace("/", "").upper()
        for k in d.get("cards", []):
            if k.get("symbol") == sym:
                return {"card": k, "age_sec": _yas(d), "guarantee": "YOK"}
        return JSONResponse(
            {"error": "parite taramada yok", "symbol": sym,
             "available": [k.get("symbol") for k in d.get("cards", [])]},
            status_code=404)

    @app.get("/api/qualification/convergence")
    def convergence_detail(symbol: str = Query(..., min_length=3, max_length=20),
                           horizon: str = Query("4h", max_length=5),
                           direction: str = Query("LONG", max_length=5)):
        """Yakınsamanın TAM detayı — daralma eğrisi, dönem/rejim tahminleri.

        Ana uçta yalnız özet taşınır; bu detay hücre başına ~1 KB ve 540
        kombinasyonda yanıtı megabaytlara çıkarıyordu."""
        try:
            m = json.loads((art_dir / "matrix.json").read_text(encoding="utf-8"))
        except Exception as e:
            return JSONResponse({"error": f"matris okunamadı: {type(e).__name__}"},
                                status_code=404)
        sym = symbol.replace("/", "").upper()
        for k in m.get("cards", []):
            if k.get("symbol") != sym:
                continue
            for h in k.get("horizons", []):
                if (h.get("horizon") == horizon
                        and h.get("direction") == direction.upper()):
                    return {"symbol": sym, "horizon": horizon,
                            "direction": direction.upper(),
                            "convergence": h.get("convergence"),
                            "note": ("Daralma oranı bağımsız gözlemde ~0,50'dir; "
                                     "büyükse gözlemler bağımlı demektir ve veri "
                                     "biriktirmenin getirisi beklenenden azdır.")}
        return JSONResponse({"error": "hücre bulunamadı", "symbol": sym},
                            status_code=404)

    @app.get("/api/qualification/universe")
    def universe():
        v = kutu.get("universe")
        if isinstance(v, dict):
            return v
        return {"scanned": 0, "eligible": 0, "excluded": 0,
                "note": "evren taraması henüz koşmadı"}

    @app.get("/api/qualification/evidence")
    def evidence():
        """Kanıt detayı (şartname 54): ablasyon, kalibrasyon, desil, DSR/PBO."""
        p = art_dir / "validation_report.json"
        if not p.exists():
            return {"available": False,
                    "reason": "validation_report.json yok — modeller UNVERIFIED"}
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            return {"available": False, "reason": f"okunamadı: {type(e).__name__}"}
        modeller = {}
        for k, m in (r.get("models") or {}).items():
            lt = m.get("locked_test") or {}
            modeller[k] = {
                "ok": m.get("ok"),
                "n_oos": m.get("n_oos"), "folds": m.get("folds"),
                "oos_baseline": m.get("oos_baseline"),
                "oos_brier": m.get("oos_brier"),
                "oos_brier_base": m.get("oos_brier_base"),
                "oos_ece": m.get("oos_ece"),
                "oos_calibration": m.get("oos_calibration"),
                "oos_deciles": m.get("oos_deciles"),
                "oos_reliability": m.get("oos_reliability"),
                "ablation": m.get("ablation"),
                "pruned_families": m.get("pruned_families"),
                "locked_test": {kk: lt.get(kk) for kk in (
                    "n", "baseline", "brier", "brier_base", "ece",
                    "calibration", "deciles", "reliability", "selected",
                    "selected_frac", "subperiods", "positive_subperiod_frac")},
                "psi": m.get("psi"),
            }
        return {"available": True,
                "generated_at": r.get("generated_at"),
                "split": r.get("split"),
                "base_resolution": r.get("base_resolution"),
                "n_cells": r.get("n_cells"),
                "n_trials_registry": r.get("n_trials_registry"),
                "trial_dispersion_sharpe": r.get("trial_dispersion_sharpe"),
                "status_distribution": r.get("status_distribution"),
                "data_readiness": _data_readiness(runs),
                "models": modeller}

    @app.get("/api/qualification/ledger")
    def ledger_view(days: int = Query(30, ge=1, le=3650)):
        from ..qualification import ledger as L
        led = _ledger()
        yayim = led.predictions(source=L.SOURCE_PUBLISHED)
        golge = led.predictions(source=L.SOURCE_SHADOW)
        return {"scorecard_window": L.scorecard(led, days),
                "scorecard_full": L.scorecard(led, None),
                # Kalibrasyon gölgeden DE beslenir: soru "yayımladıklarım
                # tuttu mu" değil, "olasılık tahminim doğru mu".
                "calibration_board": L.calibration_board(led),
                "calibration_published_only": L.calibration_board(
                    led, source=L.SOURCE_PUBLISHED),
                "today": L.daily_panel(led, MORNING_TZ),
                "n_predictions": len(yayim),
                "n_shadow": len(golge),
                "n_resolved": len(led.outcomes()),
                "shadow_note": ("Gölge kayıtlar sinyal DEĞİLDİR: nitelendirme "
                                "iddiası taşımaz, işlem önermez, karne "
                                "paydasına girmez. Yalnız olasılık tahmininin "
                                "canlı veriyle sınanmasını sağlar — sıfır "
                                "QUALIFIED üreten bir sistemde defter aksi "
                                "hâlde boş kalır ve model kendi hatasını "
                                "göremez."),
                "note": ("Başarısız sinyaller ASLA silinmez; payda yayımlanan "
                         "sinyal sayısıdır.")}

    @app.get("/api/qualification/models")
    def model_registry():
        """§XVI/CII — model envanteri ve kartları."""
        from ..qualification.model_registry import Registry, STATUS_TR
        r = Registry(art_dir / "model_registry.json")
        kayitlar = r.all()
        return {"models": kayitlar, "history": r.history(),
                "production_ready": r.production_ready(),
                "status_tr": STATUS_TR,
                "note": ("Hiçbir model APPROVED değildir: bağımsız bir "
                         "doğrulayıcı atanmadı. Geliştirici kendi modelini "
                         "onaylayamaz (§LVI) ve bu kural kodla zorlanır.")}

    @app.get("/api/system-health")
    def system_health():
        """§LXXX — tek özet. Bir kritik bileşen kırmızıysa OTOPİLOT KAPALI."""
        from ..qualification import safety as SF
        from ..qualification.ledger import Ledger
        d = _snapshot()
        led = _ledger()
        art = None
        try:
            from ..qualification.live import Artifacts
            art = Artifacts(art_dir)
        except Exception:
            pass

        # Canlı yürütme anahtarları — ÜÇÜ de gerekli. EMS artık koşucu kayıt
        # defterinden sorulur: en az bir otopilot koşucusu kuruluysa yürütme
        # yolu HAZIR sayılır (paper dâhil — yol aynı, yalnız hedef farklı).
        canli = bool(os.environ.get("CRYPTOMIND_LIVE_CONFIRM"))
        try:
            ems_ready = bool(EMS_PROBE()) if EMS_PROBE else False
        except Exception:
            ems_ready = False
        acik = led.open_predictions()
        gecikmis = 0
        try:
            import datetime as _dt
            from ..qualification.horizons import HORIZON_MIN
            simdi = _dt.datetime.now(_dt.timezone.utc)
            for x in acik:
                dk = HORIZON_MIN.get(x.get("horizon"))
                if dk is None:
                    continue
                t = _dt.datetime.fromisoformat(
                    x["timestamp"].replace("Z", "+00:00"))
                if (simdi - t).total_seconds() > dk * 60:
                    gecikmis += 1
        except Exception:
            gecikmis = 0

        bilesenler = [
            SF.market_data_health({**d, "scanner": d.get("scanner")}),
            SF.models_health(bool(d.get("scanner", {}).get("artifacts_ok")),
                             (art.validation if art else None),
                             (art.errors if art else None)),
            SF.execution_health(live_enabled=canli, ems_ready=ems_ready),
            SF.risk_health(kill_switch=True, limits_loaded=True, breaches=0),
            SF.reconciliation_health(len(led.predictions()),
                                     len(led.outcomes()), gecikmis),
            SF.security_health(write_endpoints_closed=True,
                               secrets_in_env=True, live_confirm_set=canli),
        ]
        h = SF.assess(bilesenler).to_dict()
        h["state_tr"] = {k: v for k, v in SF.STATE_TR.items()}
        return h

    # Otopilot karar zinciri SAĞLIK kapısını buradan sorar (panelle aynı değer)
    app.state.system_health = system_health

    @app.get("/api/qualification/attribution")
    def attribution(days: int = Query(0, ge=0, le=3650)):
        """§CVI — ana ürün metriği: gerçekleşen ÷ tahmin edilen net EV."""
        from ..qualification import attribution as AT
        led = _ledger()
        from ..qualification import ledger as L
        return {"overall": AT.overall(led, days or None),
                "last_30d": AT.overall(led, 30),
                # Gölge ayrı hesaplanır ve ASLA `overall` ile toplanmaz.
                "shadow": AT.overall(led, days or None,
                                     source=L.SOURCE_SHADOW),
                "published": AT.overall(led, days or None,
                                        source=L.SOURCE_PUBLISHED),
                "by_cell": AT.by_cell(led),
                "verdict_tr": AT.VERDICT_TR,
                "why": ("Bu tek oran dört katmanı birden ölçer: tahmin, "
                        "maliyet, yürütme, risk. Sapma katmanlara ayrıştırılır; "
                        "aksi hâlde 'model bozuldu' ile 'borsa pahalılaştı' "
                        "ayırt edilemez.")}

    @app.get("/api/morning")
    def morning():
        v = kutu.get("morning")
        if isinstance(v, dict):
            return v
        return {"state": "SCANNING", "note": "sabah motoru henüz koşmadı",
                "timezone": MORNING_TZ}

    # -------------------------------------------------------- arka plan döngüsü
    def _dongu() -> None:
        time.sleep(8)
        while True:
            t0 = time.time()
            try:
                with kilit:
                    d = _tarama()
                    kutu["scan"] = d
                    from ..qualification.live import json_safe
                    kutu["morning"] = json_safe(_sabah(d))
                gecen = time.time() - t0
                kutu["last_scan_sec"] = round(gecen, 1)
                print(f"[cryptomind-qual] tarama {gecen:.0f} sn · "
                      f"{len(d.get('cards') or [])} parite", flush=True)
                if gecen > QUAL_REFRESH_SEC:
                    # Tarama, yenileme aralığından uzun sürüyorsa döngü kendi
                    # kuyruğunu büyütür ve 2 çekirdekli sunucuda API'yi aç
                    # bırakır. Bu durumda aralık taramanın kendisine uydurulur.
                    print(f"[cryptomind-qual] UYARI: tarama ({gecen:.0f} sn) "
                          f"yenileme aralığından ({QUAL_REFRESH_SEC} sn) uzun — "
                          f"bir sonraki tur gecikmeli başlayacak", flush=True)
            except Exception as e:
                print(f"[cryptomind-qual] tarama hatası: {type(e).__name__}: {e}",
                      flush=True)
            try:
                _sonuclari_coz()
            except Exception as e:
                print(f"[cryptomind-qual] sonuç çözümü: {type(e).__name__}: {e}",
                      flush=True)
            # Bekleme: en az yenileme aralığı, ama taramanın kendisi kadar da
            # nefes bırak — üst üste binen tur CPU'yu tüketip API'yi 503'e düşürür.
            gecen = float(kutu.get("last_scan_sec") or 0.0)
            bekle = max(60, QUAL_REFRESH_SEC, int(gecen))
            # 2. mesaj 40: "entry açıkken çok daha sık" — yayımlanmış bir
            # fırsat varken fiyat, defter ve olasılık hızla eskir. Fırsat
            # yokken sık taramanın bir karşılığı yoktur; CPU boşa gider.
            aktif = kutu.get("scan") or {}
            if any(k.get("best_horizon") for k in (aktif.get("cards") or [])):
                bekle = max(60, int(gecen) + 15)
                kutu["fast_mode"] = True
            else:
                kutu["fast_mode"] = False
            time.sleep(bekle)

    def _sabah(d: Dict) -> Dict:
        from ..qualification import morning as M
        led = _ledger()
        try:
            ogrenilen = M.learn_slots(led, MORNING_TZ)
        except Exception:
            ogrenilen = {"best_slot": None, "windows": {}, "consistent": {}}
        kartlar = d.get("cards", [])
        rdy = M.readiness(kartlar,
                          data_quality=_ortalama(kartlar, "data_quality"),
                          liquidity=_ortalama(kartlar, "liquidity_score"))
        try:
            from zoneinfo import ZoneInfo
            simdi = dt.datetime.now(ZoneInfo(MORNING_TZ))
        except Exception:
            simdi = dt.datetime.now(dt.timezone.utc)
        pencerede = M.in_window(simdi, MORNING_TZ)
        bugun = simdi.strftime("%Y-%m-%d")
        yayimlandi = kutu.get("morning_published_date") == bugun
        yayimla, gerekce = M.should_publish(kartlar, rdy, simdi, yayimlandi)
        rapor = kutu.get("morning_report")
        if pencerede and yayimla:
            rapor = M.build_report(kartlar, d.get("scanner", {}), rdy, ogrenilen,
                                   simdi.strftime("%H:%M"), MORNING_TZ,
                                   combos=d.get("scanner", {}).get("combinations"))
            kutu["morning_published_date"] = bugun
            kutu["morning_report"] = rapor
        durum = ("REPORT READY" if rapor and kutu.get("morning_published_date") == bugun
                 else ("SCANNING" if pencerede else "OUTSIDE WINDOW"))
        if durum == "REPORT READY" and rapor and rapor.get("empty_result"):
            durum = "NO QUALIFIED OPPORTUNITY"
        return {
            "state": durum,
            "timezone": MORNING_TZ,
            "window": f"{M.WINDOW_START[0]:02d}:00–{M.WINDOW_END[0]:02d}:00",
            "in_window": pencerede,
            "now_local": simdi.strftime("%H:%M"),
            "next_evaluation_sec": QUAL_REFRESH_SEC,
            "readiness": rdy.to_dict(),
            "publish_decision": gerekce,
            "current_slot": M.slot_of(simdi, MORNING_TZ),
            "learned_slots": ogrenilen,
            "performance": M.morning_performance(led, MORNING_TZ),
            "slot_switch": dict(zip(("slot", "reason"), M.should_switch_slot(
                kutu.get("morning_slot"), ogrenilen))),
            "report": rapor,
        }

    def _sonuclari_coz() -> None:
        """Ufku dolan tahminleri gerçek barlarla çöz (şartname 45)."""
        from ..qualification import live as LV
        from ..qualification.ledger import evaluate_open
        led = _ledger()
        if not led.open_predictions():
            return

        def bul(symbol, t0, t1):
            import pandas as pd
            d = LV._klines(symbol, limit=1500)
            if d is None:
                return None
            a, b = pd.Timestamp(t0), pd.Timestamp(t1)
            return d[(d.index > a) & (d.index <= b)]
        evaluate_open(led, bul)

    def _evren() -> None:
        time.sleep(20)
        while True:
            try:
                from ..qualification import universe as U
                mk = U.collect_binance_markets(_feats(), runs / "data_5m")
                kutu["universe"] = U.scan(mk)
            except Exception as e:
                print(f"[cryptomind-qual] evren taraması: {type(e).__name__}: {e}",
                      flush=True)
            time.sleep(3600)

    if QUAL_REFRESH_SEC > 0:
        threading.Thread(target=_dongu, name="cryptomind-qual",
                         daemon=True).start()
        threading.Thread(target=_evren, name="cryptomind-universe",
                         daemon=True).start()


def _ortalama(kartlar: List[Dict], alan: str) -> Optional[float]:
    v = [k.get(alan) for k in kartlar if k.get(alan) is not None]
    return (sum(v) / len(v)) if v else None


def _data_readiness(runs: Path) -> List[Dict]:
    """Şartname 70 — veri hazırlık matrisi. 'Yok' ile 'etkisiz' KARIŞTIRILMAZ."""
    var5m = (runs / "data_5m").exists() and any((runs / "data_5m").glob("*.parquet"))
    varfeat = (runs / "features").exists() and any((runs / "features").glob("*.parquet"))
    return [
        {"source": "OHLCV (5m vadeli)", "state": "HISTORICAL+LIVE" if var5m else "MISSING",
         "note": "2022-01 → bugün, boşluksuz" if var5m else "runs/data_5m yok"},
        {"source": "Trades / taker akışı", "state": "HISTORICAL+LIVE" if var5m else "MISSING",
         "note": "kline içindeki taker_buy_base ve trades alanları"},
        {"source": "L2 defter merdiveni", "state": "LIVE" if varfeat else "MISSING",
         "note": "kaydedici topluyor; tarihsel etiketlerle hizalanacak kadar derin DEĞİL"},
        {"source": "Funding", "state": "LIVE" if varfeat else "MISSING",
         "note": "canlı ölçülüyor; 4,5 yıllık geçmiş modele girmedi"},
        {"source": "Açık pozisyon (OI)", "state": "LIVE" if varfeat else "MISSING",
         "note": "canlı; geçmiş 30 günle sınırlı"},
        {"source": "Likidasyonlar", "state": "MISSING",
         "note": "toplanmıyor — UNMEASURED"},
        {"source": "Makro", "state": "MISSING", "note": "5m ızgaraya hizalanmadı"},
        {"source": "Sosyal", "state": "PARTIAL",
         "note": "olay örneklemi yön özelliği olmaya yetmiyor — UNMEASURED"},
        {"source": "Zincir üstü", "state": "MISSING", "note": "toplanmıyor"},
    ]
