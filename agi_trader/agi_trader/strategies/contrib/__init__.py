"""
TOPLULUK KATKISI KURULUMLARI — dışarıdan gelen strateji önerilerinin giriş kapısı.

Bu paket, depoya katkı yapan herkesin kendi kurulumunu ekleyebilmesi için vardır. Tek bir
kural geçerlidir ve istisnası yoktur:

    KATKI SHADOW DOĞAR. Sinyal üretir, EMİR VERMEZ — paper'da bile.
    PAPER'a terfi yalnız `scripts/cm_verify_contribution.py` kapılarından
    ÖLÇÜLEREK geçilirse mümkündür (bkz. strategies/lifecycle.py `gates`).

Neden bu kadar katı: bu depodaki ölçümler defalarca "mantıklı görünen" kurulumların gerçek
veride kenarı olmadığını gösterdi — 21 videodan çıkarılan 10 kurulum 7 günlük gerçek veride
t = −20,4 verdi; 1 günlük ölçümde "kazandırıyor" görünen seans etkisi 7 günde silindi. Bir
katkının iyi niyetli, iyi yazılmış ve hatta zekice olması, kenarı olduğunu göstermez. Yalnız
ölçüm gösterir.

--------------------------------------------------------------------------- ARAYÜZ
Her katkı `contrib/` altında tek bir .py dosyasıdır ve iki şey tanımlar:

    META = {
        "name": "benim_kurulumum",       # [a-z0-9_], benzersiz, mevcut sleeve adlarıyla çakışamaz
        "author": "@github_kullanici",
        "source": "nereden geldi (URL / makale / kitap / kendi fikrim)",
        "claim": "ne iddia ediliyor",
        "claim_evidence": "iddianın kanıtı — YOKSA 'YOK' yazın (dürüstlük puanı düşürmez)",
        "mechanism": "mekanik tanım: hangi koşulda tetiklenir",
        "exit_mode": "FIXED_TARGET" | "PARTIAL_AND_RUN" | "DYNAMIC_PEAK",
        "time_stop_min": 90,             # dakika
        "urgency": 0 | 1 | 2,            # 0 = acele yok (maker), 2 = acil
        "regimes": ["TREND YUKARI", "RANGE / YATAY", "VOLATİL", "TREND AŞAĞI"],
    }

    def fire(f, p, price, atr_abs):
        \"\"\"Tetiklenmiyorsa None; tetikleniyorsa sözlük döndürür.\"\"\"
        return {"direction": "LONG", "size": 0.7,
                "stop_hint": ..., "target_hint": ..., "note": "neden tetiklendi"}

`f` komitenin özellik sözlüğüdür (`committee.fast_features` + `sleeves_fast.extra_features`
+ `sleeves_video.video_features`). YENİ VERİ ÇEKİLMEZ: aynı DataFrame'den türetilmiş
özellikler kullanılır — bir katkı ağ isteği yapamaz, dosya okuyamaz.

--------------------------------------------------------------------------- GÜVENLİK
Yükleyici kasıtlı olarak paranoyaktır:
  * META şeması ve `fire` imzası doğrulanır; uymayan dosya YÜKLENMEZ ve sebebi kaydedilir.
  * Ad çakışması reddedilir (bir katkı mevcut bir sleeve'in yerine geçemez).
  * `size` [0, 1] aralığına kelepçelenir; spot'ta SHORT üretilemez.
  * Her `fire` çağrısı ayrı ayrı try/except içindedir: bir katkının patlaması diğerlerini ya
    da koşucuyu düşüremez. Hata SESSİZCE YUTULMAZ — `FIRE_ERRORS`'a yazılır ve uçtan görünür
    (bu depoda sessiz `except: pass` yüzünden iki kez arıza yaşandı).
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil
import re
import sys
from typing import Callable, Dict, List, Optional

EXIT_FIXED, EXIT_PARTIAL_RUN, EXIT_DYNAMIC_PEAK = "FIXED_TARGET", "PARTIAL_AND_RUN", "DYNAMIC_PEAK"
VALID_EXIT_MODES = {EXIT_FIXED, EXIT_PARTIAL_RUN, EXIT_DYNAMIC_PEAK}
VALID_REGIMES = {"TREND YUKARI", "RANGE / YATAY", "VOLATİL", "TREND AŞAĞI"}
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,39}$")
REQUIRED_META = ("name", "author", "source", "claim", "claim_evidence", "mechanism",
                 "exit_mode", "time_stop_min", "urgency", "regimes")

# Yükleme sonuçları — hem panel hem CI bunları okur.
CONTRIB: Dict[str, Dict] = {}          # ad -> {"meta":..., "fire":...}
LOAD_ERRORS: List[Dict] = []           # yüklenemeyen dosyalar + SEBEP
FIRE_ERRORS: List[Dict] = []           # çalışma anında patlayan katkılar (son 50)


def _meta_hatalari(meta, modul: str, mevcut_adlar) -> List[str]:
    """META'yı doğrula. Dönen liste boşsa geçerlidir."""
    h: List[str] = []
    if not isinstance(meta, dict):
        return ["META bir sözlük değil"]
    for k in REQUIRED_META:
        if k not in meta:
            h.append(f"META['{k}'] eksik")
    if h:
        return h
    ad = str(meta["name"])
    if not NAME_RE.match(ad):
        h.append(f"name '{ad}' geçersiz — [a-z][a-z0-9_]{{2,39}} olmalı")
    if ad in mevcut_adlar:
        h.append(f"name '{ad}' MEVCUT bir sleeve ile çakışıyor — katkı var olanın yerine geçemez")
    if ad in CONTRIB:
        h.append(f"name '{ad}' başka bir katkı tarafından zaten kullanılıyor")
    if meta["exit_mode"] not in VALID_EXIT_MODES:
        h.append(f"exit_mode '{meta['exit_mode']}' geçersiz — {sorted(VALID_EXIT_MODES)}")
    try:
        ts = int(meta["time_stop_min"])
        if not (5 <= ts <= 1440):
            h.append("time_stop_min 5–1440 dakika aralığında olmalı")
    except Exception:
        h.append("time_stop_min tam sayı olmalı")
    if meta["urgency"] not in (0, 1, 2):
        h.append("urgency 0, 1 veya 2 olmalı")
    rej = meta["regimes"]
    if not isinstance(rej, (list, tuple)) or not rej:
        h.append("regimes boş olamaz")
    else:
        for r in rej:
            if r not in VALID_REGIMES:
                h.append(f"bilinmeyen rejim '{r}' — {sorted(VALID_REGIMES)}")
    for k in ("author", "source", "claim", "claim_evidence", "mechanism"):
        if not str(meta.get(k) or "").strip():
            h.append(f"META['{k}'] boş — kaynağını ve iddianı yazmak ZORUNLUDUR "
                     f"(kanıt yoksa claim_evidence'a 'YOK' yaz)")
    return h


def _fire_hatalari(fn) -> List[str]:
    if not callable(fn):
        return ["`fire` bulunamadı ya da çağrılabilir değil"]
    try:
        par = list(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return ["`fire` imzası okunamadı"]
    if par[:4] != ["f", "p", "price", "atr_abs"]:
        return [f"`fire` imzası (f, p, price, atr_abs) olmalı — gelen: ({', '.join(par)})"]
    if len(par) > 5 or (len(par) == 5 and par[4] != "df"):
        return [f"beşinci parametre yalnız `df` olabilir — gelen: ({', '.join(par)})"]
    return []


def _df_ister(fn) -> bool:
    """`fire` beşinci parametre `df` tanımladıysa kendi göstergesini hesaplamak istiyordur."""
    try:
        return list(inspect.signature(fn).parameters)[:5] == ["f", "p", "price", "atr_abs", "df"]
    except (TypeError, ValueError):
        return False


def load(mevcut_adlar: Optional[set] = None) -> Dict[str, Dict]:
    """`contrib/` altındaki katkıları keşfet, DOĞRULA ve kaydet.

    Doğrulamayı geçmeyen dosya yüklenmez; sebebi `LOAD_ERRORS`'a yazılır. Bu bilinçli:
    "yüklenmedi" ile "yüklendi ama hiç tetiklenmiyor" ayırt edilebilir olmalı."""
    CONTRIB.clear()
    LOAD_ERRORS.clear()
    mevcut = set(mevcut_adlar or ())
    for m in pkgutil.iter_modules(__path__):
        if m.name.startswith("_") or m.name.upper() == "SABLON":
            continue
        try:
            tam = f"{__name__}.{m.name}"
            # Modül zaten yüklüyse YENİDEN yükle: katkı sahibi dosyasını düzenleyip
            # doğrulayıcıyı tekrar çalıştırdığında ESKİ sürümün ölçülmesi, ölçümün
            # kendisini yalan hâline getirir.
            mod = (importlib.reload(sys.modules[tam]) if tam in sys.modules
                   else importlib.import_module(tam))
        except Exception as e:
            LOAD_ERRORS.append({"modul": m.name, "hatalar": [f"import edilemedi: {type(e).__name__}: {e}"]})
            continue
        meta = getattr(mod, "META", None)
        hatalar = _meta_hatalari(meta, m.name, mevcut) + _fire_hatalari(getattr(mod, "fire", None))
        if hatalar:
            LOAD_ERRORS.append({"modul": m.name, "hatalar": hatalar})
            continue
        CONTRIB[str(meta["name"])] = {"meta": dict(meta), "fire": mod.fire, "modul": m.name,
                                      "df_ister": _df_ister(mod.fire)}
    return CONTRIB


# --------------------------------------------------------------------- kayıt defterleri
def _reg(alan: str, don=lambda v: v) -> Dict:
    return {ad: don(v["meta"][alan]) for ad, v in CONTRIB.items()}


def sleeve_tr() -> Dict[str, str]:
    return {ad: f"{v['meta'].get('title_tr') or ad} (katkı)" for ad, v in CONTRIB.items()}


def sleeve_exit_mode() -> Dict[str, str]:
    return _reg("exit_mode")


def sleeve_time_stop_min() -> Dict[str, int]:
    return _reg("time_stop_min", int)


def sleeve_urgency() -> Dict[str, int]:
    return _reg("urgency", int)


def regime_sleeves() -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {r: [] for r in VALID_REGIMES}
    for ad, v in CONTRIB.items():
        for r in v["meta"]["regimes"]:
            out.setdefault(r, []).append(ad)
    return out


def all_sleeves() -> List[str]:
    return sorted(CONTRIB)


def sources() -> List[Dict]:
    """Panel/uç için künye: kim, nereden, ne iddia etti, kanıtı neydi."""
    return [{"sleeve": ad, **{k: v["meta"].get(k) for k in
                              ("author", "source", "claim", "claim_evidence", "mechanism")},
             "stage": "SHADOW", "modul": v["modul"]} for ad, v in sorted(CONTRIB.items())]


# ------------------------------------------------------------------------- tetikleme
def fire_contrib_sleeves(f: Dict, allowed: List[str], p, allow_short: bool = False,
                         now_ts: Optional[float] = None, df=None) -> List[Dict]:
    """Katkı kurulumlarını çalıştır. Her biri YALITILMIŞ — biri patlarsa diğerleri sürer.

    `df`: özellikleri üreten AYNI bar çerçevesi. Yalnız `fire` beşinci parametre `df`
    tanımlayan katkılara geçilir — böylece `f`'te bulunmayan göstergeleri (DI±, MACD,
    SAR, mum formasyonları …) kendileri hesaplayabilir. Çerçeve `f` ile aynı olduğu için
    katkı, sistemin geri kalanından FAZLA bilgi görmez; ileriye bakış ayrıca statik
    denetimle (shift(-n)) kollanır."""
    out: List[Dict] = []
    if not f.get("ok") or not CONTRIB:
        return out
    price = float(f.get("price") or f.get("close") or 0.0)
    if price <= 0:
        return out
    atr_abs = max(1e-12, float(f.get("atr_pct") or 0.3) / 100.0 * price)
    izin = set(allowed or ())
    for ad, v in CONTRIB.items():
        if ad not in izin:
            continue
        try:
            r = (v["fire"](f, p, price, atr_abs, df) if (v.get("df_ister") and df is not None)
                 else v["fire"](f, p, price, atr_abs))
        except Exception as e:
            # SESSİZ YUTMA YASAK: katkının patladığı, "sinyal yok"tan ayırt edilebilmeli.
            FIRE_ERRORS.append({"sleeve": ad, "hata": f"{type(e).__name__}: {e}", "ts": now_ts})
            del FIRE_ERRORS[:-50]
            continue
        if not r:
            continue
        yon = str(r.get("direction") or "LONG").upper()
        if yon not in ("LONG", "SHORT") or (yon == "SHORT" and not allow_short):
            continue
        try:
            boyut = float(r.get("size", 0.5))
        except Exception:
            boyut = 0.5
        out.append({
            "kind": ad, "direction": yon,
            "size": min(1.0, max(0.0, boyut)),
            "exit_mode": v["meta"]["exit_mode"],
            "stop_hint": r.get("stop_hint"), "target_hint": r.get("target_hint"),
            "note": f"[katkı · {v['meta']['author']}] {str(r.get('note') or '')}".strip(),
        })
    return out
