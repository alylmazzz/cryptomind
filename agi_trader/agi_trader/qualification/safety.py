"""MERKEZİ GÜVENLİK KAPISI VE SİSTEM SAĞLIĞI (§LXXX, XL, XLIV, LXIII).

TEK CÜMLE: bir kritik bileşen kırmızıysa OTOPİLOT KAPANIR.

Bu modülün var olma sebebi, kapıların dağınık olmasının kendisidir. Veri
bayatlığı bir yerde, model sürüklenmesi başka yerde, mutabakat farkı üçüncü
bir yerde kontrol edilirse; hepsi ayrı ayrı "sorun yok" derken sistem bütün
olarak işlemez durumda olabilir. Burada tek bir özet vardır ve **fail-closed**
çalışır: bilinmeyen bir bileşen SAĞLIKLI SAYILMAZ.

ALTI BİLEŞEN
  MARKET_DATA     canlı bar akışı taze ve tam mı
  MODELS          artefaktlar yüklü, kalibre, sürüklenmemiş mi
  EXECUTION       yürütme yolu ve idempotency hazır mı
  RISK            risk limitleri ve kill-switch etkin mi
  RECONCILIATION  tahmin↔sonuç mutabakatı çalışıyor mu
  SECURITY        sır yönetimi ve yazma uçları kapalı mı

DURUMLAR: GREEN · WATCH · NOT_CONFIGURED · DEGRADED · RED · UNKNOWN
`UNKNOWN` bir bileşen için GREEN sayılmaz — ölçülmemiş bir sağlık, sağlık
değildir. `NOT_CONFIGURED` ise "bozuk" değil "kurulmadı" demektir: otopiloti
kapatır ama sistemi arızalı göstermez.

⚠️ BU MODÜL EMİR GÖNDERMEZ. Yalnız "otopilot açılabilir mi" sorusunu cevaplar.
Canlı yürütme ayrıca üç bağımsız anahtar ister ve üçü de kapalıdır.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional

GREEN = "GREEN"
WATCH = "WATCH"
DEGRADED = "DEGRADED"
RED = "RED"
UNKNOWN = "UNKNOWN"
# "Kurulmadı" ile "bozuldu" AYNI ŞEY DEĞİLDİR — ölçülerek değil, kullanıcı
# geri bildirimiyle bulundu: panel "SİSTEM bozulmuş" yazıyordu, oysa 6
# bileşenden 5'i sağlıklıydı ve altıncısı KASITLI olarak kurulmamıştı.
# DEGRADED bir arıza bildirir ve müdahale çağırır; NOT_CONFIGURED bir tasarım
# kararını bildirir ve müdahale gerektirmez.
NOT_CONFIGURED = "NOT_CONFIGURED"

# Sıralama: kötüden iyiye. Genel durum en KÖTÜ bileşenden gelir.
SEVERITY = {RED: 5, UNKNOWN: 4, DEGRADED: 3, NOT_CONFIGURED: 2,
            WATCH: 1, GREEN: 0}
COMPONENTS = ("MARKET_DATA", "MODELS", "EXECUTION", "RISK",
              "RECONCILIATION", "SECURITY")
# Otopiloti kapatan durumlar. NOT_CONFIGURED de kapatır — yürütme katmanı
# yoksa otomatik işlem YAPILAMAZ — ama sistemi "bozuk" göstermez.
BLOCKING = (RED, UNKNOWN, DEGRADED, NOT_CONFIGURED)

# ── SİSTEM MODU ────────────────────────────────────────────────────────────
# Bu ayrım olmadan "sağlık" sorusu cevaplanamaz: ölçüm yapan bir sistemde
# yürütme katmanının olmaması sağlıklı; işlem yapan bir sistemde felakettir.
MEASUREMENT_ONLY = "MEASUREMENT_ONLY"   # ölçer, emir GÖNDERMEZ (bugünkü hâl)
TRADING_READY = "TRADING_READY"         # yürütme kurulu
MODE_TR = {MEASUREMENT_ONLY: "ölçüm modu", TRADING_READY: "işlem modu"}

# Ölçüm modunda kurulmamış olması BEKLENEN bileşenler — genel sağlığa
# katılmazlar. Listede olmayan bir bileşen kurulmamışsa bu bir eksikliktir.
EXPECTED_ABSENT_IN_MEASUREMENT = ("EXECUTION",)

MAX_DATA_AGE_SEC = 900.0
MAX_SCAN_AGE_SEC = 1800.0


@dataclass
class ComponentHealth:
    name: str
    state: str
    detail: str = ""
    metrics: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SystemHealth:
    overall: str
    autopilot: bool
    components: List[Dict]
    blocking: List[str]
    checked_at: str
    note: str = ""
    # Sistem hangi modda çalışıyor (ölçüm / işlem). Genel sağlığın nasıl
    # yorumlanacağını bu belirler.
    mode: str = MEASUREMENT_ONLY
    mode_label: str = ""
    # Otopilotun neden kapalı olduğu. "Kapalı" tek başına arıza ima eder;
    # gerekçe olmadan kullanıcı bunu bozukluk sanır — nitekim sandı.
    autopilot_reason: str = ""
    # Kasıtlı olarak kurulmamış, bu modda BEKLENEN bileşenler. Genel sağlığa
    # katılmazlar ama gizlenmezler de.
    not_configured: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


def _en_kotu(durumlar: List[str]) -> str:
    if not durumlar:
        return UNKNOWN
    return max(durumlar, key=lambda s: SEVERITY.get(s, 3))


def market_data_health(scan: Optional[Dict]) -> ComponentHealth:
    if not scan or not scan.get("generated_at"):
        return ComponentHealth("MARKET_DATA", UNKNOWN,
                               "tarama henüz koşmadı — sağlık ÖLÇÜLMEDİ")
    kartlar = scan.get("cards") or []
    if not kartlar:
        return ComponentHealth("MARKET_DATA", RED, "hiçbir parite taranamadı")
    yaslar = [k.get("data_age_sec") for k in kartlar
              if k.get("data_age_sec") is not None]
    bayat = [k["symbol"] for k in kartlar
             if (k.get("data_age_sec") or 0) > MAX_DATA_AGE_SEC]
    kaliteler = [k.get("data_quality") for k in kartlar
                 if k.get("data_quality") is not None]
    ihlal = int((scan.get("scanner") or {}).get("schema_violations") or 0)
    kaps = (scan.get("scanner") or {}).get("field_coverage") or {}
    teshissiz = list(kaps.get("unexplained") or [])
    m = {"pairs": len(kartlar), "stale_pairs": len(bayat),
         "max_age_sec": (max(yaslar) if yaslar else None),
         "min_data_quality": (min(kaliteler) if kaliteler else None),
         "schema_violations": ihlal,
         "undiagnosed_empty_fields": teshissiz}
    if not yaslar:
        return ComponentHealth("MARKET_DATA", UNKNOWN, "veri yaşı ölçülemedi", m)
    # Şema ihlali bir VERİ sorunu değil KOD sorunudur; sessizce geçmemeli.
    if ihlal:
        return ComponentHealth("MARKET_DATA", DEGRADED,
                               f"{ihlal} şema/birim ihlali — üretilen sayılar "
                               f"güvenilmez", m)
    # Hiçbir hücrede dolmayan ve SEBEBİ BİLİNMEYEN alan → panelde boş sütun.
    # Bu sessiz kalırsa yalnız elle denetimde bulunur (nitekim öyle bulundu).
    if teshissiz:
        return ComponentHealth("MARKET_DATA", DEGRADED,
                               f"{len(teshissiz)} alan hiçbir hücrede dolmadı "
                               f"ve teşhis edilmedi: {', '.join(teshissiz[:4])}",
                               m)
    if len(bayat) > len(kartlar) * 0.3:
        return ComponentHealth("MARKET_DATA", RED,
                               f"{len(bayat)}/{len(kartlar)} parite bayat", m)
    if kaliteler and min(kaliteler) < 0.5:
        dusuk = [k["symbol"] for k in kartlar
                 if (k.get("data_quality") or 1) < 0.5]
        return ComponentHealth("MARKET_DATA", WATCH,
                               f"düşük veri kalitesi: {', '.join(dusuk[:5])}", m)
    if bayat:
        return ComponentHealth("MARKET_DATA", WATCH,
                               f"bayat: {', '.join(bayat[:5])}", m)
    return ComponentHealth("MARKET_DATA", GREEN,
                           f"{len(kartlar)} parite taze · şema temiz", m)


def models_health(artifacts_ok: bool, validation: Optional[Dict],
                  errors: Optional[List[str]] = None) -> ComponentHealth:
    m = {"artifact_errors": list(errors or [])}
    if not artifacts_ok:
        return ComponentHealth("MODELS", DEGRADED,
                               "araştırma artefaktları yüklenemedi — sistem "
                               "RESEARCH_ONLY'de kalır", m)
    if not validation:
        return ComponentHealth("MODELS", DEGRADED,
                               "validation_report yok — modeller UNVERIFIED", m)
    modeller = validation.get("models") or {}
    calisan = [k for k, v in modeller.items() if v.get("ok")]
    m.update({"n_models": len(modeller), "n_ok": len(calisan),
              "n_trials": validation.get("n_trials_registry")})
    if not calisan:
        return ComponentHealth("MODELS", DEGRADED, "hiçbir model kurulamadı", m)
    if not validation.get("n_trials_registry"):
        return ComponentHealth("MODELS", DEGRADED,
                               "deneme kaydı YOK — DSR anlamsız", m)
    if len(calisan) < len(modeller):
        return ComponentHealth("MODELS", WATCH,
                               f"{len(calisan)}/{len(modeller)} model çalışıyor", m)
    return ComponentHealth("MODELS", GREEN,
                           f"{len(calisan)} model + {m['n_trials']} deneme", m)


def execution_health(live_enabled: bool = False,
                     ems_ready: bool = False) -> ComponentHealth:
    """Yürütme yolu.

    ÜÇ AYRI DURUM, ÜÇ AYRI ANLAM:
      RED             canlı mod açık ama EMS yok → çift emir riski, TEHLİKE
      NOT_CONFIGURED  yürütme kurulmadı, canlı mod da kapalı → TASARIM KARARI
      GREEN           yürütme yolu hazır

    Ortadaki durum eskiden DEGRADED idi ve panel "SİSTEM bozulmuş" yazıyordu.
    Bozuk bir şey yoktu: sistem ölçüm modunda ve emir göndermiyor. Yanlış
    etiket, olmayan bir arıza için müdahale çağırır."""
    m = {"live_enabled": live_enabled, "ems_ready": ems_ready}
    if live_enabled and not ems_ready:
        return ComponentHealth("EXECUTION", RED,
                               "canlı mod AÇIK ama EMS/idempotency yok — "
                               "çift emir riski", m)
    if not ems_ready:
        return ComponentHealth("EXECUTION", NOT_CONFIGURED,
                               "yürütme katmanı kurulmadı (tasarım kararı) — "
                               "sistem ölçüm modunda, emir göndermiyor", m)
    return ComponentHealth("EXECUTION", GREEN, "yürütme yolu hazır", m)


def risk_health(kill_switch: bool = True, limits_loaded: bool = True,
                breaches: int = 0) -> ComponentHealth:
    m = {"kill_switch": kill_switch, "limits_loaded": limits_loaded,
         "breaches": breaches}
    if not kill_switch or not limits_loaded:
        return ComponentHealth("RISK", RED,
                               "kill-switch ya da limitler yüklü değil", m)
    if breaches:
        return ComponentHealth("RISK", DEGRADED, f"{breaches} limit ihlali", m)
    return ComponentHealth("RISK", GREEN, "limitler etkin, ihlal yok", m)


def reconciliation_health(n_predictions: int, n_resolved: int,
                          n_open_overdue: int = 0) -> ComponentHealth:
    m = {"predictions": n_predictions, "resolved": n_resolved,
         "open_overdue": n_open_overdue}
    if n_predictions == 0:
        return ComponentHealth("RECONCILIATION", GREEN,
                               "yayımlanmış tahmin yok — mutabakat gerekmiyor", m)
    if n_open_overdue > max(5, n_predictions * 0.2):
        return ComponentHealth("RECONCILIATION", DEGRADED,
                               f"{n_open_overdue} tahmin ufku dolduğu hâlde "
                               f"çözülmedi", m)
    if n_resolved == 0:
        return ComponentHealth("RECONCILIATION", WATCH,
                               "henüz hiçbir tahmin çözülmedi", m)
    return ComponentHealth("RECONCILIATION", GREEN,
                           f"{n_resolved}/{n_predictions} çözüldü", m)


def security_health(write_endpoints_closed: bool = True,
                    secrets_in_env: bool = True,
                    live_confirm_set: bool = False) -> ComponentHealth:
    m = {"write_endpoints_closed": write_endpoints_closed,
         "secrets_in_env": secrets_in_env,
         "live_confirm_set": live_confirm_set}
    if not write_endpoints_closed:
        return ComponentHealth("SECURITY", RED,
                               "yazma uçları açık — panel salt-okunur olmalı", m)
    if not secrets_in_env:
        return ComponentHealth("SECURITY", RED,
                               "sırlar ortam değişkeninde değil", m)
    return ComponentHealth("SECURITY", GREEN,
                           "yazma uçları kapalı, sırlar ortamdan", m)


def _infer_mode(tam: List[ComponentHealth]) -> str:
    """Modu bileşenlerden türet: yürütme kurulu değilse sistem ölçüm modundadır.

    Mod bir konfigürasyon bayrağı DEĞİL, gözlemlenen bir gerçektir — yalan
    söyleyemesin diye. `execution.mode=live` yazan biri EMS kurmadan işlem
    moduna geçemez."""
    for c in tam:
        if c.name == "EXECUTION":
            return MEASUREMENT_ONLY if c.state == NOT_CONFIGURED else TRADING_READY
    return MEASUREMENT_ONLY


def assess(components: List[ComponentHealth],
           mode: Optional[str] = None) -> SystemHealth:
    """Genel durum EN KÖTÜ bileşenden gelir; ortalama ALINMAZ.

    Ortalama almak, bir bileşen kırmızıyken sistemi 'çoğunlukla sağlıklı'
    göstermek olurdu — güvenlik kapısının tam tersi.

    TEK İSTİSNA — ve bu bir gevşetme değil, bir düzeltmedir: o modda
    kurulmaması BEKLENEN bir bileşen (ölçüm modunda EXECUTION) genel sağlığı
    düşürmez. Düşürdüğü sürüm panelde "SİSTEM bozulmuş" yazıyordu; bozuk bir
    şey yoktu, yürütme katmanı hiç kurulmamıştı. Yanlış alarm, gerçek alarmı
    değersizleştirir.

    İstisna OTOPİLOTU KAPSAMAZ: kurulmamış yürütme yine de otopiloti kapatır,
    çünkü emir gönderilemez. Değişen tek şey, bunun 'arıza' değil 'mod' olarak
    bildirilmesi."""
    var = {c.name for c in components}
    tam = list(components) + [
        ComponentHealth(ad, UNKNOWN, "bileşen hiç raporlanmadı")
        for ad in COMPONENTS if ad not in var]

    mod = mode or _infer_mode(tam)
    beklenen_yok = (EXPECTED_ABSENT_IN_MEASUREMENT
                    if mod == MEASUREMENT_ONLY else ())
    # Kurulmamış AMA bu modda beklenen bileşenler
    kurulmamis = [c.name for c in tam
                  if c.state == NOT_CONFIGURED and c.name in beklenen_yok]
    # Genel sağlık bunlar HARİÇ hesaplanır. Beklenmedik bir NOT_CONFIGURED
    # (ör. işlem modunda) hâlâ genel duruma girer ve sistemi bozuk gösterir.
    sagliga_giren = [c for c in tam if c.name not in kurulmamis]
    genel = _en_kotu([c.state for c in sagliga_giren])

    # Otopilot kapısı DEĞİŞMEZ: kurulmamış bileşen de engelleyicidir.
    engel = [c.name for c in tam if c.state in BLOCKING]
    if not engel:
        gerekce = ""
    elif kurulmamis and set(engel) <= set(kurulmamis):
        gerekce = ("sistem ölçüm modunda: yürütme katmanı kurulmadı, emir "
                   "gönderilmiyor — arıza değil, tasarım kararı")
    else:
        gercek = [e for e in engel if e not in kurulmamis]
        gerekce = "engelleyen bileşen: " + ", ".join(gercek)

    return SystemHealth(
        overall=genel, autopilot=(not engel), components=[c.to_dict() for c in tam],
        blocking=engel,
        checked_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        mode=mod, mode_label=MODE_TR.get(mod, mod),
        autopilot_reason=gerekce, not_configured=kurulmamis,
        note=("Genel durum en KÖTÜ bileşenden gelir; UNKNOWN sağlıklı "
              "SAYILMAZ. Bu modda kurulmaması beklenen bileşenler genel "
              "duruma katılmaz ama otopiloti yine de kapalı tutar. Canlı "
              "yürütme ayrıca üç bağımsız anahtar ister ve üçü de kapalıdır."))


STATE_TR = {GREEN: "sağlıklı", WATCH: "izlemede", DEGRADED: "bozulmuş",
            RED: "kritik", UNKNOWN: "ölçülmedi",
            NOT_CONFIGURED: "kurulmadı"}
