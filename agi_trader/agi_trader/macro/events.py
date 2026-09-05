"""
Olay Öngörü + Zamanlama — "birbirini etkileyecek büyük olayları öngör".

İki parça:
  1) upcoming_events(): yüksek-etkili tekrarlayan makro olayların BİR SONRAKİ
     yaklaşık tarihini + kalan süresini hesaplar (CPI, FOMC, NFP, OPEX). Kesin
     tarih için ekonomik takvim API'si gerekir; bu sezgisel kestirim "tahmini"
     etiketiyle sunulur — yine de yaklaşan risk penceresini doğru işaretler.

  2) market_movers(): pariteler arası korelasyon + son momentum ile "piyasayı
     süren" varlığı (driver) ve onu TAKİP EDECEK korele varlıkları + tahmini
     zamanlamayı çıkarır (yüksek korelasyon → eşzamanlı; düşük → gecikmeli).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List


def _next_weekday(d: datetime, weekday: int, nth: int = 1) -> datetime:
    """Ayın nth. <weekday>'i (0=Pzt). nth=1 ilk, nth=-1 son."""
    if nth > 0:
        first = d.replace(day=1)
        offset = (weekday - first.weekday()) % 7
        day = 1 + offset + (nth - 1) * 7
        return first.replace(day=day)
    # son occurrence
    if d.month == 12:
        nxt = d.replace(year=d.year + 1, month=1, day=1)
    else:
        nxt = d.replace(month=d.month + 1, day=1)
    last = nxt - timedelta(days=1)
    offset = (last.weekday() - weekday) % 7
    return last - timedelta(days=offset)


def _next_monthly(now: datetime, day: int) -> datetime:
    """Bu ay <day> geçtiyse gelecek ayın <day>'i."""
    try:
        cand = now.replace(day=day, hour=13, minute=30, second=0, microsecond=0)
    except ValueError:
        cand = now
    if cand <= now:
        y, m = (now.year + (now.month // 12)), (now.month % 12 + 1)
        cand = cand.replace(year=y, month=m)
    return cand


def upcoming_events(now: datetime = None) -> List[Dict]:
    """Yaklaşan yüksek-etkili olayların tahmini takvimi (kalan güne göre sıralı)."""
    now = now or datetime.now(timezone.utc)
    evs: List[Dict] = []

    def add(name, when, impact, asset, note):
        days = (when - now).total_seconds() / 86400
        evs.append({"name": name, "date": when.strftime("%Y-%m-%d %H:%M UTC"),
                    "in_days": round(days, 1), "impact": impact, "asset": asset, "note": note})

    # CPI (ABD TÜFE) — genelde ayın ~12'si, 13:30 UTC
    add("ABD TÜFE (CPI)", _next_monthly(now, 12), "yüksek", "USD · risk varlıkları",
        "Enflasyon sürprizi → faiz beklentisi → kripto/borsa sert tepki")
    # NFP (tarım dışı istihdam) — ayın ilk Cuma'sı
    nfp = _next_weekday(now, 4, 1).replace(hour=13, minute=30, tzinfo=timezone.utc)
    if (nfp - now).days < 0:
        nfp = _next_weekday(now.replace(day=28) + timedelta(days=7), 4, 1).replace(hour=13, minute=30, tzinfo=timezone.utc)
    add("ABD İstihdam (NFP)", nfp, "yüksek", "USD · genel piyasa", "İşgücü verisi → FED patikası")
    # FOMC — yaklaşık her 6.5 haftada; bir sonraki kabaca +45 gün referanslı
    add("FOMC faiz kararı (tahmini)", now + timedelta(days=((45 - (now.day % 45)) or 45)),
        "çok yüksek", "USD · tüm varlıklar", "Faiz/QT kararı → en geniş etkili olay")
    # Opsiyon vadesi (OPEX) — ayın 3. Cuma'sı
    opex = _next_weekday(now, 4, 3).replace(hour=20, minute=0, tzinfo=timezone.utc)
    if (opex - now).days < 0:
        nm = (now.replace(day=28) + timedelta(days=7))
        opex = _next_weekday(nm, 4, 3).replace(hour=20, minute=0, tzinfo=timezone.utc)
    add("Opsiyon Vadesi (OPEX)", opex, "orta", "BTC/ETH · hisse", "Vade kaynaklı volatilite/pin riski")

    evs.sort(key=lambda e: e["in_days"])
    return evs


def market_movers(snapshots: List[Dict], corr_matrix: Dict[str, Dict[str, float]] = None) -> Dict:
    """Anlık sinyallerden 'piyasa süren' (driver) + takip edenleri (followers) çıkar.

    snapshots: [{symbol, direction, momentum?, confidence}] — en yüksek
    (momentum × güven) yönlü varlık SÜRÜCÜ olur.
    corr_matrix: {sym: {sym2: corr}} — SÜRÜCÜYE-GÖRELİ korelasyon (driver-relative).
    Verilirse follower'ların driver ile gerçek korelasyonu kullanılır (doğru). Yoksa
    her snapshot'taki BTC-referans 'correlation' alanına düşülür (yaklaşık)."""
    if not snapshots:
        return {"driver": None, "followers": []}

    def force(s):
        return abs(s.get("momentum", 50) - 50) / 50.0 * s.get("confidence", 0) * (0 if s.get("direction") == "FLAT" else 1)

    ranked = sorted(snapshots, key=force, reverse=True)
    driver = ranked[0]
    if force(driver) <= 0:
        return {"driver": None, "followers": [], "note": "Belirgin piyasa sürücüsü yok (yatay)"}

    dsym = driver["symbol"]
    drow = (corr_matrix or {}).get(dsym, {})
    followers = []
    for s in ranked[1:]:
        if corr_matrix is not None:
            corr = drow.get(s["symbol"])                    # SÜRÜCÜYE göre gerçek korelasyon
        else:
            corr = (s.get("correlation") or {}).get("value")  # yedek: BTC-referans
        if corr is None:
            continue
        a = abs(corr)
        timing = "eşzamanlı" if a >= 0.7 else "gecikmeli (1-3 bar)" if a >= 0.4 else "zayıf bağ"
        exp_dir = driver["direction"] if corr >= 0 else ("SHORT" if driver["direction"] == "LONG" else "LONG")
        followers.append({"symbol": s["symbol"], "correlation": round(float(corr), 2),
                          "expected_direction": exp_dir, "timing": timing})
    followers.sort(key=lambda f: abs(f["correlation"]), reverse=True)
    return {
        "driver": {"symbol": dsym, "direction": driver["direction"],
                   "momentum": driver.get("momentum"), "confidence": driver.get("confidence")},
        "followers": followers[:8],
        "relative": corr_matrix is not None,
        "note": f"{dsym} {driver['direction']} yönünde piyasayı sürüyor; "
                f"korele varlıklar yukarıdaki zamanlamayla takip etmeli"
                + ("" if corr_matrix is not None else " (korelasyon BTC-referanslı, yaklaşık)") + ".",
    }
