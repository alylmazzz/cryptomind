"""
KANIT MOTORU — sürekli, otomatik ve YALIN kanıt biriktirme.

NEDEN
─────
Kanıt kapıları (devre kesici, kanıt tavanı, seans kapısı, DSR) hep aynı birkaç
istatistiğe bakar: n, ortalama, standart sapma, kazanma sayısı. Bunlar için ham
işlem geçmişini bellekte ya da her döngüde yeniden yazılan bir dosyada tutmak
gereksizdir — YETERLİ İSTATİSTİK (sufficient statistics) O(1) yer tutar.

2026-09-06 ÖLÇÜMÜ (neden bu modül var):
  • İşlem kaydı 1.024 B; istatistiğin ihtiyacı olan alanlar bunun yalnız %10,8'i (111 B).
  • `runner_*.json` HER DÖNGÜDE (~25 sn) baştan yazılıyor ve içinde TRADES_KEEP=3000
    işlem taşınıyor → ledger dolduğunda döngü başına ~3 MB yeniden yazma.
  • `signals.jsonl` kayıt başına 6.626 B ve 118 MB'a ulaşmış; tek tüketicisi
    (`weight_optimizer`) bunun ~%4'ünü kullanıyor.
  • `TradeJournal.load_all()` dosyanın TAMAMINI `read_text()` ile belleğe alıyordu —
    118 MB'lık dosyada bu birkaç yüz MB'lık bir tepe demekti (OOM tuzağı).

TASARIM
───────
  • `evidence.jsonl` — SALT-EKLEME, kayıt başına ~130 B, asla yeniden yazılmaz.
    65 işlem/gün → ~8,5 KB/gün → ~3 MB/yıl. Kalıcı ve eksiksiz kanıt kaydı.
  • Okuma HER ZAMAN akış (satır satır) — dosya ne kadar büyürse büyüsün bellek sabit.
  • Yeterli istatistikler tek geçişte toplanır: n, Σx, Σx², kazanç sayısı.
  • `kac_islem_gerek()` — "bu sleeve'in kanıtlanması için kaç işlem daha lazım?"
    Kanıt toplamayı OTOMATİK ve ÖLÇÜLEBİLİR yapan sayı budur.

Bu modül ağ erişimi yapmaz, hiçbir şeyi silmez, işlem akışını bloklamaz.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

# Kanıt satırında TUTULAN alanlar — kapıların gerçekten kullandıkları.
# Bu listeyi büyütmeden önce "hangi kapı bunu okuyor?" sorusu cevaplanmalı.
ALANLAR = (
    "ts",          # kapanış zamanı (int saniye — ondalık gereksiz)
    "sym",         # parite
    "slv",         # sleeve
    "rej",         # rejim (TREND YUKARI / RANGE / VOLATİL …)
    "ss",          # seans bloğu (UTC 4 saatlik: 0,4,8,12,16,20)
    "d",           # yön (L/S)
    "ot",          # emir tipi (m/t)
    "xm",          # çıkış modu (F/P/D)
    "r",           # çıkış sebebi
    "np",          # net_pct_realized  ← ana ölçüm
    "nu",          # net_pnl (USDT)
    "w",           # kazandı mı (0/1)
    "no",          # notional
    "pk",          # peak_net_pct (tepe yakalama için)
    "cp",          # gidiş-dönüş maliyet %
    "sp",          # stop %
    "hs",          # tutma süresi (sn)
    "lv",          # merdiven basamak sayısı
    "ev",          # giriş anında VAADEDİLEN EV (plan hedefiyle) — kalibrasyon ölçümü için
    "eva",         # giriş anında ULAŞILABİLİR hedefle EV — hangisi öngörüyor?
)

_MOD = {"FIXED_TARGET": "F", "PARTIAL_AND_RUN": "P", "DYNAMIC_PEAK": "D"}


def yol(output_dir: str, tag: str = "") -> Path:
    p = Path(output_dir)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[2] / p
    p = p / "live"
    p.mkdir(parents=True, exist_ok=True)
    return p / (f"evidence_{tag}.jsonl" if tag else "evidence.jsonl")


def satir(trade: Dict, rejim: Optional[str] = None) -> Dict:
    """İşlem kaydı → kanıt satırı. Yalnız kapıların kullandığı alanlar."""
    ts = float(trade.get("closed_ts") or 0.0)
    return {
        "ts": int(ts),
        "sym": trade.get("symbol"),
        "slv": trade.get("sleeve") or trade.get("trigger") or "?",
        "rej": rejim,
        "ss": (time.gmtime(ts).tm_hour // 4) * 4 if ts else None,
        "d": "L" if str(trade.get("direction")) == "LONG" else "S",
        "ot": "m" if str(trade.get("order_type")) == "maker" else "t",
        "xm": _MOD.get(str(trade.get("exit_mode")), "?"),
        "r": trade.get("reason"),
        "np": round(float(trade.get("net_pct_realized") or 0.0), 4),
        "nu": round(float(trade.get("net_pnl") or 0.0), 4),
        "w": 1 if trade.get("win") else 0,
        "no": round(float(trade.get("notional") or 0.0), 2),
        "pk": round(float(trade.get("peak_net_pct") or 0.0), 4),
        "cp": round(float(trade.get("cost_pct_roundtrip") or 0.0), 4),
        "sp": round(float(trade.get("stop_pct") or 0.0), 4),
        "hs": int(float(trade.get("hold_sec") or 0)),
        "lv": int(trade.get("levels_hit") or 0),
        "ev": (None if trade.get("ev_pct") is None else round(float(trade["ev_pct"]), 4)),
        "eva": (None if trade.get("ev_achievable_pct") is None else round(float(trade["ev_achievable_pct"]), 4)),
    }


def kaydet(trade: Dict, output_dir: str, tag: str = "", rejim: Optional[str] = None) -> bool:
    """Tek işlemi kanıt defterine EKLE. Hata işlem akışını DURDURMAZ (ölçüm, ticaret değil)."""
    try:
        with open(yol(output_dir, tag), "a", encoding="utf-8") as f:
            f.write(json.dumps(satir(trade, rejim), ensure_ascii=False, separators=(",", ":")) + "\n")
        return True
    except Exception:
        return False


def oku(output_dir: str, tag: str = "", since: Optional[float] = None,
        limit: Optional[int] = None) -> Iterator[Dict]:
    """AKIŞ okuma — dosya ne kadar büyürse büyüsün bellek sabit kalır.
    `limit` verilirse yalnız SON `limit` satır döner (deque ile, yine sabit bellek)."""
    p = yol(output_dir, tag)
    if not p.exists():
        return
    if limit:
        from collections import deque
        buf: deque = deque(maxlen=int(limit))
        with open(p, encoding="utf-8") as f:
            for line in f:
                buf.append(line)
        kaynak: Iterable[str] = buf
    else:
        kaynak = open(p, encoding="utf-8")
    try:
        for line in kaynak:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if since is not None and float(r.get("ts") or 0) < since:
                continue
            yield r
    finally:
        if hasattr(kaynak, "close"):
            kaynak.close()      # type: ignore[union-attr]


# ─────────────────────────────────────────────────── yeterli istatistikler
class Biriktirici:
    """n, Σx, Σx², kazanç — O(1) bellek. t, ortalama, Wilson bunlardan TAM çıkar."""

    __slots__ = ("n", "s", "s2", "w", "usd")

    def __init__(self):
        self.n = 0; self.s = 0.0; self.s2 = 0.0; self.w = 0; self.usd = 0.0

    def ekle(self, x: float, kazandi: bool, usd: float = 0.0) -> None:
        self.n += 1; self.s += x; self.s2 += x * x; self.w += 1 if kazandi else 0; self.usd += usd

    @property
    def ort(self) -> float:
        return self.s / self.n if self.n else 0.0

    @property
    def sd(self) -> float:
        if self.n < 2:
            return 0.0
        v = max(0.0, self.s2 / self.n - self.ort ** 2)
        return math.sqrt(v * self.n / (self.n - 1))

    @property
    def t(self) -> float:
        sd = self.sd
        return 0.0 if (self.n < 2 or sd <= 0) else self.ort / (sd / math.sqrt(self.n))

    def to_dict(self) -> Dict:
        return {"n": self.n, "ort_pct": round(self.ort, 4), "sd": round(self.sd, 4),
                "t": round(self.t, 2), "kazanma": round(self.w / self.n, 3) if self.n else None,
                "net_usd": round(self.usd, 3)}


def kac_islem_gerek(b: Biriktirici, hedef_t: float = 2.0) -> Optional[int]:
    """ŞU ANKİ etki büyüklüğü sürerse |t| = hedef_t'ye ulaşmak için KAÇ işlem daha lazım?

    t = ort/(sd/√n) → n* = (hedef_t · sd / ort)².  Kanıt toplamayı ölçülebilir yapan sayı budur.

    İki özel durum AYRI ele alınır (testler bu ayrımı yakaladı):
      • ETKİ YOK (|ort| ≈ 0 ya da ort/sd sıfıra yakın) → None. "Bekleyerek kanıtlanmaz"
        demektir; uydurma bir sayı vermek sonsuz beklemeyi haklı gösterirdi.
      • DAĞILIM YOK (sd = 0, tüm sonuçlar aynı) → t sonsuz, ek işlem GEREKMEZ → 0.
        Bunu None döndürmek "asla kanıtlanmaz" anlamına gelirdi; tam TERSİ doğru."""
    if b.n < 5 or abs(b.ort) < 1e-9:
        return None
    if b.sd <= 0:
        return 0                                   # sapma yok → t sonsuz, kanıt zaten var
    if abs(b.ort) / b.sd < 1e-4:                   # etki büyüklüğü ölçülemez ölçüde küçük
        return None
    n_gerek = (hedef_t * b.sd / abs(b.ort)) ** 2
    if not math.isfinite(n_gerek) or n_gerek > 100000:
        return None
    return max(0, int(math.ceil(n_gerek)) - b.n)


def ozet(output_dir: str, tag: str = "", since: Optional[float] = None,
         min_n: int = 3) -> Dict:
    """Tek geçişte bütün kanıt kesitleri. Bellek: kesit sayısı kadar (yüzlerce bayt)."""
    kesit = {"sleeve": {}, "sleeve_rejim": {}, "seans": {}, "emir": {}, "cikis_modu": {}, "parite": {}}
    genel = Biriktirici()
    ilk = son = None
    for r in oku(output_dir, tag, since=since):
        x = float(r.get("np") or 0.0); w = bool(r.get("w")); u = float(r.get("nu") or 0.0)
        genel.ekle(x, w, u)
        ts = int(r.get("ts") or 0)
        ilk = ts if ilk is None else min(ilk, ts); son = ts if son is None else max(son, ts)
        for ad, anahtar in (("sleeve", r.get("slv")),
                            ("sleeve_rejim", f"{r.get('slv')}|{r.get('rej')}"),
                            ("seans", r.get("ss")), ("emir", r.get("ot")),
                            ("cikis_modu", r.get("xm")), ("parite", r.get("sym"))):
            if anahtar is None:
                continue
            d = kesit[ad].setdefault(str(anahtar), Biriktirici())
            d.ekle(x, w, u)
    out = {"genel": genel.to_dict(), "ilk_ts": ilk, "son_ts": son}
    for ad, m in kesit.items():
        out[ad] = {k: {**v.to_dict(), "kalan_islem_t2": kac_islem_gerek(v)}
                   for k, v in sorted(m.items(), key=lambda kv: -kv[1].n) if v.n >= min_n}
    return out


def dondur(output_dir: str, tag: str = "", max_satir: int = 200_000) -> Optional[int]:
    """Kanıt defteri çok büyürse EN ESKİ satırları arşive taşı (silme YOK).
    200.000 satır ≈ 8 yıl (65 işlem/gün) — pratikte hiç tetiklenmez, ama sınır yazılıdır."""
    p = yol(output_dir, tag)
    if not p.exists():
        return None
    n = sum(1 for _ in open(p, encoding="utf-8"))
    if n <= max_satir:
        return None
    tut = max_satir // 2
    with open(p, encoding="utf-8") as f:
        satirlar = f.readlines()
    ars = p.with_suffix(f".{int(time.time())}.arsiv.jsonl")
    with open(ars, "w", encoding="utf-8") as f:
        f.writelines(satirlar[:-tut])
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.writelines(satirlar[-tut:])
    os.replace(tmp, p)
    return n - tut
