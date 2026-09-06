"""
HABER ETKİ MOTORU — hangi TÜR haber, hangi PARİTEDE, hangi YÖNDE, yüzde KAÇ hareket yapıyor?

NEDEN VAR
─────────
`news_scanner.EVENT_PRIOR` 19 olay türü için elle yazılmış yön/şiddet değerleri taşıyor
(LISTING +0,6 · HACK −0,9 · TOKEN_UNLOCK −0,3 …). Bunların HİÇBİRİ ölçülmedi —
`event_study.py`'nin hesap ağırlıkları için söylediği şey burada da geçerli:

    "Ölçülmemiş bir ağırlık, modeli o yöne eğer ve sahte güven üretir."

Ayrıca `scan()` her taramada olay türünü, kaynak katmanını ve duyguyu ZATEN üretiyordu;
bu bilgi hiçbir yerde saklanmıyordu. Yani veri vardı, öğrenme yoktu.

NASIL ÇALIŞIR (kaçırılan-fırsat motorunun aynı deseni: gözle → ufuk dolunca çöz)
──────────────────────────────────────────────────────────────────────────────
  1. GÖZLEM  — haber görüldüğü anda kaydedilir: (ts, kategori, parite, kaynak katmanı,
     duygu, o anki fiyat, o anki ATR%). ~140 B.
  2. ÇÖZÜM   — 5 dk / 1 sa / 4 sa / 24 sa ufukları dolunca o anki fiyattan GERÇEKLEŞEN
     getiri yazılır. Bekleyen gözlemler diskte tutulur; süreç yeniden başlasa da kaybolmaz.
  3. ÖĞRENME — (kategori) ve (kategori × parite) kesitlerinde yeterli istatistik.

İKİ AYRI SORU — ve bunları birbirinin kanıtı saymak en sık yapılan hatadır:
  • BÜYÜKLÜK: haber sonrası hareket normalden büyük mü?  → |getiri| / ATR-normalize
  • YÖN     : haber yönü öngörüyor mu?                   → işaretli getiri + t testi
Bir kategori "büyük hareket yapıyor" olabilir ama YÖNÜ öngörülemez olabilir; o durumda
o kategoriye göre AL/SAT yapmak kayıptır, yalnız pozisyon boyutu küçültülür.

İSTATİSTİK KAPISI: bir kesit ancak `MIN_GOZLEM` gözlem VE |t| ≥ 2 ise "ölçüldü" sayılır.
Aksi hâlde `olculdu=False` döner ve EVENT_PRIOR bir VARSAYIM olarak kalır — kanıt değil.

Bellek: gözlemler salt-ekleme JSONL; okuma akış. Bekleyen kuyruk ufuk süresiyle sınırlı
(24 saat), dolayısıyla sınırsız büyüyemez.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

UFUKLAR: Dict[str, int] = {"5m": 300, "1h": 3600, "4h": 14400, "24h": 86400}
MIN_GOZLEM = 12          # bu sayının altında istatistik YAPILMAZ
T_ESIK = 2.0             # "ölçüldü" için |t| eşiği
MAX_BEKLEYEN = 5000      # kuyruk tavanı (koruma; normalde ufuk süresiyle sınırlı)


def _yol(output_dir: str, tag: str, ad: str) -> Path:
    p = Path(output_dir)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[2] / p
    p = p / "live"
    p.mkdir(parents=True, exist_ok=True)
    return p / (f"{ad}_{tag}.jsonl" if tag else f"{ad}.jsonl")


@dataclass
class Gozlem:
    ts: float
    sym: str
    kat: str                 # olay kategorisi (news_scanner.EVENT_TAXONOMY)
    tier: int                # kaynak katmanı 1/2/3
    duygu: float             # -1..+1 (sözlük tabanlı, kaba)
    p0: float                # gözlem anındaki fiyat
    atr: float               # gözlem anındaki ATR% (normalizasyon için)
    kalan: List[str] = field(default_factory=lambda: list(UFUKLAR))

    def to_dict(self) -> Dict:
        return {"ts": round(self.ts, 1), "sym": self.sym, "kat": self.kat, "tier": self.tier,
                "duygu": round(self.duygu, 3), "p0": self.p0, "atr": round(self.atr, 4),
                "kalan": self.kalan}

    @classmethod
    def from_dict(cls, d: Dict) -> "Gozlem":
        return cls(float(d["ts"]), str(d["sym"]), str(d["kat"]), int(d.get("tier") or 3),
                   float(d.get("duygu") or 0.0), float(d["p0"]), float(d.get("atr") or 0.3),
                   list(d.get("kalan") or UFUKLAR))


class HaberEtkiMotoru:
    """Gözle → çöz → öğren. Ağ erişimi yok; fiyatlar dışarıdan verilir."""

    def __init__(self, output_dir: str = "runs", tag: str = ""):
        self.output_dir, self.tag = output_dir, tag
        self.kayit = _yol(output_dir, tag, "news_impact")     # çözülmüş gözlemler
        self.bekleyen_yol = _yol(output_dir, tag, "news_pending")
        self.bekleyen: List[Gozlem] = []
        self._yukle()

    # ---------------------------------------------------------------- kalıcılık
    def _yukle(self) -> None:
        try:
            if self.bekleyen_yol.exists():
                self.bekleyen = [Gozlem.from_dict(json.loads(l))
                                 for l in self.bekleyen_yol.read_text(encoding="utf-8").splitlines() if l.strip()]
        except Exception:
            self.bekleyen = []

    def _kaydet_bekleyen(self) -> None:
        try:
            tmp = self.bekleyen_yol.with_suffix(".tmp")
            tmp.write_text("\n".join(json.dumps(g.to_dict(), ensure_ascii=False, separators=(",", ":"))
                                     for g in self.bekleyen), encoding="utf-8")
            tmp.replace(self.bekleyen_yol)
        except Exception:
            pass

    # ---------------------------------------------------------------- 1) GÖZLE
    def gozle(self, sym: str, kat: str, tier: int, duygu: float, fiyat: float,
              atr_pct: float, now: Optional[float] = None,
              tekil_pencere_sn: int = 1800) -> bool:
        """Haberi kaydet. AYNI (parite, kategori) için `tekil_pencere_sn` içinde ikinci
        gözlem alınmaz — aynı olayın birden çok başlıkla tekrar sayılması, örneklemi
        şişirip t'yi sahte biçimde büyütürdü (haber kaynakları aynı başlığı tekrarlar)."""
        now = time.time() if now is None else now
        if not (fiyat and fiyat > 0) or not kat or kat == "OTHER":
            return False
        for g in self.bekleyen:
            if g.sym == sym and g.kat == kat and now - g.ts < tekil_pencere_sn:
                return False
        if len(self.bekleyen) >= MAX_BEKLEYEN:
            self.bekleyen = self.bekleyen[-(MAX_BEKLEYEN // 2):]
        self.bekleyen.append(Gozlem(now, sym, kat, int(tier or 3), float(duygu or 0.0),
                                    float(fiyat), float(atr_pct or 0.3)))
        self._kaydet_bekleyen()
        return True

    # ---------------------------------------------------------------- 2) ÇÖZ
    def coz(self, fiyatlar: Dict[str, float], now: Optional[float] = None) -> List[Dict]:
        """Ufku dolan gözlemleri GERÇEKLEŞEN getiriyle kapat. Her döngüde çağrılır; ucuz."""
        now = time.time() if now is None else now
        yeni: List[Dict] = []
        kalanlar: List[Gozlem] = []
        for g in self.bekleyen:
            px = fiyatlar.get(g.sym)
            dolan = [u for u in g.kalan if now - g.ts >= UFUKLAR[u]]
            if px and px > 0:
                for u in dolan:
                    r = (px / g.p0 - 1.0) * 100.0
                    # ATR-normalize: "normalden büyük mü?" sorusu için ölçek bağımsız kat
                    olcek = max(1e-6, g.atr * math.sqrt(UFUKLAR[u] / 60.0))
                    satir = {"ts": int(g.ts), "sym": g.sym, "kat": g.kat, "tier": g.tier,
                             "duygu": g.duygu, "ufuk": u, "r": round(r, 4),
                             "z": round(r / olcek, 3), "atr": g.atr}
                    self._ekle(satir)
                    yeni.append(satir)
                g.kalan = [u for u in g.kalan if u not in dolan]
            elif dolan and now - g.ts > UFUKLAR["24h"] * 1.5:
                g.kalan = []          # fiyat hiç gelmedi ve çok eskidi → düş
            if g.kalan:
                kalanlar.append(g)
        if len(kalanlar) != len(self.bekleyen):
            self.bekleyen = kalanlar
            self._kaydet_bekleyen()
        return yeni

    def _ekle(self, satir: Dict) -> None:
        try:
            with open(self.kayit, "a", encoding="utf-8") as f:
                f.write(json.dumps(satir, ensure_ascii=False, separators=(",", ":")) + "\n")
        except Exception:
            pass

    # ---------------------------------------------------------------- 3) ÖĞREN
    def satirlar(self, ufuk: Optional[str] = None) -> Iterator[Dict]:
        if not self.kayit.exists():
            return
        with open(self.kayit, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if ufuk and r.get("ufuk") != ufuk:
                    continue
                yield r

    def ozet(self, ufuk: str = "4h", min_n: int = MIN_GOZLEM) -> Dict:
        """(kategori) ve (kategori × parite) kesitlerinde BÜYÜKLÜK ve YÖN — ayrı ayrı."""
        kat: Dict[str, List[Dict]] = {}
        kat_sym: Dict[str, List[Dict]] = {}
        for r in self.satirlar(ufuk):
            kat.setdefault(r["kat"], []).append(r)
            kat_sym.setdefault(f"{r['kat']}|{r['sym']}", []).append(r)

        def hesap(rows: List[Dict]) -> Dict:
            n = len(rows)
            rs = [float(x["r"]) for x in rows]
            zs = [float(x["z"]) for x in rows]
            ort = sum(rs) / n
            sd = math.sqrt(max(0.0, sum((x - ort) ** 2 for x in rs) / max(1, n - 1)))
            t = ort / (sd / math.sqrt(n)) if (n > 1 and sd > 0) else 0.0
            buyukluk = sum(abs(z) for z in zs) / n          # ATR-normalize mutlak hareket
            yukari = sum(1 for x in rs if x > 0) / n
            return {
                "n": n,
                "yon_ort_pct": round(ort, 4),
                "yon_t": round(t, 2),
                "yon_olculdu": bool(n >= min_n and abs(t) >= T_ESIK),
                "yukari_orani": round(yukari, 3),
                "buyukluk_z": round(buyukluk, 3),           # 1,0 ≈ normal oynaklık
                "buyukluk_olculdu": bool(n >= min_n and buyukluk >= 1.3),
                "medyan_pct": round(sorted(rs)[n // 2], 4),
                "p90_pct": round(sorted(rs)[min(n - 1, int(0.9 * n))], 4),
            }

        return {
            "ufuk": ufuk,
            "toplam_gozlem": sum(len(v) for v in kat.values()),
            "bekleyen": len(self.bekleyen),
            "kategori": {k: hesap(v) for k, v in sorted(kat.items(), key=lambda kv: -len(kv[1])) if len(v) >= 3},
            "kategori_parite": {k: hesap(v) for k, v in sorted(kat_sym.items(), key=lambda kv: -len(kv[1]))
                                if len(v) >= max(3, min_n // 2)},
        }

    def prior_karsilastir(self, ufuk: str = "4h") -> List[Dict]:
        """ELLE YAZILMIŞ `EVENT_PRIOR` ile ÖLÇÜLEN yönü karşılaştır — varsayım tutuyor mu?"""
        try:
            from .news_scanner import EVENT_PRIOR
        except Exception:
            EVENT_PRIOR = {}
        o = self.ozet(ufuk)
        out = []
        for k, v in o["kategori"].items():
            p = EVENT_PRIOR.get(k)
            if p is None:
                continue
            uyum = None
            if v["yon_olculdu"]:
                uyum = (p > 0 and v["yon_ort_pct"] > 0) or (p < 0 and v["yon_ort_pct"] < 0) or (p == 0)
            out.append({"kategori": k, "varsayim_prior": p, "olculen_pct": v["yon_ort_pct"],
                        "t": v["yon_t"], "n": v["n"], "olculdu": v["yon_olculdu"], "uyumlu": uyum})
        return sorted(out, key=lambda x: -x["n"])
