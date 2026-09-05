"""
KAÇIRILAN FIRSAT ATIF MOTORU — "yapılmayan işlem" de bir karardır ve ölçülür.

Her aday için sistemin NEDEN işlem açmadığı (hangi kapı, hangi rol, hangi eşik) ve o anda
elde OLAN / OLMAYAN bilgiler kaydedilir; aday ufuk boyunca hedef/stop'a karşı izlenir.
Çözülünce iki küme karşılaştırılır — hedefe ulaşanlar (KAÇIRILAN KAZANÇ) ve stop olanlar
(DOĞRU KAÇINMA):

  • KAPI İSABETİ            kapı kazananı mı engelliyor, kaybedeni mi? (isabet + Wilson aralığı)
  • GÖZ ARDI EDİLEN ÖZELLİK kapının hesaba KATMADIĞI ama kaçırılan kazançlarda sistematik
                            biçimde var olan özellikler (lift = P(özellik|kazanç)/P(özellik|zarar))
  • BİLGİ VARLIĞI/YOKLUĞU   ağır bağlam, haber, defter derinliği, doğrulanmış ücret, nitelendirme
                            hücresi, rol verisi — yokken kaçırma oranı vs varken
  • KÖR NOKTALAR            hiç DEĞERLENDİRİLMEYENLER de izlenir: Top-K dışı, açık-pozisyon tavanı,
                            kaynak RED, nakit/portföy modu, HALT, bayat veri, maker dolmadı / max chase

Çıktı: insan-okur Türkçe anlatı (KACIRILANLAR_*.md), JSONL, panel raporu ve — kanıt eşiği
geçilirse — CHALLENGER önerisi. Üretim parametresi buradan DOĞRUDAN değişmez.

Dürüstlük kuralları: n küçükse "ölçülmedi" denir; ufuk içinde hem hedef hem stop görülürse
AMBIGUOUS sayılır (kazanç sayılmaz); kaçırılan kazancın maliyeti gerçek komisyon/kayma ile
net olarak yazılır; öneriler yalnız kapı ZARARLI verdiğinde ve sınır içinde üretilir.
"""
from __future__ import annotations

import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional

MAX_RECORDS = 600
MIN_N_GATE = 10          # kapı isabeti için asgari çözülmüş aday
MIN_N_PROPOSAL = 20      # challenger önerisi için asgari
MIN_N_FEATURE = 6
PROPOSAL_COOLDOWN = 12 * 3600

# Kapı → hesaba kattığı girdiler (bunun dışındaki destekleyici özellikler "göz ardı edilen"dir)
GATE_CONSIDERS: Dict[str, set] = {
    "KOMİSYON": {"hedef_yuzdesi", "maliyet", "spread", "derinlik"},
    "SPREAD": {"spread"}, "DERİNLİK": {"derinlik", "emir_boyutu"},
    "RİSK": {"acik_pozisyon", "korelasyon", "maruziyet", "duraklatma", "p_win", "rr"},
    "DENETÇİ": {"giris", "stop", "hedef", "baglam_yasi", "emir_boyutu"},
    "OY": {"rol_oylari"}, "GÜVEN": {"rol_oylari", "rol_verisi"},
    "NEGATİF_EV": {"p_win", "hedef_yuzdesi", "stop_yuzdesi", "maliyet"},
    "MAX_CHASE": {"fiyat", "giris_bolgesi"}, "SAĞLIK": {"sistem_sagligi"},
    "NİTELENDİRME": {"nitelendirme_hucresi"}, "HABER_RİSKİ": {"haber"}, "OLAY": {"makro_takvim"},
    "KORELASYON": {"acik_pozisyon", "korelasyon"}, "MARUZİYET": {"maruziyet"},
    "AÇIK_POZİSYON": {"acik_pozisyon"}, "PARİTE_DURAKLATILDI": {"duraklatma"},
    "TOP_K": {"ilgi_puani"}, "MAX_OPEN": {"acik_pozisyon"}, "KAYNAK": {"bellek"},
    "PORTFÖY": {"genislik", "korelasyon", "haber_seviyesi"}, "NAKİT": {"haber_seviyesi", "genislik"},
    "HALT": {"gunluk_zarar", "drawdown"}, "MANAGE_ONLY": set(), "BAYAT": {"tazelik"},
    "MAKER_DOLMADI": {"dolum"}, "LIFECYCLE": {"asama"}, "TIER3": {"kaynak_katmani"}, "CHALLENGER": set(),
    "REJİM_SEÇİCİ": {"rejim"}, "STOP_RİSKİ": {"uyaranlar"}, "GÜNLÜK_TAVAN": {"islem_sayisi"},
    "SLEEVE_DURAKLATILDI": {"sleeve_performansi"}, "YAŞAM_DÖNGÜSÜ": {"sleeve_kaniti"}, "SLEEVE_TAVANI": {"acik_pozisyon"}, "ROTASYON": {"kalan_ev"},
}
GATE_TR = {
    "KOMİSYON": "komisyon kapısı (brüt/maliyet)", "SPREAD": "spread kapısı", "DERİNLİK": "defter derinliği",
    "RİSK": "risk rolü", "DENETÇİ": "denetçi rolü", "OY": "oy eşiği", "GÜVEN": "güven eşiği",
    "NEGATİF_EV": "negatif net EV", "MAX_CHASE": "max chase (kovalama yok)", "SAĞLIK": "sistem sağlığı",
    "NİTELENDİRME": "nitelendirme hücresi", "HABER_RİSKİ": "haber riski", "OLAY": "makro olay takvimi",
    "KORELASYON": "korelasyon", "MARUZİYET": "maruziyet odası", "AÇIK_POZİSYON": "açık pozisyon tavanı",
    "PARİTE_DURAKLATILDI": "parite duraklatma (ders)", "TOP_K": "Tier-A elemesi (Top-K dışı)",
    "MAX_OPEN": "açık pozisyon tavanı (değerlendirilmedi)", "KAYNAK": "kaynak bütçesi (RED)",
    "PORTFÖY": "portföy modu (giriş kapalı)", "NAKİT": "nakit modu", "HALT": "kill-switch",
    "MANAGE_ONLY": "yalnız pozisyon yönetimi", "BAYAT": "bayat veri", "MAKER_DOLMADI": "maker emri dolmadı",
    "LIFECYCLE": "yaşam döngüsü aşaması", "TIER3": "yalnız Tier-3 kaynak", "CHALLENGER": "challenger farkı",
    "REJİM_SEÇİCİ": "rejim seçici sleeve'i kapattı", "STOP_RİSKİ": "stop-risk skoru (giriş uyaranları)",
    "GÜNLÜK_TAVAN": "günlük işlem tavanı", "SLEEVE_DURAKLATILDI": "sleeve devre kesici (duraklatıldı)",
    "YAŞAM_DÖNGÜSÜ": "gölge aşaması (kanıt yok — emir verilmez, ölçüm sürer)",
    "SLEEVE_TAVANI": "sleeve başına açık tavanı", "ROTASYON": "rotasyon (daha iyi fırsat yok)",
}
# kapı → challenger öneri kuralı (param, delta, sınır lessons.BOUNDS'tan)
GATE_PROPOSAL = {
    "KOMİSYON": ("min_gross_to_cost", -0.5), "OY": ("theta", -0.05), "GÜVEN": ("min_confidence", -0.05),
    "TOP_K": ("top_k", +2), "MAX_OPEN": ("max_open", +1), "DENETÇİ": ("rr", -0.2),
    "GÜNLÜK_TAVAN": ("max_trades_per_day", +40),
}
PROPOSAL_BOUNDS = {"min_gross_to_cost": (1.5, 5.0), "theta": (0.1, 0.6), "min_confidence": (0.2, 0.8),
                   "top_k": (3, 20), "max_open": (1, 8), "rr": (1.0, 3.0), "max_trades_per_day": (10, 400)}

# Destekleyici özellik yüklemleri (LONG için; SHORT'ta aynalanır) → adı: (yüklem, Türkçe)
SUPPORT: Dict[str, tuple] = {
    "hacim_patlamasi": (lambda f, s: (f.get("vol_ratio") or 0) >= 1.5, "hacim patlaması (×1,5+)"),
    "derin_dip_z": (lambda f, s: (s * (f.get("z") or 0)) <= -2.0, "derin dip (z ≤ −2)"),
    "rsi_asiri": (lambda f, s: ((f.get("rsi") or 50) <= 30) if s > 0 else ((f.get("rsi") or 50) >= 70), "RSI aşırı bölge"),
    "goreli_guc_ust": (lambda f, s: (f.get("rs_rank") or 0) >= 0.8 if s > 0 else (f.get("rs_rank") if f.get("rs_rank") is not None else 1) <= 0.2, "göreli güç uç %20"),
    "trend_uyumlu": (lambda f, s: bool(f.get("trend_up")) if s > 0 else not f.get("trend_up"), "trend yönle uyumlu"),
    "ema_kesisim": (lambda f, s: bool(f.get("ema_cross_up")) and s > 0, "EMA9×21 yukarı kesişim"),
    "kirilim": (lambda f, s: bool(f.get("breakout_up")) if s > 0 else bool(f.get("breakdown")), "20-bar kırılım"),
    "sikisma_oncesi": (lambda f, s: f.get("bb_prev_pctile") is not None and f["bb_prev_pctile"] <= 20, "sıkışma sonrası (bant %20 yüzdelik)"),
    "likidite_supurme": (lambda f, s: bool(f.get("swept_low")) if s > 0 else bool(f.get("swept_high")), "likidite süpürme + geri alım"),
    "vwap_uzak": (lambda f, s: (s * (f.get("dist_vwap_pct") or 0)) <= -0.5, "VWAP'tan ≥ %0,5 uzak (dönüş potansiyeli)"),
    "range_kenari": (lambda f, s: bool(f.get("range_ok")) and ((f.get("range_pos") or 1) <= 0.2 if s > 0 else (f.get("range_pos") or 0) >= 0.8), "range kenarı"),
    "haber_katalizor": (lambda f, s: bool(f.get("news_confirmed")), "doğrulanmış haber katalizörü"),
    "dusuk_spread": (lambda f, s: (f.get("spread_bps") if f.get("spread_bps") is not None else 99) <= 3.0, "düşük spread (≤ 3 bps)"),
    "asiri_4sa": (lambda f, s: (s * (f.get("move_4h_pct") or 0)) <= -2.0, "4 saatte aşırı hareket (tersine dönüş)"),
    "trend_skoru": (lambda f, s: (f.get("trend_score") or 0) >= 0.7 and s > 0, "trend skoru ≥ 0,7"),
    "defter_dengesi": (lambda f, s: (f.get("obi") or 0.5) >= 0.65 if s > 0 else (f.get("obi") or 0.5) <= 0.35, "emir defteri dengesi yönde"),
}
# Uyaran (zarar tarafını haklı çıkaran) yüklemler
WARN: Dict[str, tuple] = {
    "haber_riski": (lambda f, s: bool(f.get("news_severe")), "ağır haber riski"),
    "trend_ters": (lambda f, s: (not f.get("trend_up")) if s > 0 else bool(f.get("trend_up")), "trend ters yönde"),
    "kirilma_ters": (lambda f, s: bool(f.get("breakdown")) if s > 0 else bool(f.get("breakout_up")), "ters yönde kırılım"),
    "spread_genis": (lambda f, s: (f.get("spread_bps") or 0) > 8.0, "geniş spread"),
    "hacim_yok": (lambda f, s: f.get("vol_ratio") is not None and f["vol_ratio"] < 0.7, "hacim yok"),
    "asiri_uzama": (lambda f, s: bool(f.get("extended")), "EMA'dan aşırı uzamış"),
}
INFO_TR = {
    "slow_ctx": "ağır bağlam (formasyon/indikatör/yapı/mover)", "news": "haber verisi", "book_depth": "defter derinliği",
    "fees_verified": "doğrulanmış ücret", "qual_cell": "nitelendirme hücresi", "roles_full": "tüm rollerin verisi",
}


def _wilson(k: int, n: int, z: float = 1.96):
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    adj = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - adj) / den), min(1.0, (c + adj) / den))


def normalize_gate(veto: str) -> str:
    v = str(veto or "").strip()
    u = v.upper()
    if u.startswith("KOMİSYON"):
        return "KOMİSYON"
    if u.startswith("SPREAD"):
        return "SPREAD"
    if u.startswith("DERİNLİK"):
        return "DERİNLİK"
    if u.startswith("OY "):
        return "OY"
    if u.startswith("GÜVEN"):
        return "GÜVEN"
    if "NEGATİF NET EV" in u:
        return "NEGATİF_EV"
    if u.startswith("MAX CHASE"):
        return "MAX_CHASE"
    if u.startswith("SAĞLIK"):
        return "SAĞLIK"
    if u.startswith("STOP RİSKİ"):
        return "STOP_RİSKİ"
    if u.startswith("NİTELENDİRME"):
        return "NİTELENDİRME"
    if u.startswith("HABER RİSKİ"):
        return "HABER_RİSKİ"
    if u.startswith("OLAY"):
        return "OLAY"
    if u.startswith("KORELASYON"):
        return "KORELASYON"
    if u.startswith("MARUZİYET"):
        return "MARUZİYET"
    if u.startswith("AÇIK POZİSYON"):
        return "AÇIK_POZİSYON"
    if u.startswith("PARİTE DURAKLATILDI"):
        return "PARİTE_DURAKLATILDI"
    if u.startswith("DENETÇİ"):
        return "DENETÇİ"
    if u.startswith("KILL-SWITCH"):
        return "HALT"
    if u.startswith("CHALLENGER"):
        return "CHALLENGER"
    if u.startswith("LIFECYCLE") or u.startswith("YAŞAM"):
        return "LIFECYCLE"
    tok = v.split(":")[0].split(" ")[0].strip()
    return tok.upper() if tok else "?"


def _sign(direction: str) -> float:
    return 1.0 if str(direction).upper() == "LONG" else -1.0


def _clean(x):
    """NaN/inf → None (JSON uyumu; np.float64 dahil), iç içe sözlük/liste."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, float):
        return float(x) if math.isfinite(x) else None
    return x


class MissedEngine:
    def __init__(self, path: Optional[Path] = None, journal_md: Optional[Path] = None,
                 journal_jsonl: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.md = journal_md
        self.jsonl = journal_jsonl
        self.records: Deque[Dict] = deque(maxlen=MAX_RECORDS)
        self.blind: Dict[str, int] = {}                 # gölge açılamayan kör noktalar (ör. BAYAT)
        self._last_proposal: Dict[str, float] = {}
        self.proposed: Dict[str, float] = {}
        self._dirty = False
        self.load()

    # ------------------------------------------------------------ kalıcılık
    def load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            self.records = deque(d.get("records", []), maxlen=MAX_RECORDS)
            self.blind = d.get("blind", {})
            self._last_proposal = d.get("last_proposal", {})
            self.proposed = d.get("proposed", {})
        except Exception:
            pass

    def save(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"records": list(self.records), "blind": self.blind,
                                       "last_proposal": self._last_proposal, "proposed": self.proposed,
                                       "saved_ts": time.time()}, ensure_ascii=False, default=str), encoding="utf-8")
            tmp.replace(self.path)
        except Exception:
            pass

    # ------------------------------------------------------------ kayıt
    def _open_for(self, symbol: str, gate: Optional[str] = None) -> bool:
        """Aynı parite + aynı kapı için tek açık gölge (veto ve susturulan-sleeve gölgeleri birlikte yaşayabilir)."""
        return any(r["symbol"] == symbol and r["outcome"] is None and (gate is None or r.get("gate") == gate) for r in self.records)

    @staticmethod
    def _features_from_trace(trace: Dict, book: Optional[Dict], news: Optional[Dict]) -> Dict:
        f = dict(trace.get("fast") or {})
        f["spread_bps"] = (book or {}).get("spread_bps")
        f["depth_usd"] = min(float((book or {}).get("bid_depth_usd") or 0.0), float((book or {}).get("ask_depth_usd") or 0.0))
        f["news_confirmed"] = bool((news or {}).get("confirmed"))
        f["news_severe"] = bool((news or {}).get("severe_risk"))
        f["news_score"] = (news or {}).get("score")
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in f.items()
                if isinstance(v, (int, float, bool)) or v is None}

    def _add(self, rec: Dict) -> Optional[Dict]:
        if self._open_for(rec["symbol"], rec.get("gate")):
            return None
        rec.setdefault("outcome", None)
        rec.setdefault("mfe_pct", 0.0)
        rec.setdefault("mae_pct", 0.0)
        rec["id"] = f"{rec['symbol']}@{int(rec['ts'])}"
        for k in ("features", "plan", "context", "info"):
            if k in rec:
                rec[k] = _clean(rec[k])
        self.records.append(rec)
        self._dirty = True
        return rec

    def on_vetoed(self, trace: Dict, info: Dict, book: Optional[Dict], news: Optional[Dict],
                  horizon_sec: float, now: float, context: Optional[Dict] = None) -> Optional[Dict]:
        """Komite değerlendirdi ama veto etti: plan + vetolar + oylar + özellikler + bilgi bayrakları."""
        plan = trace.get("plan")
        if not plan or not trace.get("direction"):
            return None
        vetoes = [str(x) for x in (trace.get("vetoes") or [])]
        gates = [normalize_gate(v) for v in vetoes] or ["?"]
        votes = [{"role": v.get("role"), "score": v.get("score"), "confidence": v.get("confidence"),
                  "data_ok": v.get("data_ok", True), "veto": v.get("veto")} for v in (trace.get("votes") or [])]
        no_data = [v["role"] for v in votes if not v.get("data_ok", True)]
        info = dict(info or {})
        info["roles_full"] = len(no_data) == 0
        info["roles_no_data"] = no_data
        rec = {"kind": "veto", "symbol": trace["symbol"], "ts": float(now), "direction": trace["direction"],
               "sleeve": trace.get("trigger"), "gate": gates[0], "gates": gates, "vetoes": vetoes,
               "reason_detail": "; ".join(vetoes)[:300],
               "plan": {k: plan.get(k) for k in ("entry", "stop", "target", "stop_pct", "target_pct", "rr")},
               "features": self._features_from_trace(trace, book, news), "info": info, "votes": votes,
               "context": {**(context or {}), "ev_pct": (trace.get("ticket") or {}).get("ev_pct"),
                           "score": trace.get("score"), "confidence": trace.get("confidence"),
                           "competition": [c.get("kind") for c in (trace.get("competition") or [])]},
               "expires": float(now) + float(horizon_sec)}
        return self._add(rec)

    def on_unevaluated(self, symbol: str, gate: str, cheap: Dict, price: float, direction: str,
                       stop_pct: float, target_pct: float, horizon_sec: float, now: float,
                       info: Optional[Dict] = None, context: Optional[Dict] = None, detail: str = "") -> Optional[Dict]:
        """Hiç değerlendirilmeyen aday (Top-K dışı, tavan, RED, nakit, HALT…) — genel planla gölge."""
        s = _sign(direction)
        rec = {"kind": "unevaluated", "symbol": symbol, "ts": float(now), "direction": direction, "sleeve": None,
               "setup": (cheap or {}).get("kind"),
               "gate": gate, "gates": [gate], "vetoes": [], "reason_detail": detail or GATE_TR.get(gate, gate),
               "plan": {"entry": price, "stop": price * (1 - s * stop_pct / 100.0), "target": price * (1 + s * target_pct / 100.0),
                        "stop_pct": round(stop_pct, 4), "target_pct": round(target_pct, 4), "rr": round(target_pct / max(1e-9, stop_pct), 3)},
               "features": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in (cheap or {}).items()
                            if isinstance(v, (int, float, bool)) or v is None},
               "info": dict(info or {}), "votes": [], "context": dict(context or {}),
               "expires": float(now) + float(horizon_sec)}
        return self._add(rec)

    def on_execution_miss(self, trace: Dict, plan: Dict, gate: str, detail: str, horizon_sec: float, now: float,
                          info: Optional[Dict] = None) -> Optional[Dict]:
        """Karar AÇ dedi ama yürütme gerçekleşmedi (maker dolmadı / max chase) — yürütme kaynaklı kaçırma."""
        rec = {"kind": "execution", "symbol": trace.get("symbol"), "ts": float(now), "direction": trace.get("direction") or plan.get("direction"),
               "sleeve": trace.get("trigger"), "gate": gate, "gates": [gate], "vetoes": [], "reason_detail": detail,
               "plan": {k: plan.get(k) for k in ("entry", "stop", "target", "stop_pct", "target_pct", "rr")},
               "features": self._features_from_trace(trace, None, None), "info": dict(info or {}), "votes": [],
               "context": {"ev_pct": (trace.get("ticket") or {}).get("ev_pct")}, "expires": float(now) + float(horizon_sec)}
        return self._add(rec)

    def count_blind(self, gate: str, n: int = 1) -> None:
        self.blind[gate] = self.blind.get(gate, 0) + int(n)

    # ------------------------------------------------------------ takip / çözüm
    def update(self, bars: Dict[str, "object"], now: float, cost_pct: float = 0.1) -> List[Dict]:
        """bars: symbol → DataFrame(high, low, close). MFE/MAE izler, ilk-geçişle çözer, atıf yazar."""
        resolved = []
        for r in self.records:
            if r["outcome"] is not None:
                continue
            df = bars.get(r["symbol"])
            if df is not None and len(df):
                hi = float(df["high"].iloc[-1]); lo = float(df["low"].iloc[-1])
                s = _sign(r["direction"])
                e = float(r["plan"]["entry"])
                fav = ((hi if s > 0 else lo) / e - 1.0) * 100.0 * s
                adv = ((lo if s > 0 else hi) / e - 1.0) * 100.0 * s
                r["mfe_pct"] = round(max(float(r.get("mfe_pct") or 0.0), fav), 4)
                r["mae_pct"] = round(min(float(r.get("mae_pct") or 0.0), adv), 4)
                t, st = float(r["plan"]["target"]), float(r["plan"]["stop"])
                hit_t = hi >= t if s > 0 else lo <= t
                hit_s = lo <= st if s > 0 else hi >= st
                if hit_t and hit_s:
                    r["outcome"] = "AMBIGUOUS"
                elif hit_t:
                    r["outcome"] = "TARGET"
                elif hit_s:
                    r["outcome"] = "STOP"
            r["cycles_seen"] = int(r.get("cycles_seen") or 0) + 1
            if r["outcome"] is None and now >= float(r["expires"]):
                r["outcome"] = "TIMEOUT"
            if r["outcome"] is not None:
                r["resolved_ts"] = now
                r["attribution"] = self.attribute(r, cost_pct)
                self._journal(r)
                resolved.append(r)
        if resolved or self._dirty:
            self.save()
            self._dirty = False
        return resolved

    # ------------------------------------------------------------ atıf
    def attribute(self, r: Dict, cost_pct: float = 0.1) -> Dict:
        s = _sign(r["direction"])
        f = r.get("features") or {}
        gate = r.get("gate") or "?"
        considered = set()
        for g in r.get("gates") or [gate]:
            considered |= GATE_CONSIDERS.get(g, set())
        supportive = [name for name, (pred, _) in SUPPORT.items() if _safe(pred, f, s)]
        warning = [name for name, (pred, _) in WARN.items() if _safe(pred, f, s)]
        # kapının zaten baktığı şeyler "göz ardı" sayılmaz (ör. komisyon kapısı spread'e bakar)
        ignore_map = {"dusuk_spread": "spread", "haber_katalizor": "haber", "defter_dengesi": "derinlik"}
        ignored = [n for n in supportive if ignore_map.get(n, n) not in considered]
        info = r.get("info") or {}
        missing = [k for k in ("slow_ctx", "news", "book_depth", "fees_verified", "qual_cell", "roles_full")
                   if k in info and not info.get(k)]
        out = {"gate": gate, "gate_tr": GATE_TR.get(gate, gate), "why_blocked": r.get("reason_detail"),
               "ignored_supportive": ignored, "ignored_supportive_tr": [SUPPORT[n][1] for n in ignored],
               "warnings_present": warning, "warnings_present_tr": [WARN[n][1] for n in warning],
               "missing_info": missing, "missing_info_tr": [INFO_TR.get(k, k) for k in missing],
               "roles_no_data": info.get("roles_no_data") or [], "mfe_pct": r.get("mfe_pct"), "mae_pct": r.get("mae_pct")}
        tp = float(r["plan"].get("target_pct") or 0.0)
        sp = float(r["plan"].get("stop_pct") or 0.0)
        mins = (float(r.get("resolved_ts") or r["ts"]) - float(r["ts"])) / 60.0
        mae = abs(float(r.get("mae_pct") or 0.0)); mfe = float(r.get("mfe_pct") or 0.0)
        out["minutes"] = round(mins, 1)
        out["mae_over_stop"] = (round(mae / sp, 2) if sp > 0 else None)
        out["mfe_over_target"] = (round(mfe / tp, 2) if tp > 0 else None)
        if r["outcome"] == "TARGET":
            out["path"] = (f"hedefe {mins:.0f} dk'da ulaştı; önce aleyhe %{mae:.2f} gitti (stop'un %{(mae / sp * 100) if sp else 0:.0f}'i)"
                           + (" — neredeyse stop oluyordu" if sp and mae / sp >= 0.6 else " — temiz yol"))
        elif r["outcome"] == "STOP":
            out["path"] = (f"{mins:.0f} dk'da stop; lehte en fazla %{mfe:.2f} gitti (hedefin %{(mfe / tp * 100) if tp else 0:.0f}'i)"
                           + (" — hedefe yaklaşıp döndü, geri-verme çıkışı kurtarırdı" if tp and mfe / tp >= 0.5 else " — hiç çalışmadı"))
        else:
            out["path"] = f"{mins:.0f} dk ufuk; MFE %{mfe:.2f} / MAE %{mae:.2f}"
        if r["outcome"] == "TARGET":
            out["verdict"] = "KAÇIRILAN KAZANÇ"
            out["net_missed_pct"] = round(tp - cost_pct, 4)
            out["how"] = self._how_missed(r, out)
        elif r["outcome"] == "STOP":
            out["verdict"] = "DOĞRU KAÇINMA"
            out["net_avoided_pct"] = round(float(r["plan"].get("stop_pct") or 0.0) + cost_pct, 4)
            out["how"] = self._how_avoided(r, out)
        elif r["outcome"] == "AMBIGUOUS":
            out["verdict"] = "BELİRSİZ (aynı barda hedef+stop)"
            out["how"] = "aynı bar içinde hem hedef hem stop görüldü — kazanç sayılmadı"
        else:
            out["verdict"] = "NÖTR (ufuk doldu)"
            out["how"] = f"ufuk içinde hedefe de stopa da değmedi (MFE %{r.get('mfe_pct')}, MAE %{r.get('mae_pct')})"
        return out

    @staticmethod
    def _how_missed(r: Dict, a: Dict) -> str:
        gate = a["gate"]
        parts = [f"{a['gate_tr']} engelledi: {a['why_blocked']}"]
        if a["ignored_supportive_tr"]:
            parts.append("kapının hesaba KATMADIĞI destekleyiciler vardı: " + ", ".join(a["ignored_supportive_tr"]))
        if a["missing_info_tr"]:
            parts.append("EKSİK bilgi: " + ", ".join(a["missing_info_tr"]) +
                         (" (veri olmayan roller: " + ", ".join(a["roles_no_data"]) + " — oy/güven bu yüzden düşük kaldı)"
                          if a["roles_no_data"] and gate in ("OY", "GÜVEN") else ""))
        if r.get("kind") == "unevaluated":
            parts.append("aday hiç DEĞERLENDİRİLMEDİ — komite çalışmadı; sebep işlem mantığı değil, kapasite/durum")
        if r.get("kind") == "execution":
            parts.append("karar AÇ demişti; kayıp YÜRÜTME kaynaklı (dolum/kovalama)")
        fix = {"KOMİSYON": "brüt/maliyet eşiği bu kapıda zarar üretiyorsa challenger'a −0,5 önerilir (sınır 1,5)",
               "OY": "oy eşiği (theta) challenger'a −0,05 önerilir; veri olmayan rollerin ağırlığı sıfırlanmalı",
               "GÜVEN": "güven eşiği challenger'a −0,05; ağır bağlam yoksa hafif bağlam rolleri devreye girmeli",
               "TOP_K": "Tier-A ilgi puanı bu kurulumu yakalamıyor → top_k +2 ve ilgi puanına ilgili özellik eklenmeli",
               "MAX_OPEN": "kapasite doluydu → max_open +1 (yalnız kanıtla) ya da EV'ye göre pozisyon değiştirme",
               "MAX_CHASE": "giriş bölgesi dar; maker beklemesi kısaltılabilir (maker_wait_bars) ya da chase toleransı ölçülmeli",
               "REJİM_SEÇİCİ": "bu sleeve bu rejimde kapalı; kanıt birikirse REGIME_SLEEVES listesine eklenmeli (kod değişikliği, challenger değil)",
               "YAŞAM_DÖNGÜSÜ": "sleeve GÖLGE aşamasında: ölçülmüş kenarı yok, emir vermez; gölge sonuçları biriktikçe replay kapılarıyla PAPER'a terfi edebilir",
               "MAKER_DOLMADI": "maker dolum olasılığı düşük; entry optimizer taker'ı seçmeliydi (P(fill) kalibrasyonu)",
               "NEGATİF_EV": "p_win öncülü düşük kalmış olabilir; sleeve bazlı p_win ölçülmeli (meta-tahsisçi)"}
        if gate in fix:
            parts.append("NASIL DÜZELİR: " + fix[gate])
        return " · ".join(parts)

    @staticmethod
    def _how_avoided(r: Dict, a: Dict) -> str:
        parts = [f"{a['gate_tr']} doğru engelledi: {a['why_blocked']}"]
        if a["warnings_present_tr"]:
            parts.append("uyaran bilgiler VARDI: " + ", ".join(a["warnings_present_tr"]))
        if a["ignored_supportive_tr"]:
            parts.append("destekleyici görünen ama yanıltan özellikler: " + ", ".join(a["ignored_supportive_tr"]))
        return " · ".join(parts)

    # ------------------------------------------------------------ rapor
    def report(self, now: Optional[float] = None) -> Dict:
        now = time.time() if now is None else now
        done = [r for r in self.records if r.get("outcome") in ("TARGET", "STOP")]
        gates: Dict[str, Dict] = {}
        for r in self.records:
            for g in r.get("gates") or [r.get("gate", "?")]:
                gs = gates.setdefault(g, {"gate": g, "gate_tr": GATE_TR.get(g, g), "n": 0, "missed": 0, "avoided": 0,
                                          "timeout": 0, "ambiguous": 0, "open": 0, "missed_net_pct": 0.0})
                gs["n"] += 1
                o = r.get("outcome")
                if o == "TARGET":
                    gs["missed"] += 1
                    gs["missed_net_pct"] = round(gs["missed_net_pct"] + float((r.get("attribution") or {}).get("net_missed_pct") or 0.0), 3)
                elif o == "STOP":
                    gs["avoided"] += 1
                elif o == "TIMEOUT":
                    gs["timeout"] += 1
                elif o == "AMBIGUOUS":
                    gs["ambiguous"] += 1
                else:
                    gs["open"] += 1
        for g, gs in gates.items():
            k, n = gs["avoided"], gs["avoided"] + gs["missed"]
            gs["precision"] = round(k / n, 3) if n else None
            lo, hi = _wilson(k, n)
            gs["wilson"] = [round(lo, 3), round(hi, 3)]
            if n < MIN_N_GATE:
                gs["verdict"] = f"ÖLÇÜLMEDİ ({n}/{MIN_N_GATE})"
            elif hi < 0.5:
                gs["verdict"] = "ZARARLI — kazananı engelliyor"
            elif lo > 0.5:
                gs["verdict"] = "KORUYOR — kaybedeni engelliyor"
            else:
                gs["verdict"] = "BELİRSİZ"
        # özellik lift'i
        missed = [r for r in done if r["outcome"] == "TARGET"]
        avoided = [r for r in done if r["outcome"] == "STOP"]
        feats = []
        for name, (pred, tr) in SUPPORT.items():
            k1 = sum(1 for r in missed if _safe(pred, r.get("features") or {}, _sign(r["direction"])))
            k2 = sum(1 for r in avoided if _safe(pred, r.get("features") or {}, _sign(r["direction"])))
            n1, n2 = len(missed), len(avoided)
            if n1 + n2 < MIN_N_FEATURE:
                continue
            p1 = (k1 + 0.5) / (n1 + 1.0); p2 = (k2 + 0.5) / (n2 + 1.0)
            feats.append({"feature": name, "feature_tr": tr, "in_missed": k1, "n_missed": n1, "in_avoided": k2,
                          "n_avoided": n2, "lift": round(p1 / p2, 2),
                          "note": ("kaçırılan kazançlarda belirgin → göz ardı ediliyor" if p1 / p2 >= 1.5 and k1 >= 3
                                   else "zararlarda belirgin → yanıltıcı" if p2 / p1 >= 1.5 and k2 >= 3 else "ayırt etmiyor")})
        feats.sort(key=lambda x: -x["lift"])
        # bilgi varlığı / yokluğu
        info_rows = []
        for key, tr in INFO_TR.items():
            with_ = [r for r in done if (r.get("info") or {}).get(key) is True]
            without = [r for r in done if (r.get("info") or {}).get(key) is False]
            if len(with_) + len(without) < MIN_N_FEATURE:
                continue
            mw = sum(1 for r in with_ if r["outcome"] == "TARGET") / len(with_) if with_ else None
            mo = sum(1 for r in without if r["outcome"] == "TARGET") / len(without) if without else None
            info_rows.append({"info": key, "info_tr": tr, "n_with": len(with_), "n_without": len(without),
                              "missed_rate_with": (None if mw is None else round(mw, 3)),
                              "missed_rate_without": (None if mo is None else round(mo, 3)),
                              "note": ("bilgi YOKKEN daha çok kaçırılıyor → yokluğu karar kalitesini düşürüyor"
                                       if mw is not None and mo is not None and mo > mw + 0.15 else
                                       "bilgi VARKEN daha çok kaçırılıyor → bilgi yanlış yönlendiriyor"
                                       if mw is not None and mo is not None and mw > mo + 0.15 else "fark ölçülmedi")})
        kinds = {}
        for r in self.records:
            kinds[r.get("kind")] = kinds.get(r.get("kind"), 0) + 1
        winner_profile = self._profile(missed, "TARGET")
        stop_profile = self._profile(avoided, "STOP")
        recent = [{"id": r["id"], "symbol": r["symbol"], "ts": r["ts"], "direction": r["direction"], "sleeve": r.get("sleeve"),
                   "kind": r.get("kind"), "gate": r.get("gate"), "outcome": r.get("outcome"), "mfe_pct": r.get("mfe_pct"),
                   "attribution": r.get("attribution")} for r in list(self.records)[::-1] if r.get("outcome")][:20]
        total_missed_net = round(sum(float((r.get("attribution") or {}).get("net_missed_pct") or 0.0) for r in missed), 3)
        return {"n_records": len(self.records), "n_open": sum(1 for r in self.records if r["outcome"] is None),
                "n_missed": len(missed), "n_avoided": len(avoided), "missed_net_pct_sum": total_missed_net,
                "gates": sorted(gates.values(), key=lambda g: -g["n"]), "features": feats, "info": info_rows,
                "kinds": kinds, "blind": self.blind, "recent": recent, "proposals": self.proposals(now, dry_run=True),
                "winner_profile": winner_profile, "stop_profile": stop_profile, "interest_weights": self.interest_weights(),
                "note": ("Kaçırılan kazanç = veto/atlanan aday ufuk içinde hedefe ULAŞTI (stop'a değmeden). "
                         "n küçükken sonuç 'ÖLÇÜLMEDİ'; kapı yalnız Wilson üst sınırı < 0,5 ise ZARARLI. "
                         "Öneriler challenger'a gider, üretim parametresi doğrudan değişmez.")}

    def _profile(self, rows: List[Dict], outcome: str) -> Dict:
        """Kazanan/stop profili: nasıl ulaştı / neden ulaşamadı — strateji geliştirme girdisi."""
        if not rows:
            return {"n": 0}
        import statistics as _st
        mins = [float((r.get("attribution") or {}).get("minutes")) for r in rows if (r.get("attribution") or {}).get("minutes") is not None]
        ratio_key = "mae_over_stop" if outcome == "TARGET" else "mfe_over_target"
        ratios = [float((r.get("attribution") or {}).get(ratio_key)) for r in rows if (r.get("attribution") or {}).get(ratio_key) is not None]
        by_setup: Dict[str, int] = {}
        by_gate: Dict[str, int] = {}
        feats: Dict[str, int] = {}
        warns: Dict[str, int] = {}
        for r in rows:
            k = r.get("sleeve") or r.get("setup") or (r.get("features") or {}).get("kind") or r.get("kind") or "?"
            by_setup[str(k)] = by_setup.get(str(k), 0) + 1
            by_gate[r.get("gate") or "?"] = by_gate.get(r.get("gate") or "?", 0) + 1
            a = r.get("attribution") or {}
            for f in a.get("ignored_supportive_tr") or []:
                feats[f] = feats.get(f, 0) + 1
            for w in a.get("warnings_present_tr") or []:
                warns[w] = warns.get(w, 0) + 1
        n = len(rows)
        prof = {"n": n, "median_minutes": round(_st.median(mins), 1) if mins else None,
                ("median_mae_over_stop" if outcome == "TARGET" else "median_mfe_over_target"): (round(_st.median(ratios), 2) if ratios else None),
                "by_setup": dict(sorted(by_setup.items(), key=lambda kv: -kv[1])), "by_gate": dict(sorted(by_gate.items(), key=lambda kv: -kv[1])),
                "common_supportive": dict(sorted(feats.items(), key=lambda kv: -kv[1])[:6]),
                "common_warnings": dict(sorted(warns.items(), key=lambda kv: -kv[1])[:6])}
        if outcome == "TARGET":
            near = sum(1 for x in ratios if x >= 0.6)
            prof["near_stop_share"] = round(near / len(ratios), 2) if ratios else None
            prof["lesson"] = (f"{n} kazananın {near}'i stop'un %60+'ına kadar aleyhe gitti → stop çok dar değil ama giriş erken; "
                              f"medyan {prof['median_minutes']} dk'da hedef" if ratios else "ölçülmedi")
        else:
            close_ = sum(1 for x in ratios if x >= 0.5)
            prof["near_target_share"] = round(close_ / len(ratios), 2) if ratios else None
            prof["lesson"] = (f"{n} stop'un {close_}'i hedefin yarısına ulaşıp döndü → yarı-tepe geri-verme/kısmi TP bunları kurtarır; "
                              f"en sık uyaran: {next(iter(prof['common_warnings']), '—')}" if ratios else "ölçülmedi")
        return prof

    def interest_weights(self, min_n: int = 8) -> Dict[str, float]:
        """Tier-A vekil tetikleyicileri (dip/breakout) için öğrenilmiş ağırlık: isabet = kazanan/(kazanan+stop).
        n < min_n → 1,0; ağırlık [0,5 · 1,5]. Kör bir 'daha çok işlem' değil, ölçülmüş eleme."""
        stats: Dict[str, List[int]] = {"dip": [0, 0], "breakout": [0, 0]}
        for r in self.records:
            if r.get("kind") != "unevaluated" or r.get("outcome") not in ("TARGET", "STOP"):
                continue
            k = r.get("setup") or (r.get("features") or {}).get("kind")
            if k in stats:
                stats[k][0 if r["outcome"] == "TARGET" else 1] += 1
        out = {}
        for k, (w, l) in stats.items():
            n = w + l
            out[k] = 1.0 if n < min_n else float(min(1.5, max(0.5, 0.5 + w / n)))
            out[k + "_n"] = n
        return out

    def proposals(self, now: Optional[float] = None, current: Optional[Dict] = None, dry_run: bool = False) -> Dict[str, float]:
        """Kanıtı olan kapılar için challenger önerisi (param → yeni değer). Soğuma 12 sa.
        dry_run: yalnız mevcut (daha önce üretilmiş) önerileri döndürür, durum DEĞİŞMEZ (panel raporu için)."""
        now = time.time() if now is None else now
        current = current or {}
        if dry_run:
            return {GATE_PROPOSAL[g][0]: v for g, v in self.proposed.items() if g in GATE_PROPOSAL}
        out: Dict[str, float] = {}
        rep_gates = {}
        for r in self.records:
            if r.get("outcome") not in ("TARGET", "STOP"):
                continue
            for g in r.get("gates") or [r.get("gate")]:
                gs = rep_gates.setdefault(g, [0, 0])
                gs[0 if r["outcome"] == "STOP" else 1] += 1
        for g, (k_avoid, k_miss) in rep_gates.items():
            n = k_avoid + k_miss
            if n < MIN_N_PROPOSAL or g not in GATE_PROPOSAL:
                continue
            lo, hi = _wilson(k_avoid, n)
            if hi >= 0.5:
                continue                                   # kapı zararlı değil (ya da ölçülemedi)
            last = self._last_proposal.get(g)
            if last is not None and now - float(last) < PROPOSAL_COOLDOWN:
                if g in self.proposed:
                    out[GATE_PROPOSAL[g][0]] = self.proposed[g]
                continue
            param, delta = GATE_PROPOSAL[g]
            lo_b, hi_b = PROPOSAL_BOUNDS[param]
            cur = float(current.get(param, (lo_b + hi_b) / 2.0))
            new = float(min(hi_b, max(lo_b, cur + delta)))
            if abs(new - cur) < 1e-9:
                continue
            out[param] = round(new, 4)
            self.proposed[g] = round(new, 4)
            self._last_proposal[g] = now
            self._journal_note(f"ÖNERİ → challenger: {GATE_TR.get(g, g)} {n} adayda isabet %{k_avoid/n*100:.0f} "
                               f"(Wilson üst {hi:.2f} < 0,5) → {param} {cur} → {new}", now)
        if out:
            self.save()
        return out

    # ------------------------------------------------------------ günlük
    def _journal(self, r: Dict) -> None:
        a = r.get("attribution") or {}
        rec = {"ts": r.get("resolved_ts"), "type": "missed", "id": r["id"], "symbol": r["symbol"], "kind": r.get("kind"),
               "gate": r.get("gate"), "outcome": r.get("outcome"), "attribution": a, "plan": r.get("plan"),
               "features": r.get("features"), "info": r.get("info")}
        self._append_jsonl(rec)
        if r.get("outcome") == "TARGET" or (r.get("outcome") == "STOP" and r.get("kind") != "unevaluated"):
            emoji = "👁️🟢" if r["outcome"] == "TARGET" else "👁️🔴"
            lines = [f"\n### {emoji} {a.get('verdict')} · {r['symbol']} {r['direction']} · "
                     f"{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(float(r['ts'])))} · sleeve {r.get('sleeve') or '—'} · {r.get('kind')}",
                     f"- Plan: giriş {r['plan'].get('entry')} · stop %{r['plan'].get('stop_pct')} · hedef %{r['plan'].get('target_pct')} · "
                     f"MFE %{r.get('mfe_pct')} · MAE %{r.get('mae_pct')}"
                     + (f" · **net kaçırılan %{a.get('net_missed_pct')}**" if r["outcome"] == "TARGET" else f" · kaçınılan zarar %{a.get('net_avoided_pct')}"),
                     f"- Neden yapılmadı: {a.get('gate_tr')} — {a.get('why_blocked')}",
                     f"- Yol: {a.get('path')}",
                     f"- Nasıl/neden: {a.get('how')}"]
            self._append_md("\n".join(lines))

    def _journal_note(self, text: str, now: float) -> None:
        self._append_jsonl({"ts": now, "type": "proposal", "text": text})
        self._append_md(f"\n- 🥊 {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(now))} {text}")

    def _append_jsonl(self, rec: Dict) -> None:
        if not self.jsonl:
            return
        try:
            self.jsonl.parent.mkdir(parents=True, exist_ok=True)
            with open(self.jsonl, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass

    def _append_md(self, text: str) -> None:
        if not self.md:
            return
        try:
            self.md.parent.mkdir(parents=True, exist_ok=True)
            new = not self.md.exists()
            with open(self.md, "a", encoding="utf-8") as fh:
                if new:
                    fh.write("# CryptoMind — Kaçırılan Fırsatlar ve Neden Yapılmadı\n\n"
                             "Veto edilen, elenen ya da yürütülemeyen her aday ufuk boyunca izlenir. Hedefe ulaşanlar için "
                             "hangi kapının engellediği, hangi özelliklerin göz ardı edildiği ve hangi bilginin eksik olduğu yazılır.\n")
                fh.write(text + "\n")
        except Exception:
            pass

    def status(self) -> Dict:
        return self.report()


def _safe(pred: Callable, f: Dict, s: float) -> bool:
    try:
        return bool(pred(f, s))
    except Exception:
        return False
