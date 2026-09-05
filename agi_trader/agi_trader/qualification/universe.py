"""Global parite evreni ve uygunluk taraması — şartname 3, 4, 10 (2. mesaj 1).

SABİT LİSTE YOK
Sistem "BTC/ETH/SOL/DOGE/AVAX" gibi elle yazılmış bir listeyle sınırlı
kalmaz. Bağlı borsadaki BÜTÜN aktif USDT/USDC marketleri çekilir ve her biri
on bir kapıdan geçirilir. Geçen ELIGIBLE, geçmeyen EXCLUDED olur ve
**geçemediği kapı adıyla birlikte** kaydedilir — sessizce kaybolmaz.

NEDEN HER MARKET İŞLENMEZ
Bir marketin listelenmiş olması, üzerinde net +%1 hedefin ÖLÇÜLEBİLİR olduğu
anlamına gelmez. Yeterli geçmiş yoksa taban oranı hesaplanamaz; defter inceyse
maliyet hedefi yer; veri bayatsa hesap bugünü temsil etmez. Bu kapılar
"fırsatı kaçırmamak" için değil, OLMAYAN fırsatı uydurmamak için vardır.

MarketEligibilityScore YÖN İÇERMEZ (şartname 4)
Skor bir marketin ÖLÇÜLEBİLİRLİĞİNİ puanlar; al/sat sinyali değildir ve
öyle sunulmaz.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

QUOTES = ("USDT", "USDC")

# Kapılar — her biri (ad, açıklama). Sıra raporda korunur.
GATES = [
    ("market_active", "market aktif mi"),
    ("trading_enabled", "işlem açık mı"),
    ("ohlcv_history", "yeterli tarihsel bar var mı"),
    ("trade_history", "yeterli işlem geçmişi var mı"),
    ("l2_available", "L2 defter erişilebilir mi"),
    ("spread_acceptable", "spread kabul edilebilir mi"),
    ("liquidity_sufficient", "defter derinliği yeterli mi"),
    ("volume_sufficient", "günlük hacim yeterli mi"),
    ("book_synchronized", "defter senkron mu"),
    ("data_fresh", "veri güncel mi"),
    ("cost_calculable", "maliyet hesaplanabiliyor mu"),
]


@dataclass
class EligibilityThresholds:
    """Eşikler config'den yönetilir; başlangıçta MUHAFAZAKÂR."""
    min_bars: int = 30_000            # 5m barda ~104 gün
    min_trades_per_bar: float = 5.0
    max_spread_bps: float = 15.0
    min_depth_usd: float = 25_000.0   # ±%0,5 bandında tek taraf
    min_daily_volume_usd: float = 5_000_000.0
    max_staleness_sec: float = 900.0
    # Kaydedici defteri 300 saniyede bir örnekliyor; 60 saniyelik eşik
    # HER ZAMAN düşerdi — kapı marketi değil kendi örnekleme aralığımızı
    # ölçmüş olurdu. Eşik bir örnekleme aralığı + pay olarak belirlenir.
    max_book_age_sec: float = 420.0


@dataclass
class MarketStatus:
    symbol: str
    quote: str
    eligible: bool
    score: float
    failed_gates: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


def _score_component(x: Optional[float], iyi: float, kotu: float) -> float:
    """[0,1]'e doğrusal eşleme; `iyi` 1, `kotu` 0. Eksikse 0 (varsayım yok)."""
    if x is None or not math.isfinite(x):
        return 0.0
    if iyi == kotu:
        return 1.0
    v = (x - kotu) / (iyi - kotu)
    return float(max(0.0, min(1.0, v)))


def evaluate_market(m: Dict, th: EligibilityThresholds = EligibilityThresholds()
                    ) -> MarketStatus:
    """Tek bir marketin uygunluğu.

    `m` beklenen alanlar (eksik olan kapıyı DÜŞÜRÜR, varsayılmaz):
      symbol, quote, active, spot/trading, bars, trades_per_bar,
      has_l2, spread_bps, depth_usd, volume_usd_24h, data_age_sec,
      book_age_sec, cost_model
    """
    dusen: List[str] = []
    neden: List[str] = []

    def kapi(ad: str, ok: Optional[bool], aciklama: str):
        if not ok:
            dusen.append(ad)
            neden.append(aciklama)

    kapi("market_active", bool(m.get("active")), "market aktif değil")
    kapi("trading_enabled", bool(m.get("trading")), "işlem kapalı")
    bars = m.get("bars")
    kapi("ohlcv_history", bars is not None and bars >= th.min_bars,
         f"tarihsel bar {bars} < {th.min_bars}")
    tpb = m.get("trades_per_bar")
    kapi("trade_history", tpb is not None and tpb >= th.min_trades_per_bar,
         f"bar başına işlem {tpb} < {th.min_trades_per_bar}")
    kapi("l2_available", bool(m.get("has_l2")), "L2 defter yok")
    sp = m.get("spread_bps")
    kapi("spread_acceptable", sp is not None and sp <= th.max_spread_bps,
         f"spread {sp} bps > {th.max_spread_bps}")
    dp = m.get("depth_usd")
    kapi("liquidity_sufficient", dp is not None and dp >= th.min_depth_usd,
         f"derinlik {dp} $ < {th.min_depth_usd:,.0f} $")
    vol = m.get("volume_usd_24h")
    kapi("volume_sufficient", vol is not None and vol >= th.min_daily_volume_usd,
         f"24s hacim {vol} $ < {th.min_daily_volume_usd:,.0f} $")
    ba = m.get("book_age_sec")
    kapi("book_synchronized", ba is not None and ba <= th.max_book_age_sec,
         f"defter yaşı {ba} sn > {th.max_book_age_sec}")
    da = m.get("data_age_sec")
    kapi("data_fresh", da is not None and da <= th.max_staleness_sec,
         f"veri yaşı {da} sn > {th.max_staleness_sec}")
    kapi("cost_calculable", m.get("cost_model") in ("MEASURED_L2_VWAP", "ESTIMATED"),
         "maliyet modeli kurulamadı")

    # Skor — YÖN İÇERMEZ, yalnız ölçülebilirlik
    bilesen = {
        "liquidity": _score_component(dp, th.min_depth_usd * 8, th.min_depth_usd / 4),
        "spread": _score_component(sp, 1.0, th.max_spread_bps * 2),
        "depth": _score_component(dp, th.min_depth_usd * 4, 0.0),
        "data_continuity": _score_component(bars, th.min_bars * 8, th.min_bars / 4),
        "volatility_adequacy": _score_component(m.get("rv_24h_pct"), 0.25, 0.02),
        "trading_activity": _score_component(tpb, th.min_trades_per_bar * 20,
                                             th.min_trades_per_bar / 4),
        "book_integrity": _score_component(
            None if ba is None else th.max_book_age_sec - ba, th.max_book_age_sec, 0.0),
        "historical_sample": _score_component(bars, 400_000, th.min_bars),
        "cost_efficiency": _score_component(
            None if sp is None else (th.max_spread_bps - sp), th.max_spread_bps, 0.0),
    }
    skor = 100.0 * sum(bilesen.values()) / len(bilesen)
    return MarketStatus(
        symbol=str(m.get("symbol")), quote=str(m.get("quote") or ""),
        eligible=not dusen, score=round(skor, 1),
        failed_gates=dusen, reasons=neden,
        metrics={**{k: m.get(k) for k in
                    ("bars", "trades_per_bar", "spread_bps", "depth_usd",
                     "volume_usd_24h", "data_age_sec", "book_age_sec",
                     "cost_model", "rv_24h_pct")},
                 "score_components": {k: round(v, 3) for k, v in bilesen.items()}})


def scan(markets: Sequence[Dict],
         th: EligibilityThresholds = EligibilityThresholds()) -> Dict:
    """Bütün marketleri tara; ELIGIBLE/EXCLUDED ayrımını ve NEDENLERİ döndür."""
    sonuc = [evaluate_market(m, th) for m in markets
             if (m.get("quote") in QUOTES)]
    uygun = [s for s in sonuc if s.eligible]
    disi = [s for s in sonuc if not s.eligible]
    sayac: Dict[str, int] = {}
    for s in disi:
        for g in s.failed_gates:
            sayac[g] = sayac.get(g, 0) + 1
    return {
        "scanned": len(sonuc),
        "eligible": len(uygun),
        "excluded": len(disi),
        "quotes": list(QUOTES),
        "gates": [{"gate": g, "description": d, "failed_count": sayac.get(g, 0)}
                  for g, d in GATES],
        "eligible_symbols": [s.symbol for s in
                             sorted(uygun, key=lambda x: -x.score)],
        "markets": [s.to_dict() for s in
                    sorted(sonuc, key=lambda x: (not x.eligible, -x.score))],
        "thresholds": asdict(th),
        "scanned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": ("Skor yalnız ÖLÇÜLEBİLİRLİĞİ puanlar; al/sat yönü içermez "
                 "ve sinyal olarak kullanılamaz."),
    }


def bar_inventory(data_dir: Optional[Path]) -> Dict[str, int]:
    """Parite başına tarihsel 5m bar sayısı.

    İki kaynak: (a) parquet dosyaları varsa doğrudan sayılır, (b) yoksa
    araştırma koşusunun ürettiği `data_inventory.json` okunur. Sunucuda ham
    veri tutulmuyor; envanter dosyası bu yüzden ARTEFAKTIN parçasıdır.
    Hiçbiri yoksa sayı **None kalır ve kapı DÜŞER** — tahmin edilmez."""
    out: Dict[str, int] = {}
    if not data_dir:
        return out
    d = Path(data_dir)
    if d.exists():
        for p in d.glob("*_5m.parquet"):
            try:
                import pyarrow.parquet as pq
                out[p.name.split("_")[0]] = pq.ParquetFile(p).metadata.num_rows
            except Exception:
                pass
    if out:
        return out
    for aday in (d.parent / "qualification" / "data_inventory.json",
                 d / "data_inventory.json"):
        if aday.exists():
            try:
                ham = json.loads(aday.read_text(encoding="utf-8"))
                return {k: int(v) for k, v in (ham.get("bars") or ham).items()}
            except Exception:
                pass
    return out


def collect_binance_markets(recorder_rows=None, data_dir: Optional[Path] = None,
                            timeout: float = 20.0) -> List[Dict]:
    """Binance USD-M vadeli borsasından canlı market listesi + ölçümler.

    Kaydedicinin (recorder) topladığı gerçek spread/derinlik varsa onlar
    kullanılır; yoksa ilgili kapılar DÜŞER — tahmin üretilmez.
    """
    import urllib.request

    def _get(url):
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())

    info = _get("https://fapi.binance.com/fapi/v1/exchangeInfo")
    tick = {t["symbol"]: t for t in
            _get("https://fapi.binance.com/fapi/v1/ticker/24hr")}

    olcum: Dict[str, Dict] = {}
    if recorder_rows is not None and len(recorder_rows):
        for sym, g in recorder_rows.groupby("symbol"):
            son = g.sort_values("ts").iloc[-1]
            yas = (time.time() - son["ts"].timestamp()) if hasattr(son["ts"], "timestamp") else None
            olcum[str(sym)] = {
                "spread_bps": float(son.get("spread_bps", float("nan"))),
                "depth_usd": float(min(son.get("bid_depth_usd", 0) or 0,
                                       son.get("ask_depth_usd", 0) or 0)),
                "book_age_sec": yas,
                "has_l2": True,
                "cost_model": ("MEASURED_L2_VWAP"
                               if son.get("bid_cum_1bps") is not None
                               and son.get("bid_cum_1bps") == son.get("bid_cum_1bps")
                               else "ESTIMATED"),
            }

    bars = bar_inventory(data_dir)

    out: List[Dict] = []
    for s in info.get("symbols", []):
        q = s.get("quoteAsset")
        if q not in QUOTES:
            continue
        sym = s["symbol"]
        t = tick.get(sym, {})
        o = olcum.get(sym, {})
        try:
            hacim = float(t.get("quoteVolume", 0.0))
            islem = float(t.get("count", 0.0)) / 288.0     # 24s → 5m bar başına
            son_fiyat = float(t.get("lastPrice", 0.0) or 0.0)
            degisim = abs(float(t.get("priceChangePercent", 0.0) or 0.0))
        except Exception:
            hacim = islem = son_fiyat = degisim = 0.0
        out.append({
            "symbol": sym, "quote": q,
            "active": s.get("status") == "TRADING",
            "trading": s.get("contractType") == "PERPETUAL",
            "bars": bars.get(sym),
            "trades_per_bar": islem,
            "has_l2": o.get("has_l2", False),
            "spread_bps": o.get("spread_bps"),
            "depth_usd": o.get("depth_usd"),
            "volume_usd_24h": hacim,
            "data_age_sec": 0.0 if son_fiyat > 0 else None,
            "book_age_sec": o.get("book_age_sec"),
            "cost_model": o.get("cost_model"),
            "rv_24h_pct": degisim / 100.0,
        })
    return out
