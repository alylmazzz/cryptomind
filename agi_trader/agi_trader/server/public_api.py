"""
CryptoMind — HERKESE AÇIK, SALT-OKUNUR API.

Tam dashboard (`server/app.py`) yerel/tek-kullanıcı içindir: kimlik bilgisi
yazma (`POST /api/credentials`), otonom motoru başlat/durdur (`POST /api/auto/*`),
ağırlık öğrenme, optimizasyon uygulama gibi DEĞİŞTİRİCİ uçlar içerir. Bunlar
internete açılırsa kimlik doğrulaması olmadan kötüye kullanılabilir.

Bu modül, mindcorplab.com/cryptomind için yalnız GET olan, yalnız OKUYAN bir
alt küme sunar:

  GET /api/health       — servis + veri sağlığı
  GET /api/strategy     — OOS-doğrulanmış strateji künyesi (gerçek runs/ dosyaları)
  GET /api/trend        — trend-takip PAPER portföyü (equity, pozisyon, eğri)
  GET /api/risk         — risk & sağlık metrikleri (Sharpe, VaR/CVaR, drift)
  GET /api/prices       — canlı fiyat şeridi
  GET /api/analyze      — çok katmanlı canlı analiz (önbellekli)
  GET /api/chart        — mum + trend çizgisi + S/R + seviyeler
  GET /api/events       — makro takvim (kural-temelli tahmini) + piyasa sürücüleri
  GET /api/correlation  — gerçek getiri korelasyon matrisi

GÜVENLİK/DÜRÜSTLÜK NOTLARI
  • Hiçbir POST/DELETE ucu YOK → panelden sistem durumu değiştirilemez.
  • Borsa API anahtarı gerekmez; yalnız halka açık piyasa verisi kullanılır.
  • Emir gönderilmez: portföy PAPER (kağıt üzerinde), gerçek para yok.
  • Ağır analiz uçları tek seferde bir istek çalıştırır (paylaşımlı sunucuyu korur).
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

from ..config import load_config
from ..monitor import risk_report

# Önbellek ömürleri (saniye) — paylaşımlı ARM sunucuda CPU/bellek koruması
TTL_CHART = 300
TTL_PRICES = 20
TTL_CORR = 3600
TTL_EVENTS = 3600
TTL_STATE = 20          # trend_state.json disk okuması

# Panelde analiz edilen pariteler. İlk 5'i config.yaml'daki KALİBRE allowlist
# (BNB/XRP/ADA walk-forward testini geçemediği için dışarıda). 2026-09-02'de 10
# parite daha eklendi — ÖLÇÜLEREK seçildi: nitelendirme evreninde (24 sa hacim
# ≥ 20 M$), Binance/Bybit/OKX/MEXC dördünde spot listeli, MEXC'te 1 dk veri
# hatasız + spread ≤ 10 bps + ±%2 derinlik ≥ 45 k$, 1 dk σ ≥ %0,07 (hareket var).
# Elenenler: TRX (σ %0,03), FET/APT (spread ~20 bps), ZEC/XMR (dört borsada yok).
# Beyaz liste aynı zamanda keyfi sembolle sunucuda ağır iş tetiklenmesini engeller.
# Ortamdan ezilebilir: CRYPTOMIND_SYMBOLS="BTC/USDT,ETH/USDT,..."
ALLOWED_SYMBOLS = [s.strip().upper() for s in os.environ["CRYPTOMIND_SYMBOLS"].split(",") if s.strip()]     if os.environ.get("CRYPTOMIND_SYMBOLS") else [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "AVAX/USDT",
    "LINK/USDT", "SUI/USDT", "NEAR/USDT", "PEPE/USDT", "AAVE/USDT",
    "UNI/USDT", "LTC/USDT", "DOT/USDT", "FIL/USDT", "BCH/USDT",
]
ALLOWED_TFS = ["1h", "4h", "1d", "1w"]

# Çok-zaman-dilimli katman için panelin kullandığı TF'ler. Tam config (15m…1M)
# analizi ~2× yavaşlatıyor ve 4 saatlik karara katkısı sınırlı; yerel tam
# dashboard config.yaml'daki listeyi kullanmaya devam eder.
PANEL_TFS = ["1h", "4h", "1d"]

# Analiz hattı parite başına ~8-20 sn sürer (çok-zaman-dilimli veri çekimi + AI
# topluluğu). Bu YÜZDEN istek anında çalıştırılmaz: arka plan yenileyici belirli
# aralıklarla anlık görüntüyü üretir, uçlar hazır sonucu ANINDA döndürür.
DEFAULT_TF = "4h"
REFRESH_SEC = int(os.environ.get("CRYPTOMIND_REFRESH", "900"))


def create_public_app(config=None) -> FastAPI:
    from .safe_json import SafeJSONResponse
    app = FastAPI(title="CryptoMind Public API", docs_url=None, redoc_url=None, default_response_class=SafeJSONResponse)
    cfg = config or load_config()
    cfg.data["timeframes"] = list(PANEL_TFS)
    cfg.data["symbols"] = list(ALLOWED_SYMBOLS)

    runs = Path(cfg.get("output_dir", "runs"))
    if not runs.is_absolute():
        runs = Path(__file__).resolve().parents[2] / runs

    cache: Dict[str, tuple] = {}          # key -> (ts, value)
    # RLock şart: ağır uçlar kilidi tutarken orch() de aynı kilidi ister
    # (düz Lock ile ilk analiz isteğinde kilitlenme olurdu).
    heavy_lock = threading.RLock()        # ağır işler sıraya girsin
    orch_box: Dict[str, object] = {}      # tembel Orchestrator

    # ------------------------------------------------------------- yardımcılar
    def cached(key: str, ttl: float, producer):
        hit = cache.get(key)
        now = time.time()
        if hit and now - hit[0] < ttl:
            return hit[1]
        val = producer()
        cache[key] = (now, val)
        return val

    def orch():
        """Orchestrator'ı ilk ağır istekte kur (servis açılışı hızlı kalsın)."""
        if "o" not in orch_box:
            with heavy_lock:
                if "o" not in orch_box:
                    from ..agents import Orchestrator
                    orch_box["o"] = Orchestrator(cfg)
        return orch_box["o"]

    def read_state() -> Optional[Dict]:
        def _read():
            p = runs / "trend_state.json"
            if not p.exists():
                return None
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
        return cached("state", TTL_STATE, _read)

    # ------------------------------------------------------------------ health
    @app.get("/api/health")
    def health():
        st = read_state()
        last = st.get("last_rebalance") if st else None
        stale = None
        if last:
            try:
                import datetime as _dt
                d = _dt.date.fromisoformat(last)
                stale = (_dt.date.today() - d).days
            except Exception:
                pass
        snap = cache.get("analyze")
        return {"ok": True, "service": "cryptomind",
                "light_mode": os.environ.get("AGI_LIGHT_MODE", "") in ("1", "true", "yes"),
                "paper_state": bool(st), "last_rebalance": last,
                "days_since_update": stale,
                "engine_loaded": "o" in orch_box,
                "refresh_sec": REFRESH_SEC,
                "snapshot_age_sec": (int(time.time() - snap[0]) if snap else None),
                "snapshot_symbols": (len(snap[1].get("signals", [])) if snap else 0)}

    # ---------------------------------------------------------------- strategy
    @app.get("/api/strategy")
    def strategy():
        """Yayınlanan stratejinin künyesi — sayılar gerçek araştırma çıktılarından
        (runs/selected_universe.json, runs/portfolio_trend.json) okunur."""
        def _load(name):
            p = runs / name
            if not p.exists():
                return None
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None

        sel = _load("selected_universe.json") or {}
        crypto_only = (_load("portfolio_trend.json") or {}).get("metrics") or {}
        st = read_state() or {}
        return {
            "name": "Trend200 + Mom20 · vol-hedefli · diversifiye",
            "rule": "Fiyat > 200 günlük SMA VE 20 günlük momentum > 0 → pozisyon; "
                    "aksi halde NAKİT. Pozisyon boyutu gerçekleşen volatiliteye göre "
                    "ölçeklenir; portföy hedef-vol ile kaldıraçlanır (max 2.5×).",
            "universe": st.get("pairs", []),
            "universe_size": len(st.get("pairs", [])),
            "rebalance": "günlük",
            "fitted_parameters": 0,
            "oos_window": "2022-2026 (4.5 yıl, çok rejimli)",
            "oos_metrics": sel.get("metrics") or {},
            "crypto_only_metrics": {k: crypto_only.get(k) for k in
                                    ("sharpe", "cagr", "dd", "calmar") if k in crypto_only},
            "costs_modeled": "taker %0.04 + volatiliteye bağlı kayma (~%0.2 gidiş-dönüş)",
            # Kullanıcının "%1" sorusu iki AYRI şeydir — panelde ayrıştırılır.
            "one_percent": {
                "per_trade": {
                    "feasible": True,
                    "explain": ("TEK İŞLEMDE %1'lik hareketi yakalamak mümkündür ve "
                                "sistem bunu destekler: formasyon paneli, hedefi mevcut "
                                "fiyattan ≥%1 uzakta olan kurulumları '≥%1 ✓' ile "
                                "işaretler, grafikte ±%1 bandı çizilir."),
                },
                "per_day_compound": {
                    "feasible": False,
                    "annual_equivalent_pct": 3678,
                    "required_annual_vol_pct": 2685,
                    "required_leverage": 192,
                    "daily_vol_pct": 140,
                    "explain": ("HER GÜN %1 bileşik ise yıllık %3.678 (36,8 kat) eder. "
                                "Getiri = Sharpe × Volatilite kimliği gereği, bu "
                                "stratejinin Sharpe'ı (1,37) ile bu getiri %2.685 yıllık "
                                "volatilite ister: 192× kaldıraç, günlük %140 dalgalanma. "
                                "Tek ters gün sermayeyi siler. Sharpe 20'lik kurumsal bir "
                                "HFT masası bile bu hedef için günlük %9,6 vol taşımak "
                                "zorundadır. Bu bir ayar sorunu değil, matematiksel bir "
                                "sınırdır."),
                },
                "realistic_target": {
                    "monthly_pct": "1,5 – 3",
                    "annual_pct": "20 – 50",
                    "max_drawdown_pct": "10 – 20",
                },
            },
            "honest_notes": [
                "TEK İŞLEMDE %1 mümkündür; HER GÜN %1 (yıllık %3.678) değildir — "
                "aynı Sharpe'la 192× kaldıraç ve günlük %140 volatilite gerektirir.",
                "Parametreler sabittir ve geçmişe uydurulmamıştır (fit yok) — "
                "4.5 yıllık pencerenin tamamı doğal olarak örneklem dışıdır.",
                "Aynı kod tabanındaki karmaşık 1 saatlik TA-konsensüs botu örneklem "
                "dışında EDGE ÜRETMEDİ (2022-2025 her yıl zararda); yayınlanan strateji "
                "bilinçli olarak o değil, basit trend-takiptir.",
                "Gerçekçi beklenti ayda ~%1.5-2'dir. 'Garantili günlük kâr' mümkün değildir.",
                "Panel PAPER (kağıt) portföydür: gerçek emir gönderilmez, "
                "borsa anahtarı kullanılmaz. Yatırım tavsiyesi değildir.",
            ],
        }

    def read_track_state(track: str) -> Optional[Dict]:
        """Ray başına paper durum dosyası (base eski adı korur)."""
        name = "trend_state.json" if track == "base" else f"trend_state_{track}.json"

        def _read():
            q = runs / name
            if not q.exists():
                return None
            try:
                return json.loads(q.read_text(encoding="utf-8"))
            except Exception:
                return None
        return cached(f"state_{track}", TTL_STATE, _read)

    def _track_metrics(d: Optional[Dict], track: str) -> Dict:
        """Bir rayın ÖLÇÜLMÜŞ metrikleri. Tahmin yok; yalnız kaydedilmiş özsermaye eğrisi."""
        if not d:
            return {"track": track, "available": False,
                    "reason": "bu ray henüz hiç adım atmadı (durum dosyası yok)"}
        import math as _m
        init = float(d.get("initial", 10000) or 10000)
        eq = float(d.get("equity", init) or init)
        hist = [h for h in (d.get("history") or []) if isinstance(h.get("equity"), (int, float))]
        curve = [float(h["equity"]) for h in hist if _m.isfinite(float(h["equity"]))]
        rets = [curve[i] / curve[i - 1] - 1 for i in range(1, len(curve)) if curve[i - 1]]
        n = len(rets)
        mean = sum(rets) / n if n else None
        var = (sum((r - mean) ** 2 for r in rets) / n) if n else None
        sd = (var ** 0.5) if var else None
        sharpe = (mean / sd * (365 ** 0.5)) if (mean is not None and sd) else None
        peak, mdd = (curve[0] if curve else init), 0.0
        for v in curve:
            peak = max(peak, v)
            mdd = max(mdd, 1 - v / peak) if peak else mdd
        risk = d.get("risk") or {}
        weights = d.get("weights") or {}
        return {
            "track": track, "available": True,
            "initial": init, "equity": round(eq, 2),
            "return_pct": round((eq / init - 1) * 100, 2),
            "days": n,
            "mean_daily_pct": (round(mean * 100, 4) if mean is not None else None),
            "vol_daily_pct": (round(sd * 100, 4) if sd else None),
            # ÖLÇÜLMEMİŞ ≠ SIFIR: n<5 günde Sharpe yayımlanmaz (gürültü ile edge ayrılmaz).
            "sharpe": (round(sharpe, 2) if (sharpe is not None and n >= 5) else None),
            "sharpe_note": (None if n >= 5 else f"{n} gün < 5 gün — Sharpe ölçülmedi"),
            "max_drawdown_pct": round(mdd * 100, 2),
            "peak_equity": round(float(d.get("peak_equity") or peak), 2),
            "dd_locked": bool(d.get("dd_locked", False)),
            "ruined": bool(d.get("ruined", False)),
            "invested_pct": round(sum(float(v) for v in weights.values()) * 100, 1),
            "target_vol_pct": (round(float(risk.get("target_vol", 0)) * 100, 1) if risk else None),
            "max_lev": risk.get("max_lev"),
            "max_exposure": risk.get("max_exposure"),
            "dd_gates_pct": ({"soft": round(float(risk.get("dd_soft", 0)) * 100, 1),
                              "hard": round(float(risk.get("dd_hard", 0)) * 100, 1),
                              "kill": round(float(risk.get("dd_kill", 0)) * 100, 1)} if risk else None),
            "last_rebalance": d.get("last_rebalance"),
            "equity_curve": [{"date": h.get("date"), "equity": h.get("equity")} for h in hist[-180:]],
        }

    # --------------------------------------------------------- risk rayları
    @app.get("/api/trend/tracks")
    def trend_tracks():
        """Aynı sinyal, farklı kaldıraç — yan yana ÖLÇÜM.

        Kaldıraç Sharpe'ı artırmaz; ortalamayı da sapmayı da aynı katsayıyla
        büyütür. Bu uç, "daha yüksek kazanç" isteğinin bedelini (düşüş) aynı
        tabloda gösterir ki seçim tahminle değil ölçümle yapılsın."""
        rows = [_track_metrics(read_track_state(t), t)
                for t in ("base", "aggressive", "extreme", "max")]
        live = [r for r in rows if r.get("available") and (r.get("days") or 0) >= 1]
        return {
            "tracks": rows,
            "measured_tracks": len(live),
            "note": ("Kaldıraç ölçeği büyütür, kenarı DEĞİL: her rayın Sharpe'ı aynı "
                     "olmalıdır; ayrışıyorsa fark kaldıraçtan değil düşüş kısıcısından gelir."),
            "warning": ("Agresif raylar PAPER'dır ve gerçek para taahhüdü değildir. "
                        "48 günlük ölçüm gerçekleşen Sharpe 2,61 verdi; backtest beklentisi "
                        "1,36 idi. Kaldıraç kararı bu ikisinden HANGİSİNE inandığınıza bağlıdır — "
                        "beklenen Sharpe doğruysa aynı getiri iki katı kaldıraç ve iki katı düşüş ister."),
        }

    # ------------------------------------------------------------------- trend
    @app.get("/api/trend")
    def trend():
        d = read_state()
        if not d:
            return {"available": False, "reason": "paper portföy durumu henüz oluşmadı"}
        eq = float(d.get("equity", 0))
        init = float(d.get("initial", 10000))
        sigs = d.get("last_signals", {}) or {}
        weights = d.get("weights", {}) or {}
        in_market = [s for s, v in weights.items() if abs(float(v)) > 1e-4]
        hist = d.get("history", []) or []
        curve = [{"date": h.get("date"), "equity": h.get("equity")} for h in hist[-180:]]
        return {"available": True, "mode": "trend_follow_paper",
                "initial": init,
                "equity": round(eq, 2), "return_pct": round((eq / init - 1) * 100, 2),
                "invested_pct": round(sum(float(v) for v in weights.values()) * 100, 1),
                "in_market": in_market, "cash": len(in_market) == 0,
                "weights": weights, "signals": sigs, "pairs": d.get("pairs", []),
                "days": len(hist),
                "last_rebalance": d.get("last_rebalance"), "equity_curve": curve}

    # -------------------------------------------------------------------- risk
    @app.get("/api/risk")
    def risk():
        d = read_state()
        if not d:
            return {"available": False, "reason": "paper portföy durumu henüz oluşmadı"}
        return risk_report(d)

    # ------------------------------------------------------------------ prices
    _PRICE_MAP = {
        "BTC/USDT": ("BTC/USDT", None), "ETH/USDT": ("ETH/USDT", None),
        "SOL/USDT": ("SOL/USDT", None), "DOLAR/TL": ("USDTRY=X", None),
        "BIST/TL": ("XU100.IS", None), "BIST/DOLAR": ("XU100.IS", "USDTRY=X"),
        "ALTIN": ("GLD", None), "S&P500": ("SPY", None), "NASDAQ": ("QQQ", None),
    }

    DEFAULT_PRICE_STRIP = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ALTIN", "S&P500", "DOLAR/TL"]

    def _price(under: str, allow_fetch: bool = True):
        """Önbellekten fiyat; taze değilse ve motor boştaysa çeker."""
        hit = cache.get(f"px:{under}")
        if hit and time.time() - hit[0] < TTL_PRICES:
            return hit[1]
        if allow_fetch and heavy_lock.acquire(timeout=8):
            try:
                val = orch().data.fetch_price(under)
                cache[f"px:{under}"] = (time.time(), val)
                return val
            except Exception:
                pass
            finally:
                heavy_lock.release()
        return hit[1] if hit else (None, None)

    @app.get("/api/prices")
    def prices(symbols: str = Query(default="")):
        want = [s.strip() for s in symbols.split(",") if s.strip()][:8] or DEFAULT_PRICE_STRIP
        out = []
        for d in want:
            under, div = _PRICE_MAP.get(d, (d, None))
            p, chg = _price(under)
            if div and p:
                p2, chg2 = _price(div)
                if p2:
                    p, chg = p / p2, (chg or 0) - (chg2 or 0)
                else:
                    p = None
            out.append({"symbol": d, "price": p,
                        "change_pct": (round(chg, 2) if chg is not None else None)})
        return {"prices": out}

    # ----------------------------------------------------------------- analyze
    def _compute_analyze() -> Dict:
        """Ağır hat — YALNIZ arka plan yenileyici çağırır."""
        o = orch()
        o.primary_tf = DEFAULT_TF
        signals = o.run(list(ALLOWED_SYMBOLS))
        return {"signals": [s.to_dict() for s in signals],
                "env": o.describe_environment(), "tf": DEFAULT_TF,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}

    def _signal_for(symbol: str) -> Optional[Dict]:
        hit = cache.get("analyze")
        if not hit:
            return None
        for s in hit[1].get("signals", []):
            if s.get("symbol") == symbol:
                return s
        return None

    @app.get("/api/analyze")
    def analyze():
        """Son anlık görüntüyü ANINDA döndürür (istek anında hesaplama yapmaz).
        Yaş bilgisi `age_sec` ile açıkça bildirilir."""
        hit = cache.get("analyze")
        if not hit:
            return {"pending": True, "signals": [], "tf": DEFAULT_TF,
                    "refresh_sec": REFRESH_SEC,
                    "note": "Motor ilk anlık görüntüyü hazırlıyor (~1 dk)."}
        out = dict(hit[1])
        out["age_sec"] = int(time.time() - hit[0])
        out["refresh_sec"] = REFRESH_SEC
        return out

    # ------------------------------------------------------------------- chart
    def _build_chart(symbol: str, tf: str, bars: int) -> Dict:
        from .chart import build_chart
        return build_chart(orch(), symbol, tf, bars=bars, signal=_signal_for(symbol))

    @app.get("/api/chart")
    def chart(symbol: str = Query(default="BTC/USDT"), tf: str = Query(default=DEFAULT_TF),
              bars: int = Query(default=160)):
        symbol = symbol.strip().upper()
        if symbol not in ALLOWED_SYMBOLS:
            symbol = ALLOWED_SYMBOLS[0]
        tf = tf if tf in ALLOWED_TFS else DEFAULT_TF
        bars = max(60, min(int(bars), 300))
        key = f"ch:{symbol}|{tf}|{bars}"
        hit = cache.get(key)
        if hit and time.time() - hit[0] < TTL_CHART:
            return hit[1]

        # Grafik tek zaman dilimi çektiği için ucuzdur (~1 sn); yine de yenileyici
        # çalışırken kilidi beklemeyelim — bayat kopya varsa onu ver.
        if not heavy_lock.acquire(timeout=20):
            if hit:
                return hit[1]
            return JSONResponse(status_code=503,
                                content={"error": "motor meşgul, birazdan tekrar deneyin"})
        try:
            out = _build_chart(symbol, tf, bars)
            cache[key] = (time.time(), out)
            return out
        except Exception as e:
            if hit:
                return hit[1]
            return JSONResponse(status_code=503,
                                content={"error": f"{type(e).__name__}: {e}"})
        finally:
            heavy_lock.release()

    # ---------------------------------------------------------------- patterns
    def _build_patterns(symbol: str, tf: str, bars: int, top_n: int) -> Dict:
        from ..analysis.chart_patterns import detect_chart_patterns, pattern_consensus
        o = orch()
        df = o.data.fetch_ohlcv(symbol, tf)
        if df is None or len(df) < 60:
            return {"symbol": symbol, "tf": tf, "patterns": [], "reason": "yetersiz veri"}
        # TESPİT penceresi ÇİZİM penceresinden geniş olmalı: 160 barda üçgen/kama
        # gibi çok pivotlu formasyonlar oluşamaz (ölçüldü: 160 barda 1, 320 barda 4-5).
        # Formasyon daha eskiden başlayabilir; çizerken sol kenara kırpılır.
        det_bars = max(bars * 2, 320)
        det = df.tail(det_bars)
        pats = detect_chart_patterns(det, top_n=top_n * 2)

        # Görünür pencere: tespit penceresinin son `bars` mumu → 0..bars-1
        offset = max(0, len(det) - bars)
        visible = []
        for p in pats:
            # görünür pencereden ÖNCE bitmiş formasyonu gösterme (artık geçersiz)
            if p["end_i"] < offset:
                continue
            visible.append(p)
        pats = visible[:top_n]
        for p in pats:
            p["start_i"] = max(0, p["start_i"] - offset)
            p["end_i"] = max(0, p["end_i"] - offset)
            for pt in p["points"]:
                pt["i"] = max(0, pt["i"] - offset)
            for ln in p["lines"]:
                ln["x0"] = max(0, ln["x0"] - offset)
                ln["x1"] = max(0, ln["x1"] - offset)
            for c in p["curve"]:
                c["x"] = max(0, c["x"] - offset)
        price = float(det["close"].iloc[-1])
        # "Hangi yönde yüzde kaçlık işlem optimal?" — yalnız GEÇERLİ formasyonlardan.
        # ATR, hedefin kaç barda ulaşılabileceğini söyler (uzak hedef = düşük değer).
        try:
            from ..analysis.indicators import atr as _atr
            atr_pct = float(_atr(det).iloc[-1]) / price * 100.0
        except Exception:
            atr_pct = 0.0
        try:
            from ..analysis.chart_patterns import trade_recommendation
            rec = trade_recommendation(pats, price, atr_pct, tf)
        except Exception as e:
            rec = {"available": False, "reason": f"{type(e).__name__}: {e}"}
        return {"symbol": symbol, "tf": tf, "bars": int(min(bars, len(det))),
                "detect_bars": int(len(det)),
                "price": price,
                "min_move_pct": 1.0,
                "atr_pct": round(atr_pct, 3),
                # kullanıcının "%1'lik oynamayı öngör" isteği: fiyatın ±%1 seviyeleri
                "one_pct_levels": {"up": round(price * 1.01, 8),
                                   "down": round(price * 0.99, 8)},
                "patterns": pats,
                "n_invalid": sum(1 for p in pats if not p.get("valid", True)),
                "recommendation": rec,
                "consensus": pattern_consensus(pats),
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}

    @app.get("/api/patterns")
    def patterns(symbol: str = Query(default="BTC/USDT"),
                 tf: str = Query(default=DEFAULT_TF),
                 bars: int = Query(default=160),
                 top: int = Query(default=5)):
        """Grafiğe çizilebilir en önemli N formasyon (geometri + hedef + %uzaklık)."""
        symbol = symbol.strip().upper()
        if symbol not in ALLOWED_SYMBOLS:
            symbol = ALLOWED_SYMBOLS[0]
        tf = tf if tf in ALLOWED_TFS else DEFAULT_TF
        bars = max(60, min(int(bars), 300))
        top = max(1, min(int(top), 8))
        key = f"pat:{symbol}|{tf}|{bars}|{top}"
        hit = cache.get(key)
        if hit and time.time() - hit[0] < TTL_CHART:
            return hit[1]
        if not heavy_lock.acquire(timeout=20):
            if hit:
                return hit[1]
            return JSONResponse(status_code=503,
                                content={"error": "motor meşgul", "patterns": []})
        try:
            out = _build_patterns(symbol, tf, bars, top)
            cache[key] = (time.time(), out)
            return out
        except Exception as e:
            if hit:
                return hit[1]
            return JSONResponse(status_code=503,
                                content={"error": f"{type(e).__name__}: {e}",
                                         "patterns": []})
        finally:
            heavy_lock.release()

    # --------------------------------------------------------------- harmonics
    def _build_harmonics(symbol: str, tf: str, bars: int,
                         pattern: Optional[str]) -> Dict:
        from ..analysis.harmonics import detect_harmonics_rich
        o = orch()
        df = o.data.fetch_ohlcv(symbol, tf)
        if df is None or len(df) < 60:
            return {"symbol": symbol, "tf": tf, "patterns": [],
                    "available": {}, "reason": "yetersiz veri"}
        det_bars = max(bars * 2, 320)
        det = df.tail(det_bars)
        out = detect_harmonics_rich(det, pattern=pattern, top_n=6,
                                    include_forming=True, bars_ahead=12)

        # görünür pencereye hizala (0..bars-1); projekte D sağa taşabilir
        offset = max(0, len(det) - bars)
        keep = []
        for p in out["patterns"]:
            if max((pt["i"] for pt in p["points"]), default=0) < offset:
                continue                       # tamamen görünür pencerenin solunda
            for pt in p["points"]:
                pt["i"] = pt["i"] - offset
            for lg in p["legs"]:
                lg["x0"] -= offset
                lg["x1"] -= offset
            if p.get("prz"):
                p["prz"]["i"] -= offset
                p["prz"]["i_from"] -= offset
            keep.append(p)
        out["patterns"] = keep
        out["symbol"], out["tf"] = symbol, tf
        out["bars"] = int(min(bars, len(det)))
        out["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        return out

    @app.get("/api/harmonics")
    def harmonics(symbol: str = Query(default="BTC/USDT"),
                  tf: str = Query(default=DEFAULT_TF),
                  bars: int = Query(default=160),
                  pattern: str = Query(default="")):
        """Harmonik XABCD formasyonları — geometri, bacak oranları, PRZ ve işlem planı.

        pattern boşsa hepsi; "butterfly"/"cypher"/… ile tek formasyon filtrelenir."""
        symbol = symbol.strip().upper()
        if symbol not in ALLOWED_SYMBOLS:
            symbol = ALLOWED_SYMBOLS[0]
        tf = tf if tf in ALLOWED_TFS else DEFAULT_TF
        bars = max(60, min(int(bars), 300))
        pat = (pattern or "").strip().lower() or None
        key = f"harm:{symbol}|{tf}|{bars}|{pat}"
        hit = cache.get(key)
        if hit and time.time() - hit[0] < TTL_CHART:
            return hit[1]
        if not heavy_lock.acquire(timeout=20):
            if hit:
                return hit[1]
            return JSONResponse(status_code=503,
                                content={"error": "motor meşgul", "patterns": []})
        try:
            out = _build_harmonics(symbol, tf, bars, pat)
            cache[key] = (time.time(), out)
            return out
        except Exception as e:
            if hit:
                return hit[1]
            return JSONResponse(status_code=503,
                                content={"error": f"{type(e).__name__}: {e}",
                                         "patterns": []})
        finally:
            heavy_lock.release()

    # ------------------------------------------------------------------- mover
    def _build_mover() -> Dict:
        """Bugün hangi parite ≥%1 oynar? Tüm izlenen pariteler sıralanır."""
        from ..analysis.mover import rank_movers, load_validation
        o = orch()
        panel = {}
        for sym in ALLOWED_SYMBOLS:
            try:
                df = o.data.fetch_ohlcv(sym, "1d")
            except Exception:
                df = None
            if df is not None and len(df) >= 250:
                panel[sym] = df
        if not panel:
            return {"picks": [], "reason": "günlük veri alınamadı"}
        out = rank_movers(panel, validation=load_validation())
        out["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        return out

    @app.get("/api/mover")
    def mover():
        """Günün "%1 hareket adayı" sıralaması — kanıtlı, kalibre, taban oranlı.

        SADECE BÜYÜKLÜK. Yön tahmini yapılmaz (ölçüldü: yön AUC 0,47-0,50)."""
        hit = cache.get("mover")
        if hit and time.time() - hit[0] < 3600:
            out = dict(hit[1])
            out["age_sec"] = int(time.time() - hit[0])
            return out
        if not heavy_lock.acquire(timeout=25):
            if hit:
                return hit[1]
            return JSONResponse(status_code=503,
                                content={"error": "motor meşgul", "picks": []})
        try:
            out = _build_mover()
            cache["mover"] = (time.time(), out)
            return out
        except Exception as e:
            if hit:
                return hit[1]
            return JSONResponse(status_code=503,
                                content={"error": f"{type(e).__name__}: {e}",
                                         "picks": []})
        finally:
            heavy_lock.release()

    # ----------------------------------------------------------------- candles
    def _build_candles(symbol: str, tf: str, lookback: int) -> Dict:
        from ..analysis.candles import (detect_candles, candle_summary,
                                        CANDLE_MEASURED)
        o = orch()
        df = o.data.fetch_ohlcv(symbol, tf)
        if df is None or len(df) < 40:
            return {"available": False, "symbol": symbol, "tf": tf,
                    "reason": "yeterli bar yok"}
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        pats = detect_candles(df, lookback=lookback)
        return {"available": True, "symbol": symbol, "tf": tf,
                "patterns": pats,
                "summary": candle_summary(pats),
                "measured": CANDLE_MEASURED,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}

    @app.get("/api/candles")
    def candles(symbol: str = Query(default="BTC/USDT"),
                tf: str = Query(default=DEFAULT_TF),
                lookback: int = Query(default=3)):
        """Mum formasyonları — bağlam şartlı, ölçülmüş kanıtla birlikte.

        DİKKAT: yön üstünlüğü ölçüldü ve bulunamadı; anlamlı çıkan iki formasyon
        TERS yönde. Ayrıntı `measured` ve her formasyonun `evidence` alanında."""
        symbol = symbol.strip().upper()
        if symbol not in ALLOWED_SYMBOLS:
            symbol = ALLOWED_SYMBOLS[0]
        tf = tf if tf in ALLOWED_TFS else DEFAULT_TF
        lookback = max(1, min(int(lookback), 10))
        key = f"cdl:{symbol}|{tf}|{lookback}"
        hit = cache.get(key)
        if hit and time.time() - hit[0] < 600:
            return hit[1]
        if not heavy_lock.acquire(timeout=20):
            if hit:
                return hit[1]
            return JSONResponse(status_code=503,
                                content={"available": False, "error": "motor meşgul"})
        try:
            out = _build_candles(symbol, tf, lookback)
            cache[key] = (time.time(), out)
            return out
        except Exception as e:
            if hit:
                return hit[1]
            return JSONResponse(status_code=503,
                                content={"available": False,
                                         "error": f"{type(e).__name__}: {e}"})
        finally:
            heavy_lock.release()

    # -------------------------------------------------------------- indicators
    def _build_indicators(symbol: str, tf: str) -> Dict:
        from ..analysis.indicator_board import build_board, BOARD_EVIDENCE
        o = orch()
        df = o.data.fetch_ohlcv(symbol, tf)
        if df is None or len(df) < 210:
            return {"available": False, "symbol": symbol, "tf": tf,
                    "reason": "yeterli bar yok"}
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        # MİKROYAPI: kaydediciden bu paritenin son satırı. Varsa funding, defter
        # eğimi, derinlik dengesizliği gibi FİYATTAN TÜRETİLEMEYEN göstergeler
        # eklenir; yoksa o bölüm atlanır (uydurulmaz).
        micro = None
        try:
            from ..data.recorder import load_features
            feat = load_features()
            key = symbol.replace("/", "")
            g = feat[feat["symbol"] == key]
            if len(g):
                micro = g.iloc[-1]
        except Exception:
            micro = None
        out = build_board(df, micro=micro)
        out["symbol"] = symbol
        out["tf"] = tf
        out["microstructure"] = micro is not None
        out["evidence"] = BOARD_EVIDENCE
        out["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        return out

    @app.get("/api/indicators")
    def indicators(symbol: str = Query(default="BTC/USDT"),
                   tf: str = Query(default=DEFAULT_TF)):
        """129 göstergenin AL / SAT / NÖTR dağılımı.

        DİKKAT: ham sayım bağımsız kanıt değildir (13 EMA aynı şeyi ölçer) ve
        ölçüldü ki bu konsensüsü takip etmek örneklem dışında para KAYBETTİRİR.
        Uç, ölçüm sonucunu `evidence` alanında birlikte döndürür."""
        symbol = symbol.strip().upper()
        if symbol not in ALLOWED_SYMBOLS:
            symbol = ALLOWED_SYMBOLS[0]
        tf = tf if tf in ALLOWED_TFS else DEFAULT_TF
        key = f"ind:{symbol}|{tf}"
        hit = cache.get(key)
        if hit and time.time() - hit[0] < 600:
            return hit[1]
        if not heavy_lock.acquire(timeout=20):
            if hit:
                return hit[1]
            return JSONResponse(status_code=503,
                                content={"available": False, "error": "motor meşgul"})
        try:
            out = _build_indicators(symbol, tf)
            cache[key] = (time.time(), out)
            return out
        except Exception as e:
            if hit:
                return hit[1]
            return JSONResponse(status_code=503,
                                content={"available": False,
                                         "error": f"{type(e).__name__}: {e}"})
        finally:
            heavy_lock.release()

    # ------------------------------------------------------------------ social
    @app.get("/api/social")
    def social():
        """X/haber istihbaratı: toplama durumu + ÖLÇÜLMÜŞ hesap etki skorları.

        Elle yazılmış ağırlıklar KULLANILMAZ — yalnız istatistiksel kapıyı
        geçen hesaplar skor alır (bkz. sentiment/event_study.py)."""
        def _f():
            from ..sentiment.event_study import load_scores
            from ..sentiment.collector import load_events
            scores = load_scores()
            try:
                ev = load_events()
                n_ev, srcs = int(len(ev)), sorted(ev["source"].unique().tolist()) if len(ev) else []
                first = str(ev["ts"].min())[:16] if len(ev) else None
            except Exception:
                n_ev, srcs, first = 0, [], None
            x_live = False
            try:
                from ..sentiment.twitter_intelligence import TwitterIntelligence
                x_live = bool(TwitterIntelligence(cfg).live())
            except Exception:
                pass
            top = [a for a in scores.get("accounts", []) if a.get("measured")][:10]
            return {
                "collector": {"events": n_ev, "sources": srcs, "since": first,
                              "x_live": x_live},
                "x_note": ("X akışı AÇIK (anahtar bulundu)" if x_live else
                           "X akışı KAPALI — X API anahtarı yok. Ücretsiz haber "
                           "kaynakları (CryptoPanic + RSS) kullanılıyor. Kendi "
                           "anahtarınızı /#hesap sayfasından girebilirsiniz."),
                "measured_accounts": top,
                "n_accounts": scores.get("n_accounts", 0),
                "n_measured": scores.get("n_measured", 0),
                "min_events": scores.get("min_events", 20),
                "study_note": scores.get("note", "henüz ölçüm yapılmadı"),
                "direction_warning": scores.get(
                    "direction_warning",
                    "Yön ve büyüklük ayrı ölçülür; büyüklük etkisi yön öngörüsü demek değildir."),
            }
        return cached("social", 900, _f)

    # ------------------------------------------------------------------ events
    @app.get("/api/events")
    def events():
        def _f():
            from ..macro import upcoming_events
            try:
                return {"calendar": upcoming_events()}
            except Exception as e:
                return {"calendar": [], "error": f"{type(e).__name__}: {e}"}
        return cached("events", TTL_EVENTS, _f)

    # ------------------------------------------------------------- correlation
    @app.get("/api/correlation")
    def correlation(tf: str = Query(default="1d")):
        tf = tf if tf in ALLOWED_TFS else "1d"

        def _compute():
            import numpy as np
            import pandas as pd
            o = orch()
            rets: Dict[str, "pd.Series"] = {}
            for s in ALLOWED_SYMBOLS:
                try:
                    df = o.data.fetch_ohlcv(s, tf)
                except Exception:
                    df = None
                if df is not None and len(df) > 30:
                    rets[s] = df["close"].pct_change().dropna()
            if len(rets) < 2:
                return {"symbols": list(rets), "matrix": [], "tf": tf,
                        "note": "korelasyon için en az 2 parite gerekli"}
            n = min(min(len(v) for v in rets.values()), 200)
            frame = pd.DataFrame({k: v.tail(n).reset_index(drop=True) for k, v in rets.items()})
            corr = frame.corr().round(3)
            m = corr.values
            off = m[~np.eye(len(m), dtype=bool)]
            avg_abs = float(np.mean(np.abs(off))) if off.size else 0.0
            return {"symbols": list(corr.columns), "matrix": corr.values.tolist(),
                    "tf": tf, "bars": n, "avg_abs_corr": round(avg_abs, 3),
                    "concentration": ("YÜKSEK" if avg_abs > 0.7 else
                                      "ORTA" if avg_abs > 0.4 else "DÜŞÜK")}

        key = f"corr:{tf}"
        hit = cache.get(key)
        if hit and time.time() - hit[0] < TTL_CORR:
            return hit[1]
        if not heavy_lock.acquire(timeout=15):
            return hit[1] if hit else {"symbols": [], "matrix": [], "tf": tf,
                                       "note": "motor meşgul"}
        try:
            out = _compute()
            cache[key] = (time.time(), out)
            return out
        except Exception as e:
            return hit[1] if hit else {"symbols": [], "matrix": [], "tf": tf,
                                       "note": f"{type(e).__name__}"}
        finally:
            heavy_lock.release()

    # ------------------------------------------------------- arka plan yenileyici
    def _refresh_cycle() -> None:
        """Ağır anlık görüntüyü üret. Kilit parça parça alınır ki panel istekleri
        (grafik/fiyat) döngü ortasında da yanıt alabilsin."""
        with heavy_lock:
            out = _compute_analyze()
            cache["analyze"] = (time.time(), out)
        for sym in ALLOWED_SYMBOLS:
            with heavy_lock:
                try:
                    cache[f"ch:{sym}|{DEFAULT_TF}|160"] = (time.time(),
                                                          _build_chart(sym, DEFAULT_TF, 160))
                except Exception:
                    pass
                try:
                    cache[f"pat:{sym}|{DEFAULT_TF}|160|5"] = (
                        time.time(), _build_patterns(sym, DEFAULT_TF, 160, 5))
                except Exception:
                    pass
                try:
                    cache[f"harm:{sym}|{DEFAULT_TF}|160|None"] = (
                        time.time(), _build_harmonics(sym, DEFAULT_TF, 160, None))
                except Exception:
                    pass
                try:
                    cache[f"ind:{sym}|{DEFAULT_TF}"] = (
                        time.time(), _build_indicators(sym, DEFAULT_TF))
                except Exception:
                    pass
                try:
                    cache[f"cdl:{sym}|{DEFAULT_TF}|3"] = (
                        time.time(), _build_candles(sym, DEFAULT_TF, 3))
                except Exception:
                    pass
        with heavy_lock:
            try:
                cache["mover"] = (time.time(), _build_mover())
            except Exception:
                pass
        for disp in DEFAULT_PRICE_STRIP:
            under, div = _PRICE_MAP.get(disp, (disp, None))
            for u in filter(None, (under, div)):
                _price(u)

    def _refresher() -> None:
        time.sleep(3)                      # uvicorn portu bağlasın
        while True:
            try:
                _refresh_cycle()
            except Exception as e:          # döngü asla ölmesin
                print(f"[cryptomind] yenileme hatası: {type(e).__name__}: {e}", flush=True)
            time.sleep(max(60, REFRESH_SEC))

    if REFRESH_SEC > 0:
        threading.Thread(target=_refresher, name="cryptomind-refresh", daemon=True).start()

    # NET +%1 nitelendirme katmanı — kendi arka plan döngüsüyle ayrı modülde.
    # Bağlanamazsa panel kalanıyla çalışmaya devam eder (fail-open DEĞİL:
    # uçlar yoksa panel o bölümü göstermez, yanlış sayı üretmez).
    try:
        from .qualification_api import register as _register_qual
        _register_qual(app, runs)
    except Exception as e:
        print(f"[cryptomind] nitelendirme katmanı yüklenemedi: "
              f"{type(e).__name__}: {e}", flush=True)

    # Otopilot koşucuları orkestratörün SON anlık görüntüsünü buradan okur —
    # panelde gösterilen sinyalle aynı kaynak (ayrı hesaplama yok).
    def _cached_value(key: str):
        hit = cache.get(key)
        return (hit[1], time.time() - hit[0]) if hit else (None, None)

    def _mover_pick(symbol: str):
        mv, _ = _cached_value("mover")
        if not mv:
            return None
        key = symbol.replace("/", "").upper()
        for p in mv.get("picks") or []:
            s = str(p.get("symbol", "")).replace("/", "").upper()
            if s == key:
                return p
        return None

    def _context_for(symbol: str) -> Optional[Dict]:
        """Komite stratejisinin YAVAŞ bağlamı — yalnız önbellekten okur, HESAPLAMAZ.
        Panelde gösterilen grafik/formasyon/gösterge/mover ile birebir aynı veri."""
        if symbol not in ALLOWED_SYMBOLS:
            return None
        chart, age = _cached_value(f"ch:{symbol}|{DEFAULT_TF}|160")
        pat, _ = _cached_value(f"pat:{symbol}|{DEFAULT_TF}|160|5")
        harm, _ = _cached_value(f"harm:{symbol}|{DEFAULT_TF}|160|None")
        ind, _ = _cached_value(f"ind:{symbol}|{DEFAULT_TF}")
        cdl, _ = _cached_value(f"cdl:{symbol}|{DEFAULT_TF}|3")
        ev, _ = _cached_value("events")
        corr, _ = _cached_value("corr:1d")
        soc, _ = _cached_value("social")
        sig = _signal_for(symbol)
        sig_age = (time.time() - cache["analyze"][0]) if cache.get("analyze") else None
        ages = [a for a in (age, sig_age) if a is not None]
        return {"symbol": symbol, "tf": DEFAULT_TF, "signal": sig, "chart": chart,
                "patterns": pat, "harmonics": harm, "indicators": ind, "candles": cdl,
                "mover_pick": _mover_pick(symbol), "events": (ev or {}).get("calendar"),
                "corr": corr, "social": soc,
                "age_sec": (max(ages) if ages else None)}

    def _candidate_symbols() -> List[str]:
        """Simülatör için aday pariteler: yavaş bağlamı OLAN pariteler, mover
        olasılığına göre sıralı (veri olmayan parite aday DEĞİLDİR)."""
        mv, _ = _cached_value("mover")
        prob = {}
        for p in (mv or {}).get("picks") or []:
            s = str(p.get("symbol", "")).replace("/", "").upper()
            try:
                prob[s] = float(p.get("probability") or p.get("prob") or 0.0)
            except (TypeError, ValueError):
                pass
        return sorted(ALLOWED_SYMBOLS, key=lambda s: -prob.get(s.replace("/", ""), 0.0))

    app.state.signal_for = _signal_for
    app.state.context_for = _context_for
    app.state.candidate_symbols = _candidate_symbols
    app.state.allowed_symbols = list(ALLOWED_SYMBOLS)
    app.state.runs_dir = runs
    app.state.config = cfg
    return app
