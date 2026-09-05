"""
Web Dashboard (FastAPI) — Frontend/Dashboard rolü.

Tüm motoru tek panelde toplar:
  • Canlı analiz kartları (yön, güven, alış/satış %, sonraki maks/min, katmanlar)
  • Canlı fiyat akışı (SSE)                          [#1]
  • Likidasyon haritası + CVD/Order Flow             [#2]
  • Backtest + walk-forward                          [#3]
  • Çoklu-borsa arbitraj                             [#8]
  • Ağırlık öğrenme tetikleme                        [#7]

Çalıştırma:  python serve.py   (veya)  python -m agi_trader.server
"""
from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import Body, FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from ..config import load_config
from ..agents import Orchestrator
from ..providers import cred_schema, valid_keys

_HTML = (Path(__file__).parent / "dashboard.html")

# Panelden girilebilecek kimlik bilgileri — TEK KAYNAK: providers.py kataloğu.
# Yeni borsa/sağlayıcı eklemek için providers.PROVIDERS'a ekle; panel otomatik güncellenir.
CRED_SCHEMA = cred_schema()
_VALID_KEYS = valid_keys()


def _mask(val):
    if not val:
        return ""
    return val[-4:] if len(val) > 4 else "••"


def create_app(config=None) -> FastAPI:
    app = FastAPI(title="AGI Trader Dashboard")
    cfg = config or load_config()
    _orch0 = Orchestrator(cfg)
    from ..auto import AutoTrader
    state = {"orch": _orch0, "cfg": cfg, "cache": {}, "cache_ts": {},
             "auto": AutoTrader(_orch0, cfg)}
    app.state.auto = state["auto"]   # test/debug erişimi (idiomatik FastAPI)
    app.state.orch = _orch0

    # ---- streaming (SSE) altyapısı (#1) ----
    from .stream import LiveStreamer
    streamer = LiveStreamer(state["orch"].data, cfg)
    state["streamer"] = streamer
    streamer.start()

    def _orch() -> Orchestrator:
        return state["orch"]

    @app.get("/", response_class=HTMLResponse)
    def index():
        return _HTML.read_text(encoding="utf-8")

    @app.get("/api/env")
    def env():
        return _orch().describe_environment()

    # ---- canlı fiyat şeridi (kripto + BIST + USD/TRY) ----
    # Görünen ad -> (underlying, böl_underlying|None). Böl varsa: fiyat = a/b.
    _PRICE_MAP = {
        "BTC/USDT": ("BTC/USDT", None), "ETH/USDT": ("ETH/USDT", None),
        "SOL/USDT": ("SOL/USDT", None), "DOLAR/TL": ("USDTRY=X", None),
        "BIST/TL": ("XU100.IS", None), "BIST/DOLAR": ("XU100.IS", "USDTRY=X"),
    }

    def _price_cached(underlying: str):
        now = time.time()
        c = state.setdefault("price_cache", {})
        hit = c.get(underlying)
        if hit and now - hit[2] < 8:          # 8 sn önbellek
            return hit[0], hit[1]
        p, chg = _orch().data.fetch_price(underlying)
        c[underlying] = (p, chg, now)
        return p, chg

    @app.get("/api/prices")
    def prices(symbols: str = Query(default="BTC/USDT,ETH/USDT,BIST/DOLAR,DOLAR/TL,BIST/TL,SOL/USDT")):
        disp = [s.strip() for s in symbols.split(",") if s.strip()]
        out = []
        for d in disp:
            under, div = _PRICE_MAP.get(d, (d, None))
            p, chg = _price_cached(under)
            if div:
                p2, chg2 = _price_cached(div)
                if p and p2:
                    p = p / p2
                    chg = (chg or 0) - (chg2 or 0)   # oran değişimi ≈ fark
                else:
                    p = None
            out.append({"symbol": d, "price": p, "change_pct": (round(chg, 2) if chg is not None else None)})
        return {"prices": out}

    @app.get("/api/analyze")
    def analyze(symbols: str = Query(default=""), tf: str = Query(default="")):
        orch = _orch()
        syms = [s.strip() for s in symbols.split(",") if s.strip()] or orch.config.symbols
        if tf:
            orch.primary_tf = tf
        # 60 sn önbellek (ağır pipeline tekrar etmesin)
        key = f"{','.join(syms)}|{orch.primary_tf}"
        now = time.time()
        if key in state["cache"] and now - state["cache_ts"].get(key, 0) < 60:
            return state["cache"][key]
        signals = orch.run(syms)
        out = {"signals": [s.to_dict() for s in signals],
               "env": orch.describe_environment()}
        state["cache"][key] = out
        state["cache_ts"][key] = now
        return out

    @app.get("/api/stream")
    def stream(symbols: str = Query(default="BTC/USDT,ETH/USDT")):
        syms = [s.strip() for s in symbols.split(",") if s.strip()]

        def gen():
            q = streamer.subscribe(syms)
            try:
                while True:
                    try:
                        data = q.get(timeout=30)
                        yield f"data: {json.dumps(data)}\n\n"
                    except queue.Empty:
                        yield ": keep-alive\n\n"
            finally:
                streamer.unsubscribe(q)

        return StreamingResponse(gen(), media_type="text/event-stream")

    # ---- grafik + formasyon görselleştirme ----
    @app.get("/api/chart")
    def chart(symbol: str = Query(default="BTC/USDT"), tf: str = Query(default=""),
              bars: int = Query(default=200)):
        from .chart import build_chart
        orch = _orch()
        tf = tf or orch.primary_tf
        # giriş/stop/TP seviyelerini, varsa /api/analyze önbelleğinden bul (hızlı)
        sig = None
        for out in state["cache"].values():
            for s in out.get("signals", []):
                if s.get("symbol") == symbol and s.get("timeframe") == tf:
                    sig = s
                    break
            if sig:
                break
        return build_chart(orch, symbol, tf, bars=bars, signal=sig)

    # ---- FAZ6: Trend-takip paper portföyü (OOS-doğrulanmış çekirdek strateji) ----
    @app.get("/api/trend")
    def trend():
        import json
        from pathlib import Path
        p = Path(config.get("output_dir", "runs")) / "trend_state.json"
        if not p.exists():
            return {"available": False, "reason": "trend_daemon henüz çalışmadı (python trend_daemon.py)"}
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            return {"available": False, "reason": str(e)}
        eq = float(d.get("equity", 0)); init = float(d.get("initial", 10000))
        sigs = d.get("last_signals", {})
        in_market = [s for s, v in sigs.items() if v.get("in_market")]
        curve = [{"date": h.get("date"), "equity": h.get("equity")} for h in d.get("history", [])[-180:]]
        return {"available": True, "mode": "trend_follow_paper",
                "equity": round(eq, 2), "return_pct": round((eq/init-1)*100, 2),
                "invested_pct": round(sum(d.get("weights", {}).values())*100, 1),
                "in_market": in_market, "cash": len(in_market) == 0,
                "weights": d.get("weights", {}), "signals": sigs,
                "last_rebalance": d.get("last_rebalance"), "equity_curve": curve}

    # ---- Ops/İzleme: risk & sağlık metrikleri (drift, VaR/CVaR, maruziyet) ----
    @app.get("/api/risk")
    def risk():
        import json
        from pathlib import Path
        from ..monitor import risk_report
        p = Path(config.get("output_dir", "runs")) / "trend_state.json"
        if not p.exists():
            return {"available": False, "reason": "trend_daemon henüz çalışmadı"}
        try:
            state = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            return {"available": False, "reason": str(e)}
        return risk_report(state)

    # ---- RL v2 (PPO) durum + çoklu-varlık eğitim ----
    @app.get("/api/rl")
    def rl_status():
        try:
            from ..ai.rl_agent import _HAS_RL, _MEM_CACHE
            return {"available": _HAS_RL, "cached_agents": list(_MEM_CACHE.keys()),
                    "note": "PPO ajanı ai_ensemble katmanında otomatik kullanılır (symbol+tf başına)."}
        except Exception as e:
            return {"available": False, "error": str(e)}

    @app.post("/api/rl/train")
    def rl_train(body: Dict = Body(default={})):
        """Tüm izlenen pariteler üzerinde TEK paylaşılan PPO ajanı eğit (multi-asset)."""
        from ..ai.rl_agent import RLAgent, _HAS_RL
        if not _HAS_RL:
            return {"ok": False, "reason": "stable-baselines3/gymnasium kurulu değil"}
        orch = _orch()
        syms = [s.strip() for s in str(body.get("symbols", "")).split(",") if s.strip()] or orch.config.symbols
        dfs, ok = [], []
        for s in syms:
            try:
                df = orch.data.fetch_ohlcv(s, orch.primary_tf)
                if df is not None and len(df) > 120:
                    dfs.append(df); ok.append(s)
            except Exception:
                continue
        if not dfs:
            return {"ok": False, "reason": "yeterli veri yok"}
        ag = RLAgent("multiasset", timesteps=int(body.get("timesteps", 20000)))
        meta = ag.fit_or_load(dfs, force=True)
        preds = {s: ag.predict(d) for s, d in zip(ok, dfs)}
        return {"ok": meta.get("available", False), "meta": meta, "symbols": ok,
                "predictions": preds}

    # ---- bot komutları (genel + Telegram webhook) ----
    @app.get("/api/command")
    def command(cmd: str = Query(...)):
        from ..bot import process_command
        return {"reply": process_command(_orch(), state["auto"], cmd)}

    @app.post("/api/webhook/telegram")
    def telegram_webhook(body: Dict = Body(...)):
        from ..bot import process_command
        msg = (body.get("message") or body.get("edited_message") or {})
        text = msg.get("text", "")
        chat = (msg.get("chat") or {}).get("id")
        if not text:
            return {"ok": True, "skip": "metin yok"}
        reply = process_command(_orch(), state["auto"], text)
        # yanıtı aynı sohbete gönder (bot token varsa)
        token = state["cfg"].secret("TELEGRAM_BOT_TOKEN")
        if token and chat:
            try:
                import requests
                requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                              json={"chat_id": chat, "text": reply}, timeout=8)
            except Exception:
                pass
        return {"ok": True, "reply": reply}

    # ---- sektör rotasyonu ----
    @app.get("/api/sectors")
    def sectors(tf: str = Query(default="1d")):
        from ..macro import compute_sector_rotation
        return compute_sector_rotation(_orch(), tf)

    # ---- TradingView webhook (harici strateji sinyali) ----
    @app.post("/api/webhook/tradingview")
    def tv_webhook(body: Dict = Body(...)):
        secret = state["cfg"].secret("TRADINGVIEW_WEBHOOK_SECRET")
        if secret and str(body.get("secret", "")) != secret:
            return JSONResponse({"ok": False, "reason": "geçersiz secret"}, status_code=403)
        symbol = str(body.get("symbol", "")).strip()
        action = str(body.get("action") or body.get("side") or "").strip()
        if not symbol or not action:
            return {"ok": False, "reason": "symbol ve action gerekli"}
        # TradingView 'BTCUSDT' -> 'BTC/USDT' normalizasyonu (yalnız kripto için)
        if "/" not in symbol and symbol.upper().endswith("USDT"):
            symbol = symbol[:-4].upper() + "/USDT"
        price = body.get("price")
        res = state["auto"].external_signal(symbol, action, price, source="TradingView")
        return res

    # ---- olay öngörü + zamanlama (makro takvim + contagion) ----
    @app.get("/api/events")
    def events(symbols: str = Query(default="")):
        from ..macro import upcoming_events, market_movers
        # contagion için: önce autotrader overview, yoksa son analiz önbelleği
        snaps = list(state["auto"].overview.values())
        if not snaps:
            for out in state["cache"].values():
                for s in out.get("signals", []):
                    snaps.append({"symbol": s.get("symbol"), "direction": s.get("direction"),
                                  "momentum": s.get("momentum_score", 50), "confidence": s.get("confidence", 0),
                                  "price": s.get("entry"), "correlation": s.get("correlation_badge")})
        # SÜRÜCÜYE-GÖRELİ korelasyon matrisi (driver-relative contagion için)
        corr_matrix = _build_corr_matrix([s["symbol"] for s in snaps if s.get("symbol")])
        return {"calendar": upcoming_events(), "movers": market_movers(snaps, corr_matrix)}

    def _build_corr_matrix(syms):
        import pandas as pd
        syms = list(dict.fromkeys(syms))  # tekilleştir, sırayı koru
        if len(syms) < 2:
            return None
        orch = _orch()
        rets = {}
        for s in syms:
            df = orch.data.fetch_ohlcv(s, orch.primary_tf)
            if df is not None and len(df) > 30:
                rets[s] = df["close"].pct_change().dropna()
        if len(rets) < 2:
            return None
        n = min(min(len(v) for v in rets.values()), 200)
        frame = pd.DataFrame({k: v.tail(n).reset_index(drop=True) for k, v in rets.items()})
        cm = frame.corr()
        return {a: {b: (None if a == b else round(float(cm.loc[a, b]), 3)) for b in cm.columns}
                for a in cm.index}

    # ---- otonom işlem motoru (AutoTrader) + komuta merkezi ----
    @app.get("/api/auto")
    def auto_status():
        return state["auto"].full_state()

    @app.post("/api/auto/start")
    def auto_start(body: Dict = Body(default={})):
        syms = None
        if body.get("symbols"):
            syms = [s.strip() for s in str(body["symbols"]).split(",") if s.strip()]
        return state["auto"].start(interval=body.get("interval"), symbols=syms)

    @app.post("/api/auto/stop")
    def auto_stop():
        return state["auto"].stop()

    @app.post("/api/auto/reset")
    def auto_reset():
        return state["auto"].reset()

    @app.post("/api/auto/close_all")
    def auto_close_all():
        return state["auto"].close_all()

    # ---- alarmlar (fiyat + formasyon) ----
    @app.get("/api/alarms")
    def alarms(limit: int = Query(default=30)):
        eng = _orch().alarms
        return {"recent": eng.recent(limit), "price_alarms": eng.list_price_alarms(),
                "notify_enabled": _orch().notifier.enabled}

    @app.post("/api/alarms")
    def add_alarm(body: Dict = Body(...)):
        eng = _orch().alarms
        sym = str(body.get("symbol", "")).strip()
        try:
            price = float(body.get("price"))
        except (TypeError, ValueError):
            return {"ok": False, "reason": "geçerli fiyat gerekli"}
        if not sym:
            return {"ok": False, "reason": "parite gerekli"}
        return {"ok": True, "alarm": eng.add_price_alarm(sym, price, body.get("direction", "cross"))}

    @app.delete("/api/alarms/{alarm_id}")
    def del_alarm(alarm_id: int):
        return {"ok": _orch().alarms.remove_price_alarm(alarm_id)}

    # ---- korelasyon matrisi (portföy yoğunlaşma riski) ----
    @app.get("/api/correlation")
    def correlation(symbols: str = Query(default=""), tf: str = Query(default="")):
        import pandas as pd
        orch = _orch()
        syms = [s.strip() for s in symbols.split(",") if s.strip()] or orch.config.symbols
        tf = tf or orch.primary_tf
        rets: Dict[str, pd.Series] = {}
        for s in syms:
            df = orch.data.fetch_ohlcv(s, tf)
            if df is not None and len(df) > 30:
                rets[s] = df["close"].pct_change().dropna()
        if len(rets) < 2:
            return {"symbols": list(rets), "matrix": [], "tf": tf,
                    "note": "korelasyon için en az 2 parite gerekli"}
        n = min(min(len(v) for v in rets.values()), 200)
        frame = pd.DataFrame({k: v.tail(n).reset_index(drop=True) for k, v in rets.items()})
        corr = frame.corr().round(3)
        # aşırı yoğunlaşma uyarısı: ortalama |korelasyon| yüksekse
        import numpy as np
        m = corr.values
        off = m[~np.eye(len(m), dtype=bool)]
        avg_abs = float(np.mean(np.abs(off))) if off.size else 0.0
        return {"symbols": list(corr.columns), "matrix": corr.values.tolist(), "tf": tf,
                "avg_abs_corr": round(avg_abs, 3),
                "concentration": ("YÜKSEK" if avg_abs > 0.7 else "ORTA" if avg_abs > 0.4 else "DÜŞÜK")}

    # ---- dışa aktarım: sinyaller (JSON) / işlem günlüğü (CSV) ----
    @app.get("/api/export")
    def export(what: str = Query(default="signals"), fmt: str = Query(default="json")):
        if what == "journal":
            entries = _orch().journal.load_all()
            if fmt == "csv":
                import io, csv
                buf = io.StringIO()
                if entries:
                    cols = sorted({k for e in entries for k in e.keys()})
                    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
                    w.writeheader()
                    for e in entries:
                        w.writerow({k: e.get(k) for k in cols})
                return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=journal.csv"})
            return JSONResponse(entries)
        # signals: son analiz önbelleğinden topla
        sigs = []
        for out in state["cache"].values():
            sigs.extend(out.get("signals", []))
        return JSONResponse({"signals": sigs})

    # ---- #2 likidasyon + CVD ----
    @app.get("/api/liquidations")
    def liquidations(symbol: str = Query(default="BTC/USDT")):
        from ..onchain.liquidation import liquidation_map, cvd_analysis
        em = _orch().data
        return {"symbol": symbol,
                "liquidation_map": liquidation_map(_orch().whale, symbol),
                "cvd": cvd_analysis(_orch().whale, symbol)}

    # ---- #8 arbitraj ----
    @app.get("/api/arbitrage")
    def arbitrage(symbols: str = Query(default="")):
        from ..data.arbitrage import scan_arbitrage
        syms = [s.strip() for s in symbols.split(",") if s.strip()] or _orch().config.symbols
        return {"opportunities": scan_arbitrage(_orch().data, syms)}

    # ---- #3 backtest ----
    @app.get("/api/backtest")
    def backtest(symbol: str = Query(default="BTC/USDT"), tf: str = Query(default="4h"),
                 bars: int = Query(default=600)):
        from ..backtest.engine import run_backtest
        return run_backtest(_orch(), symbol, tf, bars)

    # ---- parametre optimizasyonu (grid search + Monte Carlo doğrulama) ----
    @app.get("/api/optimize")
    def optimize(symbol: str = Query(default="BTC/USDT"), tf: str = Query(default="4h"),
                 bars: int = Query(default=800)):
        from ..backtest.engine import optimize_parameters
        return optimize_parameters(_orch(), symbol, tf, bars)

    @app.post("/api/optimize/apply")
    def optimize_apply(body: Dict = Body(...)):
        """Optimizer'ın bulduğu en iyi parametreleri canlı risk motoruna uygula.
        atr_mult → stop mesafesi; rr → TP merdiveni; ÇALIŞAN motora anında yansır."""
        try:
            atr_mult = float(body.get("atr_mult"))
            rr = float(body.get("rr"))
        except (TypeError, ValueError):
            return {"ok": False, "reason": "geçerli atr_mult ve rr gerekli"}
        cfg = state["cfg"]
        tp_ladder = [round(rr * 0.7, 2), round(rr, 2), round(rr * 1.4, 2)]
        cfg.data.setdefault("risk", {})["atr_stop_mult"] = atr_mult
        cfg.data["risk"]["tp_r_multiples"] = tp_ladder
        # çalışan risk motoruna yerinde uygula (öğrenilmiş ağırlıklar korunur)
        orch = state["orch"]
        orch.risk.atr_stop_mult = atr_mult
        orch.risk.tp_r_multiples = list(tp_ladder)
        state["cache"].clear()
        return {"ok": True, "applied": {"atr_stop_mult": atr_mult, "tp_r_multiples": tp_ladder},
                "note": "Canlı uygulandı (bellek). Kalıcı için config.yaml'a yazın."}

    # ---- çoklu-parite tarama (+ Twitter sosyal) ----
    @app.get("/api/scan")
    def scan(symbols: str = Query(default=""), tf: str = Query(default=""),
             top: int = Query(default=25), onchain: bool = Query(default=False)):
        from ..scan import MultiPairScanner
        sc = state.get("scanner")
        if sc is None:
            sc = MultiPairScanner(_orch())
            state["scanner"] = sc
        syms = [s.strip() for s in symbols.split(",") if s.strip()] or None
        return sc.scan(symbols=syms, tf=tf or None, top=top, include_onchain=onchain)

    # ---- #7 ağırlık öğrenme ----
    @app.post("/api/learn")
    @app.get("/api/learn")
    def learn():
        from ..learn.weight_optimizer import optimize_weights
        return optimize_weights(_orch())

    # ---- kimlik bilgileri (credentials) ----
    @app.get("/api/credentials")
    def get_credentials():
        cfg = state["cfg"]
        fields = []
        for f in CRED_SCHEMA:
            val = cfg.secret(f["key"])
            fields.append({**f, "set": bool(val), "masked": _mask(val)})
        return {"fields": fields}

    @app.post("/api/credentials")
    def set_credentials(body: Dict[str, str] = Body(...)):
        cfg = state["cfg"]
        updates = {k: v for k, v in body.items() if k in _VALID_KEYS}
        if not updates:
            return {"saved": False, "reason": "geçerli alan yok"}
        cfg.save_secrets(updates)

        # ilgili bileşenleri yeni anahtarlarla yeniden başlat
        orch = state["orch"]
        try:
            from ..data import ExchangeManager
            from ..onchain import WhaleFlowEngine
            from ..sentiment import TwitterIntelligence
            from ..notify import Notifier
            orch.data = ExchangeManager(cfg)
            orch.whale = WhaleFlowEngine(cfg, orch.data)
            orch.twitter = TwitterIntelligence(cfg)
            orch.notifier = Notifier(cfg)
            state["streamer"].em = orch.data
            state["cache"].clear()
            state.pop("scanner", None)
        except Exception as e:
            return {"saved": True, "applied": False, "error": str(e)}

        return {"saved": True, "applied": True, "updated": list(updates.keys()),
                "env": orch.describe_environment()}

    # ---- sistem / günlük ----
    @app.get("/api/journal")
    def journal():
        orch = _orch()
        return {
            "summary": orch.journal.summary(),
            "weights": orch.decision.base_weights,
            "execution": orch.execution.status(),
            "env": orch.describe_environment(),
        }

    return app
