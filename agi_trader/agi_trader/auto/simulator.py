"""
1.000 $ SANAL SİMÜLATÖR — sistemin kendi komite koşucusu.

Kullanıcı hesabından bağımsız (sahip: SYSTEM_UID=0), sunucu açılınca kendiliğinden
kurulur ve 7/24 döner. Gerçek anlık veriyle (anahtarsız public OHLCV), en düşük
komisyonlu borsada, maker-öncelikli emirle, 12 rollü komiteyle işlem açar; her
kapanışta ders çıkarır. Salt-okunur uçları herkese açıktır (/api/simulator*):
sanal olduğu ve anahtar taşımadığı için gizlenecek bir şey yoktur — panelin
"açık kayıt" felsefesiyle aynı.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, Query

from ..strategies import committee as CM
from ..strategies import fees as FE
from ..strategies import video_scalp as VS
from . import live_runner as LR

SYSTEM_UID = 0
SIM_CAPITAL = 1000.0
SIM_LABEL = "1.000 $ sanal simülatör"

# AĞIR katman: public_api.ALLOWED_SYMBOLS (orkestratör + formasyon + gösterge + mover)
HEAVY_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "AVAX/USDT",
                 "LINK/USDT", "SUI/USDT", "NEAR/USDT", "PEPE/USDT", "AAVE/USDT",
                 "UNI/USDT", "LTC/USDT", "DOT/USDT", "FIL/USDT", "BCH/USDT"]
# HAFİF katman (2026-09-02 ölçüldü: 4 borsada spot, MEXC spread ≤ 12 bps, ±%2 derinlik
# ≥ 30 k$, 1 dk σ ≥ %0,08): rejim + seviye + ATR bağlamı sunucu içinde ucuza üretilir.
LIGHT_SYMBOLS = ["ADA/USDT", "XRP/USDT", "XLM/USDT", "TRUMP/USDT", "PENDLE/USDT",
                 "SHIB/USDT", "OP/USDT", "RENDER/USDT", "ARKM/USDT", "ENA/USDT",
                 "WLD/USDT", "TIA/USDT", "ARB/USDT", "GALA/USDT", "ONDO/USDT",
                 "ICP/USDT", "BONK/USDT", "ATOM/USDT", "JUP/USDT", "WIF/USDT",
                 "EIGEN/USDT", "SEI/USDT", "LDO/USDT", "VIRTUAL/USDT", "STRK/USDT"]


def extra_symbols(limit: int = 10, max_age_days: float = 7.0) -> List[str]:
    """scripts/cm_pair_scout.py'nin ölçtüğü ek pariteler (runs/live/universe_extra.json); yoksa boş."""
    import json as _json, time as _t, os as _os
    if _os.environ.get("CRYPTOMIND_EXTRA_SYMBOLS", "1") != "1":
        return []
    p = Path(__file__).resolve().parents[2] / "runs" / "live" / "universe_extra.json"
    try:
        d = _json.loads(p.read_text(encoding="utf-8"))
        if _t.time() - float(d.get("ts") or 0) > max_age_days * 86400:
            return []
        core = set(HEAVY_SYMBOLS + LIGHT_SYMBOLS)
        return [s for s in (d.get("symbols") or []) if s not in core][:limit]
    except Exception:
        return []


# SERMAYE TAHSİSİ (2026-09-04 ölçümü): scalping katmanı 105 işlemde −%0,62 (komisyon brütün 3 katı),
# trend katmanı 48 günde +%4,29 / Sharpe 2,68 / DD %2,0. Risk sermayesi ölçülmüş kazanan katmana
# aittir; scalping KAPATILMAZ (kapatılan katman bir daha ölçülemez) ama ÖLÇÜM BÜTÇESİYLE yaşar.
# Limitler `capital_allocator.scalp_risk_budget` ile paydan TÜRETİLİR — pay %2 iken 200 $'lık emir
# açmak payı anlamsız kılardı.
SCALP_BASELINE = {"max_order_usdt": 200.0, "risk_per_trade_pct": 1.0, "max_open": 5,
                  "max_exposure_pct": 75.0, "max_trades_per_day": 200}


def scalp_limits(weight_pct: Optional[float] = None) -> Dict:
    """Scalping payından koşucu limitleri. Pay verilmezse ölçüm bütçesi (CA.MEASURE_BUDGET_PCT)."""
    from . import capital_allocator as CA
    import os as _os
    if weight_pct is None:
        try:
            weight_pct = float(_os.environ.get("CRYPTOMIND_SCALP_WEIGHT_PCT", CA.MEASURE_BUDGET_PCT))
        except (TypeError, ValueError):
            weight_pct = CA.MEASURE_BUDGET_PCT
    return CA.scalp_risk_budget(float(weight_pct), SIM_CAPITAL, SCALP_BASELINE)


def default_config(exchange_id: str) -> LR.RunnerConfig:
    lim = scalp_limits()
    return LR.RunnerConfig.from_dict({
        "exchange_id": exchange_id, "mode": "paper", "market_type": "spot",
        "strategy": "committee", "symbols_mode": "auto",
        # 15 parite — public_api.ALLOWED_SYMBOLS ile aynı evren (yavaş bağlamı olanlar);
        # otomatik modda mover sırasına göre her 15 dk yeniden sıralanır.
        "symbols": HEAVY_SYMBOLS + LIGHT_SYMBOLS + extra_symbols() + extra_symbols() + extra_symbols(),
        "capital_usdt": SIM_CAPITAL,
        "max_order_usdt": lim["max_order_usdt"], "max_open": lim["max_open"],
        "risk_per_trade_pct": lim["risk_per_trade_pct"], "max_exposure_pct": lim["max_exposure_pct"],
        "daily_loss_limit_pct": 0.05, "max_drawdown_pct": 0.15,
        "max_trades_per_day": lim["max_trades_per_day"], "halt_action": "flatten",
        # kanıt boyutu emir tavanını AŞAMAZ: ölçüm bütçesinde emir tavanı 10 $ iken 25 $'lık
        # "kanıt tavanı" bağlamaz ve kapı sessizce devre dışı kalırdı
        "params": CM.CommitteeParams(loop_sec=30, min_hold_sec=900, max_hold_sec=3600,
                                     rr=1.6, giveback=0.5, min_gross_to_cost=2.0,
                                     probe_notional_usdt=min(25.0, lim["max_order_usdt"])).to_dict(),
        "chain": {}, "require_paper_proof": True, "paper_proof_trades": 20,
        "label": SIM_LABEL, "top_k": 10,
    })


def pick_venue(registry: LR.RunnerRegistry, candidates: Optional[List[str]] = None) -> Dict:
    """En ucuz borsa; canlı veri çekilebiliyor mu diye yoklanır (1 dk BTC/USDT)."""
    from ..execution.broker import Broker

    def probe(ex: str) -> bool:
        b = Broker(ex, "paper", client_factory=registry.client_factory, paper_capital=1.0)
        df = b.fetch_ohlcv("BTC/USDT", "1m", limit=5)
        return len(df) >= 3
    return FE.cheapest_venue(candidates, p_maker=0.5, probe=probe)


def ensure_simulator(registry: LR.RunnerRegistry, venue: Optional[str] = None,
                     start: bool = True) -> Dict:
    """Varsa (restore_all geri yüklediyse) dokunma; yoksa kur ve başlat."""
    existing = [r for r in registry.all_for(SYSTEM_UID) if r.cfg.strategy == "committee"]
    if existing:
        r = existing[0]
        # kod-tanımlı evren/limitler diskteki eski konfigi ezer (pozisyon/işlem/ders KORUNUR)
        # sistem simülatörü kod-tanımlıdır. `risk_per_trade_pct` LİSTEDE YOKTU: boyut hesabının
        # (`_size`: sermaye × risk% / stop%) ana çarpanı olduğu için sermaye tahsisi uygulanmıyordu.
        changed = r.sync_config(default_config(r.cfg.exchange_id),
                                fields=("symbols", "max_open", "max_order_usdt", "max_exposure_pct", "top_k",
                                        "max_trades_per_day", "risk_per_trade_pct", "exit", "params"))
        if start and not r.running:
            r.start()
        return {"created": False, "exchange": r.cfg.exchange_id, "running": r.running,
                "cycle": r.cycle, "trades": len(r.trades), "synced": changed}
    if venue is None:
        v = pick_venue(registry)
        venue = v["exchange_id"]
        note = f"en ucuz venue {venue} (maker {v['maker_bps']} / taker {v['taker_bps']} bps)" + \
            (f"; atlandı: {v['skipped']}" if v.get("skipped") else "")
    else:
        note = f"venue elle: {venue}"
    cfg = default_config(venue)
    restore = None
    p = LR.state_path(registry.output_dir, SYSTEM_UID, venue)
    if p.exists():
        try:
            restore = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            restore = None
    r = registry.create(SYSTEM_UID, cfg, restore=restore)
    r._log("system", "🧪 " + note)
    if start:
        r.start()
    return {"created": True, "exchange": venue, "running": r.running, "note": note}


def get_simulator(registry: LR.RunnerRegistry) -> Optional[LR.LiveRunner]:
    rs = [r for r in registry.all_for(SYSTEM_UID) if r.cfg.strategy == "committee"]
    return rs[0] if rs else None


def create_public_router(registry: LR.RunnerRegistry) -> APIRouter:
    """GET-only, anahtar içermez, kullanıcıya özgü hiçbir şey taşımaz."""
    r = APIRouter()

    def _not_ready():
        return {"configured": False, "note": "simülatör henüz kurulmadı (kasa/venue bekleniyor)",
                "venues": FE.ranked_venues()}

    @r.get("/api/simulator")
    def simulator():
        s = get_simulator(registry)
        if s is None:
            return _not_ready()
        st = s.full_state()
        st.pop("broker", None)
        st["public"] = True
        st["strategy_card"] = CM.describe()
        st["venues"] = FE.ranked_venues()
        return st

    @r.get("/api/simulator/journal")
    def journal(limit: int = Query(default=60, ge=1, le=500)):
        s = get_simulator(registry)
        if s is None:
            return _not_ready()
        les = s.lessons.status()
        md = s.lessons.md
        text = ""
        try:
            if md and Path(md).exists():
                text = Path(md).read_text(encoding="utf-8")[-20000:]
        except Exception:
            text = ""
        return {"configured": True, "trades": s.trades[-limit:][::-1],
                "lessons": les["lessons"], "roles": les["roles"], "veto_stats": les["veto_stats"],
                "feature_stats": les["feature_stats"], "shadows_recent": les["shadows_recent"],
                "shadows_open": les["shadows_open"], "overrides": les["overrides"],
                "p_win": les["p_win"], "n_trades": les["n_trades"], "markdown": text}

    @r.get("/api/simulator/trades")
    def trades_page(page: int = Query(default=1, ge=1), per_page: int = Query(default=25, ge=1, le=200)):
        """SAYFALI işlem listesi — tamamı görünür (bellekteki tam defter, en yeni sayfa 1). `seq` = kronolojik
        1-tabanlı sıra: sayfa 3 (25'lik) → #(N−50)…#(N−74). Panel hangi sayfaya tıklarsa o sıradaki işlemler gelir."""
        s = get_simulator(registry)
        if s is None:
            return _not_ready()
        with s._lock:
            allt = list(s.trades)
        total = len(allt)
        pages = max(1, (total + per_page - 1) // per_page)
        page = min(page, pages)
        # kronolojik sıra numarası: en eski = 1
        end = total - (page - 1) * per_page                    # bu sayfanın en yeni işleminin seq'i
        start = max(0, end - per_page)
        rows = []
        for i in range(end - 1, start - 1, -1):
            t = dict(allt[i]); t["seq"] = i + 1
            t.pop("prev_hash", None)
            rows.append(t)
        return {"configured": True, "total": total, "page": page, "per_page": per_page, "pages": pages, "order": "desc",
                "from_seq": (end if total else 0), "to_seq": (start + 1 if total else 0), "trades": rows,
                "note": "seq = kronolojik sıra (ilk işlem #1); sayfa 1 = en yeni"}

    @r.get("/api/simulator/equity")
    def equity(max_points: int = Query(default=2000, ge=50, le=20000)):
        """TAM özsermaye geçmişi (başlangıç anından bugüne): son 6 sa 30 sn çözünürlük, öncesi 5 dk kova;
        eski sürümde kaybolan başlangıç kapanan işlem defterinden (gerçekleşmiş özsermaye) tamamlanır (`src: ledger`)."""
        s = get_simulator(registry)
        if s is None:
            return _not_ready()
        return {"configured": True, **s.equity_series(max_points=max_points)}

    @r.get("/api/simulator/feed")
    def feed():
        """Hafif canlı akış — üst-sağ İŞLEM GÜNLÜĞÜ paneli için.
        Durum etiketi + olaylar + işlemler + tetikleyiciye en yakın pariteler +
        kısa vadeli piyasa OLASILIKLARI (mover modeli, 4h konsensüs, rejim).
        Olasılıklar ölçülmüş modellerden okunur; burada hesaplanmaz, uydurulmaz."""
        import time as _t
        s = get_simulator(registry)
        if s is None:
            return {**_not_ready(), "status": {"state": "NOT_READY", "label": "SİMÜLATÖR KURULUYOR"}}
        with s._lock:
            pending = [{"symbol": k, "side": o["order"]["side"], "price": o["order"]["price"],
                        "notional": o["notional"], "bars": o["bars"]} for k, o in s.pending.items()]
            positions = [p.to_dict() for p in s.positions.values()]
            trades = s.trades[-5:][::-1]
            events = list(s.events)[:20]
            decisions = list(s.last_decisions.values())
            p = s.params
            g = s.guard.state
            cycle, last_ts, running = s.cycle, s.last_cycle_ts, s.running
            symbols = list(s.cfg.symbols)
            st = s.stats()
        if positions:
            state, label = "OPEN", f"{len(positions)} AÇIK POZİSYON"
        elif pending:
            state, label = "PENDING", f"MAKER EMRİ BEKLİYOR · {pending[0]['symbol']}"
        elif trades or st["closed_trades"]:
            state, label = "ACTIVE", "SONRAKİ İŞLEM BEKLENİYOR"
        else:
            state, label = "WAITING_FIRST", "İLK İŞLEM BEKLENİYOR"
        if g.halted:
            state, label = "HALT", "HALT — " + "; ".join(g.reasons)[:80]
        # tetikleyiciye yakınlık (komite fast özeti) — eksik şart listesiyle
        nearest = []
        for d in decisions:
            f = d.get("fast") or {}
            if not f:
                continue
            tpl = d.get("template") or "mean_reversion"
            miss = []
            if tpl == "mean_reversion":
                z, rsi = f.get("z"), f.get("rsi")
                if z is None or z > -p.dip_z:
                    miss.append(f"z {z} > -{p.dip_z}")
                if rsi is None or rsi > p.rsi_max:
                    miss.append(f"RSI {rsi} > {p.rsi_max:.0f}")
                if not f.get("bar_up"):
                    miss.append("yeşil bar yok")
                score = (z if z is not None else 9.0)
            else:
                band = p.pullback_band_atr * (f.get("atr_pct") or 0.3)
                dist = f.get("dist_ema_pct")
                if not f.get("trend_up"):
                    miss.append("trend ↓")
                if dist is None or not (-band <= dist <= band * 0.5):
                    miss.append(f"EMA uzaklık %{dist} (bant ±%{band:.2f})")
                if not f.get("bar_up"):
                    miss.append("yeşil bar yok")
                if (f.get("rsi") or 0) >= 65:
                    miss.append("RSI ≥ 65")
                score = abs(dist) if dist is not None else 9.0
            nearest.append({"symbol": d.get("symbol"), "template": tpl, "z": f.get("z"), "rsi": f.get("rsi"),
                            "dist_ema_pct": f.get("dist_ema_pct"), "trend_up": f.get("trend_up"),
                            "bar_up": f.get("bar_up"), "missing": miss, "ready": not miss,
                            "tier": d.get("tier"), "news": d.get("news"), "freshness": d.get("freshness"), "cvd": d.get("cvd"),
                            "result": str(d.get("result", ""))[:120], "_k": score})
        nearest.sort(key=lambda x: (len(x["missing"]), x["_k"]))
        for x in nearest:
            x.pop("_k", None)
        # kısa vadeli piyasa olasılıkları — mover (bugün ≥%1 oynar mı), 4h konsensüs, rejim
        cons = {"LONG": 0, "SHORT": 0, "FLAT": 0}
        regs: Dict[str, int] = {}
        movers = []
        confs = []
        ups = []
        for sym in symbols:
            slow = registry.ctx.slow_for(sym) or {}
            sig = slow.get("signal") or {}
            d = str(sig.get("direction") or "FLAT").upper()
            cons[d if d in cons else "FLAT"] += 1
            if sig.get("confidence") is not None:
                confs.append(float(sig["confidence"]))
            pu = (sig.get("forecast") or {}).get("prob_up")
            if isinstance(pu, (int, float)):
                ups.append(float(pu))
            reg = ((slow.get("chart") or {}).get("regime") or {}).get("label") or next(
                (l.get("detail", {}).get("label") for l in sig.get("layer_breakdown") or []
                 if l.get("layer") == "_regime"), None)
            if reg:
                regs[reg] = regs.get(reg, 0) + 1
            mp = slow.get("mover_pick") or {}
            if mp.get("probability") is not None:
                movers.append({"symbol": sym, "probability": mp.get("probability"),
                               "base_rate": mp.get("base_rate"), "lift": mp.get("lift"),
                               "expected_move_pct": mp.get("expected_move_pct"),
                               "trusted": mp.get("model_trusted")})
        movers.sort(key=lambda m: -(m["probability"] or 0))
        outlook = {
            "consensus_4h": cons,
            "avg_confidence": round(sum(confs) / len(confs), 3) if confs else None,
            "avg_prob_up": round(sum(ups) / len(ups), 3) if ups else None,
            "regimes": regs,
            "movers": movers[:5],
            "p_win_committee": round(s.lessons.p_win(), 3),
            "note": ("Olasılıklar ölçülmüş modellerden: mover = bugün ≥%1 oynama olasılığı (yön değil), "
                     "4h konsensüs = 11 katmanlı motorun yönü, rejim = HMM. Garanti yok."),
        }
        mn = registry.ctx.market() or {}
        return {"configured": True, "ts": _t.time(),
                "risk_mode": s.risk, "cash_mode": s.cash_mode,
                "portfolio_mode": s.portfolio, "best_action": s.best_action(),
                "top_opportunities": s.top_opportunities(3), "resource": s.resource,
                "scan": s.scan[:8], "fee_info": s.fee_info, "venue_compare": s.venue_compare,
                "challenger": s.challenger.evaluate(),
                "missed": (lambda m: {"n_missed": m["n_missed"], "n_avoided": m["n_avoided"], "n_open": m["n_open"],
                                      "missed_net_pct_sum": m["missed_net_pct_sum"],
                                      "top_gate": (m["gates"][0]["gate_tr"] if m["gates"] else None),
                                      "last": (m["recent"][0] if m["recent"] else None)})(s.missed.report()),
                "news_market": {"level": mn.get("level", 0), "risk_off_score": mn.get("risk_off_score"),
                                "items": (mn.get("items") or [])[:4], "age_sec": mn.get("age_sec"),
                                "last_error": mn.get("last_error")},
                "tiers": {"heavy": len([x for x in symbols if x in HEAVY_SYMBOLS]),
                          "light": len([x for x in symbols if x not in HEAVY_SYMBOLS])},
                "status": {"state": state, "label": label, "cycle": cycle, "last_cycle_ts": last_ts,
                           "running": running, "halted": g.halted, "symbols": len(symbols),
                           "loop_sec": p.loop_sec, "closed_trades": st["closed_trades"],
                           "equity": st["equity"], "net_pnl": st["net_pnl"], "fees_paid": st["fees_paid"]},
                "positions": positions, "pending": pending, "trades": trades, "events": events,
                "nearest": nearest[:6], "outlook": outlook,
                "venue": {"exchange_id": s.venue.exchange_id, "maker_bps": s.venue.maker_bps,
                          "taker_bps": s.venue.taker_bps}}

    @r.get("/api/allocation")
    def allocation():
        """SERMAYE TAHSİSİ — hangi katmana ne kadar risk, ÖLÇÜLMÜŞ sonuca göre.

        Trend katmanı (günlük yeniden dengeleme) ile scalping katmanının (1 dk komite) gerçekleşen
        Sharpe'ları karşılaştırılır; pay ölçüme göre verilir. Eşik altı katman kapatılmaz, ölçüm
        bütçesiyle yaşar — kapatılan katman bir daha ölçülemez."""
        from . import capital_allocator as CA
        base = Path(registry.output_dir)
        if not base.is_absolute():
            base = Path(__file__).resolve().parents[2] / base
        ts = CA.load_trend_state(base / "trend_state.json")
        sim = get_simulator(registry)
        trades = list(sim.trades) if sim is not None else []
        cap = float(sim.cfg.capital_usdt) if sim is not None else SIM_CAPITAL
        rep = CA.report(ts, trades, cap)
        rep["configured"] = True
        rep["applied"] = {"scalp_runner_limits": (sim.cfg.to_dict() if sim is None else
                                                  {k: getattr(sim.cfg, k) for k in
                                                   ("max_order_usdt", "max_open", "risk_per_trade_pct",
                                                    "max_exposure_pct", "max_trades_per_day")}),
                          "baseline": SCALP_BASELINE,
                          "note": "koşucu limitleri paydan türetilir (scalp_limits); pay artarsa limitler büyür"}
        return rep

    @r.get("/api/video-sources")
    def video_sources():
        """Video kaynaklı kurulumların künyesi: hangi kanal, ne iddia etti, kanıtı neydi, ne ALINMADI,
        ölçtüğümüzde ne çıktı. İddialar taşınmaz — mekanik çekirdek taşınır."""
        from ..strategies import sleeves_video as SV
        card = SV.describe()
        s = get_simulator(registry)
        if s is not None:
            live = {}
            for t in s.trades:
                k = t.get("sleeve") or t.get("trigger")
                if k in SV.SOURCES:
                    a = live.setdefault(k, {"n": 0, "wins": 0, "net": 0.0})
                    a["n"] += 1; a["wins"] += int(bool(t.get("win"))); a["net"] = round(a["net"] + float(t.get("net_pnl") or 0.0), 4)
            states = s.allocator.sleeve_states()
            for row in card["sources"]:
                row["live"] = live.get(row["sleeve"], {"n": 0, "wins": 0, "net": 0.0})
                row["state"] = states.get(row["sleeve"], "UNPROVEN")
                row["evidence_live"] = s.allocator.evidence(row["sleeve"])
        return {"configured": s is not None, **card}

    @r.get("/api/simulator/missed")
    def missed(limit: int = Query(default=20, ge=1, le=100)):
        """Kaçırılan fırsatlar: neden yapılmadı, hangi özellik göz ardı edildi, hangi bilgi eksikti."""
        s = get_simulator(registry)
        if s is None:
            return _not_ready()
        rep = s.missed.report()
        rep["recent"] = rep["recent"][:limit]
        text = ""
        try:
            md = s.missed.md
            if md and Path(md).exists():
                text = Path(md).read_text(encoding="utf-8")[-16000:]
        except Exception:
            text = ""
        return {"configured": True, **rep, "markdown": text}

    @r.get("/api/simulator/universe")
    def universe():
        """Bütün paritelerin görünürlüğü: tarama puanı, tazelik, rejim, son karar, CVD, pozisyon, gölge."""
        s = get_simulator(registry)
        if s is None:
            return _not_ready()
        with s._lock:
            rows = s.universe()
        return {"configured": True, "n": len(rows), "rows": rows, "extra": extra_symbols(),
                "paused_sleeves": s.allocator.paused_sleeves(), "rotations_today": getattr(s, "_rotations_today", 0)}

    @r.get("/api/simulator/calibration")
    def calibration():
        """p_win kalibrasyonu (Brier, güvenilirlik kovaları) + Monte Carlo (drawdown / iflas olasılığı)."""
        from ..learn import calibration as CAL
        s = get_simulator(registry)
        if s is None:
            return _not_ready()
        return {"configured": True, "reliability": CAL.reliability_table(s.trades),
                "monte_carlo": CAL.monte_carlo(s.trades, capital=float(s.cfg.capital_usdt), daily_loss_limit_pct=float(s.cfg.daily_loss_limit_pct) * 100.0),
                "ledger": s.full_state().get("ledger"), "alerts": s.alerts.status()}

    @r.get("/api/research")
    def research():
        """Strateji araştırma fabrikası: 250'lik kütüphane, huni, gölge modüller, hasat kutusu."""
        from ..research import library as LIB
        from ..research.harvester import Harvester
        s = get_simulator(registry)
        inbox = Harvester(Path(registry.output_dir) / "research" / "inbox").inbox_summary()
        if s is None:
            return {"configured": False, "library": LIB.summary(), "inbox": inbox}
        return {"configured": True, **s.lab.status(s.lifecycle.status()), "inbox": inbox,
                "allocator": s.allocator.status(), "thompson": s.allocator.thompson_ranking(LIB.sleeves_implemented())}

    @r.get("/api/simulator/roles")
    def roles():
        s = get_simulator(registry)
        card = CM.describe()
        if s is not None:
            rel = {x["role"]: x for x in s.lessons.status()["roles"]}
            for role in card["roles"]:
                role.update(rel.get(role["id"], {}))
                role["effective_weight"] = round(CM._weight(role["id"], s.lessons.learned()), 3)
        return card

    return r
