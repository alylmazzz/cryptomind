"""
LiveRunner — kullanıcı × borsa başına otopilot döngüsü.

İki strateji:
  video_dip_scalp  videodaki kurulum + CryptoMind karar zinciri (decision_chain)
  committee        çok-sleeve komite (strategies/committee + sleeves_fast) + ders motoru
                   + maker-öncelikli limit + çıkış motoru + portföy modu + challenger

Karar zinciri (committee):
  RAW DATA (MarketStateStore, tek fetch) → tazelik doğrulama → TIER-A ucuz tarayıcı (ilgi puanı)
  → TOP-K aday → rejim → izinli sleeve'ler → roller → EV YARIŞMASI → giriş optimizasyonu
  (bölge/optimal/max chase) → ücret/venue → EV → portföy riski → politika kapıları → emir
  → pozisyon yönetimi (kısmi/tepe/yarı-tepe/trailing/edge-decay/zaman) → mutabakat → ders
  → challenger.

Kaynak bütçesi: RSS pm2 tavanının %75/%85'inde YELLOW/RED; RED'de yeni ağır değerlendirme
durur, çıkış/risk katmanı DURMAZ. Her failure → NO TRADE (fail-safe).
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional

import numpy as np
import pandas as pd

from ..data.market_state import MarketStateStore, RateLimitCoordinator
from ..execution.broker import Broker, BrokerError, make_client_order_id
from ..execution import fee_adapter as FA
from ..execution import tca as TCA
from ..execution import venue_router as VR
from ..learn.allocator import MetaAllocator
from ..learn.challenger import Challenger
from ..learn import evidence as EV
from ..learn.lessons import LessonEngine
from ..learn.missed import MissedEngine
from ..notify.alerts import AlertBus
from ..research.lab import ResearchLab
from ..sentiment.news_impact import HaberEtkiMotoru
import hashlib
from ..risk.live_guard import LiveGuard, audit, live_enabled, reconcile
from ..strategies import committee as CM
from ..strategies import exit_engine as XE
from ..strategies import fees as FE
from ..strategies import portfolio_mode as PM
from ..strategies import reentry as RE
from ..strategies import sleeves_fast as SF
from ..strategies import video_scalp as VS
from ..strategies.lifecycle import Lifecycle
from ..strategies.light_context import LightContextCache
from . import decision_chain as DC

CONFIRM_PHRASE = "CANLI İŞLEMİ ONAYLIYORUM"
DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
MAX_SYMBOLS = 64
STRATEGIES = ("video_dip_scalp", "committee")
SYMBOL_REFRESH_CYCLES = 30            # otomatik parite listesi ~15 dk'da bir
MEM_CAP_MB = float(os.environ.get("CRYPTOMIND_MEM_CAP_MB", "1400"))
# Bellek adli tip: RSS iki kez (Tur 13 ve 2026-09-05) tavana dayanip RED durumunda
# `top_k = 0` yaparak GIRISLERI TAMAMEN DURDURDU; iki seferde de cozum "tavani yukselt"
# oldu. Tavan yukseltmek olcum degildir. Asagidaki iki arac, buyumeyi TAHMIN yerine
# OLCU haline getirir:
#   _malloc_trim() : Python nesneyi biraktiktan sonra glibc'in elinde TUTTUGU serbest
#                    bellegi isletim sistemine geri verir (gc.collect() bunu yapmaz).
#   CRYPTOMIND_TRACEMALLOC=1 : en cok bellek ayiran 10 kod satirini yayimlar.
_TRACEMALLOC_ON = os.environ.get("CRYPTOMIND_TRACEMALLOC", "").strip() in ("1", "true", "TRUE")


def _malloc_trim() -> bool:
    """glibc'in arena'larindaki serbest bellegi OS'a iade et. Yalniz Linux/glibc'te
    anlamli; baska yerde sessizce False doner (hata degil, yok)."""
    try:
        import ctypes
        return bool(ctypes.CDLL("libc.so.6").malloc_trim(0))
    except Exception:
        return False


def _tracemalloc_top(n: int = 10):
    if not _TRACEMALLOC_ON:
        return None
    try:
        import tracemalloc
        if not tracemalloc.is_tracing():
            tracemalloc.start(1)
            return {"note": "izleme yeni basladi - ilk anlik goruntu bir sonraki dongude"}
        snap = tracemalloc.take_snapshot().statistics("lineno")[:n]
        return [{"yer": str(st.traceback[0]), "mb": round(st.size / 1048576, 2), "blok": st.count}
                for st in snap]
    except Exception as e:
        return {"hata": f"{type(e).__name__}: {e}"}
COMPARE_VENUES = ["mexc", "binance", "bybit", "okx"]
VENUE_COMPARE = os.environ.get("CRYPTOMIND_VENUE_COMPARE", "0") == "1"
CVD_ENABLED = os.environ.get("CRYPTOMIND_CVD", "1") == "1"       # Top-K adayları için son işlemler (taker akışı)
TRADES_KEEP = 800                       # SICAK liste: panel sayfalama + bellek içi istatistik.
                                        # 3000 idi. Durum dosyası HER DÖNGÜDE (~25 sn) baştan yazılır;
                                        # 3000 × 1.024 B ≈ 3 MB/döngü ≈ 10 GB/gün gereksiz yazma demekti.
                                        # KALICI ve EKSİKSİZ kayıt artık `evidence.jsonl` (salt-ekleme,
                                        # ~130 B/işlem) — hiçbir kanıt kaybolmuyor, yalnız sıcak liste kısaldı.
EQUITY_HISTORY_MAX = 40000              # tam özsermaye geçmişi nokta tavanı (5 dk kova ≈ 140 gün)
EQUITY_FULL_RES_SEC = 6 * 3600          # bu süre tam çözünürlük (30 sn), öncesi kovaya indirgenir
EQUITY_BUCKET_SEC = 300


def _rec_hash(rec: Dict, prev: str) -> str:
    body = {k: v for k, v in rec.items() if k not in ("hash", "prev_hash", "exit_detail")}
    return hashlib.sha256((prev + json.dumps(body, sort_keys=True, default=str)).encode("utf-8")).hexdigest()


def verify_ledger(trades: List[Dict]) -> Dict:
    """Kapanan işlem defteri sha256 zinciri: her kayıt öncekinin hash'ini taşır (kurcalamaya karşı kanıt)."""
    prev = ""; chained = 0; first_break = None
    for i, t in enumerate(trades):
        if "hash" not in t:
            continue
        if t.get("prev_hash", "") != prev or _rec_hash(t, prev) != t["hash"]:
            first_break = first_break if first_break is not None else i
        prev = t["hash"]; chained += 1
    return {"n": len(trades), "chained": chained, "ok": first_break is None, "first_break": first_break}


def _rss_mb() -> Optional[float]:
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 1048576.0
    except Exception:
        return None


@dataclass
class RunnerConfig:
    exchange_id: str = "binance"
    mode: str = "paper"                       # paper | testnet | live
    market_type: str = "spot"
    strategy: str = "video_dip_scalp"
    symbols: List[str] = field(default_factory=lambda: list(DEFAULT_SYMBOLS))
    symbols_mode: str = "fixed"               # fixed | auto (CryptoMind adaylarından)
    capital_usdt: float = VS.VIDEO_CAPITAL_USDT
    max_order_usdt: float = 100.0
    max_open: int = 3
    risk_per_trade_pct: float = 1.0
    max_exposure_pct: float = 60.0
    daily_loss_limit_pct: float = VS.DEFAULT_DAILY_LOSS_PCT
    max_drawdown_pct: float = 0.15
    max_trades_per_day: int = 40
    halt_action: str = "flatten"              # flatten | hold
    params: Dict = field(default_factory=lambda: VS.ScalpParams().to_dict())
    chain: Dict = field(default_factory=lambda: DC.ChainConfig().to_dict())
    require_paper_proof: bool = True
    paper_proof_trades: int = 20
    label: str = ""
    top_k: int = 10                           # Tier-B derin analiz aday sayısı (kaynak durumuna göre düşer)
    max_open_per_sleeve: int = 2              # strateji çeşitlendirmesi: aynı sleeve'den en çok 2 açık pozisyon
    exit: Dict = field(default_factory=lambda: XE.ExitParams().__dict__.copy())
    reentry: Dict = field(default_factory=lambda: RE.ReentryParams().to_dict())

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict]) -> "RunnerConfig":
        c = cls()
        for k, v in (d or {}).items():
            if not hasattr(c, k) or v is None:
                continue
            cur = getattr(c, k)
            try:
                if isinstance(cur, bool):
                    setattr(c, k, bool(v))
                elif isinstance(cur, list):
                    setattr(c, k, [str(x).strip().upper() for x in v if str(x).strip()])
                elif isinstance(cur, dict):
                    setattr(c, k, dict(v))
                else:
                    setattr(c, k, type(cur)(v))
            except (TypeError, ValueError):
                pass
        return c.validated()

    def validated(self) -> "RunnerConfig":
        if self.mode not in ("paper", "testnet", "live"):
            self.mode = "paper"
        if self.market_type not in ("spot", "future"):
            self.market_type = "spot"
        if self.strategy not in STRATEGIES:
            self.strategy = "video_dip_scalp"
        if self.symbols_mode not in ("fixed", "auto"):
            self.symbols_mode = "fixed"
        syms = []
        for s in self.symbols:
            if "/" not in s:
                s = s.replace("USDT", "") + "/USDT"
            if s not in syms:
                syms.append(s)
        self.symbols = (syms or list(DEFAULT_SYMBOLS))[:MAX_SYMBOLS]
        self.capital_usdt = float(min(10_000_000.0, max(50.0, self.capital_usdt)))
        self.max_order_usdt = float(min(self.capital_usdt, max(10.0, self.max_order_usdt)))
        self.max_open = int(min(MAX_SYMBOLS, max(1, self.max_open)))
        self.risk_per_trade_pct = float(min(5.0, max(0.1, self.risk_per_trade_pct)))
        self.max_exposure_pct = float(min(100.0, max(5.0, self.max_exposure_pct)))
        self.daily_loss_limit_pct = float(min(0.5, max(0.005, self.daily_loss_limit_pct)))
        self.max_drawdown_pct = float(min(0.6, max(0.02, self.max_drawdown_pct)))
        self.max_trades_per_day = int(min(500, max(1, self.max_trades_per_day)))
        if self.halt_action not in ("flatten", "hold"):
            self.halt_action = "flatten"
        if self.strategy == "committee":
            self.params = CM.CommitteeParams.from_dict(self.params).to_dict()
        else:
            self.params = VS.ScalpParams.from_dict(self.params).to_dict()
        self.chain = DC.ChainConfig.from_dict(self.chain).to_dict()
        self.paper_proof_trades = int(min(500, max(1, self.paper_proof_trades)))
        self.top_k = int(min(40, max(3, self.top_k)))
        self.max_open_per_sleeve = int(min(10, max(1, self.max_open_per_sleeve)))
        xp = XE.ExitParams()
        for k, v in (self.exit or {}).items():
            if hasattr(xp, k) and v is not None:
                try:
                    setattr(xp, k, type(getattr(xp, k))(v))
                except (TypeError, ValueError):
                    pass
        self.exit = xp.validated().__dict__.copy()
        self.reentry = RE.ReentryParams.from_dict(self.reentry).to_dict()
        if self.market_type == "spot":
            self.params["allow_short"] = False
        return self


@dataclass
class Position:
    symbol: str
    direction: str
    entry: float
    stop: float
    target: float
    stop_pct: float
    target_pct: float
    amount: float
    notional: float
    opened_ts: float
    entry_fee: float = 0.0
    peak_pnl_pct: float = 0.0
    last_price: float = 0.0
    client_id: str = ""
    order_id: str = ""
    decision: Dict = field(default_factory=dict)
    mode: str = "paper"
    order_type: str = "taker"
    trigger: str = ""
    template: str = ""
    partial_tp: float = 0.0
    partial_fraction: float = 0.0
    partial_done: bool = False
    realized: float = 0.0          # kısmi kapanıştan gerçekleşen brüt
    fees_partial: float = 0.0
    amount_initial: float = 0.0
    # çıkış motoru izi
    exit_mode: str = XE.PARTIAL_AND_RUN
    highest_high: float = 0.0
    lowest_low: float = 0.0
    peak_net_pct: float = 0.0
    armed: bool = False
    trail_stop: Optional[float] = None
    hard_stop: float = 0.0
    atr_pct: float = 0.3
    cost_pct_roundtrip: float = 0.2
    time_stop_sec: int = 3600
    sleeve: str = ""
    current_ev_pct: Optional[float] = None
    cont_prob: Optional[float] = None
    be_locked: bool = False
    remaining_ev_pct: Optional[float] = None
    # ─── kâr merdiveni v2 ───────────────────────────────────────────────────
    ladder: List[Dict] = field(default_factory=list)   # [{price, frac, source, net_pct, hit}]
    levels_hit: int = 0
    lock_price: Optional[float] = None                 # kâr kilidinin ürettiği stop
    locked_net_pct: float = 0.0                        # kilitlenen NET kâr (%) — panelde gösterilir
    realized_net_pct: float = 0.0                      # merdivenden gerçekleşen NET (% · başlangıç notional'a göre)
    reentry_count: int = 0                             # aynı harekete kaçıncı giriş
    parent_exit_ts: float = 0.0                        # yeniden giriş ise önceki çıkışın zamanı

    def sign(self) -> float:
        return 1.0 if self.direction == "LONG" else -1.0

    def unrealized(self, price: Optional[float] = None) -> float:
        p = price if price else self.last_price
        if not p:
            return 0.0
        return self.amount * (p - self.entry) * self.sign()

    def net_pct_now(self) -> float:
        if not self.last_price:
            return 0.0
        return (self.last_price / self.entry - 1.0) * 100.0 * self.sign() - self.cost_pct_roundtrip

    def track(self) -> XE.PositionTrack:
        t = XE.PositionTrack(self.direction, self.entry, self.hard_stop or self.stop,
                             self.target if self.exit_mode != XE.DYNAMIC_PEAK else None,
                             self.opened_ts, self.exit_mode, self.stop_pct, self.cost_pct_roundtrip,
                             self.atr_pct, self.highest_high, self.lowest_low, self.peak_pnl_pct,
                             self.peak_net_pct, self.armed, self.partial_done, self.trail_stop, self.be_locked)
        # merdiven AYNI nesne olarak verilir: decide_exit basamağı `hit` işaretler,
        # absorb() geri yazar. Kopyalasaydık basamak her döngüde yeniden ateşlenirdi.
        t.ladder = self.ladder
        t.levels_hit = int(self.levels_hit)
        t.lock_price = self.lock_price
        return t

    def absorb(self, t: XE.PositionTrack) -> None:
        self.highest_high, self.lowest_low = t.highest_high, t.lowest_low
        self.peak_pnl_pct, self.peak_net_pct = t.peak_gross_pct, t.peak_net_pct
        self.armed, self.trail_stop = t.armed, t.trail_stop
        self.ladder = t.ladder
        self.levels_hit = int(t.levels_hit)
        if t.lock_price is not None:
            self.lock_price = float(t.lock_price)
        if t.partial_done and not self.partial_done:
            self.partial_done = True
        if t.be_locked and not self.be_locked:
            self.be_locked = True
        if (t.hard_stop - self.hard_stop) * self.sign() > 0:      # stop yalnız İYİ tarafa hareket eder
            self.hard_stop = t.hard_stop
            self.stop = t.hard_stop
        if self.be_locked:
            # KİLİTLİ NET, stop her yükseldiğinde tazelenmeli. (Yalnız `_partial_close`
            # içinde yazılsaydı, kilidi sonradan yukarı yürüten `decide_exit` sonrasında
            # panelde BAYAT bir değer kalırdı — kullanıcı korunan kârı yanlış görürdü.)
            self.locked_net_pct = round(t.net_pct(self.hard_stop), 4)

    def to_dict(self, xp: Optional[XE.ExitParams] = None) -> Dict:
        d = asdict(self)
        d["decision"] = {k: v for k, v in (self.decision or {}).items()
                         if k in ("result", "ticket", "score", "confidence", "trigger", "template",
                                  "vetoes", "size_mult", "allowed", "chain", "entry", "exit_mode", "competition")}
        d["unrealized"] = round(self.unrealized(), 4)
        d["pnl_pct"] = round(((self.last_price / self.entry - 1.0) * 100.0 * self.sign())
                             if self.last_price else 0.0, 3)
        d["net_pct"] = round(d["pnl_pct"] - self.cost_pct_roundtrip, 3)
        d["age_min"] = round((time.time() - self.opened_ts) / 60.0, 1)
        d["hold_bucket"] = VS.hold_bucket(time.time() - self.opened_ts)
        t = self.track()
        xp = xp or XE.ExitParams()
        d["giveback_level"] = t.giveback_level(xp)
        d["chandelier"] = XE.chandelier_stop(t, xp) if self.armed else None
        d["peak_capture_now"] = XE.peak_capture_ratio(d["net_pct"], self.peak_net_pct)
        d["remaining_ev_pct"] = self.remaining_ev_pct
        # kâr merdiveni — panelde "hangi tepeye kadar geldik, ne kilitlendi"
        d["ladder"] = t.ladder_state()
        d["levels_hit"] = self.levels_hit
        d["lock_price"] = self.lock_price
        d["locked_net_pct"] = round(self.locked_net_pct, 4)
        d["retain_eff"] = round(t.retain_eff(xp), 3)
        d["realized_net_pct"] = round(self.realized_net_pct, 4)
        # TOPLAM net = gerçekleşen (merdiven) + kâğıt üstü kalan
        d["total_net_pct"] = round(self.realized_net_pct
                                   + d["net_pct"] * (self.amount / max(1e-12, self.amount_initial or self.amount)), 4)
        return d


class Context:
    """CryptoMind sistemlerine erişim — hepsi isteğe bağlı çağrılabilirler."""

    def __init__(self, cm_signal=None, qual_cell=None, system_health=None, regime=None,
                 slow_ctx=None, candidate_symbols=None, news_for=None, market_news=None):
        self.cm_signal = cm_signal
        self.qual_cell = qual_cell
        self.system_health = system_health
        self.regime = regime
        self.slow_ctx = slow_ctx
        self.candidate_symbols = candidate_symbols
        self.news_for = news_for
        self.market_news = market_news

    def _safe(self, fn, *a):
        if fn is None:
            return None
        try:
            return fn(*a)
        except Exception:
            return None

    def signal_for(self, symbol: str) -> Optional[Dict]:
        return self._safe(self.cm_signal, symbol)

    def cell_for(self, symbol: str, horizon: str, direction: str) -> Optional[Dict]:
        return self._safe(self.qual_cell, symbol, horizon, direction)

    def health(self) -> Optional[Dict]:
        return self._safe(self.system_health)

    def regime_for(self, df: pd.DataFrame) -> Optional[Dict]:
        return self._safe(self.regime, df)

    def slow_for(self, symbol: str) -> Optional[Dict]:
        return self._safe(self.slow_ctx, symbol)

    def candidates(self) -> Optional[List[str]]:
        return self._safe(self.candidate_symbols)

    def news(self, symbol: str) -> Optional[Dict]:
        return self._safe(self.news_for, symbol)

    def market(self) -> Optional[Dict]:
        return self._safe(self.market_news)


def _live_dir(output_dir: str) -> Path:
    p = Path(output_dir)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[2] / p
    p = p / "live"
    p.mkdir(parents=True, exist_ok=True)
    return p


def state_path(output_dir: str, user_id: int, exchange_id: str) -> Path:
    return _live_dir(output_dir) / f"runner_{int(user_id)}_{exchange_id}.json"


class _StoreBrokerAdapter:
    """LightContextCache'in beklediği `fetch_ohlcv(symbol, tf, limit=)` arayüzünü depoya bağlar."""

    def __init__(self, store: MarketStateStore, exchange_id: str):
        self.store, self.ex = store, exchange_id

    def fetch_ohlcv(self, symbol: str, tf: str = "4h", limit: int = 300):
        return self.store.get_ohlcv(self.ex, symbol, tf, limit)


class LiveRunner:
    def __init__(self, user_id: int, cfg: RunnerConfig, broker: Broker,
                 ctx: Optional[Context] = None, server_config=None,
                 output_dir: str = "runs", manage_only: bool = False,
                 store: Optional[MarketStateStore] = None):
        self.user_id = int(user_id)
        self.cfg = cfg
        self.broker = broker
        self.ctx = ctx or Context()
        self.server_config = server_config
        self.output_dir = output_dir
        self.chain = DC.ChainConfig.from_dict(cfg.chain)
        self.guard = LiveGuard(server_config, output_dir,
                               state_file=f"live/guard_{self.user_id}_{cfg.exchange_id}.json",
                               daily_loss_limit=cfg.daily_loss_limit_pct,
                               max_drawdown=cfg.max_drawdown_pct)
        self.positions: Dict[str, Position] = {}
        self.pending: Dict[str, Dict] = {}
        self._skip_entry_cycle: Dict[str, int] = {}
        self.trades: List[Dict] = []
        self.paper_history: List[Dict] = []
        self.events: Deque[Dict] = deque(maxlen=300)
        self.equity_curve: List[Dict] = []
        self.equity_history: List[Dict] = []       # TAM geçmiş: son 6 sa 30 sn çözünürlük, öncesi 5 dk kova (kalıcı)
        self.realized_net = 0.0
        self.fees_paid = 0.0
        self.gross_pnl = 0.0
        self.cycle = 0
        self.last_cycle_ts: Optional[float] = None
        self.last_cycle_sec: Optional[float] = None
        self.last_data_ts: Optional[float] = None
        self.last_decisions: Dict[str, Dict] = {}
        self.scan: List[Dict] = []
        self._rs_ranks: Dict[str, float] = {}
        self._promoted: Dict = {}
        self._client_factory = None
        self.day = ""
        self.day_trades = 0
        self.running = False
        self.manage_only = manage_only
        self.reconcile_ok = True
        self.reconcile_note = ""
        self.created_ts = time.time()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._path = state_path(output_dir, self.user_id, cfg.exchange_id)
        self.venue = FE.venue_fee(cfg.exchange_id)
        self.store = store or MarketStateStore(
            fetch_ohlcv=lambda ex, s, tf, lim: self.broker.fetch_ohlcv(s, tf, limit=lim),
            fetch_book=lambda ex, s: self.broker.fetch_book_top(s))
        self.light = LightContextCache(_StoreBrokerAdapter(self.store, cfg.exchange_id),
                                       events_fn=lambda: (self.ctx.slow_for("BTC/USDT") or {}).get("events"))
        self.risk: Dict = {"level": 0, "label": "NORMAL", "reasons": []}
        self.portfolio: Dict = {"mode": PM.RISK_ON, "level": 0, "label": PM.LABEL_TR[PM.RISK_ON], "reasons": [],
                                "actions": {"new_entries": True, "size_mult": 1.0}}
        self.cash_mode = False
        self.resource: Dict = {"state": "GREEN", "rss_mb": None, "cap_mb": MEM_CAP_MB}
        self._rss_hist: List[Dict] = []          # (ts, rss) - buyume HIZINI gormek icin
        self._trim_stats = {"n": 0, "last_freed_mb": None}
        self.fee_info: Dict = {}
        self.venue_compare: Dict[str, Dict] = {}
        self.exit_state: Dict[str, Dict] = {}     # parite → son çıkış (yeniden giriş kapısı)
        self.reentry_blocks: Deque[Dict] = deque(maxlen=200)   # kapının NE engellediği ölçülebilsin
        live = _live_dir(output_dir)
        tag = f"{self.user_id}_{cfg.exchange_id}"
        self.lessons = LessonEngine(live / f"lessons_{tag}.json",
                                    journal_md=live / f"GUNLUK_{tag}.md",
                                    journal_jsonl=live / f"journal_{tag}.jsonl")
        self.challenger = Challenger(live / f"challenger_{tag}.json")
        self.lifecycle = Lifecycle(live / "lifecycle.json")
        self.allocator = MetaAllocator(live / f"allocator_{tag}.json", SF.REGIME_SLEEVES)
        self.missed = MissedEngine(live / f"missed_{tag}.json", journal_md=live / f"KACIRILANLAR_{tag}.md",
                                   journal_jsonl=live / f"missed_{tag}.jsonl")
        self.haber = HaberEtkiMotoru(output_dir, tag)
        self.lab = ResearchLab(live, tag, cfg.exchange_id, fetch_tickers=self._fetch_tickers_safe,
                               taker_bps=self.venue.taker_bps, maker_bps=self.venue.maker_bps)
        self._public_broker: Optional[Callable[[str], Broker]] = None
        self._venue_compare_ts: Dict[str, float] = {}
        self.alerts = AlertBus()
        self._ledger_hash = ""
        self.error_times: Deque[float] = deque(maxlen=300)
        self._refresh_params()

    def _corr_with_open(self, sym: str) -> Optional[float]:
        """Yeni adayın açık pozisyonlarla 1m getiri korelasyonu (en yüksek |ρ|) — korelasyon bütçesi."""
        others = [x.symbol for x in self.positions.values() if x.symbol != sym]
        if not others:
            return None
        try:
            tf = f"{self.params.bar_minutes}m"
            cols = {}
            for s_ in [sym] + others:
                df = self.store.get_ohlcv(self.cfg.exchange_id, s_, tf, 150)
                if df is not None and len(df) >= 40:
                    cols[s_] = df["close"].astype(float).pct_change().dropna().tail(120).to_numpy()
            if sym not in cols or len(cols) < 2:
                return None
            n = min(len(v) for v in cols.values())
            if n < 30:
                return None
            a = cols[sym][-n:]
            best = None
            for s_, v in cols.items():
                if s_ == sym:
                    continue
                b = v[-n:]
                if a.std() <= 0 or b.std() <= 0:
                    continue
                c = abs(float(np.corrcoef(a, b)[0, 1]))
                best = c if best is None else max(best, c)
            return best
        except Exception:
            return None

    def _fetch_tickers_safe(self) -> Dict:
        try:
            return self.broker.fetch_tickers()
        except Exception:
            return {}

    def _cost_pct_est(self) -> float:
        return round((2.0 * self.venue.taker_bps + 4.0) / 100.0, 4)

    def _proxy_trigger(self, sym: str) -> Optional[Dict]:
        """Değerlendirilmeyen adaylar için ucuz tetikleyici vekili (kör-nokta gölgesi)."""
        try:
            cf = self.store.cheap_features(self.cfg.exchange_id, sym, f"{self.params.bar_minutes}m")
        except Exception:
            return None
        if not cf:
            return None
        z = cf.get("z20"); vr = cf.get("vol_ratio") or 1.0
        dip = z is not None and z <= -1.5 and cf.get("bar_up")
        brk = (cf.get("dist_hi20_pct") is not None and cf["dist_hi20_pct"] <= 0.0 and vr >= 1.3)
        if not (dip or brk):
            return None
        sig = float(cf.get("sigma_1m_pct") or 0.1)
        p = self.params
        stop = float(min(p.max_stop_pct, max(p.min_stop_pct, 3.0 * sig * (15 ** 0.5))))
        target = float(max(p.rr_min, getattr(p, "rr", 1.6)) * stop)
        return {"cheap": {**cf, "kind": "dip" if dip else "breakout"}, "price": float(cf["price"]),
                "stop_pct": stop, "target_pct": target, "kind": "dip" if dip else "breakout"}

    def _blind_context(self) -> Dict:
        return {"portfolio_mode": self.portfolio.get("mode"), "resource": self.resource.get("state"),
                "cash_mode": self.cash_mode, "open_positions": len(self.positions) + len(self.pending),
                "max_open": self.cfg.max_open, "top_k": self._effective_top_k()}

    def _record_blind_spots(self, frames: Dict, cands: List[str], block_gate: Optional[str], now: float) -> None:
        """Hiç değerlendirilmeyen adayları gölgeye al: giriş kapalıysa (HALT/KAYNAK/PORTFÖY/NAKİT…) ilk 5 aday;
        girişler açıksa Top-K DIŞINDA kalan ve vekil tetikleyicisi olan ilk 5 parite."""
        horizon = float(self.params.max_hold_sec)
        n = 0
        pool = cands[:] if block_gate else [s for s in frames if s not in cands]
        gate = block_gate or "TOP_K"
        for sym in pool:
            if n >= 5:
                break
            if sym in self.positions or sym in self.pending:
                continue
            px = self._proxy_trigger(sym)
            if not px:
                continue
            rec = self.missed.on_unevaluated(sym, gate, px["cheap"], px["price"], "LONG", px["stop_pct"], px["target_pct"],
                                             horizon, now, info={"slow_ctx": bool(self.ctx.slow_for(sym)), "news": bool(self.ctx.news(sym))},
                                             context=self._blind_context(),
                                             detail=(f"{gate}: aday değerlendirilmedi (vekil tetikleyici {px['kind']}, z20 {px['cheap'].get('z20')})"))
            if rec:
                n += 1

    def _regime_of(self, decision: Optional[Dict]) -> Optional[str]:
        return (decision or {}).get("regime")

    # ------------------------------------------------------------ parametreler
    def _refresh_params(self) -> None:
        """Konfig + yalnız TERFİ ETMİŞ geçersiz kılmalar → etkin parametreler.
        Ders motorunun yeni önerileri doğrudan uygulanmaz; challenger'a gider."""
        promoted = dict(self._promoted) if self.cfg.strategy == "committee" else {}
        merged = {**self.cfg.params, **promoted}
        if self.cfg.strategy == "committee":
            self.params = CM.CommitteeParams.from_dict(merged)
        else:
            self.params = VS.ScalpParams.from_dict(merged)
        if self.cfg.market_type == "spot":
            self.params.allow_short = False
        self.xparams = XE.ExitParams(**{k: v for k, v in (self.cfg.exit or {}).items()
                                        if k in XE.ExitParams().__dict__}).validated()
        self.xparams.min_hold_sec = int(self.params.min_hold_sec)
        self.xparams.retain_fraction = float(min(0.70, max(0.35, getattr(self.params, "giveback", 0.5))))
        self.xparams = self.xparams.validated()
        self.rparams = RE.ReentryParams.from_dict(self.cfg.reentry)
        if self.cfg.strategy == "committee":
            self.chain = DC.ChainConfig.from_dict({**self.cfg.chain,
                                                   **{k: v for k, v in promoted.items() if k in ("min_gross_to_cost",)}})

    # ------------------------------------------------------------ kontrol
    def start(self) -> Dict:
        with self._lock:
            if self.running:
                return {"ok": False, "reason": "zaten çalışıyor"}
            self.running = True
            self.manage_only = False
            self._thread = threading.Thread(target=self._loop, daemon=True,
                                            name=f"cm-runner-{self.user_id}-{self.cfg.exchange_id}")
            self._thread.start()
        self._log("system", f"▶️ Otopilot başladı · {self.cfg.mode.upper()} · {self.cfg.strategy} · "
                            f"{self.cfg.exchange_id} (maker {self.venue.maker_bps} / taker {self.venue.taker_bps} bps) · "
                            f"{len(self.cfg.symbols)} parite · Top-K {self.cfg.top_k}")
        audit("RUNNER_START", {"user": self.user_id, "exchange": self.cfg.exchange_id,
                               "mode": self.cfg.mode, "strategy": self.cfg.strategy,
                               "symbols": self.cfg.symbols}, self.output_dir)
        self.save()
        return {"ok": True}

    def stop(self, reason: str = "kullanıcı") -> Dict:
        with self._lock:
            self.running = False
        self._log("system", f"⏹️ Otopilot durdu ({reason})")
        audit("RUNNER_STOP", {"user": self.user_id, "exchange": self.cfg.exchange_id,
                              "reason": reason}, self.output_dir)
        self.save()
        return {"ok": True}

    def resume(self) -> Dict:
        r = self.guard.resume(operator=f"user:{self.user_id}")
        self._log("system", "🔓 HALT kaldırıldı (elle)")
        self.save()
        return r

    def update_params(self, params: Optional[Dict] = None, chain: Optional[Dict] = None) -> Dict:
        with self._lock:
            if params:
                self.cfg.params = {**self.cfg.params, **params}
            if chain:
                self.cfg.chain = DC.ChainConfig.from_dict({**self.cfg.chain, **chain}).to_dict()
            self.cfg.validated()
            self._refresh_params()
        self._log("system", "⚙️ Parametreler güncellendi")
        self.save()
        return {"ok": True, "params": self.cfg.params, "chain": self.cfg.chain}

    def sync_config(self, new: "RunnerConfig", fields=("symbols", "max_open", "max_order_usdt",
                                                        "max_exposure_pct", "top_k", "max_trades_per_day")) -> List[str]:
        """Kod-tanımlı evren/limitler geri yüklenen durumu ezer (pozisyon/işlem/ders KORUNUR)."""
        changed = []
        with self._lock:
            for k in fields:
                nv = getattr(new, k)
                if getattr(self.cfg, k) != nv:
                    if k == "symbols":
                        keep = [s for s in self.cfg.symbols if s in self.positions or s in self.pending]
                        nv = keep + [s for s in nv if s not in keep]
                    setattr(self.cfg, k, nv)
                    changed.append(k)
            if changed:
                self.cfg.validated()
                if "max_order_usdt" in changed:
                    self.broker.max_order_usdt = self.cfg.max_order_usdt
                if "exit" in changed or "params" in changed:
                    self._refresh_params()
        if changed:
            self._log("system", "🔧 Konfig senkronlandı: " + ", ".join(changed))
            self.save()
        return changed

    def close_all(self, reason: str = "MANUEL") -> Dict:
        closed = []
        with self._lock:
            for sym, o in list(self.pending.items()):
                self.broker.cancel_order(o["order"])
                self.pending.pop(sym, None)
            for sym, pos in list(self.positions.items()):
                px = pos.last_price or pos.entry
                try:
                    px = self.broker.fetch_price(sym)
                except Exception:
                    pass
                rec = self._close(pos, px, reason)
                if rec:
                    closed.append(rec)
        self.save()
        return {"ok": True, "closed": len(closed)}

    def reset_paper(self) -> Dict:
        with self._lock:
            if self.running:
                return {"ok": False, "reason": "önce durdur"}
            if self.cfg.mode != "paper":
                return {"ok": False, "reason": "yalnız paper sıfırlanır"}
            self.positions.clear(); self.pending.clear(); self.trades.clear(); self.events.clear()
            self.equity_curve.clear()
            self.realized_net = self.fees_paid = self.gross_pnl = 0.0
            self.cycle = 0
            self.last_decisions.clear()
            self.broker.paper_cash = float(self.cfg.capital_usdt)
            self.broker.paper_holdings.clear()
            self.guard.resume(operator="reset")
        self.save()
        return {"ok": True}

    # ------------------------------------------------------------ döngü
    def _loop(self):
        while self.running:
            try:
                self.run_cycle()
            except Exception as e:
                self._log("error", f"döngü hatası: {type(e).__name__}: {str(e)[:120]}")
            slept = 0.0
            while self.running and slept < self.params.loop_sec:
                time.sleep(0.5)
                slept += 0.5

    def _refresh_symbols(self) -> None:
        if self.cfg.symbols_mode != "auto":
            return
        cands = self.ctx.candidates() or []
        if not cands:
            return
        keep = [s for s in self.cfg.symbols if s in self.positions or s in self.pending]
        new = keep + [s for s in cands if s not in keep]
        new += [s for s in self.cfg.symbols if s not in new]
        new = new[:MAX_SYMBOLS]
        if new != self.cfg.symbols:
            self._log("system", f"🎯 Parite listesi güncellendi ({len(new)}): {', '.join(new[:8])}…")
            self.cfg.symbols = new

    def _update_resource(self) -> None:
        rss = _rss_mb()
        st = "GREEN"
        if rss is not None:
            frac = rss / MEM_CAP_MB
            st = "RED" if frac >= 0.85 else "YELLOW" if frac >= 0.75 else "GREEN"
        prev = self.resource.get("state")
        # HİSTEREZİS: RED yalnız 2 ardışık okumada (haber taramasının ~90 sn'lik geçici tepesi girişleri durdurmasın)
        if st == "RED" and prev != "RED":
            self._red_streak = getattr(self, "_red_streak", 0) + 1
            if self._red_streak < 2:
                st = "YELLOW"
        else:
            self._red_streak = 0
        now_ = time.time()
        if rss is not None:
            self._rss_hist.append({"ts": now_, "rss": float(rss)})
            self._rss_hist = [r for r in self._rss_hist if now_ - r["ts"] <= 6 * 3600][-720:]
        # Buyume hizi: son saatte MB/saat. Bu sayi olmadan "sizinti mi, normal isinma mi"
        # sorusu tartisma konusu kalir; varken tavana ne zaman carpilacagi hesaplanabilir.
        growth = None
        eta_h = None
        if len(self._rss_hist) >= 4:
            a, b = self._rss_hist[0], self._rss_hist[-1]
            dt_h = (b["ts"] - a["ts"]) / 3600.0
            if dt_h >= 0.25:
                growth = round((b["rss"] - a["rss"]) / dt_h, 1)
                if growth and growth > 0 and rss is not None:
                    kalan = MEM_CAP_MB * 0.85 - rss
                    eta_h = round(max(0.0, kalan) / growth, 1)
        self.resource = {"state": st, "rss_mb": (None if rss is None else round(rss)), "cap_mb": MEM_CAP_MB,
                         "cycle_sec": self.last_cycle_sec, "store": self.store.status(),
                         "errors_10m": sum(1 for t in self.error_times if now_ - t <= 600),
                         "rss_growth_mb_per_h": growth,
                         "hours_to_red": eta_h,
                         "trim": dict(self._trim_stats),
                         "tracemalloc": _tracemalloc_top()}
        if st != prev and prev is not None:
            self._log("system", f"🧠 Kaynak durumu {prev} → {st} (RSS {self.resource['rss_mb']} MB)")

    def _effective_top_k(self) -> int:
        k = self.cfg.top_k
        if self.resource.get("state") == "YELLOW":
            k = max(3, k // 2)
        elif self.resource.get("state") == "RED":
            k = 0
        return k

    def run_cycle(self, now: Optional[float] = None) -> Dict:
        now = time.time() if now is None else float(now)
        t0 = time.time()
        if self.cycle % SYMBOL_REFRESH_CYCLES == 0:
            self._refresh_symbols()
        self._update_resource()
        tf = f"{self.params.bar_minutes}m"
        # --- 1) veri: depo üzerinden (tek fetch) ---
        frames: Dict[str, pd.DataFrame] = {}
        prices: Dict[str, float] = {}
        ex_err = False
        stale = 0
        for sym in list(self.cfg.symbols):
            try:
                df = self.store.get_ohlcv(self.cfg.exchange_id, sym, tf, 150, now=now)
                if self.store.freshness(self.cfg.exchange_id, sym, tf, now=now)["state"] == "STALE":
                    stale += 1
                    self.missed.count_blind("BAYAT")
                    continue                       # STALE DATA → NO TRADE
                frames[sym] = df
                prices[sym] = float(df["close"].iloc[-1])
            except Exception as e:
                ex_err = True
                self._log("error", f"{sym} veri hatası: {type(e).__name__}")
        if prices:
            self.last_data_ts = time.time()

        cands: List[str] = []
        with self._lock:
            self.cycle += 1
            today = time.strftime("%Y-%m-%d", time.gmtime(now))
            if self.day != today:
                self._rotations_today = 0
                if self.day:
                    st_ = self.stats()
                    self.alerts.send("daily", f"GÜN SONU {self.day}: özsermaye {st_['equity']} $ · net {st_['net_pnl']:+.2f} · işlem {st_['closed_trades']} · kazanma %{st_['win_rate']}", force=True)
                self.day, self.day_trades = today, 0

            # --- 2) bekleyen maker emirleri ---
            for sym, pend in list(self.pending.items()):
                df = frames.get(sym)
                if df is None:
                    continue
                self._check_pending(sym, pend, df, prices[sym], now)

            # --- 3) çıkışlar ---
            for sym, pos in list(self.positions.items()):
                px = prices.get(sym)
                if not px:
                    continue
                pos.last_price = px
                df = frames.get(sym)
                hi = float(df["high"].iloc[-1]) if df is not None and "high" in df else px
                lo = float(df["low"].iloc[-1]) if df is not None and "low" in df else px
                self._manage_exit(pos, px, hi, lo, now)

            # --- 4) kill-switch ---
            eq = self.equity()
            g = self.guard.check(eq, last_data_ts=self.last_data_ts, exchange_error=ex_err)
            if g["halted"]:
                for sym, o in list(self.pending.items()):
                    self.broker.cancel_order(o["order"])
                    self.pending.pop(sym, None)
                if self.positions and self.cfg.halt_action == "flatten":
                    self._log("system", "🛑 HALT — açık pozisyonlar kapatılıyor: " + "; ".join(g["reasons"]))
                    self.alerts.send("halt", "KILL-SWITCH: " + "; ".join(g["reasons"]), level="critical", force=True)
                    for sym, pos in list(self.positions.items()):
                        self._close(pos, prices.get(sym) or pos.last_price or pos.entry, "HALT", now)
                elif self.cycle % 20 == 1:
                    self._log("system", "🛑 HALT sürüyor — yeni giriş yok: " + "; ".join(g["reasons"]))

            # --- 5) mutabakat (testnet/live) ---
            if self.cfg.mode != "paper" and self.cycle % 4 == 1:
                self._reconcile()

            # --- 5b) portföy modu / nakit ---
            if self.cfg.strategy == "committee":
                self._update_portfolio_mode(prices, now, stale_share=(stale / max(1, len(self.cfg.symbols))))

            # --- 6) girişler ---
            can_enter = (not g["halted"] and not self.manage_only and self.reconcile_ok
                         and self.day_trades < self.cfg.max_trades_per_day
                         and self.risk.get("level", 0) == 0
                         and bool(self.portfolio.get("actions", {}).get("new_entries", True))
                         and self.resource.get("state") != "RED")
            if not can_enter and self.cycle % 20 == 1 and self.manage_only:
                self._log("system", "ℹ️ yalnız pozisyon yönetimi — yeni giriş için START gerekir")
            health = self.ctx.health() if can_enter else None
            block_gate = None
            if not can_enter:
                block_gate = ("HALT" if g["halted"] else "MANAGE_ONLY" if self.manage_only else "MUTABAKAT" if not self.reconcile_ok
                              else "GÜNLÜK_TAVAN" if self.day_trades >= self.cfg.max_trades_per_day
                              else "NAKİT" if self.cash_mode else "RİSK_PİYASA" if self.risk.get("level", 0) > 0
                              else "PORTFÖY" if not self.portfolio.get("actions", {}).get("new_entries", True)
                              else "KAYNAK" if self.resource.get("state") == "RED" else "?")
            if self.cfg.strategy == "committee":
                self.scan = self._tier_a_scan(frames)
                cands = [r["symbol"] for r in self.scan[:self._effective_top_k()]]
                try:
                    self._record_blind_spots(frames, cands, block_gate, now)
                except Exception as e:
                    self._log("error", f"kör nokta kaydı: {type(e).__name__}")
            else:
                cands = list(frames.keys())
            for idx_c, sym in enumerate(cands):
                df = frames.get(sym)
                if df is None or not can_enter or sym in self.positions or sym in self.pending:
                    continue
                if self._skip_entry_cycle.get(sym) == self.cycle:
                    continue
                if len(self.positions) + len(self.pending) >= self.cfg.max_open:
                    if self.cfg.strategy == "committee":
                        rotated = False
                        try:
                            probe = self._probe_candidate(sym, df, prices[sym], health, now)
                            if probe and probe.get("allowed"):
                                victim = self._maybe_rotate((probe.get("ticket") or {}).get("ev_pct"), sym, now)
                                if victim is not None:
                                    self._close(victim, prices.get(victim.symbol) or victim.last_price or victim.entry, "ROTATION", now)
                                    self._try_entry_committee(sym, df, prices[sym], health, now)
                                    rotated = True
                        except Exception as e:
                            self._log("error", f"rotasyon: {type(e).__name__}: {str(e)[:80]}")
                        if not rotated:
                            try:
                                self._record_blind_spots(frames, cands[idx_c:], "MAX_OPEN", now)
                            except Exception:
                                pass
                    break
                try:
                    if self.cfg.strategy == "committee":
                        self._try_entry_committee(sym, df, prices[sym], health, now)
                    else:
                        self._try_entry(sym, df, prices[sym], health, now)
                except BrokerError as e:
                    self._log("error", f"{sym} emir: {e}")
                except Exception as e:
                    self._log("error", f"{sym} giriş hatası: {type(e).__name__}: {str(e)[:100]}")

            # --- 6b) HABER ETKİSİ: gözle (haber görüldüğünde) + çöz (ufuk dolunca) ---
            # `scan()` olay türünü ZATEN üretiyordu ama hiçbir yerde saklanmıyordu; bu yüzden
            # `EVENT_PRIOR`'daki 19 yön tahmini hiç ölçülemedi. Artık ölçülüyor.
            try:
                for sym in list(prices):
                    nw = self.ctx.news(sym) or {}
                    kat = nw.get("top_event")
                    if not kat or kat == "OTHER":
                        continue
                    hl = (nw.get("headlines") or [{}])[0]
                    self.haber.gozle(sym, kat, int(hl.get("tier") or 3),
                                     float(nw.get("score") or 0.0), prices[sym],
                                     float((self.light.get(sym) or {}).get("atr_pct") or 0.3), now)
                for r in self.haber.coz(prices, now):
                    if abs(float(r.get("z") or 0)) >= 2.0:
                        self._log("learn", f"📰 {r['sym']} {r['kat']} ({r['ufuk']}): "
                                           f"%{r['r']:+.2f} · normalin {abs(r['z']):.1f} katı")
            except Exception as e:
                self._log("error", f"haber etki motoru: {type(e).__name__}: {str(e)[:80]}")

            # --- 7) gölgeler → kaçırılanlar → dersler → challenger → araştırma lab'ı ---
            if self.cfg.strategy == "committee":
                for s in self.lessons.update_shadows(frames, now):
                    self._log("learn", f"👁️ gölge: {s['symbol']} {s['kind']} ({s.get('gate')}) → {s['outcome']}")
                for m in self.missed.update(frames, now, self._cost_pct_est()):
                    a = m.get("attribution") or {}
                    if m.get("outcome") == "TARGET":
                        self._log("learn", f"👁️🟢 KAÇIRILAN KAZANÇ {m['symbol']} · {a.get('gate_tr')} · net %{a.get('net_missed_pct')} · {a.get('path')} · "
                                           f"göz ardı: {', '.join(a.get('ignored_supportive_tr') or []) or '—'} · eksik: {', '.join(a.get('missing_info_tr') or []) or '—'}")
                    elif m.get("outcome") == "STOP":
                        self._log("learn", f"👁️🔴 STOP OLACAKTI {m['symbol']} · {a.get('gate_tr')} · {a.get('path')} · uyaranlar: {', '.join(a.get('warnings_present_tr') or []) or '—'}")
                try:
                    self.lab.step(now, self.store, list(frames.keys()), frames, self._book, self.resource.get("state", "GREEN"))
                except Exception as e:
                    self._log("error", f"araştırma lab'ı: {type(e).__name__}")
                new = self.lessons.derive(now, self.params.to_dict())
                for les in new:
                    self._log("learn", "📘 " + les["title"])
                if new:
                    self.lessons.save()
                    self._challenger_cycle()
            self.last_cycle_ts = now
            self.last_cycle_sec = round(time.time() - t0, 2)
            self._mark(now)
            if self.cycle % 20 == 0:
                import gc; gc.collect()                       # döngüsel çöp (DataFrame/dict ağları) birikmesin
                # gc.collect() nesneyi Python'a iade eder, İŞLETİM SİSTEMİNE DEĞİL: glibc
                # serbest blokları arena'sında tutar, RSS düşmez ve tavan yakınında
                # `top_k = 0` ile girişler durur. malloc_trim boşluğu OS'a geri verir.
                before = _rss_mb()
                if _malloc_trim():
                    after = _rss_mb()
                    self._trim_stats["n"] += 1
                    if before is not None and after is not None:
                        self._trim_stats["last_freed_mb"] = round(before - after, 1)
        self.save()
        return {"cycle": self.cycle, "open": len(self.positions), "halted": g["halted"], "candidates": len(cands)}

    # ------------------------------------------------------------ Tier-A
    def _tier_a_scan(self, frames: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Ucuz ilgi puanı: |z| + hacim oranı + kırılım yakınlığı + oynaklık + haber dikkati."""
        rows = []
        tf = f"{self.params.bar_minutes}m"
        cross = self.store.cross_section(self.cfg.exchange_id, list(frames.keys()), tf)
        ranks = SF.relative_strength_ranks(cross)
        iw = self.missed.interest_weights()                   # kaçırılan/kaçınılan kanıtından öğrenilmiş vekil ağırlıkları
        for sym in frames:
            f = cross.get(sym)
            if not f:
                continue
            news = self.ctx.news(sym) or {}
            vr = f.get("vol_ratio") or 1.0
            z20 = f.get("z20") or 0.0
            near_hi = (f.get("dist_hi20_pct") or 9) <= 0.15
            near_lo = (f.get("dist_lo20_pct") or 9) <= 0.15
            score = (min(3.0, abs(z20)) * (iw["dip"] if z20 < 0 else iw["breakout"])
                     + 0.8 * min(3.0, max(0.0, vr - 1.0))
                     + (1.0 * iw["breakout"] if near_hi else 1.0 * iw["dip"] if near_lo else 0.0)
                     + 0.5 * min(2.0, (f.get("sigma_1m_pct") or 0.0) / 0.1)
                     + min(1.0, float(news.get("buzz") or 0.0) / 3.0)
                     + (0.5 if news.get("confirmed") else 0.0))
            rows.append({"symbol": sym, "interest": round(float(score), 3), "z20": round(f.get("z20") or 0.0, 2),
                         "freshness": self.store.freshness(self.cfg.exchange_id, sym, tf).get("state"),
                         "vol_ratio": (None if f.get("vol_ratio") is None else round(f["vol_ratio"], 2)),
                         "sigma_1m_pct": (None if f.get("sigma_1m_pct") is None else round(f["sigma_1m_pct"], 4)),
                         "rs_rank": (None if ranks.get(sym) is None else round(ranks[sym], 2)), "trend_up": f.get("trend_up"),
                         "news_buzz": news.get("buzz"), "tier": "heavy" if self.ctx.slow_for(sym) else "light"})
        rows.sort(key=lambda r: -r["interest"])
        self._rs_ranks = ranks
        self._interest_weights = iw
        return rows

    # ------------------------------------------------------------ giriş (video)
    def _try_entry(self, sym: str, df: pd.DataFrame, price: float, health: Optional[Dict], now: float) -> None:
        sig = VS.signal(df, self.params, self.cfg.market_type)
        trace: Dict = {"symbol": sym, "ts": now, "signal": {
            "direction": sig.get("direction"), "z": _r(sig.get("z")),
            "rsi": _r(sig.get("rsi")), "reasons": sig.get("reasons", [])}}
        if not sig.get("direction"):
            trace["result"] = "SİNYAL YOK"
            self.last_decisions[sym] = trace
            return
        plan = VS.plan_trade(sig["direction"], price, sig.get("sigma_bar_pct"), self.params)
        book = self._book(sym)
        cost_pct = VS.roundtrip_cost_pct(self.params, book.get("spread_bps", 0.0))
        stats = self.stats()
        p_target = (stats["win_rate"] / 100.0) if stats["closed_trades"] >= 10 else 0.5
        regime = self.ctx.regime_for(df)
        vol = _vol_label(sig.get("sigma_bar_pct"))
        notional_guess = self._size(plan["stop_pct"], 1.0)
        z_abs = abs(float(sig.get("z") or 0.0))
        conf = float(min(0.9, max(0.5, 0.55 + 0.1 * (z_abs - self.params.dip_z))))
        inp = DC.ChainInputs(
            symbol=sym, direction=plan["direction"], entry=price,
            target_gross_pct=plan["target_pct"], stop_pct=plan["stop_pct"],
            cost_pct=cost_pct, notional=max(10.0, notional_guess), horizon="1h", confidence=conf,
            cm_signal=self.ctx.signal_for(sym), qual_cell=self.ctx.cell_for(sym, "1h", plan["direction"]),
            regime=regime, volatility=vol, system_health=health, p_target=p_target,
            bid_depth_usd=book.get("bid_depth_usd", 0.0), ask_depth_usd=book.get("ask_depth_usd", 0.0),
            spread_bps=book.get("spread_bps", 0.0))
        dec = DC.decide(inp, self.chain)
        trace.update({"plan": plan, "cost_pct": round(cost_pct, 4), "chain": dec.to_dict()})
        rg = self._reentry_gate(sym, plan["direction"], float(plan["target_pct"]), cost_pct, now)
        trace["reentry"] = rg
        if not rg["allowed"]:
            trace["result"] = f"{rg.get('gate')}: {rg['reason']}"
            self.last_decisions[sym] = trace
            return
        if not dec.allowed:
            trace["result"] = "VETO: " + "; ".join(dec.vetoes)
            self.last_decisions[sym] = trace
            return
        notional = self._size(plan["stop_pct"], dec.size_mult)
        if not self._notional_ok(sym, notional, trace):
            return
        if self.cfg.mode == "live" and not self._live_preflight(trace):
            self.last_decisions[sym] = trace
            return
        self._open_taker(sym, plan, notional, trace["chain"], now, "dip", "mean_reversion", XE.PARTIAL_AND_RUN)
        trace["result"] = f"AÇILDI {plan['direction']} {notional:.2f} USDT @ {price}"
        self.last_decisions[sym] = trace

    # ------------------------------------------------------------ bağlam / ücret / portföy
    def _slow_ctx(self, sym: str) -> Dict:
        s = self.ctx.slow_for(sym)
        if s:
            s = dict(s); s.setdefault("tier", "heavy")
            return s
        return self.light.get(sym) or {}

    def _fees(self, symbol: Optional[str] = None) -> Dict:
        """Hesaba ÖZGÜ komisyon. DÜZELTİLDİ (2026-09-06): eski sürüm her zaman
        `symbols[0]`'ın oranını çekip TÜM pariteler için kullanıyordu. Borsalar parite
        bazında farklı oran/indirim uygulayabilir (promosyonlu çiftler, farklı market
        tipleri) — yanlış ücretle kurulan maliyet kapısı, kâr eşiğini kaydırır."""
        client = None
        if self.cfg.mode != "paper":
            try:
                client = self.broker._priv()
            except Exception:
                client = None
        sym = symbol or (self.cfg.symbols[0] if self.cfg.symbols else "BTC/USDT")
        f = FA.fetch_account_fee(client, self.cfg.exchange_id, sym)
        if symbol is None or symbol == (self.cfg.symbols[0] if self.cfg.symbols else None):
            self.fee_info = f
        return f

    def _tca(self, symbol: str, side: str, qty: float, ref_price: float, fill_price: float,
             order_type: str, fee: float = 0.0, requested_qty: Optional[float] = None) -> None:
        """Her dolumu TCA'ya yaz. Bu ölçüm olmadan "varsayılan maliyet" ile "ödenen maliyet"
        arasındaki fark bilinemez; entry_optimizer ve venue_router varsayımla çalışmaya
        devam eder. Kayıt başarısız olursa işlem AKIŞI DURMAZ (ölçüm, ticaret değil)."""
        try:
            TCA.record_fill(symbol, side, float(qty), float(ref_price), float(fill_price),
                            str(order_type), fee=float(fee or 0.0),
                            requested_qty=(None if requested_qty is None else float(requested_qty)),
                            output_dir=self.output_dir)
        except Exception:
            pass

    def _evidence_report(self, ttl: float = 300.0) -> Dict:
        """Kanıt durumu — hangi sleeve kanıta ne kadar yakın. TTL'li önbellek:
        kanıt defteri akışla okunur (bellek sabit) ama her panel isteğinde okumak gereksiz."""
        now = time.time()
        hit = getattr(self, "_ev_cache", None)
        if hit and now - hit[0] < ttl:
            return hit[1]
        try:
            o = EV.ozet(self.output_dir, tag=f"{self.user_id}_{self.cfg.exchange_id}", min_n=3)
            MIN_N = 8      # n<8'de t anlamsız — sahte kanıt üretmemek için
            o["verdikt"] = {
                "kanitli_kar": [k for k, v in o.get("sleeve", {}).items()
                                if v["n"] >= MIN_N and v["t"] >= 2.0 and v["ort_pct"] > 0],
                "kanitli_zarar": [k for k, v in o.get("sleeve", {}).items()
                                  if v["n"] >= MIN_N and v["t"] <= -2.0],
                "kanita_yakin": sorted(
                    ({"sleeve": k, "n": v["n"], "t": v["t"], "kalan": v["kalan_islem_t2"]}
                     for k, v in o.get("sleeve", {}).items()
                     if v["n"] >= 5 and v["ort_pct"] > 0 and v.get("kalan_islem_t2") is not None),
                    key=lambda x: x["kalan"])[:3],
            }
        except Exception as e:
            o = {"hata": type(e).__name__}
        self._ev_cache = (now, o)
        return o

    def _ev_calib_report(self, ttl: float = 900.0) -> Dict:
        """Hangi EV öngörüyor: `ev_pct` (plan hedefi) mi `ev_achievable_pct` (ölçülmüş) mü?
        Rotasyon kapısı ŞU AN `ev_pct` kullanıyor ve o sayının 182 işlemde korelasyonu
        +0,089 (öngörü yok). Karar bu ölçüme bağlanacak."""
        now = time.time()
        hit = getattr(self, "_evc_cache", None)
        if hit and now - hit[0] < ttl:
            return hit[1]
        try:
            tag = f"{self.user_id}_{self.cfg.exchange_id}"
            ev_c, eva_c, ikisi = [], [], []
            for r in EV.oku(self.output_dir, tag=tag):
                y = r.get("np")
                if y is None:
                    continue
                if r.get("ev") is not None:
                    ev_c.append((float(r["ev"]), float(y)))
                if r.get("eva") is not None:
                    eva_c.append((float(r["eva"]), float(y)))
                if r.get("ev") is not None and r.get("eva") is not None:
                    ikisi.append(1)
            def _r(c):
                if len(c) < 5:
                    return None
                xs = [a for a, _ in c]; ys = [b for _, b in c]
                mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
                sx = (sum((a - mx) ** 2 for a in xs) / len(xs)) ** 0.5
                sy = (sum((b - my) ** 2 for b in ys) / len(ys)) ** 0.5
                if sx <= 0 or sy <= 0:
                    return None
                return round(sum((a - mx) * (b - my) for a, b in c) / len(c) / (sx * sy), 3)
            o = {"n_ev": len(ev_c), "n_eva": len(eva_c), "n_ikisi": len(ikisi),
                 "r_ev": _r(ev_c), "r_eva": _r(eva_c), "hazir": len(ikisi) >= 50,
                 "not": "n≥50 olunca scripts/cm_ev_calib.py tam raporu verir"}
        except Exception as e:
            o = {"hata": type(e).__name__}
        self._evc_cache = (now, o)
        return o

    def _news_impact_report(self, ttl: float = 600.0) -> Dict:
        """Hangi haber TÜRÜ hangi paritede ne kadar hareket yaptırıyor. TTL'li önbellek."""
        now = time.time()
        hit = getattr(self, "_ni_cache", None)
        if hit and now - hit[0] < ttl:
            return hit[1]
        try:
            o = self.haber.ozet("4h")
            o["prior_karsilastirma"] = self.haber.prior_karsilastir("4h")
        except Exception as e:
            o = {"hata": type(e).__name__}
        self._ni_cache = (now, o)
        return o

    def _tca_report(self, ttl: float = 120.0) -> Dict:
        """ÖLÇÜLEN yürütme maliyeti vs varsayılan. fills.jsonl büyüdüğü için TTL'li önbellek."""
        now = time.time()
        hit = getattr(self, "_tca_cache", None)
        if hit and now - hit[0] < ttl:
            return hit[1]
        try:
            rep = TCA.tca_report(output_dir=self.output_dir,
                                 assumed_cost_bps=float(self.venue.taker_bps + 2.0))
        except Exception as e:
            rep = {"available": False, "reason": f"{type(e).__name__}"}
        self._tca_cache = (now, rep)
        return rep

    def _update_portfolio_mode(self, prices: Dict[str, float], now: float, stale_share: float = 0.0) -> None:
        slow_map = {}
        for sym in self.cfg.symbols:
            s = self.ctx.slow_for(sym)
            if s and s.get("signal"):
                slow_map[sym] = s
        mn = self.ctx.market() or {}
        prev_level = int(self.risk.get("level", 0))
        self.risk = CM.market_risk(slow_map, mn)
        tf = f"{self.params.bar_minutes}m"
        cross = self.store.cross_section(self.cfg.exchange_id, self.cfg.symbols, tf)
        br = PM.breadth(cross)
        corr = PM.correlation_shock(self.store.returns_matrix(self.cfg.exchange_id, self.cfg.symbols[:20], tf))
        h = self.ctx.health() or {}
        g = self.guard.state
        dd = (1 - self.equity() / g.peak_equity) * 100.0 if g.peak_equity else 0.0
        prev_mode = self.portfolio.get("mode")
        proposed = PM.decide_mode(br, corr, self.risk.get("btc_regime"), int(mn.get("level") or 0),
                                  dd, str(h.get("overall") or "UNKNOWN"), stale_share,
                                  self.cfg.max_drawdown_pct * 100.0)
        # GÜN-İÇİ TEPE GERİ-VERME (portföy düzeyi yarı-tepe): gün kazancı ≥ %0,25 sermayeye ulaşıp %60'ı geri
        # verildiyse gün sonuna dek SAVUNMA (canlı: 19:22'de +3,99 $ → gece +0,0 $; kazanç ölçülmemiş sleeve'lerle eridi)
        eq_now = self.equity(); day0 = float(g.day_start_equity or self.cfg.capital_usdt)
        self._day_peak = max(getattr(self, "_day_peak", day0), eq_now) if getattr(self, "_day_peak_day", None) == self.day else eq_now
        self._day_peak_day = self.day
        gain = self._day_peak - day0
        if gain >= 0.0025 * self.cfg.capital_usdt and eq_now <= day0 + 0.4 * gain and proposed["mode"] in (PM.RISK_ON, PM.SELECTIVE):
            proposed = {**proposed, "mode": PM.DEFENSIVE, "label": PM.LABEL_TR[PM.DEFENSIVE],
                        "reasons": [f"gün-içi tepe geri-verme: tepe +{gain:.2f} $, şimdi +{eq_now - day0:.2f} $ (≥ %60 geri verildi)"] + list(proposed.get("reasons") or []),
                        "actions": {**proposed["actions"], "new_entries": False, "size_mult": 0.5, "tighten_stops": True, "flatten": False}}
        # HİSTEREZİS: mod ancak 3 ardışık döngüde aynı kararı verirse değişir (dakikada bir zıplama = kaçırılan giriş);
        # CASH güvenlik modu anında geçer.
        if proposed["mode"] == PM.CASH or prev_mode is None:
            self.portfolio = proposed; self._pm_pending = None; self._pm_count = 0
        elif proposed["mode"] == prev_mode:
            self.portfolio = proposed; self._pm_pending = None; self._pm_count = 0
        else:
            if getattr(self, "_pm_pending", None) == proposed["mode"]:
                self._pm_count = getattr(self, "_pm_count", 0) + 1
            else:
                self._pm_pending, self._pm_count = proposed["mode"], 1
            if self._pm_count >= 3:
                self.portfolio = proposed; self._pm_pending = None; self._pm_count = 0
            else:
                keep = dict(self.portfolio)
                keep["pending_mode"] = proposed["mode"]; keep["pending_count"] = self._pm_count
                self.portfolio = keep
        lvl = int(self.risk["level"])
        if lvl != prev_level:
            self._log("system", f"{'🟢' if lvl == 0 else '🟡' if lvl == 1 else '🔴'} Piyasa riski: {self.risk['label']}"
                                + (" · " + "; ".join(self.risk["reasons"]) if self.risk["reasons"] else ""))
        if self.portfolio["mode"] != prev_mode:
            self._log("system", f"📊 Portföy modu: {self.portfolio['label']}"
                                + (" · " + "; ".join(self.portfolio["reasons"][:3]) if self.portfolio["reasons"] else ""))
            audit("PORTFOLIO_MODE", {"user": self.user_id, "exchange": self.cfg.exchange_id,
                                     "mode": self.portfolio["mode"], "reasons": self.portfolio["reasons"]}, self.output_dir)
        acts = self.portfolio.get("actions", {})
        if acts.get("flatten") or lvl >= 2:
            if not self.cash_mode:
                self.alerts.send("cash", "NAKİT MODU: " + "; ".join((self.portfolio.get("reasons") or self.risk.get("reasons") or [])[:3]), level="warning", force=True)
            self.cash_mode = True
            for sym, o in list(self.pending.items()):
                self.broker.cancel_order(o["order"]); self.pending.pop(sym, None)
            for sym, pos in list(self.positions.items()):
                self._close(pos, prices.get(sym) or pos.last_price or pos.entry, "NAKİT MODU", now)
        elif acts.get("tighten_stops"):
            # DÜZELTİLDİ (2026-09-06): stop `pos.entry`'ye çekiliyordu. Giriş fiyatı NET
            # başabaş değildir (gidiş-dönüş komisyon + kayma kadar eksik) — savunma modunda
            # "korunan" pozisyon küçük ZARARLA kapanıyordu. Artık kâr kilidi seviyesine
            # (tepe × retain, tabanı net başabaş) çekilir.
            xp_ = self.xparams
            for pos in self.positions.values():
                if pos.net_pct_now() <= 0:
                    continue
                t_ = pos.track()
                lvl = t_.lock_level(xp_) or t_.breakeven_plus()
                if (lvl - pos.hard_stop) * pos.sign() > 0:
                    pos.stop = pos.hard_stop = float(lvl)
                    pos.lock_price = float(lvl)
                    pos.be_locked = True
                    pos.locked_net_pct = round(t_.net_pct(float(lvl)), 4)
        if self.cash_mode and lvl == 0 and self.portfolio["mode"] in (PM.RISK_ON, PM.SELECTIVE):
            self.cash_mode = False
            self._log("system", "🟢 Nakit modundan çıkıldı — girişler yeniden açık")

    # ------------------------------------------------------------ giriş (komite)
    def _try_entry_committee(self, sym: str, df: pd.DataFrame, price: float, health: Optional[Dict], now: float) -> None:
        h = health or {}
        overall = str(h.get("overall") or "UNKNOWN").upper()
        if overall in ("RED", "UNKNOWN"):
            self.last_decisions[sym] = {"symbol": sym, "ts": now, "result": f"VETO: SAĞLIK {overall}",
                                        "vetoes": [f"SAĞLIK {overall}"]}
            px = self._proxy_trigger(sym)                    # sağlık kapısı da kör nokta: gölgeye al
            if px:
                self.missed.on_unevaluated(sym, "SAĞLIK", px["cheap"], px["price"], "LONG", px["stop_pct"], px["target_pct"],
                                           float(self.params.max_hold_sec), now, info={"slow_ctx": bool(self.ctx.slow_for(sym))},
                                           context=self._blind_context(), detail=f"SAĞLIK {overall}: komite çalıştırılmadı (vekil tetikleyici {px['kind']})")
            return
        slow = self._slow_ctx(sym)
        book = self._book(sym)
        if CVD_ENABLED:
            try:
                book = {**book, **SF.cvd_from_trades(self.broker.fetch_trades(sym, 100), now * 1000.0)}
            except Exception:
                pass
        p = self.params
        eq = self.equity()
        open_notional = sum(x.notional for x in self.positions.values())
        g = self.guard.state
        cell = self.ctx.cell_for(sym, "1h", "LONG")
        # p_win öncülü: nitelendirme modelinin p'si yalnız hücre QUALIFIED ise (ölçülmüş yönsel edge); aksi hâlde 0,5.
        # (27 parite × 540 kombinasyonda sıfır QUALIFIED: model oynaklığı tahmin ediyor, yönü değil — canlıda
        # p=0,3'lük öncül SUI'de 99 al / 21 sat göstergeye rağmen EV'yi −%0,72'ye çekip vetoluyordu.)
        prior = (float(cell["p_model_live"]) if cell and str(cell.get("status", "")).upper() == "QUALIFIED"
                 and isinstance(cell.get("p_model_live"), (int, float)) else None)
        p_win = self.lessons.p_win(prior=prior if prior is not None else 0.5)
        fee = self._fees(sym)                        # parite BAZINDA oran (symbols[0] değil)
        size_mode = float(self.portfolio.get("actions", {}).get("size_mult", 1.0))
        ctx = {
            "symbol": sym, "price": price, "df": df, "slow": slow, "qual_cell": cell,
            "book": book, "fees": {"maker_bps": fee["maker_bps"], "taker_bps": fee["taker_bps"], "verified": fee.get("verified")},
            "open_positions": {s: {"direction": x.direction} for s, x in self.positions.items()},
            "max_open": self.cfg.max_open,
            "exposure_room": max(0.0, self.cfg.max_exposure_pct / 100.0 * eq - open_notional),
            "capital": self.cfg.capital_usdt, "max_order": self.cfg.max_order_usdt,
            "notional_fn": lambda stop_pct: self._size(stop_pct, 1.0),
            "p_win": p_win, "halted": bool(g.halted), "paused_reason": self.lessons.paused_reason(sym, now),
            "daily_loss_left_pct": (round((self.cfg.daily_loss_limit_pct * 100.0) + ((eq / g.day_start_equity - 1) * 100.0), 3)
                                    if g.day_start_equity else None),
            "market_type": self.cfg.market_type, "news": self.ctx.news(sym),
            "rs_rank": self._rs_ranks.get(sym),
            "reliable_only": bool(self.portfolio.get("actions", {}).get("reliable_sleeves_only")),
            "lifecycle": self.lifecycle, "mode": self.cfg.mode,
        }
        learned = {**self.lessons.learned(), "sleeve_reliability": self.allocator.sleeve_reliability(),
                   "paused_sleeves": list(self.allocator.paused_sleeves(now)),
                   "sleeve_states": self.allocator.sleeve_states(now=now), "mfe_by_sleeve": self._mfe_by_sleeve()}
        verdict = CM.evaluate(ctx, p, learned)
        trace = verdict.to_dict()
        trace["ts"] = now
        trace["tier"] = slow.get("tier", "light" if slow else "none")
        trace["freshness"] = self.store.freshness(self.cfg.exchange_id, sym, f"{p.bar_minutes}m").get("state")
        trace["cvd"] = {k: book.get(k) for k in ("cvd_ratio", "cvd_n", "cvd_burst") if k in book}
        info_flags = {"slow_ctx": slow.get("tier") == "heavy", "tier": slow.get("tier", "none"), "news": bool(ctx.get("news")),
                      "book_depth": min(float(book.get("bid_depth_usd") or 0.0), float(book.get("ask_depth_usd") or 0.0)) > 0.0,
                      "fees_verified": bool(fee.get("verified")), "qual_cell": cell is not None}
        nw = ctx.get("news") or {}
        if nw:
            trace["news"] = {"score": nw.get("score"), "n": nw.get("n_items"), "confirmed": nw.get("confirmed"),
                             "catalysts": list(nw.get("catalysts") or {}), "risks": list(nw.get("risks") or {}),
                             "event": nw.get("top_event")}
        if self.challenger.params:
            try:
                ch_p = CM.CommitteeParams.from_dict({**self.cfg.params, **self.challenger.params})
                ch_v = CM.evaluate(ctx, ch_p, self.lessons.learned())
                self.challenger.record(sym, verdict.allowed, ch_v.allowed, ch_v.plan, now)
                if ch_v.allowed and not verdict.allowed and ch_v.plan:
                    self.lessons.on_candidate_vetoed({**ch_v.to_dict(), "vetoes": ["CHALLENGER"]}, float(p.max_hold_sec), now)
            except Exception:
                pass
        if overall == "DEGRADED" and verdict.allowed:
            verdict.size_mult *= 0.5
            trace["notes"].append("sistem sağlığı DEGRADED → boyut ×0,5")
        if not verdict.allowed:
            self.last_decisions[sym] = trace
            for t in (trace.get("silenced") or [])[:2]:
                pl = t.get("plan") or {}
                if pl.get("stop_pct") and pl.get("target_pct"):
                    g_ = t.get("gate") or "REJİM_SEÇİCİ"
                    self.missed.on_unevaluated(sym, g_, {**(trace.get("fast") or {}), "kind": t["kind"]}, price, t["direction"],
                                               float(pl["stop_pct"]), float(pl["target_pct"]), float(p.max_hold_sec), now,
                                               info=info_flags, context=self._blind_context(),
                                               detail=(f"{g_}: '{t['kind']}' " + ("devre kesici ile duraklatıldı" if g_ == "SLEEVE_DURAKLATILDI"
                                                                 else "gölge aşamasında (kanıt yok, emir yok)" if g_ == "YAŞAM_DÖNGÜSÜ"
                                                                 else f"{trace.get('regime')} rejiminde kapalı") + f" — {t.get('note')}"))
            if verdict.trigger and verdict.plan and verdict.direction:
                self.lessons.on_candidate_vetoed(trace, float(p.max_hold_sec), now)
                self.missed.on_vetoed(trace, info_flags, book, ctx.get("news"), float(p.max_hold_sec), now, self._blind_context())
            return
        same = sum(1 for x in self.positions.values() if (x.sleeve or x.trigger) == verdict.trigger) + \
            sum(1 for o in self.pending.values() if (o.get("verdict") or {}).get("trigger") == verdict.trigger)
        if same >= self.cfg.max_open_per_sleeve:
            trace["result"] = f"SLEEVE TAVANI: {verdict.trigger} zaten {same} açık (çeşitlendirme)"
            self.last_decisions[sym] = trace
            pl = verdict.plan
            self.missed.on_unevaluated(sym, "SLEEVE_TAVANI", trace.get("fast") or {}, price, verdict.direction, float(pl["stop_pct"]),
                                       float(pl["target_pct"]), float(p.max_hold_sec), now, info=info_flags, context=self._blind_context(),
                                       detail=trace["result"])
            return
        # ── YENİDEN GİRİŞ / SALINIM-MALİYET KAPISI ────────────────────────────────
        # Beklenen salınım için planın hedefi DEĞİL, ÖLÇÜLMÜŞ ulaşılabilir hedef kullanılır
        # (85 canlı işlemin 1'i sabit hedefe ulaştı; plan hedefi iyimserdir).
        tk = trace.get("ticket") or {}
        swing = tk.get("achievable_target_pct")
        if swing is None:
            swing = (verdict.plan or {}).get("target_pct")
        cost_here = float(tk.get("fee_pct_roundtrip") or trace.get("cost_pct") or self._cost_pct_est())
        rg = self._reentry_gate(sym, verdict.direction, (None if swing is None else float(swing)),
                                cost_here, now)
        trace["reentry"] = rg
        if not rg["allowed"]:
            trace["result"] = f"{rg.get('gate')}: {rg['reason']}"
            self.last_decisions[sym] = trace
            pl = verdict.plan
            self.missed.on_unevaluated(sym, str(rg.get("gate") or "YENİDEN_GİRİŞ"), trace.get("fast") or {}, price,
                                       verdict.direction, float(pl["stop_pct"]), float(pl["target_pct"]),
                                       float(p.max_hold_sec), now, info=info_flags, context=self._blind_context(),
                                       detail=trace["result"])
            return
        corr = self._corr_with_open(sym)
        if corr is not None and corr >= 0.7:
            verdict.size_mult *= 0.5
            trace["notes"].append(f"korelasyon bütçesi: açık pozisyonla |ρ| {corr:.2f} ≥ 0,7 → boyut ×0,5")
        trace["corr_with_open"] = (None if corr is None else round(corr, 3))
        w_alloc = self.allocator.weight(verdict.trigger or "", trace.get("regime"))
        if abs(w_alloc - 1.0) > 1e-9:
            verdict.size_mult *= w_alloc
            trace["notes"].append(f"meta-tahsisçi: {verdict.trigger} güvenilirliği ölçüldü → boyut ×{w_alloc:.2f}")
        # ── KANIT KAPILARI (boyut kanıtı izler; kanıtsız kenar tam boyutla oynanmaz) ──
        # 1) seans kapısı: bu 4 sa UTC bloğunun ölçülmüş beklentisi (son 14 gün) negatif ve anlamlıysa ×0,5 / giriş yok
        sg = self._session_gate(now)
        blk = self._session_block(now)
        s_mult = float((sg.get(blk) or {}).get("mult", 1.0))
        if s_mult <= 0.0:
            trace["result"] = f"SEANS KAPISI: {blk} UTC bloğu ölçülmüş beklenti %{sg[blk]['mean_net_pct']:+.3f} (n {sg[blk]['n']}, t {sg[blk]['t_stat']}) → giriş yok"
            self.last_decisions[sym] = trace
            pl = verdict.plan
            self.missed.on_unevaluated(sym, "SEANS", trace.get("fast") or {}, price, verdict.direction, float(pl["stop_pct"]),
                                       float(pl["target_pct"]), float(p.max_hold_sec), now, info=info_flags, context=self._blind_context(),
                                       detail=trace["result"])
            return
        if s_mult < 1.0:
            verdict.size_mult *= s_mult
            trace["notes"].append(f"seans kapısı: {blk} UTC bloğu beklenti %{sg[blk]['mean_net_pct']:+.3f} (n {sg[blk]['n']}, t {sg[blk]['t_stat']}) → boyut ×{s_mult:.1f}")
        # 2) drawdown yarı-boyut: son N kapanan işlemin net'i < 0 → ×0,5
        dr = self._derisk()
        if dr["active"]:
            verdict.size_mult *= dr["mult"]
            trace["notes"].append(f"drawdown yarı-boyut: son {dr['trailing_n']} işlem net {dr['trailing_net']:+.2f} $ → boyut ×{dr['mult']:.1f}")
        notional = self._size(verdict.plan["stop_pct"], verdict.size_mult * size_mode)
        # 3) sleeve kanıt tavanı: PROVEN değilse kanıt boyutu (probe), PAUSED ise giriş yok
        cap = self.allocator.notional_cap(verdict.trigger or "", float(getattr(p, "probe_notional_usdt", 25.0)),
                                          float(self.cfg.max_order_usdt), now)
        trace["evidence_gate"] = {k: cap[k] for k in ("state", "cap_usdt", "n", "mean_net_pct", "t_stat", "label")}
        if cap["cap_usdt"] <= 0.0:
            trace["result"] = f"SLEEVE DURAKLATILDI: {verdict.trigger} ({cap['label']})"
            self.last_decisions[sym] = trace
            return
        if notional > cap["cap_usdt"]:
            trace["notes"].append(f"kanıt tavanı: {verdict.trigger} {cap['state']} (n {cap['n']}, ort net %{cap['mean_net_pct']:+.3f}, t {cap['t_stat']}) "
                                  f"→ boyut {notional:.0f} $ → {cap['cap_usdt']:.0f} $")
            notional = float(cap["cap_usdt"])
        notional = self._floor_notional(sym, notional, trace)
        if not self._notional_ok(sym, notional, trace):
            return
        if self.cfg.mode == "live" and not self._live_preflight(trace):
            self.last_decisions[sym] = trace
            return
        # Venue kıyası OPT-IN: her borsa istemcisi market tablosunu yükler (~80 MB/borsa) — canlıda ilk emirde
        # RSS 1078→1389 MB (pm2 tavanı 1400). Simülatör tek borsada (MEXC) çalışır; kıyas bilgi amaçlıdır.
        if VENUE_COMPARE and self.resource.get("state") == "GREEN" and now - self._venue_compare_ts.get(sym, 0.0) >= 1800.0:
            self._venue_compare_ts[sym] = now
            self.venue_compare[sym] = self._compare_venues(sym, notional)
        plan = verdict.plan
        entry = trace.get("entry") or {}
        opt = entry.get("optimal") or {}
        wait_bars = min(int(p.maker_wait_bars), int(SF.SLEEVE_URGENCY.get(verdict.trigger or "", p.maker_wait_bars)))
        if wait_bars < p.maker_wait_bars:
            trace["notes"].append(f"aciliyet: {verdict.trigger} maker en çok {wait_bars} bar bekler" + (" → anında taker" if wait_bars == 0 else ""))
        # kanıt kapısı: komite taker'ı MAKER'a çevirdiyse (sleeve PROVEN değil) en az 1 bar maker beklenir, kovalanmaz
        proof_maker = (verdict.order_type == "maker" and opt.get("order_type") == "taker")
        if proof_maker and wait_bars == 0:
            wait_bars = 1
        if verdict.order_type == "maker" and wait_bars > 0:
            s_ = 1.0 if plan["direction"] == "LONG" else -1.0
            if opt.get("order_type") == "maker":
                lim = float(opt["price"])
            else:                                            # en iyi teklif (LONG: bid, SHORT: ask); defter yoksa fiyat
                lim = float((book.get("bid") if s_ > 0 else book.get("ask")) or price)
            side = "buy" if plan["direction"] == "LONG" else "sell"
            cid = make_client_order_id(self.user_id, self.cfg.exchange_id, sym, self.cycle, side + "m")
            order = self.broker.limit_order(sym, side, notional, lim, cid)
            self.pending[sym] = {"order": order, "plan": plan, "verdict": trace, "created_cycle": self.cycle,
                                 "created_ts": now, "bars": 0, "notional": notional, "size_mult": verdict.size_mult, "max_bars": wait_bars,
                                 "no_chase": bool(proof_maker)}
            trace["result"] = f"MAKER BEKLİYOR {plan['direction']} {notional:.2f} USDT @ {lim:.6g} (≤{p.maker_wait_bars} bar)"
            self._log("entry", f"📝 {sym} maker limit {side} {notional:.2f} USDT @ {lim:.6g} · {verdict.trigger}")
        else:
            self._open_taker(sym, plan, notional, trace, now, verdict.trigger or "", verdict.template,
                             trace.get("exit_mode") or XE.PARTIAL_AND_RUN)
            trace["result"] = f"AÇILDI {plan['direction']} {notional:.2f} USDT @ {price} (taker)"
        self.last_decisions[sym] = trace

    def _probe_candidate(self, sym: str, df: pd.DataFrame, price: float, health: Optional[Dict], now: float) -> Optional[Dict]:
        """Tavan doluyken adayı EMİR VERMEDEN değerlendir (rotasyon kıyası için)."""
        if sym in self.positions or sym in self.pending:
            return None
        h = health or {}
        if str(h.get("overall") or "UNKNOWN").upper() in ("RED", "UNKNOWN"):
            return None
        slow = self._slow_ctx(sym); book = self._book(sym); p = self.params
        fee = self._fees(sym)                        # parite BAZINDA oran
        ctx = {"symbol": sym, "price": price, "df": df, "slow": slow, "qual_cell": None, "book": book,
               "fees": {"maker_bps": fee["maker_bps"], "taker_bps": fee["taker_bps"], "verified": fee.get("verified")},
               "open_positions": {s_: {"direction": x.direction} for s_, x in self.positions.items()},
               "max_open": self.cfg.max_open + 1, "exposure_room": max(10.0, self.cfg.max_order_usdt),
               "capital": self.cfg.capital_usdt, "max_order": self.cfg.max_order_usdt,
               "notional_fn": lambda stop_pct: min(self.cfg.max_order_usdt, self.cfg.capital_usdt * self.cfg.risk_per_trade_pct / 100.0 / max(1e-6, stop_pct / 100.0)),
               "p_win": self.lessons.p_win(prior=0.5), "halted": False, "paused_reason": self.lessons.paused_reason(sym, now),
               "daily_loss_left_pct": None, "market_type": self.cfg.market_type, "news": self.ctx.news(sym),
               "rs_rank": self._rs_ranks.get(sym), "reliable_only": False, "lifecycle": self.lifecycle, "mode": self.cfg.mode}
        learned = {**self.lessons.learned(), "sleeve_reliability": self.allocator.sleeve_reliability(),
                   "paused_sleeves": list(self.allocator.paused_sleeves(now)),
                   "sleeve_states": self.allocator.sleeve_states(now=now), "mfe_by_sleeve": self._mfe_by_sleeve()}
        v = CM.evaluate(ctx, p, learned)
        return v.to_dict()

    def _compare_venues(self, sym: str, notional: float) -> Dict:
        def book_fn(venue: str, s: str):
            if venue == self.cfg.exchange_id:
                return self.store.get_book(venue, s)
            try:
                if self._public_broker is None:
                    return None
                return self._public_broker(venue).fetch_book_top(s)
            except Exception:
                return None
        try:
            return VR.compare(sym, COMPARE_VENUES, book_fn, VR.static_fee, notional,
                              float(self.lessons.learned().get("p_maker_fill", 0.5)),
                              health_fn=lambda v: self.store.rate.state(v))
        except Exception as e:
            return {"symbol": sym, "rows": [], "best": None, "note": f"kıyas yapılamadı: {type(e).__name__}"}

    def _check_pending(self, sym: str, pend: Dict, df: pd.DataFrame, price: float, now: float) -> None:
        order = pend["order"]
        low, high = float(df["low"].iloc[-1]), float(df["high"].iloc[-1])
        fill = self.broker.paper_limit_fill(order, low, high) if self.cfg.mode == "paper" \
            else (lambda o: o if o.get("status") == "closed" else None)(self.broker.order_status(order))
        vd = pend["verdict"] or {}
        if fill:
            self.pending.pop(sym, None)
            self.lessons.on_maker_attempt(True)
            self._create_position(sym, pend["plan"], fill, now, vd, "maker", vd.get("trigger") or "",
                                  vd.get("template") or "", vd.get("exit_mode") or XE.PARTIAL_AND_RUN)
            self.last_decisions[sym] = {**vd, "result": f"AÇILDI (maker doldu) @ {fill['avg_price']:.6g}"}
            return
        pend["bars"] += 1
        if pend["bars"] < int(pend.get("max_bars") or self.params.maker_wait_bars):
            return
        self.broker.cancel_order(order)
        self.pending.pop(sym, None)
        self._skip_entry_cycle[sym] = self.cycle
        plan = pend["plan"]
        s = 1.0 if plan["direction"] == "LONG" else -1.0
        drift_pct = (price / plan["entry"] - 1.0) * 100.0 * s
        chase = (vd.get("entry") or {}).get("max_chase")
        if chase is not None and (price - float(chase)) * s > 0:
            self.lessons.on_maker_attempt(False)
            self.last_decisions[sym] = {**vd, "result": f"MAKER DOLMADI · fiyat MAX CHASE'i ({float(chase):.6g}) aştı → giriş geçersiz"}
            self._log("system", f"↩️ {sym} maker dolmadı; max chase aşıldı, kovalanmadı")
            self.missed.on_execution_miss(vd, plan, "MAX_CHASE", f"maker {plan['entry']:.6g} dolmadı, fiyat max chase {float(chase):.6g} üstüne kaçtı",
                                          float(self.params.max_hold_sec), now)
            return
        cost_taker = (2.0 * self.venue.taker_bps + 4.0) / 100.0
        ratio = (plan["target_pct"] - drift_pct) / max(1e-9, cost_taker)
        if pend.get("no_chase"):
            self.lessons.on_maker_attempt(False)
            self.last_decisions[sym] = {**vd, "result": f"MAKER DOLMADI · kanıt kapısı: sleeve kanıtlanmadan taker'a kovalanmaz (kayma %{drift_pct:+.2f})"}
            self._log("system", f"↩️ {sym} maker dolmadı; kanıt kapısı → taker'a kovalanmadı")
            self.missed.on_execution_miss(vd, plan, "MAKER_DOLMADI", f"maker {plan['entry']:.6g} dolmadı; kanıt kapısı (taker yok), kayma %{drift_pct:+.2f}",
                                          float(self.params.max_hold_sec), now)
            return
        if drift_pct < 0.5 * plan["stop_pct"] and ratio >= self.params.chase_taker_ratio:
            self.lessons.on_maker_attempt(False, chased=True)
            plan2 = {**plan, "entry": price, "stop": price * (1 - s * plan["stop_pct"] / 100),
                     "target": price * (1 + s * plan["target_pct"] / 100)}
            self._open_taker(sym, plan2, pend["notional"], vd, now, vd.get("trigger") or "", vd.get("template") or "",
                             vd.get("exit_mode") or XE.PARTIAL_AND_RUN)
            self.last_decisions[sym] = {**vd, "result": f"AÇILDI (maker dolmadı → taker, edge yeniden hesaplandı) @ {price:.6g}"}
        else:
            self.lessons.on_maker_attempt(False)
            self.last_decisions[sym] = {**vd, "result": f"MAKER DOLMADI · kayma %{drift_pct:+.2f} · brüt/maliyet {ratio:.1f} → vazgeçildi"}
            self._log("system", f"↩️ {sym} maker dolmadı, vazgeçildi (kayma %{drift_pct:+.2f})")
            self.missed.on_execution_miss(vd, plan, "MAKER_DOLMADI", f"maker {plan['entry']:.6g} dolmadı; kayma %{drift_pct:+.2f}, brüt/maliyet {ratio:.1f} < {self.params.chase_taker_ratio}",
                                          float(self.params.max_hold_sec), now)

    def _open_taker(self, sym: str, plan: Dict, notional: float, decision: Dict, now: float,
                    trigger: str, template: str, exit_mode: str) -> None:
        side = "buy" if plan["direction"] == "LONG" else "sell"
        cid = make_client_order_id(self.user_id, self.cfg.exchange_id, sym, self.cycle, side)
        o = self.broker.market_order(sym, side, notional, cid, ref_price=plan["entry"])
        self._create_position(sym, plan, o, now, decision, "taker", trigger, template, exit_mode)

    def _create_position(self, sym: str, plan: Dict, o: Dict, now: float, decision: Dict,
                         order_type: str, trigger: str, template: str, exit_mode: str) -> None:
        pos = Position(symbol=sym, direction=plan["direction"], entry=float(o["avg_price"]),
                       stop=plan["stop"], target=plan["target"], stop_pct=plan["stop_pct"], target_pct=plan["target_pct"],
                       amount=float(o["amount"]), notional=float(o["filled_usdt"]), opened_ts=now,
                       entry_fee=float(o.get("fee_usdt") or 0.0), last_price=float(o["avg_price"]),
                       client_id=str(o.get("client_id") or ""), order_id=str(o.get("id") or ""),
                       decision=decision or {}, mode=self.cfg.mode, order_type=order_type, trigger=trigger,
                       template=template, partial_fraction=float(plan.get("partial_fraction") or 0.0),
                       amount_initial=float(o["amount"]), exit_mode=exit_mode, sleeve=trigger)
        s = pos.sign()
        pos.stop = pos.entry * (1.0 - s * pos.stop_pct / 100.0)
        pos.hard_stop = pos.stop
        pos.target = pos.entry * (1.0 + s * pos.target_pct / 100.0)
        pos.highest_high = pos.lowest_low = pos.entry
        pos.atr_pct = float((decision or {}).get("atr_hint") or ((decision or {}).get("fast") or {}).get("atr_pct") or 0.3)
        fee_in = (pos.entry_fee / max(1e-9, pos.notional)) * 100.0
        exit_cost = (self.venue.taker_bps + 2.0) / 100.0
        pos.cost_pct_roundtrip = round(fee_in + exit_cost, 4)
        pos.time_stop_sec = int(SF.SLEEVE_TIME_STOP_MIN.get(trigger, max(1, self.params.max_hold_sec // 60)) * 60)
        if pos.partial_fraction > 0 and exit_mode == XE.PARTIAL_AND_RUN:
            pr = float(getattr(self.params, "partial_tp_r", 1.0))
            pos.partial_tp = pos.entry * (1.0 + s * pr * pos.stop_pct / 100.0)
            near = plan.get("partial_tp_near")
            if near and (float(near) - pos.entry) * s / pos.entry * 100.0 >= 1.5 * pos.cost_pct_roundtrip:
                pos.partial_tp = float(near)            # yakın yapısal seviye = kısmi kâr (karşı-olgusal kanıt)
        # KÂR MERDİVENİ (T1…Tn) — R katları, yapısal seviyeler varsa onlara oturtulur.
        # Bu, "altı tepeyi önceden bilmek" değil; ölçekli çıkış basamaklarıdır. Her basamak
        # yalnız NET kârı maliyetin katını aşarsa kurulur (komisyon için işlem yapılmaz).
        pos.ladder = self._build_ladder(pos, plan, decision)
        prev_exit = self.exit_state.get(sym) or {}
        if str(prev_exit.get("direction") or "") == pos.direction:
            pos.reentry_count = int(prev_exit.get("count") or 0)
            pos.parent_exit_ts = float(prev_exit.get("ts") or 0.0)
        self._tca(sym, "buy" if pos.direction == "LONG" else "sell", pos.amount,
                  float(plan.get("entry") or pos.entry), pos.entry, order_type, pos.entry_fee, pos.amount)
        self.positions[sym] = pos
        self.fees_paid += pos.entry_fee
        self.day_trades += 1
        self.alerts.send(f"open:{sym}", f"AÇILDI {sym} {pos.direction} {pos.notional:.0f} $ @ {pos.entry:.6g} · stop %{pos.stop_pct} · hedef %{pos.target_pct} · {trigger}")
        self._log("entry", f"✅ {sym} {pos.direction} AÇILDI · {pos.notional:.2f} USDT @ {pos.entry:.6g} · "
                           f"stop %{pos.stop_pct} · hedef %{pos.target_pct} · {order_type} · {trigger or '-'} · çıkış {exit_mode}")
        audit("ORDER_OPEN", {"user": self.user_id, "exchange": self.cfg.exchange_id, "mode": self.cfg.mode,
                             "symbol": sym, "side": pos.direction, "notional": pos.notional, "client_id": pos.client_id,
                             "order_id": pos.order_id, "order_type": order_type, "sleeve": trigger, "exit_mode": exit_mode},
              self.output_dir)

    def _notional_ok(self, sym: str, notional: float, trace: Dict) -> bool:
        rules = self.broker.market_rules(sym)
        if notional < rules["min_notional"]:
            trace["result"] = f"BOYUT {notional:.2f} < asgari {rules['min_notional']} USDT (maruziyet/tavan)"
            self.last_decisions[sym] = trace
            return False
        return True

    def _floor_notional(self, sym: str, notional: float, trace: Dict) -> float:
        """Yığılan haircut'lar (sleeve ×0,6 · inceleme ×0,6 · portföy ×0,7 · rol çarpanları) boyutu borsa
        asgarisinin altına düşürebilir; karar AÇ ise ve oda/nakit varsa asgariye TAMAMLANIR (kanıt: TIA 4,76 $)."""
        rules = self.broker.market_rules(sym)
        mn = float(rules["min_notional"])
        if 0.0 < notional < mn:
            eq = self.equity()
            open_notional = sum(p.notional for p in self.positions.values()) + sum(p["notional"] for p in self.pending.values())
            room = max(0.0, self.cfg.max_exposure_pct / 100.0 * eq - open_notional)
            cash = self.broker.paper_cash if self.cfg.mode == "paper" else room
            if room >= mn and cash >= mn:
                trace["notes"].append(f"boyut {notional:.2f} $ → asgari {mn:.0f} $'a tamamlandı (yığılan çarpanlar)")
                return mn
        return notional

    def _live_preflight(self, trace: Dict) -> bool:
        le = live_enabled(self.server_config) if self.server_config is not None else \
            {"live": False, "missing": ["sunucu konfigürasyonu yok"]}
        if not le["live"]:
            trace["result"] = "CANLI KAPALI: " + ", ".join(le["missing"])
            return False
        if not self.reconcile_ok:
            trace["result"] = "MUTABAKAT SAPMASI — emir yok"
            return False
        return True

    def _size(self, stop_pct: float, mult: float) -> float:
        risk_usdt = self.cfg.capital_usdt * self.cfg.risk_per_trade_pct / 100.0
        notional = risk_usdt / max(1e-6, stop_pct / 100.0)
        eq = self.equity()
        open_notional = sum(p.notional for p in self.positions.values()) + sum(p["notional"] for p in self.pending.values())
        room = max(0.0, self.cfg.max_exposure_pct / 100.0 * eq - open_notional)
        notional = min(notional, self.cfg.max_order_usdt, room)
        if self.cfg.mode == "paper":
            notional = min(notional, max(0.0, self.broker.paper_cash))
        return float(max(0.0, notional * mult))

    def _book(self, sym: str) -> Dict:
        try:
            return self.store.get_book(self.cfg.exchange_id, sym)
        except Exception as e:
            # `ok: False` şart: sıfır spread "bedava" değil "ölçülmedi" demektir
            return {"spread_bps": 0.0, "bid_depth_usd": 0.0, "ask_depth_usd": 0.0,
                    "ok": False, "stale": True, "why": type(e).__name__}

    # ------------------------------------------------------------ çıkış
    def _reentry_gate(self, sym: str, direction: str, expected_swing_pct: Optional[float],
                      cost_pct: float, now: float, cont_prob: Optional[float] = None) -> Dict:
        """Yeniden giriş / maliyet kapısı. Aday hâlâ tüm komite kapılarından geçer;
        bu yalnız EK bir kısıt: kârla çıkılan harekete geri girişe izin verir,
        zararla çıkılan fikre hemen dönüşü ve komisyon-altı salınımları engeller."""
        try:
            d = RE.decide(self.exit_state, sym, direction, now, expected_swing_pct,
                          float(cost_pct or 0.0), self.rparams, cont_prob)
        except Exception:
            return {"allowed": True, "reason": "kapı hatası — geçirildi", "reentry_count": 0, "gate": None}
        if not d["allowed"]:
            self.reentry_blocks.append({"ts": now, "symbol": sym, "direction": direction,
                                        "gate": d.get("gate"), "reason": d.get("reason")})
        return d

    def _build_ladder(self, pos: Position, plan: Optional[Dict] = None,
                      decision: Optional[Dict] = None) -> List[Dict]:
        """Pozisyonun kâr merdivenini kur. Yapısal seviyeler (yakın direnç, swing high,
        planın hedefi) varsa basamaklar onlara oturtulur — böylece kâr, gerçek likidite
        bölgelerinde alınır, keyfî R noktalarında değil."""
        struct: List[float] = []
        for v in ((plan or {}).get("partial_tp_near"), (plan or {}).get("target")):
            if v:
                struct.append(float(v))
        fast = (decision or {}).get("fast") or {}
        lv = (decision or {}).get("levels") or {}
        keys = ("resistance", "swing_high", "prior_swing_high") if pos.direction == "LONG" \
            else ("support", "swing_low", "prior_swing_low")
        for k in keys:
            v = fast.get(k, lv.get(k))
            if isinstance(v, (int, float)) and v > 0:
                struct.append(float(v))
        xp = XE.ExitParams(**{**self.xparams.__dict__,
                              "time_stop_sec": pos.time_stop_sec or self.xparams.time_stop_sec})
        t = pos.track()
        lad = t.build_ladder(xp, struct)
        return self._fit_ladder_to_size(pos, lad)

    def _fit_ladder_to_size(self, pos: Position, ladder: List[Dict]) -> List[Dict]:
        """Merdiveni EMİR BOYUTUNA uydur.

        ÖLÇÜLEN KISIT: kâğıt modda `min_notional` 10 $ (muhafazakâr varsayılan) ve
        canlı kâğıt koşumunda `max_order_usdt` da 10 $. Yani 4 basamaklı bir merdivenin
        ilk dilimi 2,50 $ olur ve borsa asgarisinin ALTINDA kalır — hiçbir basamak
        ateşlenemez. (Aynı sebeple eski tek `partial_tp` de 200 işlemin yalnız 7'sinde
        çalışabilmişti.) Sessizce çalışmayan bir özellik, olmayan bir özellikten kötüdür:
        burada basamaklar BİRLEŞTİRİLİR; hiçbiri sığmıyorsa merdiven KURULMAZ ve pozisyon
        notuna sebebi yazılır — kâr kilidi zaten bağımsız çalışmaya devam eder."""
        if not ladder:
            return []
        try:
            mn = float(self.broker.market_rules(pos.symbol)["min_notional"])
        except Exception:
            mn = 10.0
        notional = float(pos.notional or 0.0)
        if notional <= 0 or mn <= 0:
            return ladder
        min_frac = mn / notional                       # bir dilimin taşıması gereken asgari pay
        out: List[Dict] = []
        carry = 0.0
        for lv in ladder:
            f = float(lv.get("frac") or 0.0) + carry
            if f < min_frac:                           # bu dilim tek başına asgariyi geçmiyor → sonrakine taşı
                carry = f
                continue
            # dilimi aldıktan sonra KALAN da asgariyi geçmeli, yoksa kalan kapatılamaz
            used = sum(x["frac"] for x in out) + f
            if (1.0 - used) < min_frac:
                carry = f
                continue
            out.append({**lv, "frac": round(f, 4)})
            carry = 0.0
        if not out and not getattr(self, "_ladder_warned", False):
            self._ladder_warned = True          # her pozisyonda değil, oturumda BİR kez
            self._log("system", f"ℹ️ Kâr merdiveni kurulamıyor: emir {notional:.2f} $ · borsa asgari emri "
                                f"{mn:.2f} $ — hiçbir dilim asgariyi geçmiyor. Kâr KİLİDİ (tepe×retain) "
                                f"bağımsız çalışmaya devam ediyor. Basamaklı kâr alımı için emir boyutu "
                                f"≥ {mn * 2:.0f} $ olmalı (kanıt tavanı bunu belirler).")
        return out

    def _manage_exit(self, pos: Position, px: float, hi: float, lo: float, now: float) -> None:
        s = pos.sign()
        # ESKİ tek kısmi TP — merdiven kapalıysa (veya basamak kurulamadıysa) geri düşüş yolu.
        # Bu bir kâr ALMA'dır (koruma değil): asgari tutmayı BEKLER. 37.905 eşleştirilmiş yolda
        # erken ölçekli çıkış beklentiyi düşürdü (−0,0015 puan, t = −4,79) — kazananı kırpıyor.
        if (not pos.ladder and not pos.partial_done and pos.partial_tp and pos.partial_fraction > 0
                and (px - pos.partial_tp) * s >= 0 and now - pos.opened_ts >= self.params.min_hold_sec):
            self._partial_close(pos, px, now, pos.partial_fraction, "PARTIAL_TP")
        if self.cfg.strategy == "committee" and self.cycle % 5 == 0:
            df_ = self.store._ohlcv.get((self.cfg.exchange_id, pos.symbol, f"{self.params.bar_minutes}m"), {}).get("df")
            self._refresh_position_edge(pos, df_, now)
        xp = XE.ExitParams(**{**self.xparams.__dict__, "time_stop_sec": pos.time_stop_sec or self.xparams.time_stop_sec})
        t = pos.track()
        dec = XE.decide_exit(t, px, hi, lo, xp, now, pos.cont_prob, pos.current_ev_pct)
        pos.absorb(t)
        if not dec:
            return
        if dec.get("partial"):
            # KÂR MERDİVENİ BASAMAĞI — pozisyon KAPANMAZ, ölçekli çıkış yapılır.
            # Dolum referansı: basamak fiyatı ile güncel fiyatın KÖTÜ olanı. Seviye bar
            # içinde görülmüş olsa bile "tam o seviyeden dolduk" varsaymak iyimserdir;
            # fiyat seviyenin altına döndüyse gerçekte oradan satılır.
            lvl_px = float(dec.get("exit_price") or px)
            fill_ref = min(lvl_px, px) if pos.direction == "LONG" else max(lvl_px, px)
            before = pos.amount
            self._partial_close(pos, fill_ref, now,
                                float(dec.get("fraction") or 0.0), "LADDER_TP", extra=dec)
            if pos.amount >= before - 1e-12:
                # Basamak "geçildi" sayıldı ama emir gönderilemedi (dilim borsa asgarisinin
                # altında). Kilit yine de yükseldi; sessiz kalmamak için kaydedilir.
                self._log("system", f"⚠️ {pos.symbol} T{pos.levels_hit} kısmi çıkışı yapılamadı "
                                    f"(dilim borsa asgarisinin altında) — kâr kilidi {pos.hard_stop:.6g} "
                                    f"seviyesinde yükseltildi, pozisyon tam boyutta koşuyor")
            # merdiven tamamen tükendiyse ve kalan miktar borsa asgarisinin altındaysa kapat
            if pos.amount * px < self.broker.market_rules(pos.symbol)["min_notional"]:
                self._close(pos, px, "LADDER_SON", now, extra=dec)
            return
        # Stop bar İÇİNDE delindiyse kapanış fiyatı değil STOP SEVİYESİ doldurulur;
        # aksi hâlde zarar "kapanışa kadar bekledik" varsayımıyla olduğundan küçük yazılır.
        self._close(pos, float(dec.get("exit_price") or px), dec["reason"], now, extra=dec)

    def _refresh_position_edge(self, pos: Position, df=None, now: Optional[float] = None) -> None:
        """Açık pozisyonun KALAN EV'si: devam olasılığı (trend/CVD/defter/eğim/RSI/kalan ufuk) × hedef mesafesi −
        (1−p) × stop mesafesi − çıkış maliyeti. Rotasyon ve edge-decay bunu kullanır."""
        try:
            now = time.time() if now is None else now
            p_ = None
            sig = self.ctx.signal_for(pos.symbol) or {}
            pu = (sig.get("forecast") or {}).get("prob_up")
            if isinstance(pu, (int, float)):
                p_ = float(pu) if pos.direction == "LONG" else 1.0 - float(pu)
            if df is not None and len(df) >= 60:
                f = CM.fast_features(df, self.params)
                if f.get("ok"):
                    f["price"] = float(df["close"].iloc[-1])
                    f = SF.extra_features(df, f, self._rs_ranks.get(pos.symbol), self._book(pos.symbol), self._slow_ctx(pos.symbol))
                    horizon = float(pos.time_stop_sec or 3600)
                    remaining = max(0.0, 1.0 - (now - pos.opened_ts) / horizon)
                    px = float(df["close"].iloc[-1]); s_ = pos.sign()
                    atr_abs = max(1e-12, float(f.get("atr_pct") or 0.3) / 100.0 * px)
                    dist_t = (pos.target - px) * s_ / atr_abs if pos.target else None
                    c = XE.continuation_probability(f, pos.direction, remaining, dist_t)
                    p_ = c["p"] if p_ is None else 0.5 * (p_ + c["p"])
                    dist_t_pct = ((pos.target - px) * s_ / px * 100.0) if pos.target else max(0.5, float(f.get("atr_pct") or 0.3) * 2)
                    dist_s_pct = max(0.0, (px - pos.hard_stop) * s_ / px * 100.0)
                    exit_cost = (self.venue.taker_bps + 2.0) / 100.0
                    pos.remaining_ev_pct = round(p_ * max(0.0, dist_t_pct) - (1.0 - p_) * dist_s_pct - exit_cost, 4)
                    if now - pos.opened_ts >= self.params.min_hold_sec:
                        pos.current_ev_pct = pos.remaining_ev_pct
            if p_ is not None:
                pos.cont_prob = float(p_)
        except Exception:
            pass

    def _maybe_rotate(self, ev_b_pct: Optional[float], sym_b: str, now: float) -> Optional[Position]:
        """Fırsat rotasyonu: EV_B − geçiş maliyeti − marj > kalan EV_A olan en zayıf açık pozisyonu kapat.
        Tepe koruması silahlı ve devam olasılığı ≥ 0,5 olan kazananlar rotasyona verilmez (trend sürdürülür)."""
        if ev_b_pct is None or not self.positions or getattr(self, "_rotations_today", 0) >= 6:
            return None
        switching = 2.0 * self._cost_pct_est() + float(getattr(self.params, "rotation_margin_pct", 0.15))
        cands = []
        for pos in self.positions.values():
            if now - pos.opened_ts < self.params.min_hold_sec or pos.remaining_ev_pct is None:
                continue
            if pos.armed and (pos.cont_prob or 0.0) >= 0.5:
                continue
            if float(ev_b_pct) - switching > float(pos.remaining_ev_pct):
                cands.append(pos)
        if not cands:
            return None
        victim = min(cands, key=lambda x: float(x.remaining_ev_pct))
        self._rotations_today = getattr(self, "_rotations_today", 0) + 1
        self._log("exit", f"🔁 ROTASYON: {victim.symbol} kalan EV %{victim.remaining_ev_pct:.3f} < {sym_b} EV %{float(ev_b_pct):.3f} − geçiş %{switching:.2f}")
        return victim

    def _partial_close(self, pos: Position, price: float, now: float, fraction: float,
                       reason: str = "PARTIAL_TP", extra: Optional[Dict] = None) -> None:
        """Ölçekli çıkış. Miktar BAŞLANGIÇ miktarının payı kadardır (merdiven basamakları
        birbirini yemesin diye), kalan miktarla sınırlıdır.

        DÜZELTİLEN HATA (2026-09-06): eski sürüm kısmi kârdan sonra stop'u `pos.entry`'ye
        çekiyordu. Giriş fiyatı NET başabaş DEĞİLDİR — giriş komisyonu + çıkış komisyonu +
        kayma kadar eksiktedir; o seviyeden kapanan koşucu küçük ZARARLA kapanıyordu.
        Yeni davranış: stop, çıkış motorunun KÂR KİLİDİ seviyesine (tepe × retain, tabanı
        net başabaş) çekilir."""
        fraction = float(max(0.0, min(1.0, fraction or 0.0)))
        base = pos.amount_initial or pos.amount
        amt = min(pos.amount, base * fraction)
        if amt <= 0 or pos.amount <= 0:
            return
        rules = self.broker.market_rules(pos.symbol)
        # Kalan parça borsa asgarisinin altında kalacaksa ölçekli çıkış yapılamaz:
        # "kapatamayacağın artık" bırakmak, sonraki çıkışı emir reddine düşürür.
        if (pos.amount - amt) * price < rules["min_notional"] or amt * price < rules["min_notional"]:
            return
        side = "sell" if pos.direction == "LONG" else "buy"
        cid = make_client_order_id(self.user_id, self.cfg.exchange_id, pos.symbol, self.cycle,
                                   side + "p" + str(pos.levels_hit))
        try:
            o = self.broker.market_order(pos.symbol, side, amt * price, cid, ref_price=price, reduce_only=True, amount=amt)
        except BrokerError as e:
            self._log("error", f"{pos.symbol} kısmi kapanış reddedildi: {e}")
            pos.partial_done = True
            return
        filled = float(o["amount"]); fill_px = float(o["avg_price"])
        gross = filled * (fill_px - pos.entry) * pos.sign()
        fee = float(o.get("fee_usdt") or 0.0)
        pos.amount -= filled
        pos.realized += gross
        pos.fees_partial += fee
        pos.partial_done = True
        pos.realized_net_pct = round((pos.realized - pos.fees_partial - pos.entry_fee * (filled / max(1e-12, base)))
                                     / max(1e-9, pos.notional) * 100.0, 4)
        self.fees_paid += fee
        self._tca(pos.symbol, side, filled, price, fill_px, pos.order_type, fee, amt)
        # stop → KÂR KİLİDİ (net başabaşın altına asla inmez)
        xp = XE.ExitParams(**{**self.xparams.__dict__, "time_stop_sec": pos.time_stop_sec or self.xparams.time_stop_sec})
        t = pos.track()
        # Tepe, ŞU ANKİ fiyatı da içermeli: eski kısmi TP yolu `decide_exit`ten ÖNCE
        # çalışıyor, dolayısıyla `peak_net_pct` henüz bu barı görmemiş olabilir. Görmemiş
        # tepeyle hesaplanan kilit, kârı kilitlemek yerine başabaşa düşerdi.
        t.peak_net_pct = max(t.peak_net_pct, t.net_pct(fill_px))
        lock = t.lock_level(xp) or t.breakeven_plus()
        if (lock - pos.hard_stop) * pos.sign() > 0:
            pos.stop = pos.hard_stop = float(lock)
            pos.lock_price = float(lock)
            pos.be_locked = True
        pos.locked_net_pct = round(t.net_pct(pos.hard_stop), 4)
        lbl = f"T{pos.levels_hit}" if reason == "LADDER_TP" else "kısmi TP"
        src = (extra or {}).get("source") or ""
        self._log("exit", f"🟢 {pos.symbol} {lbl} ({src}): %{fraction*100:.0f} kapatıldı @ {fill_px:.6g} · "
                          f"brüt {gross:+.4f} · stop → {pos.hard_stop:.6g} (net %{pos.locked_net_pct:+.3f} kilitli) · "
                          f"kalan %{pos.amount / max(1e-12, base) * 100:.0f}")
        audit("ORDER_PARTIAL", {"user": self.user_id, "exchange": self.cfg.exchange_id, "mode": pos.mode,
                                "symbol": pos.symbol, "reason": reason, "level": pos.levels_hit,
                                "fraction": fraction, "fill": fill_px, "gross": round(gross, 4),
                                "locked_net_pct": pos.locked_net_pct, "client_id": cid}, self.output_dir)

    def _close(self, pos: Position, price: float, reason: str, now: Optional[float] = None,
               extra: Optional[Dict] = None) -> Optional[Dict]:
        now = time.time() if now is None else float(now)
        side = "sell" if pos.direction == "LONG" else "buy"
        cid = make_client_order_id(self.user_id, self.cfg.exchange_id, pos.symbol, self.cycle, side + "x")
        try:
            o = self.broker.market_order(pos.symbol, side, pos.amount * price, cid, ref_price=price,
                                         reduce_only=True, amount=pos.amount)
        except BrokerError as e:
            self._log("error", f"{pos.symbol} kapanış emri reddedildi: {e}")
            return None
        exit_px = float(o["avg_price"])
        exit_fee = float(o.get("fee_usdt") or 0.0)
        self._tca(pos.symbol, side, float(o["amount"]), price, exit_px, "taker", exit_fee, pos.amount)
        gross = pos.amount * (exit_px - pos.entry) * pos.sign() + pos.realized
        fees = pos.entry_fee + exit_fee + pos.fees_partial
        net = gross - fees
        hold = max(0.0, now - pos.opened_ts)
        net_pct_realized = (net / max(1e-9, pos.notional)) * 100.0
        pcr = XE.peak_capture_ratio(net_pct_realized, pos.peak_net_pct)
        rec = {"symbol": pos.symbol, "direction": pos.direction, "entry": pos.entry, "exit": exit_px,
               "amount": pos.amount_initial or pos.amount, "notional": round(pos.notional, 4),
               "gross_pnl": round(gross, 4), "fees": round(fees, 4), "net_pnl": round(net, 4),
               "pnl_pct": round((exit_px / pos.entry - 1.0) * 100.0 * pos.sign(), 4),
               "net_pct_realized": round(net_pct_realized, 4), "reason": reason,
               "opened_ts": pos.opened_ts, "closed_ts": now, "hold_sec": round(hold), "hold_bucket": VS.hold_bucket(hold),
               "peak_pnl_pct": round(pos.peak_pnl_pct, 4), "peak_net_pct": round(pos.peak_net_pct, 4),
               "peak_capture": (None if pcr is None else round(pcr, 3)), "win": net > 0,
               "mode": pos.mode, "strategy": self.cfg.strategy, "order_type": pos.order_type,
               "trigger": pos.trigger, "sleeve": pos.sleeve, "template": pos.template, "exit_mode": pos.exit_mode,
               "target": pos.target, "partial_done": pos.partial_done, "horizon_sec": float(pos.time_stop_sec or 3600),
               "levels_hit": pos.levels_hit, "locked_net_pct": pos.locked_net_pct,
               "reentry_count": pos.reentry_count, "realized_partial": round(pos.realized, 4),
               # ÇIKIŞ A/B'si için gereken alanlar. Bunlar defterde YOKTU; 200 işlemlik
               # canlı kayıttan çıkış motorunu yeniden oynatmak bu yüzden mümkün değildi
               # (stop mesafesi/ATR/maliyet geri çıkarılamıyordu). Artık kayıtlı.
               "stop_pct": round(pos.stop_pct, 4), "target_pct": round(pos.target_pct, 4),
               "cost_pct_roundtrip": round(pos.cost_pct_roundtrip, 4), "atr_pct": round(pos.atr_pct, 4),
               "entry_stop_price": round(pos.entry * (1 - pos.sign() * pos.stop_pct / 100.0), 10),
               "fee_drag": (round(fees / gross, 3) if gross > 0 else None),
               # exit_detail tam hâliyle kayıt başına 138 B tutuyordu (defterin %13,5'i) ve
               # içindeki alanların çoğu başka sütunlarda zaten var. Yalnız TEŞHİS için
               # gerekli olanlar saklanır.
               "exit_detail": {k: v for k, v in (extra or {}).items()
                               if k in ("reason", "level", "cont_prob", "mae_pct", "current_ev_pct",
                                        "intrabar", "levels_hit", "source")},
               "p_win": ((pos.decision or {}).get("ticket") or {}).get("p_win"),
               # İKİ EV birden kaydedilir: `ev_pct` PLAN hedefiyle (iyimser — 85 işlemin 1'i
               # o hedefe ulaştı), `ev_achievable_pct` ÖLÇÜLMÜŞ sleeve MFE medyanıyla.
               # 222 işlemde `ev_achievable_pct` HİÇ kaydedilmemişti (0/222), bu yüzden
               # "hangisi gerçekten öngörüyor?" sorusu cevaplanamıyordu. Artık cevaplanabilir.
               "ev_pct": ((pos.decision or {}).get("ticket") or {}).get("ev_pct"),
               "ev_achievable_pct": ((pos.decision or {}).get("ticket") or {}).get("ev_achievable_pct"),
               "achievable_target_pct": ((pos.decision or {}).get("ticket") or {}).get("achievable_target_pct")}
        try:
            RE.record_exit(self.exit_state, pos.symbol, pos.direction, reason, net,
                           pos.peak_net_pct, now, self.rparams)
        except Exception:
            pass
        # KANIT DEFTERİ — salt-ekleme, ~130 B. Kanıt kapılarının ihtiyaç duyduğu yeterli
        # istatistikler buradan tek geçişte çıkar; sıcak durum dosyasının büyümesine gerek yok.
        EV.kaydet(rec, self.output_dir, tag=f"{self.user_id}_{self.cfg.exchange_id}",
                  rejim=self._regime_of(pos.decision))
        rec["prev_hash"] = self._ledger_hash
        rec["hash"] = _rec_hash(rec, self._ledger_hash)
        self._ledger_hash = rec["hash"]
        self.trades.append(rec)
        self.alerts.send(f"close:{pos.symbol}", f"KAPANDI {pos.symbol} {reason} net {net:+.2f} $ ({net_pct_realized:+.2f}%) · tepe %{pos.peak_net_pct:.2f}",
                         level=("info" if net >= 0 else "warning"))
        if pos.mode == "paper":
            self.paper_history.append({k: rec[k] for k in ("symbol", "net_pnl", "gross_pnl", "fees", "win", "closed_ts", "hold_bucket")})
        self.gross_pnl += gross; self.fees_paid += exit_fee; self.realized_net += net
        self.positions.pop(pos.symbol, None)
        emoji = "🟢" if net > 0 else "🔴"
        self._log("exit", f"{emoji} {pos.symbol} {reason} @ {exit_px:.6g} · net {net:+.2f} USDT (brüt {gross:+.2f}, "
                          f"komisyon {fees:.2f}) · {VS.hold_bucket(hold)} · PCR {('—' if pcr is None else f'{pcr:.2f}')}")
        audit("ORDER_CLOSE", {"user": self.user_id, "exchange": self.cfg.exchange_id, "mode": pos.mode,
                              "symbol": pos.symbol, "reason": reason, "net_pnl": round(net, 4), "fees": round(fees, 4),
                              "client_id": cid, "peak_capture": rec["peak_capture"]}, self.output_dir)
        if self.cfg.strategy == "committee":
            try:
                self.allocator.record(pos.sleeve or pos.trigger or "?", self._regime_of(pos.decision), net > 0, net, net_pct_realized)
                for ev in self.allocator.check_breakers(now):
                    self._log("learn", f"⛔ DEVRE KESİCİ ({ev.get('rule', 'pencere')}): {ev['sleeve']} son {ev['n']} işlemde net %{ev['mean_net_pct']:.3f}, "
                                       f"kazanma üst sınır {ev['wilson_upper']:.2f}, t {ev['t_stat']} → {ev.get('pause_hours', 6)} sa duraklatıldı "
                                       f"(#{ev.get('pause_count', 1)}; sonrası deneme penceresi, kanıt boyutu)")
                    self.alerts.send(f"breaker:{ev['sleeve']}", f"Devre kesici: {ev['sleeve']} {ev.get('pause_hours', 6)} sa duraklatıldı (n {ev['n']}, net %{ev['mean_net_pct']:.3f})", level="warning")
            except Exception:
                pass
            try:
                self.lessons.shadows.append({"kind": "post_exit", "symbol": pos.symbol, "direction": pos.direction,
                                             "ts": now, "entry": exit_px,
                                             "target": exit_px * (1 + pos.sign() * max(0.5, pos.atr_pct) / 100.0),
                                             "stop": exit_px * (1 - pos.sign() * 3 * max(0.5, pos.atr_pct) / 100.0),
                                             "expires": now + float(pos.time_stop_sec or 3600), "outcome": None, "gate": reason})
                new = self.lessons.on_trade_closed(rec, pos.decision, now, self.params.to_dict())
                for les in new:
                    self._log("learn", "📘 " + les["title"])
                if new:
                    self._challenger_cycle()
            except Exception as e:
                self._log("error", f"ders motoru hatası: {type(e).__name__}: {str(e)[:100]}")
        return rec

    def _challenger_cycle(self) -> None:
        proposed = dict(self.lessons.overrides)
        try:
            cur = {**self.params.to_dict(), "top_k": self.cfg.top_k, "max_open": self.cfg.max_open,
                   "max_trades_per_day": self.cfg.max_trades_per_day}
            props = self.missed.proposals(time.time(), cur)
        except Exception:
            props = {}
        # kapasite parametreleri (top_k / max_open) komite gölgesiyle sınanamaz: kanıt kaçırılan-fırsat
        # motorunun kendisidir (n ≥ 20, Wilson üst < 0,5, 12 sa soğuma, sınır içinde) → doğrudan, LOGLU
        for k in ("top_k", "max_open", "max_trades_per_day"):
            if k in props and int(props[k]) != int(getattr(self.cfg, k)):
                old = getattr(self.cfg, k)
                setattr(self.cfg, k, int(props[k])); self.cfg.validated()
                self._log("learn", f"🧭 Kapasite düzeltmesi (kaçırılan-fırsat kanıtı): {k} {old} → {getattr(self.cfg, k)}")
                audit("CAPACITY_ADJUST", {"user": self.user_id, "exchange": self.cfg.exchange_id, "param": k, "from": old,
                                         "to": getattr(self.cfg, k)}, self.output_dir)
        proposed.update({k: v for k, v in props.items() if k not in ("top_k", "max_open", "max_trades_per_day")})
        pending = {k: v for k, v in proposed.items() if self._promoted.get(k) != v}
        if pending and pending != self.challenger.params:
            if self.challenger.propose(pending):
                self._log("learn", "🥊 Challenger önerildi (gölgede sınanacak): " + ", ".join(f"{k}={v}" for k, v in pending.items()))
        ev = self.challenger.evaluate()
        if ev.get("promote"):
            won = self.challenger.conclude(True)
            self._promoted = {**self._promoted, **won}
            self._log("learn", f"🏆 Challenger TERFİ etti ({ev['n_challenger_only']} gölge, Wilson alt %{ev['wilson_lower']*100:.0f}): "
                               + ", ".join(f"{k}={v}" for k, v in won.items()))
            self._refresh_params()
        elif ev.get("reject"):
            self.challenger.conclude(False)
            for k in list(self.lessons.overrides):
                if k in pending:
                    self.lessons.overrides.pop(k, None)
            self.lessons.save()
            self._log("learn", f"🗑️ Challenger reddedildi ({ev['n_challenger_only']} gölge, kazanma %{(ev['win_challenger_only'] or 0)*100:.0f})")

    # ------------------------------------------------------------ mutabakat
    def _reconcile(self) -> None:
        try:
            ex_pos = self.broker.fetch_positions()
        except Exception as e:
            self.reconcile_ok = False
            self.reconcile_note = f"borsa pozisyonu okunamadı: {type(e).__name__}"
            return
        internal = {s: p.amount * p.sign() for s, p in self.positions.items()}
        ex_f = {s: q for s, q in ex_pos.items() if s in internal or s in self.cfg.symbols}
        r = reconcile(internal, ex_f, output_dir=self.output_dir)
        self.reconcile_ok = bool(r["ok"])
        self.reconcile_note = r["note"] if not r["ok"] else "eşleşti"
        if not r["ok"]:
            self._log("error", "⚠️ MUTABAKAT SAPMASI — yeni giriş yok: "
                               + ", ".join(f"{s}: iç {d['internal']:.6g} / borsa {d['exchange']:.6g}" for s, d in r["mismatches"].items()))

    # ------------------------------------------------------------ durum
    def equity(self) -> float:
        return float(self.cfg.capital_usdt + self.realized_net + sum(p.unrealized() + p.realized for p in self.positions.values()))

    def _mark(self, now: float) -> None:
        eq = round(self.equity(), 2)
        self.equity_curve.append({"ts": now, "equity": eq})
        if len(self.equity_curve) > 2000:
            self.equity_curve = self.equity_curve[-2000:]
        self.equity_history.append({"ts": float(now), "equity": eq, "src": "live"})
        if len(self.equity_history) % 200 == 0:
            self._compact_history(now)

    def _compact_history(self, now: float) -> None:
        """Son EQUITY_FULL_RES_SEC tam çözünürlük; daha eskisi EQUITY_BUCKET_SEC kovasına (kovanın son değeri)
        indirgenir. Kapanan işlem noktaları (`src: ledger`) ve ilk nokta her zaman korunur."""
        cut = float(now) - EQUITY_FULL_RES_SEC
        old = [p for p in self.equity_history if p["ts"] < cut]
        new = [p for p in self.equity_history if p["ts"] >= cut]
        if len(old) < 2:
            return
        kept: List[Dict] = [old[0]]
        bucket = None
        for p in old[1:]:
            if p.get("src") == "ledger":
                kept.append(p); continue
            b = int(p["ts"] // EQUITY_BUCKET_SEC)
            if b == bucket and kept and kept[-1].get("src") != "ledger":
                kept[-1] = p
            else:
                kept.append(p); bucket = b
        self.equity_history = kept + new
        if len(self.equity_history) > EQUITY_HISTORY_MAX:
            self.equity_history = self.equity_history[-EQUITY_HISTORY_MAX:]

    def _backfill_history(self) -> int:
        """Geçmiş kalıcı değilken (eski sürüm) kaybolan başlangıç: kapanan işlem defterinden gerçekleşmiş
        özsermaye noktaları (sermaye + kümülatif net) — yalnız mevcut geçmişin İLK noktasından önceki dönem."""
        if not self.trades:
            return 0
        first = self.equity_history[0]["ts"] if self.equity_history else float("inf")
        pts: List[Dict] = []
        cum = 0.0
        t0 = min(float(t.get("opened_ts") or t["closed_ts"]) for t in self.trades)
        if t0 < first:
            pts.append({"ts": t0, "equity": round(float(self.cfg.capital_usdt), 2), "src": "ledger"})
        for t in sorted(self.trades, key=lambda x: float(x["closed_ts"])):
            cum += float(t.get("net_pnl") or 0.0)
            ts = float(t["closed_ts"])
            if ts < first:
                pts.append({"ts": ts, "equity": round(float(self.cfg.capital_usdt) + cum, 2), "src": "ledger"})
        if pts:
            self.equity_history = pts + self.equity_history
        return len(pts)

    def equity_series(self, max_points: int = 2000) -> Dict:
        """Panel için tam özsermaye geçmişi (başlangıçtan bugüne) + kapanan işlem işaretleri; `max_points`'e
        kova-min/maks ile indirgenir (tepe/dip kaybolmaz)."""
        with self._lock:
            if self.trades:
                t0 = min(float(t.get("opened_ts") or t["closed_ts"]) for t in self.trades)
                if not self.equity_history or t0 < self.equity_history[0]["ts"]:
                    self._backfill_history()            # başlangıç anı defterden (yeniden yükleme olmadan da)
            pts = list(self.equity_history)
            trades = list(self.trades)
        n_raw = len(pts)
        if n_raw > max_points > 2:
            step = n_raw / (max_points / 2.0)
            out: List[Dict] = []
            i = 0.0
            while int(i) < n_raw:
                chunk = pts[int(i):int(i + step)] or pts[int(i):int(i) + 1]
                lo = min(chunk, key=lambda p: p["equity"]); hi = max(chunk, key=lambda p: p["equity"])
                for p in sorted({id(lo): lo, id(hi): hi}.values(), key=lambda p: p["ts"]):
                    out.append(p)
                i += step
            pts = out
        marks = []
        cum = 0.0
        for k, t in enumerate(sorted(trades, key=lambda x: float(x["closed_ts"])), start=1):
            cum += float(t.get("net_pnl") or 0.0)
            marks.append({"ts": float(t["closed_ts"]), "equity": round(float(self.cfg.capital_usdt) + cum, 2),
                          "net_pnl": round(float(t.get("net_pnl") or 0.0), 4), "symbol": t.get("symbol"), "seq": k})
        return {"capital": float(self.cfg.capital_usdt), "start_ts": (pts[0]["ts"] if pts else None),
                "end_ts": (pts[-1]["ts"] if pts else None), "n_raw": n_raw, "n": len(pts),
                "n_ledger": sum(1 for p in pts if p.get("src") == "ledger"),
                "points": pts, "marks": marks}

    # ------------------------------------------------------------ kanıt kapıları (boyut yönetişimi)
    def _mfe_by_sleeve(self, min_n: int = 8) -> Dict[str, float]:
        """Sleeve başına ÖLÇÜLMÜŞ tepe-brüt medyanı (%) — fişteki ulaşılabilir hedef için."""
        by: Dict[str, List[float]] = {}
        for t in self.trades:
            v = t.get("peak_pnl_pct")
            if isinstance(v, (int, float)) and v > 0:
                by.setdefault(str(t.get("sleeve") or t.get("trigger") or "?"), []).append(float(v))
        out = {}
        for k, xs in by.items():
            if len(xs) >= min_n:
                xs = sorted(xs); out[k] = round(xs[len(xs) // 2], 4)
        return out

    def _derisk(self) -> Dict:
        """Drawdown'da yarı boyut: son N kapanan işlemin net'i < 0 → ×derisk_mult (anti-martingale;
        canlı karşı-olgusal: −4,78 → −2,10 $)."""
        p = self.params
        n = int(getattr(p, "derisk_trailing_n", 20))
        tail = self.trades[-n:]
        net = round(sum(float(t.get("net_pnl") or 0.0) for t in tail), 4)
        active = len(tail) >= max(5, n // 2) and net < 0.0
        return {"active": active, "trailing_n": len(tail), "window_n": n, "trailing_net": net,
                "mult": (float(getattr(p, "derisk_mult", 0.5)) if active else 1.0)}

    def _session_gate(self, now: float, days: float = 14.0) -> Dict:
        """Seans (4 sa UTC bloğu) beklentisi — son `days` günün kapanan işlemleri; n ≥ session_min_n ve
        t ≤ session_t_half → ×0,5, t ≤ session_t_block → giriş yok. Kendi kendini günceller (sabit yasak DEĞİL)."""
        p = self.params
        if not bool(getattr(p, "session_gate", True)):
            return {}
        cut = float(now) - days * 86400.0
        by: Dict[str, List[float]] = {}
        for t in self.trades:
            ots = float(t.get("opened_ts") or t.get("closed_ts") or 0.0)
            if ots < cut:
                continue
            h = int(time.gmtime(ots).tm_hour) // 4 * 4
            pct = t.get("net_pct_realized")
            if pct is None:
                nt = float(t.get("notional") or 0.0)
                pct = (float(t.get("net_pnl", 0.0)) / nt * 100.0) if nt > 0 else 0.0
            by.setdefault(f"{h:02d}-{h + 4:02d}", []).append(float(pct))
        out = {}
        min_n = int(getattr(p, "session_min_n", 15))
        for blk, xs in sorted(by.items()):
            n = len(xs); m = sum(xs) / n
            sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5 if n > 1 else 0.0
            tt = (m / (sd / n ** 0.5)) if sd > 1e-12 else (0.0 if n < 2 else (-9.0 if m < 0 else 9.0))
            mult = 1.0
            if n >= min_n and m < 0:
                if tt <= float(getattr(p, "session_t_block", -2.5)):
                    mult = 0.0
                elif tt <= float(getattr(p, "session_t_half", -1.5)):
                    mult = 0.5
            out[blk] = {"n": n, "mean_net_pct": round(m, 4), "t_stat": round(tt, 2), "mult": mult}
        return out

    def _session_block(self, now: float) -> str:
        h = int(time.gmtime(float(now)).tm_hour) // 4 * 4
        return f"{h:02d}-{h + 4:02d}"

    def governance(self, now: Optional[float] = None) -> Dict:
        """Panel: kanıt kapıları — sleeve durumları/tavanlar, drawdown yarı-boyut, seans kapısı, taker politikası."""
        now = time.time() if now is None else now
        p = self.params
        probe = float(getattr(p, "probe_notional_usdt", 25.0))
        rows = []
        # status() sleeve BAŞINA değil, BİR KEZ hesaplanır (döngü içinde çağrılınca 30 sleeve × tam
        # istatistik = gereksiz CPU; kaynak durumu RED'e yaklaşırken bu yük anlamlı)
        agg_by = {r["sleeve"]: r for r in self.allocator.status()["rows"] if r["regime"] == "*"}
        for s, st_ in self.allocator.sleeve_states(now=now).items():
            cap = self.allocator.notional_cap(s, probe, float(self.cfg.max_order_usdt), now)
            agg = agg_by.get(s, {})
            pu = self.allocator.paused_until.get(s)
            rows.append({"sleeve": s, "sleeve_tr": SF.SLEEVE_TR.get(s, s), "state": st_, "label": cap["label"],
                         "n": cap["n"], "net": agg.get("net"), "mean_net_pct": cap["mean_net_pct"], "t_stat": cap["t_stat"],
                         "cap_usdt": cap["cap_usdt"], "paused_until": (pu if pu and pu > now else None),
                         "pause_count": int(self.allocator.pause_count.get(s, 0))})
        rows.sort(key=lambda r: (-int(r["n"] or 0), r["sleeve"]))
        return {"probe_notional_usdt": probe, "full_notional_usdt": float(self.cfg.max_order_usdt),
                "sleeves": rows, "derisk": self._derisk(), "session": self._session_gate(now),
                "session_now": self._session_block(now),
                "taker_policy": ("aciliyet-0 sleeve kanıtlanmadan (PROVEN) taker giremez → maker 1 bar; maker dolmazsa kovalanmaz"
                                 if bool(getattr(p, "taker_requires_proof", True)) else "taker serbest"),
                "mfe_by_sleeve": self._mfe_by_sleeve(),
                "note": ("Kanıt boyutu: sleeve n ≥ 20 ve t ≥ 1 ile PROVEN olana dek tavan; drawdown'da (son 20 net < 0) ×0,5; "
                         "seans t ≤ −1,5 ×0,5 / ≤ −2,5 kapalı — hepsi ölçümle kendini günceller")}

    def stats(self) -> Dict:
        t = self.trades
        wins = [x for x in t if x["win"]]
        gw = sum(x["net_pnl"] for x in wins)
        gl = abs(sum(x["net_pnl"] for x in t if not x["win"]))
        buckets = {}
        for lo, hi, ad in VS.HOLD_BUCKETS:
            b = [x for x in t if x["hold_bucket"] == ad]
            buckets[ad] = {"n": len(b), "net": round(sum(x["net_pnl"] for x in b), 2),
                           "win_rate": round(100.0 * sum(1 for x in b if x["win"]) / len(b), 1) if b else None}
        reasons: Dict[str, int] = {}
        by_sleeve: Dict[str, Dict] = {}
        for x in t:
            reasons[x["reason"]] = reasons.get(x["reason"], 0) + 1
            sv = by_sleeve.setdefault(x.get("sleeve") or x.get("trigger") or "?", {"n": 0, "net": 0.0, "wins": 0})
            sv["n"] += 1; sv["net"] = round(sv["net"] + x["net_pnl"], 4); sv["wins"] += int(x["win"])
        pcrs = [x["peak_capture"] for x in t if x.get("peak_capture") is not None]
        maker_n = sum(1 for x in t if x.get("order_type") == "maker")
        gross_pos = sum(x["gross_pnl"] for x in t if x["gross_pnl"] > 0)
        return {"closed_trades": len(t), "win_rate": round(100.0 * len(wins) / len(t), 1) if t else 0.0,
                "gross_pnl": round(self.gross_pnl, 2), "fees_paid": round(self.fees_paid, 2),
                "net_pnl": round(self.realized_net, 2),
                "fee_share_of_gross_pct": (round(100.0 * self.fees_paid / abs(self.gross_pnl), 1) if abs(self.gross_pnl) > 1e-9 else None),
                "fee_drag": (round(self.fees_paid / gross_pos, 3) if gross_pos > 0 else None),
                "profit_factor": (round(gw / gl, 2) if gl > 0 else (99.9 if gw > 0 else 0.0)),
                "equity": round(self.equity(), 2), "capital": self.cfg.capital_usdt,
                "return_pct": round((self.equity() / self.cfg.capital_usdt - 1.0) * 100.0, 3),
                "open_positions": len(self.positions), "pending_orders": len(self.pending),
                "hold_buckets": buckets, "exit_reasons": reasons, "by_sleeve": by_sleeve, "day_trades": self.day_trades,
                "maker_share_pct": round(100.0 * maker_n / len(t), 1) if t else None,
                "avg_peak_capture": (round(sum(pcrs) / len(pcrs), 3) if pcrs else None),
                "avg_hold_min": round(sum(x["hold_sec"] for x in t) / 60.0 / len(t), 1) if t else None,
                "ev_per_trade": (round(self.realized_net / len(t), 4) if t else None)}

    def readiness(self) -> Dict:
        h = self.paper_history
        n = len(h)
        net = round(sum(x["net_pnl"] for x in h), 2)
        fees = round(sum(x["fees"] for x in h), 2)
        need = self.cfg.paper_proof_trades
        ok = (not self.cfg.require_paper_proof) or (n >= need and net > 0)
        why = []
        if self.cfg.require_paper_proof:
            if n < need:
                why.append(f"paper işlem {n}/{need}")
            if net <= 0:
                why.append(f"paper net {net:+.2f} USDT ≤ 0")
        return {"ok": ok, "paper_trades": n, "paper_net": net, "paper_fees": fees,
                "required_trades": need, "required": self.cfg.require_paper_proof, "missing": why}

    def universe(self) -> List[Dict]:
        """Bütün paritelerin tek tabloda görünürlüğü: tarama, karar, tazelik, CVD, pozisyon, gölge sayısı."""
        scan = {r["symbol"]: r for r in self.scan}
        rows = []
        open_sh = {}
        for rec in self.missed.records:
            if rec.get("outcome") is None:
                open_sh[rec["symbol"]] = open_sh.get(rec["symbol"], 0) + 1
        for sym in self.cfg.symbols:
            sc = scan.get(sym, {}); d = self.last_decisions.get(sym) or {}; f = d.get("fast") or {}
            pos = self.positions.get(sym)
            vr = d.get("veto_review") or {}
            rows.append({"symbol": sym, "tier": sc.get("tier") or d.get("tier"), "freshness": sc.get("freshness") or d.get("freshness"),
                         "interest": sc.get("interest"), "z20": sc.get("z20"), "vol_ratio": sc.get("vol_ratio"), "rs_rank": sc.get("rs_rank"),
                         "regime": d.get("regime"), "template": d.get("template"), "trigger": d.get("trigger"),
                         "z": f.get("z"), "rsi": f.get("rsi"), "trend_score": f.get("trend_score"), "adx": f.get("adx"),
                         "cvd": (d.get("cvd") or {}).get("cvd_ratio"), "obi": f.get("obi"),
                         "score": d.get("score"), "confidence": d.get("confidence"), "ev_pct": (d.get("ticket") or {}).get("ev_pct"),
                         "result": str(d.get("result") or "")[:90], "review": (vr.get("summary_tr") or "")[:160],
                         "stop_risk": (d.get("stop_risk") or {}).get("n"),
                         "position": (None if pos is None else {"sleeve": pos.sleeve, "net_pct": round(pos.net_pct_now(), 3), "peak_net_pct": round(pos.peak_net_pct, 3),
                                                                  "cont_prob": pos.cont_prob, "remaining_ev_pct": pos.remaining_ev_pct, "be_locked": pos.be_locked, "age_min": round((time.time() - pos.opened_ts) / 60.0, 1)}),
                         "shadows_open": open_sh.get(sym, 0), "decided_ts": d.get("ts")})
        rows.sort(key=lambda r: (-(r["interest"] or 0.0)))
        return rows

    def best_action(self) -> Dict:
        if self.guard.state.halted:
            return {"action": "HALT", "why": "; ".join(self.guard.state.reasons)}
        if self.portfolio.get("mode") == PM.CASH or self.cash_mode:
            return {"action": "CASH", "why": self.portfolio.get("label", "nakit modu")}
        for pos in self.positions.values():
            if pos.armed:
                lvl = pos.track().giveback_level(self.xparams)
                return {"action": "TRAILING PEAK", "symbol": pos.symbol,
                        "why": f"tepe koruması silahlı · yarı-tepe çıkışı {('—' if lvl is None else f'{lvl:.6g}')}"}
        if self.positions:
            return {"action": "HOLD", "why": f"{len(self.positions)} açık pozisyon yönetiliyor"}
        if self.pending:
            return {"action": "WAIT FOR ENTRY", "symbol": next(iter(self.pending)), "why": "maker emri dolum bekliyor"}
        if self.portfolio.get("mode") == PM.DEFENSIVE:
            return {"action": "WAIT", "why": self.portfolio.get("label")}
        ready = [d for d in self.last_decisions.values() if str(d.get("result", "")).startswith("AÇ")]
        if ready:
            return {"action": "BUY", "symbol": ready[0].get("symbol"), "why": ready[0].get("result")}
        return {"action": "WAIT", "why": "tetikleyici/edge yok — NO EDGE → NO TRADE"}

    def top_opportunities(self, n: int = 3) -> List[Dict]:
        rows = []
        for d in self.last_decisions.values():
            t = d.get("ticket") or {}
            if not d.get("plan"):
                continue
            rows.append({"symbol": d.get("symbol"), "sleeve": d.get("trigger"), "tier": d.get("tier"),
                         "allowed": d.get("allowed"), "result": str(d.get("result", ""))[:110],
                         "ev_pct": t.get("ev_pct"), "p_win": t.get("p_win"), "expected_profit_usdt": t.get("expected_profit_usdt"),
                         "fee_usdt": t.get("fee_usdt"), "max_loss_usdt": t.get("max_loss_usdt"),
                         "entry": d.get("entry"),
                         "plan": {k: d["plan"].get(k) for k in ("entry", "stop", "target", "stop_pct", "target_pct", "rr")},
                         "exit_mode": d.get("exit_mode"), "valid_until": d.get("valid_until"), "ts": d.get("ts")})
        rows.sort(key=lambda r: ((0 if r["allowed"] else 1), -(r["ev_pct"] if r["ev_pct"] is not None else -99)))
        return rows[:n]

    def full_state(self) -> Dict:
        with self._lock:
            g = self.guard.state
            out = {
                "configured": True, "user_id": self.user_id, "exchange": self.cfg.exchange_id,
                "mode": self.cfg.mode, "market_type": self.cfg.market_type, "strategy": self.cfg.strategy,
                "strategy_name": CM.STRATEGY_NAME if self.cfg.strategy == "committee" else VS.STRATEGY_NAME,
                "label": self.cfg.label, "running": self.running, "manage_only": self.manage_only,
                "halted": g.halted, "halt_reasons": g.reasons,
                "day_return_pct": (round((self.equity() / g.day_start_equity - 1) * 100, 3) if g.day_start_equity else None),
                "drawdown_pct": (round((1 - self.equity() / g.peak_equity) * 100, 2) if g.peak_equity else None),
                "reconcile_ok": self.reconcile_ok, "reconcile_note": self.reconcile_note,
                "cycle": self.cycle, "last_cycle_ts": self.last_cycle_ts, "last_cycle_sec": self.last_cycle_sec,
                "loop_sec": self.params.loop_sec,
                "config": self.cfg.to_dict(), "effective_params": self.params.to_dict(),
                "exit_params": dict(self.xparams.__dict__), "stats": self.stats(), "readiness": self.readiness(),
                "positions": [p.to_dict(self.xparams) for p in self.positions.values()],
                "pending": [{"symbol": s, "side": o["order"]["side"], "price": o["order"]["price"],
                             "notional": o["notional"], "bars": o["bars"], "created_ts": o["created_ts"]}
                            for s, o in self.pending.items()],
                "trades": self.trades[-50:][::-1], "events": list(self.events)[:60],
                "equity_curve": self.equity_curve[-300:], "decisions": list(self.last_decisions.values()),
                "scan": self.scan[:15], "top_k": self._effective_top_k(), "interest_weights": getattr(self, "_interest_weights", None),
                "broker": self.broker.describe(),
                "venue": {"exchange_id": self.venue.exchange_id, "maker_bps": self.venue.maker_bps,
                          "taker_bps": self.venue.taker_bps, "note": self.venue.note},
                "fee_info": self.fee_info, "venue_compare": self.venue_compare,
                "reentry": {"params": self.rparams.to_dict(), "recent_blocks": list(self.reentry_blocks)[-20:],
                            "n_blocked": len(self.reentry_blocks),
                            "last_exits": {k: {"reason": v.get("reason"), "direction": v.get("direction"),
                                               "net_pnl": v.get("net_pnl"), "count": v.get("count"), "ts": v.get("ts")}
                                           for k, v in list(self.exit_state.items())[-20:]}},
                "tca": self._tca_report(),
                "kanit": self._evidence_report(),
                "haber_etki": self._news_impact_report(),
                "ev_kalibrasyon": self._ev_calib_report(),
                "risk_mode": self.risk, "cash_mode": self.cash_mode, "portfolio_mode": self.portfolio,
                "resource": self.resource, "best_action": self.best_action(),
                "top_opportunities": self.top_opportunities(3),
                "challenger": self.challenger.evaluate() if self.cfg.strategy == "committee" else None,
                "lifecycle": self.lifecycle.status() if self.cfg.strategy == "committee" else None,
                "missed": self.missed.report() if self.cfg.strategy == "committee" else None,
                "ledger": verify_ledger(self.trades), "alerts": self.alerts.status(),
                "allocator": self.allocator.status() if self.cfg.strategy == "committee" else None,
                "governance": self.governance() if self.cfg.strategy == "committee" else None,
                "n_trades_total": len(self.trades), "equity_history_n": len(self.equity_history),
                "research_brief": ({"pairs_open": len(self.lab.pairs.open), "pairs_found": len(self.lab.pairs.pairs),
                                    "carry_enabled": self.lab.carry_enabled, "carry_open": len(self.lab.carry.open),
                                    "triangular_scans": self.lab.tri.n_scans, "triangular_found": len(self.lab.tri.history),
                                    "mm_fills": sum(v.get("n_fills", 0) for v in self.lab.mm.state.values()),
                                    "errors": self.lab.errors} if self.cfg.strategy == "committee" else None),
            }
            if self.cfg.strategy == "committee":
                out["lessons"] = self.lessons.status()
            return out

    def _log(self, typ: str, msg: str) -> None:
        now = time.time()
        self.events.appendleft({"ts": now, "type": typ, "msg": msg})
        if typ == "error":
            self.error_times.append(now)
            recent = sum(1 for t in self.error_times if now - t <= 600)
            if recent >= 10:
                self.alerts.send("errors", f"{self.cfg.exchange_id}: 10 dk'da {recent} hata — son: {msg[:120]}", level="warning", now=now)

    # ------------------------------------------------------------ kalıcılık
    def save(self) -> None:
        try:
            d = {"user_id": self.user_id, "config": self.cfg.to_dict(), "running": self.running,
                 "manage_only": self.manage_only, "positions": [asdict(p) for p in self.positions.values()],
                 "pending": self.pending, "trades": self.trades[-TRADES_KEEP:], "paper_history": self.paper_history[-2000:],
                 "realized_net": self.realized_net, "fees_paid": self.fees_paid, "gross_pnl": self.gross_pnl,
                 "cycle": self.cycle, "equity_curve": self.equity_curve[-1000:],
                 "equity_history": self.equity_history[-EQUITY_HISTORY_MAX:],
                 "paper_cash": self.broker.paper_cash, "paper_holdings": self.broker.paper_holdings,
                 "last_decisions": self.last_decisions, "promoted": self._promoted, "ledger_hash": self._ledger_hash,
                 "exit_state": self.exit_state,
                 # GÜNLÜK SAYAÇLAR — kalıcı olmalılar. Değillerdi ve her yeniden başlatma
                 # (pm2 max_memory_restart / autorestart / elle deploy) `day`i "" yapıp
                 # sonraki döngüde `day != today` dalına düşürüyor, böylece GÜNLÜK İŞLEM
                 # TAVANI ve ROTASYON TAVANI SIFIRLANIYORDU. 2026-09-06 kanıtı: rotasyon
                 # tavanı 6/gün olmasına rağmen o gün 10 rotasyon oldu (5 yeniden başlatma).
                 # Bu bir risk kapısıdır; yeniden başlatma onu gevşetmemeli.
                 "day": self.day, "day_trades": self.day_trades,
                 "rotations_today": getattr(self, "_rotations_today", 0),
                 "created_ts": self.created_ts, "saved_ts": time.time()}
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(d, ensure_ascii=False, default=str), encoding="utf-8")
            tmp.replace(self._path)
        except Exception:
            pass

    def load(self, d: Dict) -> None:
        with self._lock:
            for p in d.get("positions", []):
                try:
                    pos = Position(**{k: v for k, v in p.items() if k in Position.__dataclass_fields__})
                    if not pos.hard_stop:
                        pos.hard_stop = pos.stop
                    if not pos.ladder:
                        # v1 durumundan yüklenen açık pozisyon: merdiveni kur (yapısal seviye yok,
                        # yalnız R katları). Basamak zaten geçilmişse ilk kontrolde işaretlenir.
                        try:
                            pos.ladder = self._build_ladder(pos)
                        except Exception:
                            pos.ladder = []
                    self.positions[pos.symbol] = pos
                except Exception:
                    continue
            pend = d.get("pending") or {}
            if isinstance(pend, dict):
                self.pending = {s: o for s, o in pend.items() if isinstance(o, dict) and o.get("order")}
            self.trades = list(d.get("trades", []))
            self.paper_history = list(d.get("paper_history", []))
            self.realized_net = float(d.get("realized_net", 0.0)); self.fees_paid = float(d.get("fees_paid", 0.0))
            self.gross_pnl = float(d.get("gross_pnl", 0.0)); self.cycle = int(d.get("cycle", 0))
            self.equity_curve = list(d.get("equity_curve", []))
            self.equity_history = [{"ts": float(p["ts"]), "equity": float(p["equity"]), "src": p.get("src", "live")}
                                   for p in (d.get("equity_history") or []) if p.get("ts") is not None]
            if not self.equity_history and self.equity_curve:
                self.equity_history = [{"ts": float(p["ts"]), "equity": float(p["equity"]), "src": "live"} for p in self.equity_curve]
            try:
                nb = self._backfill_history()
                if nb:
                    self._log("system", f"📈 özsermaye geçmişi defterden tamamlandı: {nb} nokta (başlangıç anından)")
            except Exception:
                pass
            try:
                # Sıcak liste TRADES_KEEP ile sınırlı; KALICI kanıt `evidence.jsonl`'de.
                # Geri doldurma önce oradan yapılır (akış okuma — bellek sabit), böylece
                # sıcak listeyi kısaltmak hiçbir kanıtı kaybettirmez.
                kaynak = self.trades
                try:
                    ev = [{"sleeve": r.get("slv"), "regime": r.get("rej"), "win": bool(r.get("w")),
                           "net_pnl": r.get("nu"), "net_pct_realized": r.get("np"),
                           "closed_ts": r.get("ts")}
                          for r in EV.oku(self.output_dir, tag=f"{self.user_id}_{self.cfg.exchange_id}")]
                    if len(ev) > len(kaynak):
                        kaynak = ev
                except Exception:
                    pass
                na = self.allocator.backfill(kaynak)
                if na:
                    self._log("system", f"🧾 sleeve kanıt geçmişi kuruldu: {na} kayıt "
                                        f"({'kanıt defteri' if kaynak is not self.trades else 'sıcak liste'})")
            except Exception:
                pass
            self.last_decisions = dict(d.get("last_decisions") or {})
            # Günlük sayaçlar: yalnız AYNI güne aitse geri yüklenir (gün değiştiyse zaten sıfırlanmalı).
            kayitli_gun = str(d.get("day") or "")
            if kayitli_gun == time.strftime("%Y-%m-%d", time.gmtime()):
                self.day = kayitli_gun
                self.day_trades = int(d.get("day_trades") or 0)
                self._rotations_today = int(d.get("rotations_today") or 0)
            self.exit_state = {k: v for k, v in (d.get("exit_state") or {}).items() if isinstance(v, dict)}
            if not self.exit_state and self.trades:
                # v1 durumundan yükleniyorsa son çıkışları defterden kur — yeniden giriş
                # kapısı yeniden başlatmadan sonra "hiç işlem olmamış" sanmasın.
                for t in self.trades[-200:]:
                    try:
                        RE.record_exit(self.exit_state, str(t["symbol"]), str(t["direction"]),
                                       str(t.get("reason") or ""), float(t.get("net_pnl") or 0.0),
                                       float(t.get("peak_net_pct") or 0.0), float(t.get("closed_ts") or 0.0),
                                       self.rparams)
                    except Exception:
                        continue
            self._promoted = dict(d.get("promoted") or {})
            self._ledger_hash = str(d.get("ledger_hash") or (self.trades[-1].get("hash", "") if self.trades else ""))
            self.created_ts = float(d.get("created_ts", time.time()))
            if self.cfg.mode == "paper":
                self.broker.paper_cash = float(d.get("paper_cash", self.cfg.capital_usdt))
                self.broker.paper_holdings = {k: float(v) for k, v in (d.get("paper_holdings") or {}).items()}
            self._refresh_params()


# ======================================================================
# Kayıt defteri — süreçteki tüm koşucular
# ======================================================================
class RunnerRegistry:
    def __init__(self, output_dir: str = "runs", ctx: Optional[Context] = None, server_config=None,
                 creds_lookup: Optional[Callable[[int, str], Dict]] = None, client_factory=None):
        self.output_dir = output_dir
        self.ctx = ctx or Context()
        self.server_config = server_config
        self.creds_lookup = creds_lookup
        self.client_factory = client_factory
        self._r: Dict[tuple, LiveRunner] = {}
        self._lock = threading.Lock()
        self._stores: Dict[str, MarketStateStore] = {}
        self._public: Dict[str, Broker] = {}
        self.rate = RateLimitCoordinator()

    def public_broker(self, venue: str) -> Broker:
        """Venue kıyası için borsa başına TEK paylaşımlı public istemci (her çağrıda yeni ccxt örneği = bellek sızıntısı)."""
        b = self._public.get(venue)
        if b is None:
            b = Broker(venue, "paper", paper_capital=1.0, client_factory=self.client_factory)
            self._public[venue] = b
        return b

    def key(self, user_id: int, exchange_id: str) -> tuple:
        return (int(user_id), str(exchange_id))

    def get(self, user_id: int, exchange_id: str) -> Optional[LiveRunner]:
        return self._r.get(self.key(user_id, exchange_id))

    def all_for(self, user_id: int) -> List[LiveRunner]:
        return [r for (u, _), r in self._r.items() if u == int(user_id)]

    def ems_ready(self) -> bool:
        return any(True for _ in self._r.values())

    def live_running(self) -> bool:
        return any(r.running and r.cfg.mode == "live" for r in self._r.values())

    def store_for(self, exchange_id: str) -> MarketStateStore:
        """Aynı borsadaki koşucular AYNI depoyu paylaşır (aynı veri iki kez çekilmez)."""
        st = self._stores.get(exchange_id)
        if st is None:
            pub = Broker(exchange_id, "paper", paper_capital=1.0, client_factory=self.client_factory)
            st = MarketStateStore(fetch_ohlcv=lambda ex, s, tf, lim: pub.fetch_ohlcv(s, tf, limit=lim),
                                  fetch_book=lambda ex, s: pub.fetch_book_top(s), rate=self.rate)
            self._stores[exchange_id] = st
        return st

    def create(self, user_id: int, cfg: RunnerConfig, creds: Optional[Dict] = None,
               manage_only: bool = False, restore: Optional[Dict] = None) -> LiveRunner:
        fee = FE.venue_fee(cfg.exchange_id)
        fee_bps = fee.taker_bps if cfg.strategy == "committee" else VS.effective_fee_bps(VS.ScalpParams.from_dict(cfg.params))
        broker = Broker(cfg.exchange_id, cfg.mode, creds=creds, market_type=cfg.market_type,
                        fee_bps=fee_bps, maker_fee_bps=(fee.maker_bps if cfg.strategy == "committee" else None),
                        max_order_usdt=cfg.max_order_usdt, paper_capital=cfg.capital_usdt, client_factory=self.client_factory)
        r = LiveRunner(user_id, cfg, broker, self.ctx, self.server_config, self.output_dir,
                       manage_only=manage_only, store=self.store_for(cfg.exchange_id))
        r._client_factory = self.client_factory
        r._public_broker = self.public_broker
        r.lab._client_factory = self.client_factory
        if restore:
            r.load(restore)
        with self._lock:
            old = self._r.get(self.key(user_id, cfg.exchange_id))
            if old is not None:
                old.stop("yeniden yapılandırma"); old.broker.close()
            self._r[self.key(user_id, cfg.exchange_id)] = r
        return r

    def remove(self, user_id: int, exchange_id: str) -> bool:
        with self._lock:
            r = self._r.pop(self.key(user_id, exchange_id), None)
        if r is None:
            return False
        r.stop("kaldırıldı"); r.broker.close()
        return True

    def restore_all(self) -> List[Dict]:
        out = []
        for f in sorted(_live_dir(self.output_dir).glob("runner_*.json")):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                cfg = RunnerConfig.from_dict(d.get("config"))
                uid = int(d.get("user_id"))
                creds = None; manage_only = False
                if cfg.mode != "paper":
                    creds = self.creds_lookup(uid, cfg.exchange_id) if self.creds_lookup else None
                    if not creds:
                        out.append({"file": f.name, "restored": False, "why": "anahtar bulunamadı (kasa kilitli?)"})
                        continue
                    manage_only = True
                r = self.create(uid, cfg, creds=creds, manage_only=manage_only, restore=d)
                if d.get("running") and (cfg.mode == "paper" or r.positions):
                    r.running = True; r.manage_only = manage_only
                    r._thread = threading.Thread(target=r._loop, daemon=True, name=f"cm-runner-{uid}-{cfg.exchange_id}")
                    r._thread.start()
                    r._log("system", "♻️ yeniden başlatma sonrası devam" + (" (yalnız pozisyon yönetimi)" if manage_only else ""))
                out.append({"file": f.name, "restored": True, "mode": cfg.mode, "manage_only": manage_only, "positions": len(r.positions)})
            except Exception as e:
                out.append({"file": f.name, "restored": False, "why": f"{type(e).__name__}: {e}"})
        return out

    def summary(self, user_id: int) -> Dict[str, Dict]:
        out = {}
        for r in self.all_for(user_id):
            st = r.stats()
            out[r.cfg.exchange_id] = {"mode": r.cfg.mode, "strategy": r.cfg.strategy, "running": r.running,
                                      "manage_only": r.manage_only, "halted": r.guard.state.halted,
                                      "open_positions": len(r.positions), "equity": st["equity"], "net_pnl": st["net_pnl"],
                                      "fees_paid": st["fees_paid"], "closed_trades": st["closed_trades"],
                                      "cycle": r.cycle, "last_cycle_ts": r.last_cycle_ts}
        return out


# ---------------------------------------------------------------- yardımcı
def _r(x, nd: int = 3):
    try:
        return None if x is None else round(float(x), nd)
    except (TypeError, ValueError):
        return None


def _vol_label(sigma_bar_pct) -> str:
    try:
        s = float(sigma_bar_pct)
    except (TypeError, ValueError):
        return "medium"
    if s < 0.05:
        return "low"
    if s < 0.15:
        return "medium"
    if s < 0.35:
        return "high"
    return "extreme"
