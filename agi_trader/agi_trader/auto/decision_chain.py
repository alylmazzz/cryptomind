"""
Karar zinciri — CryptoMind'ın kendi sistemleri her girişte OY ve VETO verir.

Video stratejisi bir ADAY üretir ("bu parite dipte, LONG"). Aday tek başına
emir olmaz; sırayla şu kapılardan geçer ve her kapı ya geçirir, ya boyutu
ölçekler, ya da veto eder:

  1. SAĞLIK       /api/system-health — RED/UNKNOWN → hiçbir giriş yok
  2. KOMİSYON     brüt hedef / gidiş-dönüş maliyet ≥ K (videonun dersi)
  3. KONSENSÜS    orkestratörün 4h sinyali ters ve güçlüyse VETO; aynıysa ×1,25
  4. NİTELENDİRME parite×ufuk×yön hücresi — QUALIFIED ×1, NO_EDGE ×0,5 (katı: veto)
  5. FIRSAT       opportunity.engine kapıları (net getiri, EV, likidite, veri) — ret = veto
  6. REJİM        analysis.regime.position_multiplier (trende karşı ×0,5 …)

Çıktı her zaman İZ taşır: hangi kapı ne dedi. "Neden işlem açılmadı?" sorusu
panelde bu izle cevaplanır. Bu modül ağ erişimi yapmaz; girdiler dışarıdan gelir.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from ..opportunity.engine import Gates, build_price_opportunity, evaluate

PASS, VETO, SCALE, SKIP = "PASS", "VETO", "SCALE", "SKIP"

# Nitelendirme durumu → boyut çarpanı. Kaynak: qualification/state.py durumları.
QUAL_MULT = {
    "HIGH_CONFIDENCE": 1.0, "QUALIFIED": 1.0,
    "UNVERIFIED": 0.5, "RESEARCH_ONLY": 0.5, "NO_DATA": 0.5,
    "NO_EDGE": 0.5, "DEGRADED": 0.0,
}


@dataclass
class ChainConfig:
    require_health: bool = True
    min_gross_to_cost: float = 2.0
    use_consensus: bool = True
    consensus_veto_conf: float = 0.60     # ters yön bu güvenin üstündeyse veto
    consensus_boost: float = 1.25
    flat_mult: float = 0.75               # orkestratör FLAT/bilinmiyor → biraz küçült
    use_qualification: bool = True
    strict_qualification: bool = False    # True: NİTELENMEMİŞ hücre → veto
    unqualified_mult: float = 0.5
    use_opportunity_gates: bool = True
    min_net_return_pct: float = 0.30      # scalp için net eşik (NET1 programı %1'dir)
    p_timeout: float = 0.10
    use_regime: bool = True
    max_size_mult: float = 1.5

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict]) -> "ChainConfig":
        c = cls()
        for k, v in (d or {}).items():
            if hasattr(c, k) and v is not None:
                cur = getattr(c, k)
                try:
                    setattr(c, k, bool(v) if isinstance(cur, bool) else type(cur)(v))
                except (TypeError, ValueError):
                    pass
        c.min_gross_to_cost = float(min(10.0, max(1.0, c.min_gross_to_cost)))
        c.consensus_veto_conf = float(min(1.0, max(0.3, c.consensus_veto_conf)))
        c.max_size_mult = float(min(3.0, max(0.1, c.max_size_mult)))
        return c


@dataclass
class ChainInputs:
    symbol: str
    direction: str                        # LONG | SHORT
    entry: float
    target_gross_pct: float               # stratejinin brüt hedefi (%)
    stop_pct: float
    cost_pct: float                       # gidiş-dönüş maliyet (%)
    notional: float = 100.0
    horizon: str = "1h"
    cm_signal: Optional[Dict] = None      # orkestratör anlık görüntüsü (TradeSignal.to_dict)
    qual_cell: Optional[Dict] = None      # nitelendirme hücresi
    regime: Optional[Dict] = None         # analysis.regime.detect_regime çıktısı
    volatility: str = "medium"
    system_health: Optional[Dict] = None  # /api/system-health
    p_target: float = 0.5                 # strateji istatistiğinden (yoksa 0,5 öncül)
    confidence: float = 0.6               # stratejinin KENDİ güveni (orkestratörünki değil)
    bid_depth_usd: float = 0.0
    ask_depth_usd: float = 0.0
    spread_bps: float = 0.0
    data_quality: float = 1.0


@dataclass
class ChainDecision:
    allowed: bool
    size_mult: float
    steps: List[Dict] = field(default_factory=list)
    vetoes: List[str] = field(default_factory=list)
    opportunity: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {"allowed": self.allowed, "size_mult": round(self.size_mult, 3),
                "steps": self.steps, "vetoes": self.vetoes,
                "opportunity": self.opportunity}


def _step(steps: List[Dict], gate: str, status: str, note: str, mult: float = 1.0):
    steps.append({"gate": gate, "status": status, "note": note,
                  "mult": round(float(mult), 3)})


def decide(inp: ChainInputs, cfg: ChainConfig) -> ChainDecision:
    steps: List[Dict] = []
    vetoes: List[str] = []
    mult = 1.0
    opp_dict: Optional[Dict] = None

    # 1) SAĞLIK
    if cfg.require_health:
        h = inp.system_health or {}
        overall = str(h.get("overall") or h.get("state") or "UNKNOWN").upper()
        if overall in ("RED", "UNKNOWN"):
            vetoes.append(f"SAĞLIK {overall}")
            _step(steps, "SAĞLIK", VETO, f"sistem sağlığı {overall} — giriş yok")
        elif overall == "DEGRADED":
            mult *= 0.5
            _step(steps, "SAĞLIK", SCALE, "DEGRADED → boyut ×0,5", 0.5)
        else:
            _step(steps, "SAĞLIK", PASS, f"sistem {overall}")
    else:
        _step(steps, "SAĞLIK", SKIP, "kapalı")

    # 2) KOMİSYON KAPISI (videonun dersi)
    if inp.cost_pct > 0:
        ratio = inp.target_gross_pct / inp.cost_pct
        if ratio < cfg.min_gross_to_cost:
            vetoes.append("KOMİSYON")
            _step(steps, "KOMİSYON", VETO,
                  f"brüt hedef %{inp.target_gross_pct:.3f} maliyetin {ratio:.2f} katı "
                  f"< {cfg.min_gross_to_cost} — bot kazanır, hesap erir")
        else:
            _step(steps, "KOMİSYON", PASS,
                  f"brüt/maliyet {ratio:.2f} ≥ {cfg.min_gross_to_cost}")
    else:
        _step(steps, "KOMİSYON", PASS, "maliyet sıfır varsayıldı (uyarı)")

    # 3) KONSENSÜS
    if cfg.use_consensus:
        s = inp.cm_signal or {}
        d = str(s.get("direction") or "FLAT").upper()
        conf = float(s.get("confidence") or 0.0)
        cls = str(s.get("signal_class") or "")
        if cls == "acil_cikis":
            vetoes.append("KONSENSÜS acil_cikis")
            _step(steps, "KONSENSÜS", VETO, "orkestratör ACİL ÇIKIŞ sınıfında")
        elif d in ("LONG", "SHORT") and d != inp.direction and conf >= cfg.consensus_veto_conf:
            vetoes.append(f"KONSENSÜS ters {d} %{conf*100:.0f}")
            _step(steps, "KONSENSÜS", VETO,
                  f"orkestratör {d} %{conf*100:.0f} güvenle ters yönde")
        elif d == inp.direction:
            mult *= cfg.consensus_boost
            _step(steps, "KONSENSÜS", SCALE,
                  f"orkestratör aynı yönde ({d} %{conf*100:.0f}) ×{cfg.consensus_boost}",
                  cfg.consensus_boost)
        elif d in ("LONG", "SHORT"):
            mult *= cfg.flat_mult
            _step(steps, "KONSENSÜS", SCALE,
                  f"orkestratör ters ama zayıf ({d} %{conf*100:.0f}) ×{cfg.flat_mult}",
                  cfg.flat_mult)
        else:
            mult *= cfg.flat_mult
            _step(steps, "KONSENSÜS", SCALE,
                  "orkestratör FLAT/yok ×%.2f" % cfg.flat_mult, cfg.flat_mult)
    else:
        _step(steps, "KONSENSÜS", SKIP, "kapalı")

    # 4) NİTELENDİRME
    if cfg.use_qualification:
        c = inp.qual_cell or {}
        st = str(c.get("status") or "NO_DATA").upper()
        m = QUAL_MULT.get(st, 0.5)
        if st in ("QUALIFIED", "HIGH_CONFIDENCE"):
            _step(steps, "NİTELENDİRME", PASS, f"hücre {st}")
        elif st == "DEGRADED":
            vetoes.append("NİTELENDİRME DEGRADED")
            _step(steps, "NİTELENDİRME", VETO, "hücre BOZULDU")
        elif cfg.strict_qualification:
            vetoes.append(f"NİTELENDİRME {st} (katı)")
            _step(steps, "NİTELENDİRME", VETO, f"hücre {st} — katı modda giriş yok")
        else:
            m = cfg.unqualified_mult if st != "DEGRADED" else 0.0
            mult *= m
            p = c.get("p_model_live")
            ek = f", p_model=%{p*100:.1f}" if isinstance(p, (int, float)) else ""
            _step(steps, "NİTELENDİRME", SCALE,
                  f"hücre {st}{ek} → boyut ×{m}", m)
    else:
        _step(steps, "NİTELENDİRME", SKIP, "kapalı")

    # 5) FIRSAT KAPILARI (maliyet modeli + EV)
    if cfg.use_opportunity_gates:
        net_target = max(0.0, inp.target_gross_pct - inp.cost_pct)
        # derinlik ölçülmediyse notional'ın 50 katı varsayılır ve beyan edilir
        bid = inp.bid_depth_usd or inp.notional * 50.0
        ask = inp.ask_depth_usd or inp.notional * 50.0
        p_t = float(min(0.95, max(0.05, inp.p_target)))
        p_s = max(0.0, 1.0 - p_t - cfg.p_timeout)
        try:
            op = build_price_opportunity(
                inp.symbol, inp.direction, inp.entry, net_target, inp.stop_pct,
                bid_depth=bid, ask_depth=ask, spread_bps=inp.spread_bps,
                notional=inp.notional, p_target=p_t, p_stop=p_s,
                p_timeout=cfg.p_timeout, horizon=inp.horizon,
                holding_hours=1.0, data_quality=inp.data_quality,
                confidence=float(inp.confidence))
            gates = Gates(min_net_return_pct=cfg.min_net_return_pct,
                          min_expected_value_pct=0.0, calibrated=False)
            op = evaluate(op, gates)
            opp_dict = {"action": op.action, "score": op.score,
                        "net_return_pct": round(op.net_return_pct, 4),
                        "expected_value_pct": (None if not math.isfinite(op.expected_value_pct)
                                               else round(op.expected_value_pct, 4)),
                        "reject_reasons": op.reject_reasons,
                        "liquidity_score": round(op.liquidity_score, 3),
                        "depth_assumed": not bool(inp.bid_depth_usd and inp.ask_depth_usd)}
            if op.action == "NO_TRADE":
                vetoes.append("FIRSAT " + "; ".join(op.reject_reasons[:2]))
                _step(steps, "FIRSAT", VETO, "; ".join(op.reject_reasons))
            else:
                _step(steps, "FIRSAT", PASS,
                      f"skor {op.score:.0f} · EV %{op.expected_value_pct:.3f}"
                      + (" · derinlik VARSAYIM" if opp_dict["depth_assumed"] else ""))
        except Exception as e:  # kapı çalışmadıysa geçirme — fail closed
            vetoes.append(f"FIRSAT hata {type(e).__name__}")
            _step(steps, "FIRSAT", VETO, f"fırsat motoru hatası: {type(e).__name__}")
    else:
        _step(steps, "FIRSAT", SKIP, "kapalı")

    # 6) REJİM
    if cfg.use_regime:
        try:
            from ..analysis.regime import position_multiplier
            m = float(position_multiplier(inp.regime or {}, inp.direction, inp.volatility))
        except Exception:
            m = 1.0
        if m <= 0.0:
            vetoes.append("REJİM ×0")
            _step(steps, "REJİM", VETO, "rejim çarpanı 0")
        else:
            mult *= m
            lbl = (inp.regime or {}).get("label", "?")
            _step(steps, "REJİM", SCALE if m != 1.0 else PASS,
                  f"{lbl} · oynaklık {inp.volatility} ×{m}", m)
    else:
        _step(steps, "REJİM", SKIP, "kapalı")

    mult = float(min(cfg.max_size_mult, max(0.0, mult)))
    allowed = not vetoes and mult > 0.0
    return ChainDecision(allowed=allowed, size_mult=mult if allowed else 0.0,
                         steps=steps, vetoes=vetoes, opportunity=opp_dict)
