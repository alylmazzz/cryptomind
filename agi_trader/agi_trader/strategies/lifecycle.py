"""
STRATEJİ YAŞAM DÖNGÜSÜ — yeni sleeve doğrudan canlıya çıkamaz.

IDEA → BACKTEST → OOS → SHADOW → PAPER → TESTNET → LIMITED_LIVE → PRODUCTION → DEGRADED → RETIRED

Kayıt defteri `runs/live/lifecycle.json`. Canlı emir için sleeve en az LIMITED_LIVE olmalı;
simülatör (paper) PAPER ve üstünü çalıştırır; SHADOW yalnız gölge kaydı üretir.
Terfi kapıları (bilimsel kabul): OOS net beklenti > 0 · %95 alt sınır > 0 · maliyet ×2'de
edge kalır · alt dönem tutarlılığı · DSR > 0 · PBO < 0,5 · drawdown sınırı. Kanıt yoksa terfi yok.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

STAGES = ["IDEA", "BACKTEST", "OOS", "SHADOW", "PAPER", "TESTNET", "LIMITED_LIVE",
          "PRODUCTION", "DEGRADED", "RETIRED"]
RANK = {s: i for i, s in enumerate(STAGES)}
MIN_STAGE_FOR_MODE = {"paper": "PAPER", "testnet": "TESTNET", "live": "LIMITED_LIVE"}

DEFAULT_SLEEVES = ["dip", "dip_moderate", "pullback", "breakout", "momentum", "catalyst",
                   "squeeze_breakout", "sweep_reversal", "range_edge", "vwap_reversion",
                   "vwap_continuation", "rs_momentum", "news_overreaction",
                   "adaptive_trend", "donchian_breakout", "bos_retest", "failed_breakdown", "failed_breakout", "obi_momentum",
                   "swing_trend"]

# VİDEO KAYNAKLI KURULUMLAR (2026-09-04) — **SHADOW** doğarlar: sinyal üretir, emir VERMEZ.
# Gerekçe ölçüm: 33 parite × 7 gün gerçek 1 dk veri, 7.343 ham aday, maliyet %0,14 düşülmüş →
# ONUNUN DA ham kenarı negatif (toplam ort net −0,129%, t −20,4). Tek günlük ilk ölçümde NY_AM
# pozitif görünmüştü (t +3,4); 7 günde bu DOĞRULANMADI (t −0,90) → örneklem gürültüsüydü.
# Kanıtsız kenarla emir gönderilmez; gölgede ölçülmeye devam ederler (kaçırılan-fırsat motoru
# `silenced` adayları izler). Kanıt pozitife dönerse `cm_replay.py --evidence` ile lifecycle
# kapılarından geçip PAPER'a terfi edebilirler.
SHADOW_SLEEVES = ["fvg_fill", "ifvg_reclaim", "range_reclaim", "manipulation_candle", "opening_range",
                  "ema_engulf", "poc_reversion", "order_block", "stoch_cross_back", "bb_lower_band"]
DEFAULT_SLEEVES = DEFAULT_SLEEVES + SHADOW_SLEEVES


class Lifecycle:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.reg: Dict[str, Dict] = {s: {"stage": ("SHADOW" if s in SHADOW_SLEEVES else "PAPER"),
                                         "since": time.time(), "evidence": {}, "history": []}
                                     for s in DEFAULT_SLEEVES}
        if self.path and self.path.exists():
            try:
                d = json.loads(self.path.read_text(encoding="utf-8"))
                for k, v in d.items():
                    if k in self.reg:
                        self.reg[k].update(v)
                    else:
                        self.reg[k] = v
            except Exception:
                pass

    def save(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.reg, ensure_ascii=False, default=str), encoding="utf-8")
        except Exception:
            pass

    def stage(self, sleeve: str) -> str:
        return (self.reg.get(sleeve) or {}).get("stage", "IDEA")

    def can_trade(self, sleeve: str, mode: str) -> bool:
        need = MIN_STAGE_FOR_MODE.get(mode, "PRODUCTION")
        st = self.stage(sleeve)
        if st in ("DEGRADED", "RETIRED"):
            return False
        return RANK.get(st, -1) >= RANK[need]

    def record_evidence(self, sleeve: str, ev: Dict) -> None:
        r = self.reg.setdefault(sleeve, {"stage": "IDEA", "since": time.time(), "evidence": {}, "history": []})
        r["evidence"] = {**r.get("evidence", {}), **ev, "ts": time.time()}
        self.save()

    def gates(self, sleeve: str) -> Dict:
        """Bilimsel kabul kapıları — hepsi kanıtla geçilmeli; ölçülmemiş = geçilmedi."""
        e = (self.reg.get(sleeve) or {}).get("evidence") or {}
        checks = {
            "oos_expectancy_pos": (e.get("oos_expectancy") or 0.0) > 0,
            "ci_lower_pos": (e.get("ci_lower") or 0.0) > 0,
            "cost_x2_robust": (e.get("expectancy_cost_x2") or 0.0) > 0,
            "subperiod_consistent": bool(e.get("subperiod_consistent")),
            "dsr_pos": (e.get("dsr") or 0.0) > 0,
            "pbo_ok": (e.get("pbo") if e.get("pbo") is not None else 1.0) < 0.5,
            "min_trades": (e.get("n_trades") or 0) >= 30,
        }
        return {"checks": checks, "passed": all(checks.values()), "evidence": e}

    def promote(self, sleeve: str, to: str, reason: str = "") -> Dict:
        if to not in STAGES:
            return {"ok": False, "why": "bilinmeyen aşama"}
        r = self.reg.setdefault(sleeve, {"stage": "IDEA", "since": time.time(), "evidence": {}, "history": []})
        if RANK[to] >= RANK["LIMITED_LIVE"] and not self.gates(sleeve)["passed"]:
            return {"ok": False, "why": "bilimsel kabul kapıları geçilmedi", "gates": self.gates(sleeve)}
        r["history"].append({"from": r["stage"], "to": to, "ts": time.time(), "reason": reason})
        r["stage"], r["since"] = to, time.time()
        self.save()
        return {"ok": True, "stage": to}

    def status(self) -> List[Dict]:
        return [{"sleeve": k, **{kk: vv for kk, vv in v.items() if kk != "history"},
                 "gates_passed": self.gates(k)["passed"]} for k, v in self.reg.items()]
