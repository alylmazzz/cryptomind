"""
CHAMPION / CHALLENGER — öğrenme motoru üretim parametresini DOĞRUDAN değiştirmez.

Ders motorunun önerdiği geçersiz kılmalar bir ChallengerConfig'e yazılır. Koşucu her adayı
hem champion (üretim) hem challenger parametreleriyle değerlendirir; challenger'ın kararı
GÖLGE'de (emir yok) izlenir. Yeterli çözülmüş gölge (≥ MIN_N) ve champion'dan anlamlı
biçimde iyi beklenti (Wilson %95 alt sınır ile) varsa terfi eder; kötüyse atılır.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

MIN_N = 30
MIN_EDGE_PTS = 0.05          # kazanma oranı farkı (alt sınır) en az 5 puan


def _wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    p = k / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    adj = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - adj) / den)


class Challenger:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.params: Dict = {}                 # challenger geçersiz kılmaları
        self.since: Optional[float] = None
        self.decisions: List[Dict] = []        # {ts, symbol, champion_allowed, challenger_allowed, outcome}
        self.history: List[Dict] = []
        if self.path and self.path.exists():
            try:
                d = json.loads(self.path.read_text(encoding="utf-8"))
                self.params = d.get("params", {}); self.since = d.get("since")
                self.decisions = d.get("decisions", []); self.history = d.get("history", [])
            except Exception:
                pass

    def save(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"params": self.params, "since": self.since,
                                             "decisions": self.decisions[-500:], "history": self.history[-50:]},
                                            ensure_ascii=False, default=str), encoding="utf-8")
        except Exception:
            pass

    def propose(self, overrides: Dict) -> bool:
        """Ders motoru yeni geçersiz kılma üretince challenger olur (champion değişmez)."""
        if not overrides or overrides == self.params:
            return False
        self.params = dict(overrides)
        self.since = time.time()
        self.decisions = []
        self.history.append({"ts": self.since, "event": "proposed", "params": dict(overrides)})
        self.save()
        return True

    def record(self, symbol: str, champion_allowed: bool, challenger_allowed: bool,
               plan: Optional[Dict], now: float) -> None:
        if not self.params or champion_allowed == challenger_allowed:
            return
        self.decisions.append({"ts": now, "symbol": symbol, "champion": champion_allowed,
                               "challenger": challenger_allowed, "plan": plan, "outcome": None})
        self.decisions = self.decisions[-500:]

    def resolve(self, outcomes: Dict[str, str]) -> None:
        """outcomes: symbol@ts → TARGET|STOP|TIMEOUT (koşucunun gölge takibinden)."""
        for d in self.decisions:
            key = f"{d['symbol']}@{int(d['ts'])}"
            if d["outcome"] is None and key in outcomes:
                d["outcome"] = outcomes[key]

    def evaluate(self) -> Dict:
        """Challenger'ın TEK BAŞINA açtığı (champion'ın reddettiği) adaylar ne yaptı; champion'ın
        tek başına açtıkları (challenger reddetti) ne yaptı? Fark = challenger değeri."""
        ch_only = [d for d in self.decisions if d["challenger"] and not d["champion"] and d["outcome"] in ("TARGET", "STOP")]
        cp_only = [d for d in self.decisions if d["champion"] and not d["challenger"] and d["outcome"] in ("TARGET", "STOP")]
        n1 = len(ch_only); k1 = sum(1 for d in ch_only if d["outcome"] == "TARGET")
        n2 = len(cp_only); k2 = sum(1 for d in cp_only if d["outcome"] == "TARGET")
        lo1 = _wilson_lower(k1, n1) if n1 else 0.0
        win2 = (k2 / n2) if n2 else None
        promote = bool(self.params and n1 >= MIN_N and lo1 >= 0.5 + MIN_EDGE_PTS and (win2 is None or lo1 > win2))
        reject = bool(self.params and n1 >= MIN_N and (k1 / n1) < 0.45)
        return {"params": self.params, "since": self.since, "n_challenger_only": n1, "win_challenger_only": (k1 / n1 if n1 else None),
                "wilson_lower": round(lo1, 3), "n_champion_only": n2, "win_champion_only": win2,
                "promote": promote, "reject": reject, "min_n": MIN_N}

    def conclude(self, promote: bool) -> Dict:
        ev = self.evaluate()
        self.history.append({"ts": time.time(), "event": "promoted" if promote else "rejected",
                             "params": dict(self.params), "eval": ev})
        out = dict(self.params) if promote else {}
        self.params = {}; self.since = None; self.decisions = []
        self.save()
        return out
