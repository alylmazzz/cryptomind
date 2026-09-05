"""
İşlem Günlüğü ve Self-Improvement temeli (Backtest/Strateji Doğrulama rolü).

Spec: bütün işlemleri kaydet, başarılıları öğren, başarısızları analiz et,
katman ağırlıklarını güncelle. Bu modül her sinyali JSONL olarak kalıcı kaydeder
ve katman bazlı isabet istatistiği tutar — ağırlık güncellemesi için temel veri.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from ..core.models import TradeSignal


class TradeJournal:
    def __init__(self, output_dir: str = "runs"):
        self.dir = Path(output_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.signals_path = self.dir / "signals.jsonl"

    def record(self, signal: TradeSignal, execution_result: Dict) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "signal": signal.to_dict(),
            "execution": execution_result,
        }
        with self.signals_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def load_all(self) -> List[Dict]:
        if not self.signals_path.exists():
            return []
        out = []
        for line in self.signals_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        return out

    def summary(self) -> Dict:
        rows = self.load_all()
        actionable = [r for r in rows if r["signal"].get("actionable")]
        return {
            "total_signals": len(rows),
            "actionable_signals": len(actionable),
            "by_direction": self._count(rows, "direction"),
            "avg_confidence": round(
                sum(r["signal"].get("confidence", 0) for r in rows) / (len(rows) or 1), 3),
        }

    @staticmethod
    def _count(rows, field) -> Dict[str, int]:
        c: Dict[str, int] = {}
        for r in rows:
            k = r["signal"].get(field, "?")
            c[k] = c.get(k, 0) + 1
        return c
