"""
Online / Artımlı Öğrenme — kapalı-döngü self-improvement.

Otonom motor bir işlemi KAPATTIĞINDA (gerçek kâr/zarar), o işlemi açan sinyalin
katman katkıları + sonuç (kazandı/kaybetti) bu öğreniciye verilir. Her katman
için "kârlı yönü doğru işaret etti mi" ağırlıklı olarak biriktirilir. Her N
işlemde bir, katman ağırlıkları isabet oranına göre yeniden hesaplanır, karar
motoruna CANLI uygulanır ve diske yazılır (kalıcı öğrenme).

Mevcut /api/learn (fiyat-proxy) manuel toplu öğrenmedir; bu modül ise GERÇEK
işlem sonuçlarıyla otomatik, sürekli öğrenir — daha sağlam sinyaldir.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional


class OnlineLearner:
    def __init__(self, base_weights: Dict[str, float], output_dir: str = "runs",
                 update_every: int = 5):
        self.base = {k: float(v) for k, v in (base_weights or {}).items()}
        self.output_dir = output_dir
        self.update_every = max(2, int(update_every))
        # katman istatistiği: ağırlıklı isabet
        self.stats: Dict[str, Dict[str, float]] = {k: {"win_w": 0.0, "tot_w": 0.0} for k in self.base}
        self.trades_since = 0
        self.total_trades = 0
        self.history: List[Dict] = []   # ağırlık anlık görüntüleri
        self.last_weights: Optional[Dict[str, float]] = None

    def record(self, layers: Dict[str, float], direction: str, won: bool) -> Optional[Dict[str, float]]:
        """Bir kapanan işlemi öğren. layers: {katman: contribution(işaretli)}.
        Kârlı yön = won ise işlem yönü, değilse tersi. Döndürür: güncellendiyse
        yeni ağırlıklar, yoksa None."""
        dsign = 1 if direction == "LONG" else -1
        prof = dsign if won else -dsign
        for name, contrib in (layers or {}).items():
            if name.startswith("_") or abs(contrib) < 1e-4:
                continue
            st = self.stats.setdefault(name, {"win_w": 0.0, "tot_w": 0.0})
            w = abs(float(contrib))
            st["tot_w"] += w
            if (contrib > 0) == (prof > 0):
                st["win_w"] += w
        self.trades_since += 1
        self.total_trades += 1
        if self.trades_since >= self.update_every:
            return self._recompute()
        return None

    def _recompute(self) -> Dict[str, float]:
        self.trades_since = 0
        acc = self.accuracy()
        keys = set(self.base) | set(acc)
        # taban ağırlık × (0.5 + isabet) → isabetli katman büyür, isabetsiz küçülür
        raw = {k: max(0.02, self.base.get(k, 0.05) * (0.5 + acc.get(k, 0.5))) for k in keys}
        tot = sum(raw.values()) + 1e-12
        weights = {k: round(v / tot, 4) for k, v in raw.items()}
        self.last_weights = weights
        self.history.append({"ts": time.time(), "trades": self.total_trades, "weights": weights})
        if len(self.history) > 200:
            self.history = self.history[-200:]
        self._persist(weights, acc)
        return weights

    def accuracy(self) -> Dict[str, float]:
        return {k: round(s["win_w"] / s["tot_w"], 3) if s["tot_w"] > 0 else 0.5
                for k, s in self.stats.items()}

    def _persist(self, weights: Dict[str, float], acc: Dict[str, float]) -> None:
        try:
            p = Path(self.output_dir) / "learned_weights.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({
                "weights": weights, "accuracy": acc, "source": "online",
                "total_trades": self.total_trades, "updated_at": time.time(),
            }, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def status(self) -> Dict:
        acc = self.accuracy()
        best = max(acc, key=acc.get) if acc else None
        return {
            "total_trades": self.total_trades,
            "trades_until_update": max(0, self.update_every - self.trades_since),
            "update_every": self.update_every,
            "accuracy": acc,
            "best_layer": best,
            "best_layer_acc": acc.get(best) if best else None,
            "weights": self.last_weights,
            "updates": len(self.history),
        }
