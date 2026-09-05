"""
Otomatik Ağırlık Öğrenme / Self-Improvement (#7).

Journal'daki geçmiş sinyalleri GERÇEK sonraki fiyat hareketiyle değerlendirir:
her sinyal için sembolün güncel fiyatını çeker, gerçekleşen yönü belirler ve
her analiz katmanının o yönü doğru tahmin edip etmediğini sayar. Katman
ağırlıkları, isabet oranlarına orantılı olarak güncellenir ve
runs/learned_weights.json'a yazılır. Orchestrator başlangıçta bu dosyayı
varsa yükler (sürekli öğrenme döngüsü).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

DEFAULT_KEYS = ["technical", "pattern", "smc", "multi_timeframe",
                "ai_ensemble", "sentiment", "onchain", "macro"]


def _learned_path(output_dir: str) -> Path:
    return Path(output_dir) / "learned_weights.json"


def load_learned_weights(output_dir: str) -> Optional[Dict[str, float]]:
    p = _learned_path(output_dir)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8")).get("weights")
        except Exception:
            return None
    return None


def optimize_weights(orch, min_samples: int = 10) -> Dict:
    journal = orch.journal
    rows = journal.load_all()
    if len(rows) < min_samples:
        return {"message": f"Yetersiz geçmiş ({len(rows)}/{min_samples}). "
                           f"Daha fazla analiz çalıştırın, sonra tekrar öğrenin.",
                "samples": len(rows)}

    # her katman için doğru/toplam sayacı
    hits: Dict[str, int] = {k: 0 for k in DEFAULT_KEYS}
    totals: Dict[str, int] = {k: 0 for k in DEFAULT_KEYS}
    evaluated = 0

    price_cache: Dict[str, float] = {}
    for r in rows:
        sig = r.get("signal", {})
        symbol = sig.get("symbol")
        entry = sig.get("entry")
        if not symbol or not entry:
            continue
        # güncel fiyat
        if symbol not in price_cache:
            try:
                df = orch.data.fetch_ohlcv(symbol, sig.get("timeframe", "4h"), limit=5)
                price_cache[symbol] = float(df["close"].iloc[-1])
            except Exception:
                continue
        now_price = price_cache[symbol]
        realized = 1 if now_price > entry else -1 if now_price < entry else 0
        if realized == 0:
            continue
        evaluated += 1
        for b in sig.get("layer_breakdown", []):
            name = b.get("layer")
            if name not in totals:
                continue
            score = b.get("score", 0)
            if abs(score) < 0.1:
                continue
            totals[name] += 1
            if (score > 0) == (realized > 0):
                hits[name] += 1

    if evaluated < min_samples:
        return {"message": f"Değerlendirilebilir örnek az ({evaluated}). Zaman geçtikçe artar.",
                "evaluated": evaluated}

    # isabet oranı → ağırlık (0.5 taban, normalize)
    acc = {k: (hits[k] / totals[k] if totals[k] else 0.5) for k in DEFAULT_KEYS}
    raw = {k: max(0.03, acc[k]) for k in DEFAULT_KEYS}
    total = sum(raw.values())
    weights = {k: round(v / total, 4) for k, v in raw.items()}

    out = {
        "weights": weights,
        "accuracy": {k: round(acc[k], 3) for k in DEFAULT_KEYS},
        "evaluated_signals": evaluated,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    p = _learned_path(orch.config.get("output_dir", "runs"))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # canlı uygula
    orch.decision.base_weights = weights
    out["message"] = (f"{evaluated} sinyal değerlendirildi, ağırlıklar güncellendi ve uygulandı. "
                      f"En isabetli katman: {max(acc, key=acc.get)} (%{max(acc.values())*100:.0f}).")
    return out
