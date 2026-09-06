"""
İşlem Günlüğü — sinyal kaydı ve katman isabet istatistiği (ağırlık öğrenmesinin temeli).

YALINLAŞTIRMA (2026-09-06 — ölçümle)
────────────────────────────────────
Eski sürüm her sinyalin TAM `to_dict()` çıktısını yazıyordu: **kayıt başına 6.626 bayt**
ve dosya 118 MB'a ulaşmıştı. Kaydın %97,5'i tek bir `signal` bloğuydu. Oysa bu dosyanın
gerçek tüketicileri yalnız şunları okuyor:

  • `learn/weight_optimizer.optimize_weights` → symbol · entry · timeframe ·
    layer_breakdown[].layer · layer_breakdown[].score
  • `summary()` → actionable · direction · confidence

yani kaydın **~%4'ü**. Geri kalan (risk analizi, gerekçe metinleri, senaryolar, gösterge
dökümleri) hiçbir yerden OKUNMUYORDU — 113 MB salt-yazım çöpü.

Ayrıca `load_all()` dosyanın TAMAMINI `read_text().splitlines()` ile belleğe alıyordu.
118 MB'lık dosyada bu birkaç yüz MB'lık bellek tepesi demektir; 3,8 GB RAM'li ve pm2
tavanı 1600 MB olan bu sunucuda gerçek bir OOM tuzağıydı. Okuma artık AKIŞ ve SINIRLI.

Kayıt boyutu: ~6.626 B → ~260 B (**%96 azalma**). Dosya sınırlı (`MAX_SATIR`).
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from ..core.models import TradeSignal

# Tüketicilerin GERÇEKTEN okuduğu alanlar. Buraya alan eklemeden önce
# "hangi kod bunu okuyor?" sorusu cevaplanmalı — aksi hâlde salt-yazım çöpü olur.
SINYAL_ALANLARI = ("symbol", "entry", "timeframe", "direction", "bias",
                   "confidence", "actionable", "signal_class")
KATMAN_ALANLARI = ("layer", "score")
MAX_SATIR = 120_000          # ~260 B × 120k ≈ 31 MB tavan; aşınca en eskiler arşive taşınır


def _kucult(d: Dict) -> Dict:
    """Tam sinyal sözlüğü → tüketicilerin okuduğu alt küme."""
    out = {k: d.get(k) for k in SINYAL_ALANLARI if d.get(k) is not None}
    lb = d.get("layer_breakdown") or []
    if lb:
        out["layer_breakdown"] = [{k: b.get(k) for k in KATMAN_ALANLARI if b.get(k) is not None}
                                  for b in lb if isinstance(b, dict)]
    return out


class TradeJournal:
    def __init__(self, output_dir: str = "runs"):
        self.dir = Path(output_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.signals_path = self.dir / "signals.jsonl"

    # ------------------------------------------------------------------ yazma
    def record(self, signal: TradeSignal, execution_result: Dict) -> None:
        try:
            sig = _kucult(signal.to_dict())
        except Exception:
            return
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "signal": sig,
            "execution": {k: execution_result.get(k)
                          for k in ("status", "action", "reason", "filled")
                          if isinstance(execution_result, dict) and execution_result.get(k) is not None},
        }
        try:
            with self.signals_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")
        except Exception:
            return
        self._dondur_gerekirse()

    def _dondur_gerekirse(self, her: int = 500) -> None:
        """Ucuz kontrol: her `her` yazımda bir satır say, tavanı aşınca yarısını arşivle
        (SİLME yok — arşiv dosyasına taşınır)."""
        self._sayac = getattr(self, "_sayac", 0) + 1
        if self._sayac % her:
            return
        try:
            with self.signals_path.open(encoding="utf-8") as f:
                n = sum(1 for _ in f)
            if n <= MAX_SATIR:
                return
            tut = MAX_SATIR // 2
            with self.signals_path.open(encoding="utf-8") as f:
                satirlar = f.readlines()
            ars = self.signals_path.with_suffix(f".{int(time.time())}.arsiv.jsonl")
            ars.write_text("".join(satirlar[:-tut]), encoding="utf-8")
            tmp = self.signals_path.with_suffix(".tmp")
            tmp.write_text("".join(satirlar[-tut:]), encoding="utf-8")
            os.replace(tmp, self.signals_path)
        except Exception:
            pass

    # ------------------------------------------------------------------ okuma
    def iter_rows(self, limit: Optional[int] = None) -> Iterator[Dict]:
        """AKIŞ okuma — bellek sabit. `limit` verilirse yalnız SON `limit` satır."""
        if not self.signals_path.exists():
            return
        f2 = None
        if limit:
            buf: deque = deque(maxlen=int(limit))
            with self.signals_path.open(encoding="utf-8") as f:
                for line in f:
                    buf.append(line)
            kaynak: object = list(buf)
        else:
            f2 = self.signals_path.open(encoding="utf-8")
            kaynak = f2
        try:
            for line in kaynak:                      # type: ignore[union-attr]
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue
        finally:
            if f2 is not None:
                f2.close()

    def load_all(self, limit: int = 20_000) -> List[Dict]:
        """Geriye dönük uyumlu ama ARTIK SINIRLI (varsayılan son 20.000 kayıt).
        Sınırsız okuma dosya büyüdükçe belleği patlatıyordu (ölçüm: 118 MB dosya)."""
        return list(self.iter_rows(limit=limit))

    def summary(self) -> Dict:
        """Tek geçiş, SABİT bellek — satırları biriktirmez."""
        n = akt = 0
        yon: Dict[str, int] = {}
        guven = 0.0
        for r in self.iter_rows():
            s = r.get("signal") or {}
            n += 1
            if s.get("actionable"):
                akt += 1
            k = s.get("direction", "?")
            yon[k] = yon.get(k, 0) + 1
            guven += float(s.get("confidence") or 0.0)
        return {
            "total_signals": n,
            "actionable_signals": akt,
            "by_direction": yon,
            "avg_confidence": round(guven / n, 3) if n else 0.0,
        }

    @staticmethod
    def _count(rows, field) -> Dict[str, int]:
        c: Dict[str, int] = {}
        for r in rows:
            k = (r.get("signal") or {}).get(field, "?")
            c[k] = c.get(k, 0) + 1
        return c
