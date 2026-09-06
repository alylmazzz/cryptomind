"""
İşlem sonrası maliyet analizi — TCA (FAZ 8).

NEDEN: Backtest'in varsaydığı maliyet ile gerçekte ödenen maliyet ayrışırsa,
canlı sonuç backtest'in altında kalır ve NEDEN bilinmez. TCA bu farkı ölçer:
her emir için varış fiyatına göre kayma, maker/taker oranı, dolum oranı.

Bu ölçüm `risk_engine`'in kayma modelini KALİBRE eder — varsayım yerine
gözlem kullanılır. Kalibrasyon olmadan backtest gerçekçiliği bir tahmindir.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .twap import slippage_bps


def _log_path(output_dir: str = "runs") -> Path:
    p = Path(output_dir)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[2] / p
    p.mkdir(parents=True, exist_ok=True)
    return p / "fills.jsonl"


def record_fill(symbol: str, side: str, qty: float, ref_price: float,
                fill_price: float, order_type: str, fee: float = 0.0,
                requested_qty: Optional[float] = None,
                output_dir: str = "runs") -> Dict:
    """Tek dolumu kaydet. Canlı yürütmede HER emir için çağrılmalı."""
    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "symbol": symbol, "side": side.upper(), "qty": float(qty),
        "requested_qty": float(requested_qty if requested_qty is not None else qty),
        "ref_price": float(ref_price), "fill_price": float(fill_price),
        "order_type": order_type, "fee": float(fee),
        "slippage_bps": round(slippage_bps(fill_price, ref_price, side), 3),
        "fill_ratio": round(float(qty) / float(requested_qty or qty or 1), 4),
    }
    p = _log_path(output_dir)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
    _rotate(p)
    return rec


MAX_FILLS = 60_000          # ~430 B × 60k ≈ 26 MB tavan (kalibrasyon için ≥50 dolum yeter)
_YAZIM = 0


def _rotate(p: Path, her: int = 500) -> None:
    """TCA kalibrasyonu SON dolumlara bakar; dosyanın sınırsız büyümesine gerek yok.
    Tavan aşılınca en eskiler arşive TAŞINIR (silinmez). Kontrol her `her` yazımda bir —
    her dolumda satır saymak, ölçüm yapayım derken CPU harcamak olurdu."""
    global _YAZIM
    _YAZIM += 1
    if _YAZIM % her:
        return
    try:
        with open(p, encoding="utf-8") as f:
            satirlar = f.readlines()
        if len(satirlar) <= MAX_FILLS:
            return
        tut = MAX_FILLS // 2
        p.with_suffix(f".{int(time.time())}.arsiv.jsonl").write_text("".join(satirlar[:-tut]), encoding="utf-8")
        tmp = p.with_suffix(".tmp")
        tmp.write_text("".join(satirlar[-tut:]), encoding="utf-8")
        tmp.replace(p)
    except Exception:
        pass


def load_fills(output_dir: str = "runs", limit: int = 20_000) -> List[Dict]:
    """SON `limit` dolum — sabit bellek (kalibrasyon zaten son dolumlara bakar)."""
    p = _log_path(output_dir)
    if not p.exists():
        return []
    out = []
    from collections import deque
    with open(p, encoding="utf-8") as f:
        kaynak = deque(f, maxlen=limit)
    for line in kaynak:
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def tca_report(fills: Optional[List[Dict]] = None,
               assumed_cost_bps: float = 6.0,
               output_dir: str = "runs") -> Dict:
    """Gerçekleşen yürütme maliyeti vs backtest varsayımı.

    assumed_cost_bps: backtest'te kullanılan tek-yön maliyet (varsayılan 6 bps
    = taker %0,04 + kayma tabanı %0,015 + pay)."""
    fills = fills if fills is not None else load_fills(output_dir)
    if not fills:
        return {"available": False, "reason": "henüz dolum kaydı yok",
                "assumed_cost_bps": assumed_cost_bps}

    sl = np.array([f.get("slippage_bps", 0.0) for f in fills], dtype=float)
    fr = np.array([f.get("fill_ratio", 1.0) for f in fills], dtype=float)
    # HATA DÜZELTİLDİ (2026-09-06): yalnız "limit" aranıyordu; koşucu ise "maker"/"taker"
    # gönderiyor. Sonuç: `maker_share` HER ZAMAN 0,0 çıkıyordu — yani maker/taker maliyet
    # öğrenmesi tamamen KÖRDÜ. Ölçmediğini bildiğini sanmak, ölçmemekten kötüdür.
    maker = np.array([1.0 if any(k in str(f.get("order_type", "")).lower()
                                 for k in ("maker", "limit")) else 0.0
                      for f in fills])
    notional = np.array([abs(f.get("qty", 0)) * abs(f.get("fill_price", 0))
                         for f in fills], dtype=float)
    fee_bps = np.array([(f.get("fee", 0.0) / (abs(f.get("qty", 0)) *
                                              abs(f.get("fill_price", 0)) + 1e-12)) * 1e4
                        for f in fills], dtype=float)

    realized = float(np.average(sl, weights=notional + 1e-12) +
                     np.average(fee_bps, weights=notional + 1e-12))
    drift = realized - assumed_cost_bps
    return {
        "available": True,
        "n_fills": len(fills),
        "total_notional": round(float(notional.sum()), 2),
        "mean_slippage_bps": round(float(np.average(sl, weights=notional + 1e-12)), 2),
        "median_slippage_bps": round(float(np.median(sl)), 2),
        "p90_slippage_bps": round(float(np.percentile(sl, 90)), 2),
        "mean_fee_bps": round(float(np.average(fee_bps, weights=notional + 1e-12)), 2),
        "maker_share": round(float(maker.mean()), 3),
        "mean_fill_ratio": round(float(fr.mean()), 3),
        "realized_cost_bps": round(realized, 2),
        "assumed_cost_bps": assumed_cost_bps,
        "drift_bps": round(drift, 2),
        "verdict": ("gerçek maliyet varsayımın ÜSTÜNDE — backtest iyimser, "
                    "risk_engine.slippage_base yükseltilmeli"
                    if drift > 2 else
                    "gerçek maliyet varsayımla uyumlu" if abs(drift) <= 2 else
                    "gerçek maliyet varsayımın ALTINDA — maker yürütme çalışıyor"),
    }


def suggest_slippage_calibration(report: Dict, current_base: float = 0.00015) -> Dict:
    """TCA raporundan `risk.slippage_base` önerisi.

    Varsayım yerine ölçüm: kalibrasyon backtest'i gerçeğe yaklaştırır, ama
    yalnız yeterli örnek varken (≥50 dolum) anlamlıdır."""
    if not report.get("available") or report.get("n_fills", 0) < 50:
        return {"ready": False,
                "reason": f"kalibrasyon için ≥50 dolum gerekli "
                          f"({report.get('n_fills', 0)} var)"}
    realized_slip = report["mean_slippage_bps"] / 1e4
    return {"ready": True, "current": current_base,
            "suggested": round(max(0.0, realized_slip), 6),
            "note": "yalnız kayma (ücret ayrı); değişiklik config.yaml risk.slippage_base"}
