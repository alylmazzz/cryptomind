"""
ÜÇGEN / ÇOK-ADIMLI GRAF ARBİTRAJI — yön modeli gerektirmez.

  R = r1(1−f1) · r2(1−f2) · r3(1−f3)   ;  fırsat ⇔ R > 1 + kayma + gecikme rezervi
  Graf: w = −ln(r_effective); negatif döngü = arbitraj (Bellman-Ford).

Tek ticker çağrısıyla (fetch_tickers) tarama; bid/ask kullanılır (last DEĞİL — gerçek uygulanabilir fiyat).
GÖLGE: fırsat sayısı ve net getirisi loglanır; emir yok (gecikme/kısmi dolum riski ölçülmedi).
"""
from __future__ import annotations

import itertools
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _edges(tickers: Dict[str, Dict], fee: float) -> Dict[Tuple[str, str], float]:
    """(from, to) → efektif oran (1 birim from → kaç to), ücret dahil. Bid/ask yoksa atlanır."""
    e: Dict[Tuple[str, str], float] = {}
    for sym, t in tickers.items():
        if "/" not in sym or ":" in sym:
            continue
        base, quote = sym.split("/")
        bid, ask = t.get("bid"), t.get("ask")
        if not bid or not ask or bid <= 0 or ask <= 0:
            continue
        e[(quote, base)] = (1.0 / float(ask)) * (1 - fee)     # quote ile base al (ask'tan)
        e[(base, quote)] = float(bid) * (1 - fee)              # base sat (bid'e)
    return e


def find_triangles(tickers: Dict[str, Dict], start: str = "USDT", fee_bps: float = 5.0,
                   reserve_bps: float = 5.0, max_results: int = 10) -> List[Dict]:
    fee = fee_bps / 1e4
    e = _edges(tickers, fee)
    nodes = {a for a, _ in e} | {b for _, b in e}
    out = []
    thr = 1.0 + reserve_bps / 1e4
    for x, y in itertools.permutations([n for n in nodes if n != start], 2):
        r1, r2, r3 = e.get((start, x)), e.get((x, y)), e.get((y, start))
        if r1 is None or r2 is None or r3 is None:
            continue
        R = r1 * r2 * r3
        if R > thr:
            out.append({"path": [start, x, y, start], "R": round(R, 6), "net_bps": round((R - 1) * 1e4, 2),
                        "reserve_bps": reserve_bps, "fee_bps": fee_bps})
    out.sort(key=lambda o: -o["R"])
    return out[:max_results]


def bellman_ford_negative_cycle(tickers: Dict[str, Dict], fee_bps: float = 5.0, reserve_bps: float = 5.0) -> Optional[List[str]]:
    """Genel çok-adımlı arbitraj: w = −ln(r); negatif döngü var mı? Döngü düğümlerini döndürür."""
    e = _edges(tickers, fee_bps / 1e4)
    nodes = sorted({a for a, _ in e} | {b for _, b in e})
    idx = {n: i for i, n in enumerate(nodes)}
    dist = [0.0] * len(nodes)
    pred = [-1] * len(nodes)
    edges = [(idx[a], idx[b], -math.log(r)) for (a, b), r in e.items() if r > 0]
    x = -1
    for _ in range(len(nodes)):
        x = -1
        for u, v, w in edges:
            if dist[u] + w < dist[v] - reserve_bps / 1e4:
                dist[v] = dist[u] + w
                pred[v] = u
                x = v
    if x == -1:
        return None
    for _ in range(len(nodes)):
        x = pred[x]
    cyc = [x]
    v = pred[x]
    while v != x and v != -1:
        cyc.append(v)
        v = pred[v]
    cyc.append(x)
    return [nodes[i] for i in cyc[::-1]]


class TriangularShadow:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.last: List[Dict] = []
        self.history: List[Dict] = []
        self.n_scans = 0
        self.last_ts: Optional[float] = None
        self.load()

    def load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            self.history, self.n_scans, self.last_ts = d.get("history", []), int(d.get("n_scans", 0)), d.get("last_ts")
        except Exception:
            pass

    def save(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"history": self.history[-200:], "n_scans": self.n_scans, "last_ts": self.last_ts},
                                            ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def scan(self, tickers: Dict[str, Dict], now: float, fee_bps: float = 5.0, reserve_bps: float = 5.0) -> List[Dict]:
        self.n_scans += 1
        self.last_ts = now
        self.last = find_triangles(tickers, "USDT", fee_bps, reserve_bps)
        for o in self.last:
            self.history.append({"ts": now, **o})
        self.save()
        return self.last

    def status(self) -> Dict:
        return {"n_scans": self.n_scans, "last_ts": self.last_ts, "last": self.last[:5], "n_found_total": len(self.history),
                "recent": self.history[-5:][::-1], "stage": "SHADOW",
                "note": "bid/ask + ücret + rezerv sonrası R>1 sayılır; gecikme/kısmi dolum ölçülmedi, emir yok"}
