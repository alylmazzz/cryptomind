"""
FUNDING / BAZ CARRY — yön tahmini gerektirmeyen kenar (spot LONG + perp SHORT ya da tersi).

  EV = Funding + BasisConvergence − Fees − Slippage − Borrow − Rebalancing − LiquidationReserve

Funding oranları ccxt public `fetch_funding_rates()` ile TEK çağrıda (borsa başına) çekilir;
opt-in: CRYPTOMIND_RESEARCH_CARRY=1 (swap piyasa listesi bellek ister). Spot simülatörde perp
yürütme yok → GÖLGE: fırsat listesi + gölge P&L (funding tahakkuku − maliyet).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

FUNDING_PER_DAY = 3                    # 8 saatlik funding


def carry_ev(funding_rate_8h: float, days: float, taker_bps_spot: float, taker_bps_perp: float,
             spread_bps_spot: float = 2.0, spread_bps_perp: float = 2.0, slippage_bps: float = 2.0,
             borrow_daily_pct: float = 0.0, reserve_pct: float = 0.05, rebalance_pct: float = 0.02) -> Dict:
    """Pozitif funding → spot LONG + perp SHORT funding ALIR. Net EV (%) ufuk boyunca."""
    side = "SPOT_LONG_PERP_SHORT" if funding_rate_8h >= 0 else "SPOT_SHORT_PERP_LONG"
    gross = abs(float(funding_rate_8h)) * FUNDING_PER_DAY * float(days) * 100.0
    # iki bacak × giriş+çıkış taker + spread/2 + kayma
    fees = 2.0 * (taker_bps_spot + taker_bps_perp) / 100.0
    frict = (spread_bps_spot + spread_bps_perp) / 100.0 + 2.0 * slippage_bps / 100.0
    borrow = float(borrow_daily_pct) * float(days) if side == "SPOT_SHORT_PERP_LONG" else 0.0
    net = gross - fees - frict - borrow - reserve_pct - rebalance_pct
    return {"side": side, "gross_pct": round(gross, 4), "fees_pct": round(fees, 4), "friction_pct": round(frict, 4),
            "borrow_pct": round(borrow, 4), "reserve_pct": reserve_pct, "net_pct": round(net, 4), "days": days,
            "annualized_funding_pct": round(abs(float(funding_rate_8h)) * FUNDING_PER_DAY * 365 * 100.0, 2)}


def scan_funding(rates: Dict[str, Dict], days: float = 3.0, taker_bps_spot: float = 5.0, taker_bps_perp: float = 5.0,
                 min_net_pct: float = 0.05) -> List[Dict]:
    """rates: symbol → {rate (8h, ondalık), venue, mark?, index?}. Net EV'ye göre sıralı fırsatlar."""
    out = []
    for sym, r in rates.items():
        try:
            fr = float(r.get("rate") or 0.0)
        except (TypeError, ValueError):
            continue
        ev = carry_ev(fr, days, taker_bps_spot, taker_bps_perp)
        basis = None
        if r.get("mark") and r.get("index"):
            try:
                basis = (float(r["mark"]) / float(r["index"]) - 1.0) * 100.0
            except (TypeError, ValueError, ZeroDivisionError):
                basis = None
        out.append({"symbol": sym, "venue": r.get("venue"), "funding_8h_pct": round(fr * 100.0, 4), "basis_pct": (None if basis is None else round(basis, 4)),
                    **ev, "qualifies": ev["net_pct"] >= min_net_pct})
    out.sort(key=lambda x: -x["net_pct"])
    return out


def dispersion(rates_by_venue: Dict[str, Dict[str, float]], min_gap_8h: float = 0.0001) -> List[Dict]:
    """Aynı parite için borsalar arası funding farkı (long ucuz borsada / short pahalıda) — fark − maliyet."""
    syms = set()
    for v in rates_by_venue.values():
        syms |= set(v)
    out = []
    for s in syms:
        rows = [(v, float(r[s])) for v, r in rates_by_venue.items() if s in r]
        if len(rows) < 2:
            continue
        rows.sort(key=lambda x: x[1])
        lo, hi = rows[0], rows[-1]
        gap = hi[1] - lo[1]
        if gap >= min_gap_8h:
            out.append({"symbol": s, "long_venue": lo[0], "short_venue": hi[0], "gap_8h_pct": round(gap * 100, 4),
                        "gap_annualized_pct": round(gap * FUNDING_PER_DAY * 365 * 100, 2)})
    out.sort(key=lambda x: -x["gap_8h_pct"])
    return out


def fetch_funding_ccxt(exchange_id: str, symbols: List[str], client_factory: Optional[Callable] = None) -> Dict[str, Dict]:
    """Tek çağrı: fetch_funding_rates(). Hata → boş sözlük (fail-safe)."""
    try:
        if client_factory is not None:
            ex = client_factory(exchange_id, {"options": {"defaultType": "swap"}, "enableRateLimit": True})
        else:
            import ccxt
            ex = getattr(ccxt, exchange_id)({"options": {"defaultType": "swap"}, "enableRateLimit": True})
        want = {s.split(":")[0] for s in symbols}
        data = ex.fetch_funding_rates()
        out = {}
        for k, v in (data or {}).items():
            base = str(k).split(":")[0]
            if base in want and v.get("fundingRate") is not None:
                out[base] = {"rate": float(v["fundingRate"]), "venue": exchange_id, "mark": v.get("markPrice"),
                             "index": v.get("indexPrice"), "next_ts": v.get("fundingTimestamp")}
        try:
            ex.close()
        except Exception:
            pass
        return out
    except Exception:
        return {}


class CarryShadow:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.opps: List[Dict] = []
        self.open: Dict[str, Dict] = {}
        self.closed: List[Dict] = []
        self.last_ts: Optional[float] = None
        self.error: Optional[str] = None
        self.load()

    def load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            self.opps, self.open, self.closed, self.last_ts = d.get("opps", []), d.get("open", {}), d.get("closed", []), d.get("last_ts")
        except Exception:
            pass

    def save(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"opps": self.opps, "open": self.open, "closed": self.closed[-200:], "last_ts": self.last_ts},
                                            ensure_ascii=False, default=str), encoding="utf-8")
        except Exception:
            pass

    def step(self, rates: Dict[str, Dict], now: float, taker_spot: float = 5.0, taker_perp: float = 5.0, days: float = 3.0) -> List[Dict]:
        self.last_ts = now
        self.opps = scan_funding(rates, days, taker_spot, taker_perp)
        ev = []
        for o in self.opps:
            key = f"{o['symbol']}@{o.get('venue')}"
            pos = self.open.get(key)
            if pos:
                elapsed_days = (now - pos["ts"]) / 86400.0
                fr = float(rates.get(o["symbol"], {}).get("rate") or 0.0)
                pos["accrued_pct"] = round(pos.get("accrued_pct", 0.0) + abs(fr) * 100.0 * max(0.0, (now - pos.get("last_mark", pos["ts"])) / (86400.0 / FUNDING_PER_DAY)), 4)
                pos["last_mark"] = now
                pos["net_pct"] = round(pos["accrued_pct"] - pos["cost_pct"], 4)
                if elapsed_days >= days or (fr >= 0) != (pos["side"] == "SPOT_LONG_PERP_SHORT"):
                    rec = {**pos, "closed_ts": now, "reason": "HORIZON" if elapsed_days >= days else "FUNDING_FLIP", "win": pos["net_pct"] > 0}
                    self.closed.append(rec); self.open.pop(key, None); ev.append(rec)
            elif o["qualifies"]:
                self.open[key] = {"key": key, "symbol": o["symbol"], "venue": o.get("venue"), "side": o["side"], "ts": now, "last_mark": now,
                                  "cost_pct": round(o["fees_pct"] + o["friction_pct"] + o["reserve_pct"], 4), "accrued_pct": 0.0,
                                  "net_pct": 0.0, "expected_net_pct": o["net_pct"], "note": "GÖLGE — perp yürütme yok"}
                ev.append({"opened": key, "expected_net_pct": o["net_pct"]})
        if ev:
            self.save()
        return ev

    def status(self) -> Dict:
        n = len(self.closed); w = sum(1 for c in self.closed if c.get("win"))
        return {"opportunities": self.opps[:8], "open": list(self.open.values()), "n_closed": n,
                "win_rate": (round(w / n, 3) if n else None), "last_ts": self.last_ts, "error": self.error, "stage": "SHADOW",
                "note": "EV = funding − 2 bacak ücret − spread/kayma − rezerv; perp yürütme olmadığı için gölge"}
