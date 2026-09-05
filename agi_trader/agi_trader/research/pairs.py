"""
İSTATİSTİKSEL ARBİTRAJ — kointegrasyon çiftleri (Engle-Granger + Kalman dinamik beta + OU yarı-ömür).

"İki coin korele → biri düşerse al" DEĞİLDİR. Zincir:
  OLS hedge (β) → artık ADF t-istatistiği (kointegrasyon) → Kalman β_t → spread S_t = A − β_t·B
  → z_t = (S_t − μ)/σ → yarı-ömür (AR(1)) → maliyet kapısı → sinyal.
Statsmodels olmadan: DF regresyonu (Δs_t = ρ·s_{t−1} + c) t-istatistiği; kritik −3,34 (Engle-Granger 2 değişken, %5).

Spot simülatör SHORT açamaz → bu modül GÖLGE'dir: sinyal üretir, spread P&L'ini takip eder,
lifecycle kanıtı biriktirir. Emir yok.
"""
from __future__ import annotations

import itertools
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ADF_CRIT_5 = -3.5           # Engle-Granger (2 değişken, tahmini artık) %5 kritik değeri civarı; DF −2,86 DEĞİL
MIN_RET_CORR = 0.3          # ekonomik akıl: kointegre kripto çiftinde getiri korelasyonu sıfır olamaz
MAX_BETA_DISAGREE = 0.3     # Kalman β ile OLS β birbirini tutmalı (sahte regresyon kalkanı)
Z_ENTRY, Z_EXIT, Z_STOP = 2.0, 0.5, 4.0


def ols_hedge(y: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    x1 = np.column_stack([x, np.ones_like(x)])
    beta, alpha = np.linalg.lstsq(x1, y, rcond=None)[0]
    return float(beta), float(alpha)


def adf_tstat(s: np.ndarray) -> float:
    """Dickey-Fuller (sabitli, gecikmesiz): Δs_t = ρ s_{t−1} + c. ρ'nun t-istatistiği."""
    s = np.asarray(s, dtype=float)
    ds = np.diff(s)
    lag = s[:-1]
    X = np.column_stack([lag, np.ones_like(lag)])
    coef, res, *_ = np.linalg.lstsq(X, ds, rcond=None)
    resid = ds - X @ coef
    n, k = len(ds), 2
    sigma2 = float(resid @ resid) / max(1, n - k)
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = math.sqrt(max(1e-18, cov[0, 0]))
    return float(coef[0] / se)


def kalman_beta(y: np.ndarray, x: np.ndarray, delta: float = 1e-4, r: float = 1e-3) -> np.ndarray:
    """Rastgele-yürüyüş β_t (+ sabit) için Kalman filtresi; β_t dizisi döner."""
    n = len(y)
    theta = np.zeros(2)                       # [beta, alpha]
    P = np.eye(2) * 1.0
    Q = np.eye(2) * delta / (1 - delta)
    out = np.zeros(n)
    for t in range(n):
        H = np.array([x[t], 1.0])
        P = P + Q
        yhat = H @ theta
        e = y[t] - yhat
        S = H @ P @ H + r
        K = P @ H / S
        theta = theta + K * e
        P = (np.eye(2) - np.outer(K, H)) @ P
        out[t] = theta[0]
    return out


def half_life(spread: np.ndarray) -> Optional[float]:
    s = np.asarray(spread, dtype=float)
    ds = np.diff(s)
    lag = s[:-1] - s[:-1].mean()
    denom = float(lag @ lag)
    if denom <= 0:
        return None
    lam = float(lag @ ds) / denom
    if lam >= 0:
        return None                            # ortalamaya dönmüyor
    return float(-math.log(2) / math.log(1 + lam)) if (1 + lam) > 0 else None


def analyze_pair(a: np.ndarray, b: np.ndarray, min_n: int = 150) -> Optional[Dict]:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    if n < min_n:
        return None
    a, b = np.log(a[-n:]), np.log(b[-n:])
    beta, alpha = ols_hedge(a, b)
    resid = a - (beta * b + alpha)
    t = adf_tstat(resid)
    # Kalman β: ölçüm gürültüsü artık varyansından (aksi hâlde spread şoklarını β'ya yazar)
    bk = kalman_beta(a, b, delta=1e-5, r=max(1e-8, float(resid.var())))
    mu, sd = float(resid[-60:].mean()), float(resid[-60:].std(ddof=0))
    z = float((resid[-1] - mu) / sd) if sd > 0 else 0.0
    hl = half_life(resid[-120:])
    corr = float(np.corrcoef(np.diff(a), np.diff(b))[0, 1])
    disagree = abs(float(bk[-1]) - beta) / max(1e-9, abs(beta))
    # ÜÇ kapı: ADF t < kritik · getiri korelasyonu · β tutarlılığı (sahte regresyon kalkanı)
    coint = bool(t < ADF_CRIT_5 and corr >= MIN_RET_CORR and disagree <= MAX_BETA_DISAGREE)
    return {"beta_ols": round(beta, 4), "alpha_ols": round(alpha, 6), "beta_kalman": round(float(bk[-1]), 4), "adf_t": round(t, 3),
            "cointegrated": coint, "z": round(z, 3), "half_life_bars": (None if hl is None else round(hl, 1)),
            "corr": round(corr, 3), "beta_disagree": round(disagree, 3), "n": int(n)}


def scan_pairs(closes: Dict[str, np.ndarray], max_pairs: int = 10, min_n: int = 150,
               hl_range: Tuple[float, float] = (2.0, 120.0)) -> List[Dict]:
    """Elle sabit çift yok — her taramada evren içinde kointegre çiftler bulunur."""
    out = []
    syms = [s for s, c in closes.items() if c is not None and len(c) >= min_n]
    for a, b in itertools.combinations(syms, 2):
        try:
            r = analyze_pair(closes[a], closes[b], min_n)
        except Exception:
            continue
        if not r or not r["cointegrated"] or r["half_life_bars"] is None:
            continue
        if not (hl_range[0] <= r["half_life_bars"] <= hl_range[1]):
            continue
        out.append({"a": a, "b": b, **r})
    out.sort(key=lambda r: r["adf_t"])
    return out[:max_pairs]


class PairsShadow:
    """Gölge spread pozisyonları: |z| ≥ 2 giriş, |z| ≤ 0,5 çıkış, |z| ≥ 4 stop, 3×yarı-ömür zaman."""

    def __init__(self, path: Optional[Path] = None, cost_pct_roundtrip: float = 0.3):
        self.path = Path(path) if path else None
        self.cost = float(cost_pct_roundtrip)
        self.pairs: List[Dict] = []
        self.open: Dict[str, Dict] = {}
        self.closed: List[Dict] = []
        self.last_scan_ts: Optional[float] = None
        self.load()

    def load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            self.pairs, self.open, self.closed = d.get("pairs", []), d.get("open", {}), d.get("closed", [])
            self.last_scan_ts = d.get("last_scan_ts")
        except Exception:
            pass

    def save(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"pairs": self.pairs, "open": self.open, "closed": self.closed[-300:],
                                             "last_scan_ts": self.last_scan_ts}, ensure_ascii=False, default=str), encoding="utf-8")
        except Exception:
            pass

    def rescan(self, closes: Dict[str, np.ndarray], now: float) -> List[Dict]:
        self.pairs = scan_pairs(closes)
        self.last_scan_ts = now
        self.save()
        return self.pairs

    def step(self, closes: Dict[str, np.ndarray], now: float, bar_sec: float = 14400.0) -> List[Dict]:
        """Her yeni barda z'yi güncelle; gölge aç/kapat. Spread P&L'i log-fiyat farkı (≈ %)."""
        events = []
        for pr in self.pairs:
            a, b = pr["a"], pr["b"]
            ca, cb = closes.get(a), closes.get(b)
            if ca is None or cb is None or len(ca) < 150 or len(cb) < 150:
                continue
            r = analyze_pair(ca, cb)
            if not r:
                continue
            pr.update({k: r[k] for k in ("z", "beta_ols", "alpha_ols", "beta_kalman", "half_life_bars", "adf_t", "cointegrated")})
            key = f"{a}|{b}"
            la, lb = math.log(float(ca[-1])), math.log(float(cb[-1]))
            spread_now = la - r["beta_ols"] * lb - r["alpha_ols"]
            pos = self.open.get(key)
            if pos:
                pnl = (spread_now - pos["spread_entry"]) * (1.0 if pos["side"] == "LONG_SPREAD" else -1.0) * 100.0 - self.cost
                pos["pnl_pct"] = round(pnl, 4); pos["z_now"] = r["z"]
                bars = (now - pos["ts"]) / bar_sec
                reason = None
                if abs(r["z"]) <= Z_EXIT:
                    reason = "Z_EXIT"
                elif abs(r["z"]) >= Z_STOP:
                    reason = "Z_STOP"
                elif pos.get("hl") and bars >= 3 * pos["hl"]:
                    reason = "TIME"
                if reason:
                    rec = {**pos, "closed_ts": now, "reason": reason, "win": pnl > 0}
                    self.closed.append(rec); self.open.pop(key, None); events.append(rec)
            elif r["cointegrated"] and abs(r["z"]) >= Z_ENTRY and r["half_life_bars"]:
                side = "SHORT_SPREAD" if r["z"] > 0 else "LONG_SPREAD"
                self.open[key] = {"pair": key, "a": a, "b": b, "side": side, "ts": now, "z_entry": r["z"],
                                  "beta": r["beta_kalman"], "hl": r["half_life_bars"], "spread_entry": spread_now,
                                  "legs": (f"SHORT {a} / LONG {b}×β" if side == "SHORT_SPREAD" else f"LONG {a} / SHORT {b}×β"),
                                  "note": "GÖLGE — spot simülatörde short yok, emir verilmedi"}
                events.append({"opened": key, "side": side, "z": r["z"]})
        if events:
            self.save()
        return events

    def status(self) -> Dict:
        n = len(self.closed); w = sum(1 for c in self.closed if c.get("win"))
        return {"pairs": self.pairs[:10], "open": list(self.open.values()), "n_closed": n,
                "win_rate": (round(w / n, 3) if n else None),
                "net_pct_sum": round(sum(float(c.get("pnl_pct") or 0.0) for c in self.closed), 3),
                "last_scan_ts": self.last_scan_ts, "stage": "SHADOW",
                "note": "kointegrasyon haftalık taranır; z≥2 giriş / z≤0,5 çıkış / z≥4 stop; emir YOK (spot short yok)"}
