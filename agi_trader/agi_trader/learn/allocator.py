"""
META-TAHSİSÇİ — "hangi strateji en iyi?" değil, "şu rejimde hangi sleeve'in ÖLÇÜLMÜŞ edge'i var?"

  Score_i = RobustEV_i × Reliability_i × RegimeFit_i × ExecutionProbability_i

  Reliability: (sleeve, rejim) başına Beta(2,2) öncüllü Bayes güvenilirliği; n < MIN_N ise 1,0'a
               büzülür (ölçülmemişse ne ödül ne ceza). Yalnız GERÇEK kapanan işlemler sayılır
               (gölgeler ayrı motorda).
  Weight:      boyut çarpanı [0,5 · 1,2] — n ≥ MIN_N şart; ölçülmemiş sleeve 1,0.
  Thompson:    araştırma/gölge önceliği için Beta örneklemesi (canlı boyutu ETKİLEMEZ).

KANIT DURUM MAKİNESİ (2026-09-04, 85 işlemlik canlı kanıt: kazanma %50,6 ama ortalama kazanç
0,14 $ / ortalama kayıp 0,26 $ → kazanma oranı edge'i GÖSTERMEZ; dip_moderate %50 kazanıp −2,71 $
yaptı ve Beta güvenilirliği 0,5 ile TAM boyut aldı):
  UNPROVEN   kanıt yok ya da beklenti ≤ 0 → yalnız KANIT BOYUTU (probe) ile işlem
  PROVEN     n ≥ PROVE_N, ortalama net % > 0, t ≥ PROVE_T → tam boyut
  PAUSED     devre kesici (son pencere t ≤ −1 / Wilson) → giriş yok; süre her tekrarda ikiye katlanır
  PROBATION  duraklama bitti → kanıt boyutu; PROBATION_N işlemde net ≥ 0 olursa çıkar, yine
             negatifse yeniden PAUSED (kanıt sıfırlanmaz, sadece pencere yenilenir)
Ölçüldü (karşı-olgusal, aynı 85 işlem): kanıtlanmamış sleeve'lere 25 $ tavan → net −4,78 $ yerine
−0,92 $. Kanıtlanan sleeve tavanı kaldırır; kanıt olmadan boyut büyütülmez.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

MIN_N = 10
PRIOR_A, PRIOR_B = 2.0, 2.0
W_MIN, W_MAX = 0.5, 1.2
RECENT_N = 20                 # devre kesici penceresi (son işlemler)
BREAKER_MIN_N = 12
BREAKER_PAUSE_SEC = 6 * 3600
BREAKER_PAUSE_MAX_SEC = 48 * 3600
BREAKER_WIN_UPPER = 0.55      # kazanma oranı Wilson üst sınırı bunun altındaysa + net beklenti < 0 → duraklat
HISTORY_N = 200               # sleeve başına tutulan net % geçmişi (kümülatif kanıt)
PROVE_N = 20                  # kanıtlanmış sayılmak için asgari kapanan işlem
PROVE_T = 1.0                 # ortalama net % için t-istatistiği eşiği (tek yönlü ≈ %84)
PROBATION_N = 8               # duraklama sonrası deneme penceresi
CUMULATIVE_BREAK_N = 15       # kümülatif devre kesici: n ≥ 15 ve t ≤ −1,5 ve ortalama < 0
CUMULATIVE_BREAK_T = -1.5

STATE_UNPROVEN, STATE_PROVEN, STATE_PAUSED, STATE_PROBATION = "UNPROVEN", "PROVEN", "PAUSED", "PROBATION"
STATE_TR = {STATE_UNPROVEN: "kanıtlanmadı — kanıt boyutu", STATE_PROVEN: "kanıtlandı — tam boyut",
            STATE_PAUSED: "duraklatıldı — giriş yok", STATE_PROBATION: "deneme — kanıt boyutu"}


def _wilson_upper(k: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 1.0
    p = k / n; den = 1 + z * z / n; c = p + z * z / (2 * n)
    adj = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return min(1.0, (c + adj) / den)


def _t_stat(xs: List[float]) -> float:
    """Ortalama / standart hata. Tek değer ya da sıfır varyans → ±9 (işaretine göre) / 0."""
    n = len(xs)
    if n == 0:
        return 0.0
    m = sum(xs) / n
    if n < 2:
        return 0.0
    sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5
    if sd <= 1e-12:
        return 9.0 if m > 0 else (-9.0 if m < 0 else 0.0)
    return m / (sd / n ** 0.5)


class MetaAllocator:
    def __init__(self, path: Optional[Path] = None, regime_sleeves: Optional[Dict[str, List[str]]] = None):
        self.path = Path(path) if path else None
        self.stats: Dict[str, Dict[str, float]] = {}        # "sleeve|regime" -> {a, b, n, net}
        self.recent: Dict[str, List] = {}                    # sleeve -> [(won, net_pct)] son RECENT_N (duraklamada sıfırlanır)
        self.history: Dict[str, List[float]] = {}            # sleeve -> net % geçmişi (kümülatif kanıt; sıfırlanmaz)
        self.paused_until: Dict[str, float] = {}
        self.pause_count: Dict[str, int] = {}                # sleeve -> kaç kez duraklatıldı (süre katlanır)
        self.probation: Dict[str, Dict] = {}                 # sleeve -> {n, net_pct_sum, since}
        self.breaker_events: List[Dict] = []
        self.regime_sleeves = regime_sleeves or {}
        self._rng = random.Random(7)
        self.load()

    def load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            self.stats = d.get("stats", {}); self.recent = d.get("recent", {}); self.paused_until = d.get("paused_until", {})
            self.breaker_events = d.get("breaker_events", [])
            self.history = {k: [float(x) for x in v] for k, v in (d.get("history") or {}).items()}
            self.pause_count = {k: int(v) for k, v in (d.get("pause_count") or {}).items()}
            self.probation = dict(d.get("probation") or {})
        except Exception:
            pass

    def save(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"stats": self.stats, "recent": self.recent, "paused_until": self.paused_until,
                                             "history": self.history, "pause_count": self.pause_count, "probation": self.probation,
                                             "breaker_events": self.breaker_events[-50:], "saved_ts": time.time()}, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    @staticmethod
    def _key(sleeve: str, regime: Optional[str]) -> str:
        return f"{sleeve}|{regime or '*'}"

    def backfill(self, trades: List[Dict]) -> int:
        """Geçmiş kapanan işlemlerden net % geçmişini kurar (yalnız `history` boşsa; kanıt kaybolmasın).
        Döndürür: eklenen kayıt sayısı."""
        if self.history:
            return 0
        n = 0
        for t in trades:
            sl = t.get("sleeve") or t.get("trigger")
            if not sl:
                continue
            pct = t.get("net_pct_realized")
            if pct is None:
                nt = float(t.get("notional") or 0.0)
                pct = (float(t.get("net_pnl", 0.0)) / nt * 100.0) if nt > 0 else 0.0
            h = self.history.setdefault(str(sl), [])
            h.append(float(pct)); del h[:-HISTORY_N]
            n += 1
        if n:
            self.save()
        return n

    def record(self, sleeve: str, regime: Optional[str], won: bool, net: float = 0.0, net_pct: Optional[float] = None) -> None:
        for k in (self._key(sleeve, regime), self._key(sleeve, None)):
            st = self.stats.setdefault(k, {"a": 0.0, "b": 0.0, "n": 0, "net": 0.0})
            st["a" if won else "b"] += 1.0
            st["n"] += 1
            st["net"] = round(st["net"] + float(net), 4)
        pct = float(net_pct if net_pct is not None else net)
        r = self.recent.setdefault(sleeve, [])
        r.append([bool(won), pct])
        del r[:-RECENT_N]
        h = self.history.setdefault(sleeve, [])
        h.append(pct); del h[:-HISTORY_N]
        pr = self.probation.get(sleeve)
        if pr is not None:
            pr["n"] = int(pr.get("n", 0)) + 1
            pr["net_pct_sum"] = round(float(pr.get("net_pct_sum", 0.0)) + pct, 4)
            if pr["n"] >= PROBATION_N and pr["net_pct_sum"] >= 0.0:
                self.probation.pop(sleeve, None)             # deneme geçti → kümülatif kanıt karar verir
        self.save()

    def _pause(self, sleeve: str, now: float, ev: Dict) -> Dict:
        k = int(self.pause_count.get(sleeve, 0))
        dur = min(BREAKER_PAUSE_MAX_SEC, BREAKER_PAUSE_SEC * (2 ** k))
        self.pause_count[sleeve] = k + 1
        self.paused_until[sleeve] = now + dur
        self.probation[sleeve] = {"n": 0, "net_pct_sum": 0.0, "since": now}
        self.recent[sleeve] = []                    # pencere yenilenir; `history` (kümülatif kanıt) KALIR
        ev.update({"until": now + dur, "pause_hours": round(dur / 3600.0, 1), "pause_count": k + 1})
        self.breaker_events.append(ev)
        return ev

    def check_breakers(self, now: Optional[float] = None) -> List[Dict]:
        """Sleeve devre kesici. İki kural:
          (1) son ≥12 işlemde net beklenti < 0 VE (kazanma Wilson üst < 0,55 VEYA t < −1) → duraklat
          (2) deneme (probation) penceresinde ≥ 8 işlem, net < 0 ve t ≤ −1 → yeniden duraklat
        Süre her tekrarda ikiye katlanır (6 → 12 → 24 → 48 sa). Kanıt temelli (canlı: dip_moderate 22 işlemde
        −1,16 $, obi_momentum 12'de −0,67 $ günün kârını geri verdi; duraklama sonrası 'temiz sayfa' aynı
        sleeve'e 12 işlem daha tam boyut veriyordu → deneme penceresi + katlanan süre)."""
        now = time.time() if now is None else now
        events = []
        for sleeve, r in list(self.recent.items()):
            if self.paused_until.get(sleeve, 0.0) > now:
                continue
            n = len(r)
            if n == 0:
                continue
            wins = sum(1 for w, _ in r if w)
            xs = [x for _, x in r]
            mean_net = sum(xs) / n
            t_stat = _t_stat(xs)
            pr = self.probation.get(sleeve)
            in_probation = pr is not None
            fire = None
            if n >= BREAKER_MIN_N and mean_net < 0 and (_wilson_upper(wins, n) < BREAKER_WIN_UPPER or t_stat < -1.0):
                fire = "pencere"
            elif in_probation and n >= PROBATION_N and mean_net < 0 and t_stat <= -1.0:
                fire = "deneme"
            if fire:
                ev = {"ts": now, "sleeve": sleeve, "rule": fire, "n": n, "wins": wins, "mean_net_pct": round(mean_net, 4),
                      "wilson_upper": round(_wilson_upper(wins, n), 3), "t_stat": round(t_stat, 2)}
                events.append(self._pause(sleeve, now, ev))
        if events:
            self.save()
        return events

    def paused_sleeves(self, now: Optional[float] = None) -> Dict[str, float]:
        now = time.time() if now is None else now
        return {k: v for k, v in self.paused_until.items() if v > now}

    # ------------------------------------------------------------------ kanıt durumu
    def evidence(self, sleeve: str) -> Dict:
        xs = self.history.get(sleeve) or []
        n = len(xs)
        mean = (sum(xs) / n) if n else 0.0
        return {"n": n, "mean_net_pct": round(mean, 4), "t_stat": round(_t_stat(xs), 2) if n >= 2 else 0.0,
                "pos_share": (round(sum(1 for x in xs if x > 0) / n, 3) if n else None)}

    def state(self, sleeve: str, now: Optional[float] = None) -> str:
        now = time.time() if now is None else now
        if self.paused_until.get(sleeve, 0.0) > now:
            return STATE_PAUSED
        if sleeve in self.probation:
            return STATE_PROBATION
        e = self.evidence(sleeve)
        if e["n"] >= PROVE_N and e["mean_net_pct"] > 0.0 and e["t_stat"] >= PROVE_T:
            return STATE_PROVEN
        return STATE_UNPROVEN

    def sleeve_states(self, sleeves: Optional[List[str]] = None, now: Optional[float] = None) -> Dict[str, str]:
        keys = set(sleeves or []) | set(self.history) | set(self.recent) | set(self.paused_until)
        return {s: self.state(s, now) for s in sorted(keys)}

    def notional_cap(self, sleeve: str, probe_usdt: float, full_usdt: float, now: Optional[float] = None) -> Dict:
        """Kanıt durumuna göre boyut tavanı. PROVEN → tam; diğerleri → kanıt boyutu; PAUSED → 0."""
        st = self.state(sleeve, now)
        if st == STATE_PAUSED:
            cap = 0.0
        elif st == STATE_PROVEN:
            cap = float(full_usdt)
        else:
            cap = float(min(probe_usdt, full_usdt))
        return {"state": st, "cap_usdt": cap, "label": STATE_TR[st], **self.evidence(sleeve)}

    def reliability(self, sleeve: str, regime: Optional[str] = None) -> Dict:
        st = self.stats.get(self._key(sleeve, regime)) or self.stats.get(self._key(sleeve, None)) or {"a": 0.0, "b": 0.0, "n": 0}
        n = int(st.get("n", 0))
        mean = (st["a"] + PRIOR_A) / (st["a"] + st["b"] + PRIOR_A + PRIOR_B)
        return {"mean": round(mean, 3), "n": n, "measured": n >= MIN_N}

    def regime_fit(self, sleeve: str, regime: Optional[str]) -> float:
        if not regime or not self.regime_sleeves:
            return 1.0
        return 1.0 if sleeve in self.regime_sleeves.get(regime, []) else 0.5

    def score(self, sleeve: str, robust_ev_pct: Optional[float], regime: Optional[str], p_exec: float = 1.0) -> Optional[float]:
        if robust_ev_pct is None:
            return None
        rel = self.reliability(sleeve, regime)
        r = rel["mean"] if rel["measured"] else 0.5
        return round(float(robust_ev_pct) * (2.0 * r) * self.regime_fit(sleeve, regime) * float(p_exec), 5)

    def weight(self, sleeve: str, regime: Optional[str] = None) -> float:
        rel = self.reliability(sleeve, regime)
        if not rel["measured"]:
            return 1.0
        return float(min(W_MAX, max(W_MIN, 2.0 * rel["mean"])))

    def sleeve_reliability(self) -> Dict[str, float]:
        """Yalnız ölçülmüş sleeve'ler (SEÇİCİ portföy modu için)."""
        out = {}
        for k, st in self.stats.items():
            sl, rg = k.split("|", 1)
            if rg == "*" and int(st.get("n", 0)) >= MIN_N:
                out[sl] = round((st["a"] + PRIOR_A) / (st["a"] + st["b"] + PRIOR_A + PRIOR_B), 3)
        return out

    def thompson_ranking(self, sleeves: List[str], regime: Optional[str] = None) -> List[Dict]:
        """Araştırma önceliği: Beta örnekle, sırala (canlı boyut DEĞİL)."""
        rows = []
        for s in sleeves:
            st = self.stats.get(self._key(s, regime)) or self.stats.get(self._key(s, None)) or {"a": 0.0, "b": 0.0, "n": 0}
            sample = self._rng.betavariate(st["a"] + PRIOR_A, st["b"] + PRIOR_B)
            rows.append({"sleeve": s, "sample": round(sample, 3), "n": int(st.get("n", 0))})
        rows.sort(key=lambda r: -r["sample"])
        return rows

    def status(self) -> Dict:
        rows = []
        for k, st in self.stats.items():
            sl, rg = k.split("|", 1)
            rel = self.reliability(sl, None if rg == "*" else rg)
            rows.append({"sleeve": sl, "regime": rg, "n": int(st.get("n", 0)), "wins": int(st["a"]), "net": st.get("net", 0.0),
                         "reliability": rel["mean"], "measured": rel["measured"], "weight": self.weight(sl, None if rg == "*" else rg)})
        rows.sort(key=lambda r: (-r["n"], r["sleeve"]))
        now = time.time()
        return {"rows": rows, "min_n": MIN_N, "paused": {k: round(v - now) for k, v in self.paused_sleeves(now).items()},
                "recent": {k: {"n": len(v), "wins": sum(1 for w, _ in v if w), "mean_net_pct": round(sum(x for _, x in v) / len(v), 4) if v else None} for k, v in self.recent.items()},
                "states": {s: {"state": st, **self.evidence(s), "pause_count": int(self.pause_count.get(s, 0)),
                               "probation": self.probation.get(s)} for s, st in self.sleeve_states(now=now).items()},
                "breaker_events": self.breaker_events[-10:][::-1],
                "note": ("ağırlık [0,5·1,2] yalnız n ≥ min_n olan sleeve'lerde; ölçülmemiş = 1,0; devre kesici: son 12+ işlem net<0 ve "
                         "Wilson üst<0,55 → 6 sa (her tekrarda ×2, en çok 48 sa); kanıt durumu: PROVEN = n ≥ 20 ve t ≥ 1 → tam boyut, "
                         "aksi hâlde kanıt boyutu")}
