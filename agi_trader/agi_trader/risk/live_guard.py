"""
Canlı işlem koruma katmanı (FAZ 9) — gerçek para için ZORUNLU.

Bu modül alfa üretmez; SERMAYEYİ KORUR. Program hedefinin (yıllık %35-50,
DD %15-20) tutması, kötü günlerde sistemin durmasına bağlıdır: bileşik getiri
tek bir %40'lık kaybı asla telafi edemez.

Katmanlar
  1. ÜÇLÜ ONAY        — canlı emir için üç bağımsız anahtar aynı anda açık olmalı
  2. KILL-SWITCH      — günlük zarar / drawdown / veri bayatlığı / borsa hatası
  3. MUTABAKAT        — borsa pozisyonu ↔ iç durum; sapma varsa EMİR YOK
  4. KADEMELİ SERMAYE — canlı performans backtest'i tutturmadan büyütme yok
  5. DENETİM KAYDI    — her karar ve emir runs/audit.log'a

Varsayılan durum KAPALIDIR. Hiçbir konfig tek başına canlı emri açamaz.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ---- plan tablosundan sabitler ----
DAILY_LOSS_LIMIT = 0.04          # %4 günlük zarar → dur
MAX_DRAWDOWN = 0.18              # %18 drawdown → dur
STALE_DATA_SEC = 7200            # 2 saat veri gelmezse → dur
MAX_CONSEC_ERRORS = 3            # art arda borsa hatası → dur
RECONCILE_TOL = 0.01             # pozisyon sapma toleransı (%1)

# Kademeli sermaye planı: (ay, maks USDT, gereken canlı/backtest Sharpe oranı)
CAPITAL_LADDER = [(1, 500.0, 0.0), (3, 2000.0, 0.6), (6, 10000.0, 0.6)]


def _audit_path(output_dir: str = "runs") -> Path:
    p = Path(output_dir)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[2] / p
    p.mkdir(parents=True, exist_ok=True)
    return p / "audit.log"


def audit(event: str, detail: Dict, output_dir: str = "runs") -> None:
    """Her karar/emir buraya yazılır. Silinmez, döndürülmez."""
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "event": event, **detail}
    try:
        with open(_audit_path(output_dir), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


# ===========================================================================
# 1) Üçlü onay
# ===========================================================================
def live_enabled(config) -> Dict:
    """Canlı emir için ÜÇ bağımsız anahtar birden açık olmalı:
        config.execution.mode == "live"
        config.execution.allow_live == True
        ortam değişkeni CRYPTOMIND_LIVE_CONFIRM == "EVET"

    Üç ayrı yerde olmaları kasıtlıdır: konfig dosyası yanlışlıkla kopyalanabilir,
    ortam değişkeni sunucuya elle konur. İkisi birden kazara açık kalmaz."""
    mode = str(config.get("execution.mode", "paper")).lower()
    allow = bool(config.get("execution.allow_live", False))
    env = os.environ.get("CRYPTOMIND_LIVE_CONFIRM", "").strip().upper() == "EVET"
    ok = (mode == "live") and allow and env
    missing = []
    if mode != "live":
        missing.append("execution.mode != live")
    if not allow:
        missing.append("execution.allow_live = false")
    if not env:
        missing.append("CRYPTOMIND_LIVE_CONFIRM ortam değişkeni yok")
    return {"live": ok, "missing": missing,
            "note": "KAĞIT MOD" if not ok else "CANLI EMİR AÇIK"}


# ===========================================================================
# 2) Kill-switch
# ===========================================================================
@dataclass
class GuardState:
    halted: bool = False
    reasons: List[str] = field(default_factory=list)
    consec_errors: int = 0
    day_start_equity: Optional[float] = None
    day: str = ""
    peak_equity: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class LiveGuard:
    """Her döngüde `check()` çağrılır; False dönerse EMİR GÖNDERİLMEZ."""

    def __init__(self, config=None, output_dir: str = "runs",
                 state_file: str = "guard_state.json",
                 daily_loss_limit: float = DAILY_LOSS_LIMIT,
                 max_drawdown: float = MAX_DRAWDOWN):
        self.config = config
        self.output_dir = output_dir
        self.path = Path(_audit_path(output_dir)).parent / state_file
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Koşucu başına eşik (video: %20 günlük limit seçilebilir; varsayılan plan %4).
        # Eşikler yalnız SIKILAŞTIRILABİLİR yönde değil — kullanıcı bilinçli seçer,
        # ama 0 ya da negatif eşik kabul edilmez (kill-switch'i kapatmak yok).
        self.daily_loss_limit = float(max(0.001, daily_loss_limit))
        self.max_drawdown = float(max(0.001, max_drawdown))
        self.state = self._load()

    # ---------------------------------------------------------- kalıcılık
    def _load(self) -> GuardState:
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            return GuardState(**d)
        except Exception:
            return GuardState()

    def save(self) -> None:
        try:
            self.path.write_text(json.dumps(self.state.to_dict(), indent=2,
                                            ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    # ------------------------------------------------------------- kontrol
    def check(self, equity: float, last_data_ts: Optional[float] = None,
              exchange_error: bool = False) -> Dict:
        """Tüm kill-switch koşullarını değerlendirir."""
        s = self.state
        today = time.strftime("%Y-%m-%d", time.gmtime())
        if s.day != today:
            s.day, s.day_start_equity = today, float(equity)
        if s.peak_equity is None or equity > s.peak_equity:
            s.peak_equity = float(equity)

        s.consec_errors = s.consec_errors + 1 if exchange_error else 0
        reasons: List[str] = []

        if s.day_start_equity and s.day_start_equity > 0:
            day_ret = equity / s.day_start_equity - 1
            if day_ret <= -self.daily_loss_limit:
                reasons.append(f"🔴 Günlük zarar %{day_ret*100:.2f} ≤ −%{self.daily_loss_limit*100:.1f}")

        if s.peak_equity and s.peak_equity > 0:
            dd = 1 - equity / s.peak_equity
            if dd >= self.max_drawdown:
                reasons.append(f"🔴 Drawdown %{dd*100:.1f} ≥ %{self.max_drawdown*100:.0f}")

        if last_data_ts is not None:
            age = time.time() - float(last_data_ts)
            if age > STALE_DATA_SEC:
                reasons.append(f"🔴 Veri bayat: {age/3600:.1f} saat")

        if s.consec_errors >= MAX_CONSEC_ERRORS:
            reasons.append(f"🔴 Art arda {s.consec_errors} borsa hatası")

        if reasons and not s.halted:
            s.halted = True
            s.reasons = reasons
            audit("HALT", {"equity": equity, "reasons": reasons}, self.output_dir)
        elif reasons:
            s.reasons = reasons

        self.save()
        return {"can_trade": not s.halted, "halted": s.halted,
                "reasons": s.reasons,
                "day_return_pct": (round((equity / s.day_start_equity - 1) * 100, 3)
                                   if s.day_start_equity else None),
                "drawdown_pct": (round((1 - equity / s.peak_equity) * 100, 2)
                                 if s.peak_equity else None)}

    def resume(self, operator: str = "manuel") -> Dict:
        """HALT durumundan çıkış — YALNIZ elle. Otomatik resume YOKTUR:
        sistemi durduran koşul incelenmeden devam etmek, durdurmanın amacını
        ortadan kaldırır."""
        self.state.halted = False
        self.state.reasons = []
        self.state.consec_errors = 0
        self.save()
        audit("RESUME", {"operator": operator}, self.output_dir)
        return {"can_trade": True, "halted": False}


# ===========================================================================
# 3) Mutabakat
# ===========================================================================
def reconcile(internal: Dict[str, float], exchange: Dict[str, float],
              tol: float = RECONCILE_TOL, output_dir: str = "runs") -> Dict:
    """İç durum ile borsa pozisyonlarını karşılaştır.

    Sapma varsa EMİR GÖNDERİLMEZ: iç durumu yanlış olan bir sistem, düzeltmek
    isterken pozisyonu ikiye katlayabilir. Önce insan bakar."""
    syms = set(internal) | set(exchange)
    diffs = {}
    for s in syms:
        a, b = float(internal.get(s, 0.0)), float(exchange.get(s, 0.0))
        scale = max(abs(a), abs(b), 1e-9)
        if abs(a - b) / scale > tol:
            diffs[s] = {"internal": a, "exchange": b, "diff": round(a - b, 10)}
    ok = not diffs
    if not ok:
        audit("RECONCILE_MISMATCH", {"diffs": diffs}, output_dir)
    return {"ok": ok, "mismatches": diffs,
            "note": "eşleşti" if ok else "SAPMA VAR — emir gönderilmez, elle incele"}


# ===========================================================================
# 4) Kademeli sermaye
# ===========================================================================
def capital_cap(months_live: float, live_sharpe: Optional[float] = None,
                backtest_sharpe: float = 1.37) -> Dict:
    """Canlı süreye ve gerçekleşen performansa göre izin verilen maksimum sermaye.

    Canlı Sharpe backtest'in %60'ını tutturmuyorsa sermaye ARTIRILMAZ — bu,
    'backtest'te güzeldi ama canlıda çalışmıyor' durumunda kaybı sınırlar."""
    cap, reason = CAPITAL_LADDER[0][1], "başlangıç kademesi"
    for months, amount, need_ratio in CAPITAL_LADDER:
        if months_live < months:
            break
        if need_ratio > 0:
            if live_sharpe is None:
                reason = "canlı Sharpe bilinmiyor — kademe yükseltilmedi"
                break
            ratio = live_sharpe / (backtest_sharpe + 1e-12)
            if ratio < need_ratio:
                reason = (f"canlı/backtest Sharpe oranı {ratio:.2f} < {need_ratio} "
                          f"— kademe yükseltilmedi")
                break
        cap, reason = amount, f"{months} ay + performans kapısı geçildi"
    return {"max_capital_usdt": cap, "reason": reason,
            "months_live": round(float(months_live), 2),
            "live_sharpe": live_sharpe, "backtest_sharpe": backtest_sharpe}


# ===========================================================================
# Tek çağrı: emir gönderilebilir mi?
# ===========================================================================
def preflight(config, guard: LiveGuard, equity: float,
              internal_positions: Dict[str, float],
              exchange_positions: Dict[str, float],
              last_data_ts: Optional[float] = None,
              months_live: float = 0.0,
              live_sharpe: Optional[float] = None) -> Dict:
    """Emir göndermeden ÖNCE çağrılacak tek fonksiyon. Hepsi geçmeden emir yok."""
    le = live_enabled(config)
    gs = guard.check(equity, last_data_ts=last_data_ts)
    rc = reconcile(internal_positions, exchange_positions)
    cc = capital_cap(months_live, live_sharpe)

    blockers = []
    if not le["live"]:
        blockers.append("canlı mod kapalı: " + ", ".join(le["missing"]))
    if not gs["can_trade"]:
        blockers.append("kill-switch: " + "; ".join(gs["reasons"]))
    if not rc["ok"]:
        blockers.append("mutabakat sapması")
    if equity > cc["max_capital_usdt"]:
        blockers.append(f"sermaye tavanı aşıldı ({equity:.0f} > "
                        f"{cc['max_capital_usdt']:.0f} $): {cc['reason']}")

    ok = not blockers
    audit("PREFLIGHT", {"ok": ok, "blockers": blockers, "equity": equity})
    return {"ok": ok, "blockers": blockers,
            "live": le, "guard": gs, "reconcile": rc, "capital": cc}
