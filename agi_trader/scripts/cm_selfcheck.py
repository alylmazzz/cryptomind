#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CryptoMind ÖZ-DENETİM — anlık PASS/FAIL tablosu: uçlar, yazma kapısı, konfig değişmezleri, parametre sınırları,
kaynak, tazelik, defter zinciri, NaN sızıntısı, simülatör sağlığı.

  python scripts/cm_selfcheck.py --url https://mindcorplab.com/cryptomind
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
def _utf8_stdout():
    if hasattr(sys.stdout, "buffer") and getattr(sys.stdout, "encoding", "").lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def http(url: str, method: str = "GET"):
    req = urllib.request.Request(url, method=method, data=b"{}" if method == "POST" else None,
                                 headers={"Content-Type": "application/json"} if method == "POST" else {})
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            body = r.read().decode("utf-8")
            return r.status, body
    except urllib.error.HTTPError as e:
        return e.code, ""
    except Exception as e:
        return 0, str(e)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:8210"); a = ap.parse_args()
    base = a.url.rstrip("/")
    checks = []

    def add(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    for u in ("/api/simulator", "/api/simulator/feed", "/api/simulator/missed?limit=5", "/api/research", "/api/system-health", "/api/risk", "/api/trend"):
        code, body = http(base + u)
        ok = code == 200
        nan = ("NaN" in body or "Infinity" in body) if ok else False
        add(f"GET {u}", ok and not nan, f"HTTP {code}{' · NaN sızıntısı!' if nan else ''}")
    code, _ = http(base + "/api/simulator", "POST")
    add("POST kapalı (panel GET-only)", code in (403, 405), f"HTTP {code}")
    code, body = http(base + "/api/simulator")
    if code == 200:
        d = json.loads(body); c = d["config"]; e = d["effective_params"]; x = d["exit_params"]; st = d["stats"]; rs = d.get("resource") or {}
        add("mod = paper", c["mode"] == "paper", c["mode"])
        add("canlı kapıları kapalı", not d.get("running") or c["mode"] != "live", "")
        add("günlük tavan ≥ 100", c["max_trades_per_day"] >= 100, str(c["max_trades_per_day"]))
        add("emir tavanı ≤ %25 sermaye", c["max_order_usdt"] <= 0.25 * c["capital_usdt"], f"{c['max_order_usdt']}/{c['capital_usdt']}")
        add("maruziyet ≤ %75", c["max_exposure_pct"] <= 75, str(c["max_exposure_pct"]))
        add("günlük zarar limiti ≤ %5", c["daily_loss_limit_pct"] <= 0.05, str(c["daily_loss_limit_pct"]))
        add("hard stop asla kalkmaz (max_stop ≤ 5%)", e.get("max_stop_pct", 5) <= 5, str(e.get("max_stop_pct")))
        add("giveback 0,35–0,70", 0.35 <= x["retain_fraction"] <= 0.70, str(x["retain_fraction"]))
        add("BE kilidi açık", x.get("be_lock_cost_mult", 0) >= 1.0, str(x.get("be_lock_cost_mult")))
        add("erken iptal açık", 0.3 <= x.get("early_abort_mae_frac", 0) <= 0.95, str(x.get("early_abort_mae_frac")))
        add("kaynak RED değil", rs.get("state") in ("GREEN", "YELLOW"), f"{rs.get('state')} {rs.get('rss_mb')} MB")
        add("döngü ≤ loop_sec", (rs.get("cycle_sec") or 0) <= d.get("loop_sec", 30) + 5, f"{rs.get('cycle_sec')} sn")
        age = time.time() - float(d.get("last_cycle_ts") or 0)
        add("son döngü < 3 dk", age < 180, f"{age:.0f} sn önce")
        add("HALT değil", not d.get("halted"), "; ".join(d.get("halt_reasons") or []))
        add("mutabakat", d.get("reconcile_ok", True), d.get("reconcile_note", ""))
        led = d.get("ledger") or {}
        add("defter zinciri bütün", led.get("ok", True), f"{led.get('n', 0)} kayıt")
        stale = [s for s in (d.get("scan") or []) if s.get("freshness") and s["freshness"] != "LIVE"]
        add("veri tazeliği (tarama)", len(stale) <= max(2, len(d.get("scan") or []) // 4), f"{len(stale)} gecikmeli/bayat")
        pos = d.get("positions") or []
        def _inv(p):
            s_ = 1 if p["direction"] == "LONG" else -1
            return (p["hard_stop"] - p["target"]) * s_ < 0 and (p.get("be_locked") or (p["hard_stop"] - p["entry"]) * s_ < 0)
        add("pozisyon değişmezleri (stop<hedef; kilitsizse stop<giriş)", all(_inv(p) for p in pos), f"{len(pos)} açık · BE kilitli {sum(1 for p in pos if p.get('be_locked'))}")
        add("kâr faktörü ölçüldü", st["closed_trades"] == 0 or st["profit_factor"] is not None, str(st.get("profit_factor")))
    passed = sum(1 for _, ok, _ in checks if ok)
    print(f"{'DENETİM':46s} {'SONUÇ':7s} DETAY")
    for n, ok, det in checks:
        print(f"{n:46s} {'PASS' if ok else 'FAIL':7s} {det}")
    print(f"\n{passed}/{len(checks)} geçti")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    _utf8_stdout()
    sys.exit(main())
