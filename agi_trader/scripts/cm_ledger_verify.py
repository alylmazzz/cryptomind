#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEFTER ZİNCİRİ DOĞRULAMA — kapanan işlem kayıtları sha256 zinciriyle kurcalamaya karşı korunur.

  python scripts/cm_ledger_verify.py runs/live/runner_0_mexc.json
  python scripts/cm_ledger_verify.py --url https://mindcorplab.com/cryptomind
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
def _utf8_stdout():
    if hasattr(sys.stdout, "buffer") and getattr(sys.stdout, "encoding", "").lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("path", nargs="?", default=""); ap.add_argument("--url", default="")
    a = ap.parse_args()
    from agi_trader.auto.live_runner import verify_ledger
    if a.url:
        with urllib.request.urlopen(a.url.rstrip("/") + "/api/simulator", timeout=40) as r:
            trades = json.loads(r.read().decode("utf-8")).get("trades") or []
        trades = list(reversed(trades))                      # API en yeniyi önce verir
    else:
        trades = json.loads(Path(a.path).read_text(encoding="utf-8")).get("trades") or []
    v = verify_ledger(trades)
    print(f"kayıt {v['n']} · zincirli {v['chained']} · bütünlük: {'TAMAM' if v['ok'] else 'BOZUK'}" + (f" · ilk kırılma #{v['first_break']}" if not v["ok"] else ""))
    return 0 if v["ok"] else 1


if __name__ == "__main__":
    _utf8_stdout()
    sys.exit(main())
