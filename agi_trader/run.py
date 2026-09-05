#!/usr/bin/env python3
"""
AGI Trader — CLI giriş noktası.

Kullanım:
  python run.py                          # config'deki tüm pariteleri analiz et
  python run.py --symbols BTC/USDT ETH/USDT
  python run.py --timeframe 1h
  python run.py --source synthetic       # canlı veri olmadan demo
  python run.py --json runs/out.json     # sonucu JSON'a yaz
  python run.py --summary                # journal özetini göster

GÜVENLİK: Varsayılan paper-trading. Canlı emir göndermez.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Windows konsolunda Unicode (kutu/ok karakterleri) için UTF-8 çıktı
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Paket içe aktarımı (proje kökünden çalıştırılabilir)
sys.path.insert(0, str(Path(__file__).parent))

from agi_trader.config import load_config
from agi_trader.agents import Orchestrator
from agi_trader.report import format_all


def main(argv=None):
    p = argparse.ArgumentParser(description="AGI Trader — açıklanabilir kripto karar motoru")
    p.add_argument("--symbols", nargs="*", help="Analiz edilecek pariteler (örn: BTC/USDT)")
    p.add_argument("--timeframe", help="Birincil zaman dilimi (örn: 4h)")
    p.add_argument("--source", choices=["auto", "live", "synthetic"], help="Veri kaynağı")
    p.add_argument("--config", help="config.yaml yolu")
    p.add_argument("--json", help="Sonuçları bu dosyaya JSON olarak yaz")
    p.add_argument("--summary", action="store_true", help="İşlem günlüğü özetini göster ve çık")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.source:
        cfg.data["data_source"] = args.source
    if args.timeframe:
        cfg.data["primary_timeframe"] = args.timeframe
        if args.timeframe not in cfg.data["timeframes"]:
            cfg.data["timeframes"].insert(0, args.timeframe)

    orch = Orchestrator(cfg)

    if args.summary:
        print(json.dumps(orch.journal.summary(), indent=2, ensure_ascii=False))
        return 0

    symbols = args.symbols or cfg.symbols
    print(f"[AGI Trader] Analiz başlıyor — {len(symbols)} parite, kaynak: {orch.data.describe()}\n")
    signals = orch.run(symbols)

    print(format_all(signals, orch.describe_environment()))

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps([s.to_dict() for s in signals], indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(f"\n[✓] JSON yazıldı: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
