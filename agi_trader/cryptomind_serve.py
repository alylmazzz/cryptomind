#!/usr/bin/env python3
"""
CryptoMind — herkese açık paneli besleyen API sunucusu.

  python cryptomind_serve.py --host 127.0.0.1 --port 8210

  /api/*              SALT-OKUNUR panel (yalnız GET; bkz. server/public_api.py)
  /account/*          hesap + anahtar kasası (oturum + CSRF)
  /account/trading/*  otopilot: paper / testnet / canlı (oturum + CSRF + kapılar)

Üretimde nginx arkasında 127.0.0.1'e bağlanır ve /cryptomind/ altından yayımlanır.

Ortam değişkenleri:
  AGI_LIGHT_MODE=1              ağır modelleri (torch/RL/XGB/LGBM) kapat — az bellekli VPS
  CRYPTOMIND_MASTER_KEY=...     anahtar kasası (yoksa kasa KİLİTLİ, otopilot kurulamaz)
  CRYPTOMIND_LIVE_CONFIRM=EVET  canlı emir için operatör kapısı (config ile birlikte)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description="CryptoMind public API")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8210)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    import uvicorn
    from agi_trader.config import load_config
    from agi_trader.server import qualification_api as QA
    from agi_trader.server.public_api import create_public_app
    from agi_trader.server.account_api import create_account_app
    from agi_trader.server.secure_keys import vault_available, exchange_creds
    from agi_trader.server.trading_api import create_trading_router
    from agi_trader.auto.live_runner import Context, RunnerRegistry

    cfg = load_config(args.config or str(ROOT / "config.yaml"))
    runs = str(ROOT / "runs")

    # Otopilot kayıt defteri ÖNCE kurulur ki nitelendirme katmanı EXECUTION
    # sağlığını gerçek kaynaktan sorabilsin (kurulmadı ≠ bozuldu).
    registry_box = {}
    QA.EMS_PROBE = lambda: registry_box["r"].ems_ready() if "r" in registry_box else False
    QA.LIVE_PROBE = lambda: registry_box["r"].live_running() if "r" in registry_box else False

    app = create_public_app(cfg)

    def _regime(df):
        from agi_trader.analysis.regime import detect_regime
        return detect_regime(df)

    def _health():
        fn = getattr(app.state, "system_health", None)
        return fn() if fn else None

    # Haber/sosyal tarayıcı — anahtarsız RSS/Reddit/StockTwits/Binance; 10 dk'da bir
    from agi_trader.sentiment.news_scanner import NewsScanner
    from agi_trader.auto import simulator as SIM

    def _news_fetch(symbol, tf, limit):
        from agi_trader.execution.broker import Broker
        b = Broker("mexc", "paper", paper_capital=1.0)
        return b.fetch_ohlcv(symbol, tf, limit=limit)

    scanner = NewsScanner(lambda: SIM.default_config("mexc").symbols, fetch_ohlcv=_news_fetch,
                          interval_sec=int(os.environ.get("CRYPTOMIND_NEWS_INTERVAL", "1200")),
                          out_path=Path(runs) / "news" / "news_scan.json")

    ctx = Context(cm_signal=lambda s: app.state.signal_for(s),
                  qual_cell=QA.qualification_cell,
                  system_health=_health, regime=_regime,
                  slow_ctx=lambda s: app.state.context_for(s),
                  candidate_symbols=lambda: app.state.candidate_symbols(),
                  news_for=scanner.for_symbol, market_news=scanner.market)
    registry = RunnerRegistry(output_dir=runs, ctx=ctx, server_config=cfg,
                              creds_lookup=lambda uid, ex: exchange_creds(uid, ex, runs))
    registry_box["r"] = registry

    # Hesap/anahtar uçları AYRI bir uygulamadır ama aynı süreçte servis edilir.
    # Yol çakışması yok: panel `/api/...`, hesap `/account/...`. nginx ikisini
    # ayrı location bloklarıyla yayımlar; panel GET-only kalır.
    app.include_router(create_account_app(cfg, output_dir=runs).router)
    app.include_router(create_trading_router(registry, output_dir=runs, server_config=cfg))
    # 1.000 $ sanal simülatör — herkese açık, GET-only, anahtar taşımaz
    app.include_router(SIM.create_public_router(registry))

    ok, why = vault_available()
    print(f"CryptoMind API -> http://{args.host}:{args.port}", flush=True)
    print(f"  panel    : /api/*              (salt-okunur, GET)", flush=True)
    print(f"  hesap    : /account/*          (oturum + CSRF)", flush=True)
    print(f"  otopilot : /account/trading/*  (oturum + CSRF + kapılar)", flush=True)
    print(f"  simülatör: /api/simulator*     (herkese açık, sanal 1.000 $)", flush=True)
    print(f"  kasa     : {'HAZIR' if ok else 'KİLİTLİ — ' + why}", flush=True)
    for rec in registry.restore_all():
        print(f"  koşucu   : {rec}", flush=True)

    def _boot_news():
        import time as _t
        _t.sleep(int(os.environ.get("CRYPTOMIND_NEWS_BOOT_DELAY", "240")))
        if os.environ.get("CRYPTOMIND_NEWS", "1") in ("0", "false", "no"):
            print("  haber    : KAPALI (CRYPTOMIND_NEWS=0)", flush=True)
            return
        scanner.start()
        print("  haber    : tarayıcı başladı (10 dk)", flush=True)

    def _boot_simulator():
        """Sunucu portu bağlasın, ilk nitelendirme taraması (≈50 sn) bitsin; sonra
        simülatör. İkisi çakışınca RSS tepe yapıp pm2 bellek tavanına çarpıyordu."""
        import time as _t
        _t.sleep(int(os.environ.get("CRYPTOMIND_SIM_BOOT_DELAY", "90")))
        if os.environ.get("CRYPTOMIND_SIMULATOR", "1") in ("0", "false", "no"):
            print("  simülatör: KAPALI (CRYPTOMIND_SIMULATOR=0)", flush=True)
            return
        try:
            venue = os.environ.get("CRYPTOMIND_SIM_VENUE") or None
            print(f"  simülatör: {SIM.ensure_simulator(registry, venue=venue)}", flush=True)
        except Exception as e:
            print(f"  simülatör: kurulamadı — {type(e).__name__}: {e}", flush=True)

    import threading
    threading.Thread(target=_boot_simulator, name="cm-sim-boot", daemon=True).start()
    threading.Thread(target=_boot_news, name="cm-news-boot", daemon=True).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
