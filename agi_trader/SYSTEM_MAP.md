# CryptoMind — SYSTEM_MAP (2026-09-04)

Durum sözlüğü: **EXISTS** çalışıyor ve testli · **PARTIAL** kısmen · **UNVERIFIED** kod var, canlı/OOS kanıtı yok · **MISSING** yok · **DEPRECATED** kullanılmıyor. Bu harita koddan üretilmiş bir envanterdir, pazarlama metni değildir.

| Bileşen | Dosya | Durum | Not |
|---|---|---|---|
| Market state store + rate limit | `agi_trader/data/market_state.py` | EXISTS | TTL önbellek, eş zamanlı istek birleştirme, LIVE/DELAYED/STALE, devre kesici; **REST polling** (WebSocket MISSING) |
| Tier-A ucuz tarayıcı → Top-K | `auto/live_runner.py` `_tier_a_scan` | EXISTS | 40+4 parite, öğrenilmiş vekil ağırlıkları, kaynak GREEN/YELLOW/RED ile K dinamik |
| Kaynak yöneticisi | `live_runner._update_resource` | PARTIAL | RSS izleniyor; CPU/event-loop gecikmesi ölçülmüyor |
| Komite (12 rol + sleeve öz-oyu) | `strategies/committee.py`, `roles.py` | EXISTS | EV yarışması, veto incelemesi, stop-risk skoru, kapanmış-bar sinyali |
| Sleeve kütüphanesi (21 kod sleeve) | `strategies/sleeves_fast.py` | EXISTS | dip, dip_moderate, pullback, breakout, momentum, catalyst, squeeze, sweep, range, vwap×2, rs, news_overreaction, adaptive_trend, donchian, bos_retest, failed_breakdown/breakout, obi_momentum, swing_trend |
| Rejim seçici | `sleeves_fast.REGIME_SLEEVES` | EXISTS | 4 rejim; HMM yalnız ağır katmanda (`analysis/regime`), hafif katman EMA/ATR tabanlı |
| Giriş optimizasyonu | `strategies/entry_optimizer.py` | PARTIAL | bölge/optimal/max chase; **P(fill) sezgisel, kalibre edilmedi** |
| Ücret motoru | `execution/fee_adapter.py`, `strategies/fees.py` | EXISTS | anahtar yokken statik → güven ×0,85 |
| Venue router | `execution/venue_router.py` | EXISTS (opt-in) | market tablosu belleği yüzünden varsayılan kapalı |
| Çıkış motoru | `strategies/exit_engine.py` | EXISTS | hard stop, BE kilidi, zaman kilidi, erken iptal, kısmi TP, chandelier, yarı-tepe (NET), MODEL_EXIT, EDGE_DECAY, zaman |
| Devam olasılığı | `exit_engine.continuation_probability` | UNVERIFIED | sezgisel, kalibre edilmedi (fişte kaydı tutuluyor) |
| Fırsat rotasyonu | `live_runner._maybe_rotate` | EXISTS | EV_B − geçiş maliyeti − marj > kalan EV_A; günde ≤ 6 |
| Portföy modu / nakit | `strategies/portfolio_mode.py` | EXISTS | histerezis, gün-içi tepe geri-verme, breadth/korelasyon/haber/DD |
| Boyutlandırma | `live_runner._size` | EXISTS | risk/stop, tavanlar, taban 0,5, korelasyon bütçesi, sleeve tavanı; **portföy optimizasyonu MISSING** |
| Meta-tahsisçi + devre kesici | `learn/allocator.py` | EXISTS | Beta güvenilirlik, Thompson (araştırma), sleeve devre kesici (t-stat/Wilson) |
| Ders motoru | `learn/lessons.py` | EXISTS | rol güvenilirliği, dersler sınır+soğuma, gölgeler |
| Kaçırılan-fırsat atıf motoru | `learn/missed.py` | EXISTS | kör noktalar, kapı isabeti, özellik lift'i, kazanan/stop profilleri, challenger önerileri |
| Challenger | `learn/challenger.py` | EXISTS | gölge ≥30 + Wilson |
| Kalibrasyon + Monte Carlo | `learn/calibration.py` | EXISTS | Brier/kovalar; bootstrap IID (blok bootstrap MISSING) |
| Lifecycle / bilimsel kabul | `strategies/lifecycle.py` | EXISTS | hiçbir sleeve QUALIFIED değil |
| Replay / walk-forward / DSR / PBO | `auto/replay.py`, `scripts/cm_replay.py` | EXISTS | gerçek 1m/1h/4h; **ağır bağlam yok, sentetik defter** |
| Deneme kaydı + VALIDATION_REPORT | `runs/research/trials.jsonl`, `VALIDATION_REPORT.json` | EXISTS | replay her koşumda ekler |
| Haber motoru | `sentiment/news_scanner.py` | EXISTS | 9 RSS + Google + Reddit + StockTwits; taksonomi, katman, dedup; **LLM yok** |
| CVD / işlem akışı | `broker.fetch_trades` + `sleeves_fast.cvd_from_trades` | PARTIAL | Top-K için; OI/likidasyon MISSING |
| Zincir-üstü | `onchain/flow_engine.py` | UNVERIFIED | anahtarsız katman var, koşucuya bağlı değil |
| Araştırma fabrikası (250) | `research/library.py`, `harvester.py` | EXISTS | 24 gerçek Freqtrade stratejisi kutuda; hiçbiri üretimde |
| Gölge araştırma (çift/carry/üçgen/MM) | `research/pairs.py`, `carry.py`, `triangular.py`, `market_making.py` | EXISTS (SHADOW) | emir yok; carry opt-in |
| Broker (paper/testnet/live) | `execution/broker.py` | EXISTS | maker/taker, reduce-only, idempotent clientOrderId; **OMS durum makinesi PARTIAL** (UNKNOWN/RECONCILIATION_REQUIRED yok) |
| Mutabakat | `risk/live_guard.reconcile` | PARTIAL | testnet/live'da; paper'da yok |
| Kill-switch / risk limitleri | `risk/live_guard.py` | EXISTS | günlük zarar, DD, bayat veri, borsa hatası; venue/strateji düzeyinde kill MISSING |
| Prediction/outcome ledger | `server/qualification_api.py`, `trades` (hash zinciri) | EXISTS | işlem defteri sha256 zinciri |
| Uyarı kanalı | `notify/alerts.py` | EXISTS (yapılandırılmadı) | env ile Telegram/webhook |
| Güvenlik | `server/secure_keys.py`, `account_api.py` | EXISTS | kasa, para çekme izinli anahtar ret, GET-only panel; IP allowlist MISSING |
| UI | `terminal/src/*.tsx` | EXISTS | özet şeridi, fırsat/pozisyon kartları, evren, kaçırılanlar, araştırma |
| Öz-denetim / günlük rapor / defter doğrulama | `scripts/cm_selfcheck.py`, `cm_daily_report.py`, `cm_ledger_verify.py` | EXISTS | cron 00:05 UTC |
| Parite keşfi | `scripts/cm_pair_scout.py` | EXISTS | ölçülmüş ek evren (bugün 4 uygun) |
| WebSocket / event-driven veri | — | MISSING | REST + TTL |
| Emir defteri snapshot+delta senkronu | — | MISSING | REST üst defter |
| TCA (fill sonrası maliyet ayrıştırma) | `execution/tca.py` | UNVERIFIED | koşucuya bağlı değil |
| Vadeli/short yürütme | — | MISSING (spot) | short sleeve'ler yalnız vadeli izniyle |
| Eski `agi_trader/server/app.py` panosu | `server/app.py` | DEPRECATED | ana panel public_api |
