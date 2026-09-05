# CryptoMind — mindcorplab.com/cryptomind

Borsa botunun (agi_trader) halka açık paneli. **Kâğıt (paper) portföy, salt-okunur.**
Gerçek emir gönderilmez, borsa hesabına bağlanmaz, sunucuda hiçbir API anahtarı yoktur.

## Ne yayımlanıyor

| Katman | Kaynak | Not |
|---|---|---|
| Canlı paper portföy | `trend_daemon.py` → `runs/trend_state.json` | 17 varlık, günlük yeniden dengeleme |
| Risk & sağlık | `agi_trader/monitor/risk_monitor.py` | Sharpe, VaR/CVaR, drawdown, drift |
| Strateji künyesi | `runs/selected_universe.json`, `runs/portfolio_trend.json` | gerçek ölçüm dosyalarından okunur |
| Çok katmanlı analiz | `agi_trader/agents/orchestrator.py` | 5 kalibre parite, 4h, konsensüs kapısı |
| Grafik / makro / korelasyon | `server/chart.py`, `macro/events.py` | korelasyon gerçek getirilerden hesaplanır |

**Yayımlanan strateji bilinçli olarak karmaşık bot DEĞİL.** 1 saatlik TA-konsensüs botu
örneklem dışında edge üretmedi (2022-2025 her yıl zararda). Panelde OOS-doğrulanmış
Trend200+Mom20 (Sharpe 1.37 / CAGR %19.1 / DD %10.6 / Calmar 1.80) yayımlanıyor.

## Mimari

```
tarayıcı → nginx (mindcorplab.com)
             ├── /cryptomind/       → statik SPA   /var/www/cryptomind/web/
             └── /cryptomind/api/   → 127.0.0.1:8210 (pm2 "cryptomind")
                                       yalnız GET — limit_except GET HEAD
cron 01:00 UTC → /var/www/cryptomind/daily.sh → trend_state.json güncellenir
```

**Neden ayrı API?** Tam dashboard (`server/app.py`) `POST /api/credentials`,
`/api/auto/start`, `/api/learn` gibi DEĞİŞTİRİCİ uçlar içerir; internete açılırsa
kimlik doğrulaması olmadan kötüye kullanılabilir. `server/public_api.py` yalnız
okuyan uçları tanımlar — tehlikeli uçlar sunucuda **hiç mevcut değil** (404).

**Neden arka plan yenileyici?** Analiz hattı parite başına 8-20 sn sürer (çok
zaman dilimli veri çekimi). İstek anında çalıştırılsa nginx 60 sn'de timeout olurdu.
Yenileyici 15 dakikada bir anlık görüntü üretir; uçlar hazır sonucu anında döndürür
ve yaşı `age_sec` ile bildirir.

**`AGI_LIGHT_MODE=1`** (`agi_trader/core/light.py`): torch/transformers/XGBoost/
LightGBM/RL **hiç import edilmez**. Bayrağı sonradan `False` yapmak yetmiyordu —
import gerçekleştiği için torch tek başına ~400 MB RSS tutuyordu. Yerelde bayrak
kapalı olduğundan tam yığın aynen çalışmaya devam eder.

## Operasyon

```bash
# durum
ssh root@<SUNUCU_IP> 'pm2 list | grep cryptomind; curl -s localhost:8210/api/health'

# yeniden başlat / log
pm2 restart cryptomind --update-env
pm2 logs cryptomind --lines 50
tail -30 /var/www/cryptomind/runs/daemon.log

# günlük yeniden dengelemeyi elle çalıştır
/var/www/cryptomind/daily.sh
```

### Yeniden yükleme (yerelden)

```powershell
# 1) panel arayüzü
cd "Desktop\borsa botu\terminal"; npm run build     # base=/cryptomind/ (vite.config.ts)
# 2) python paketi + web -> sunucu  (WSL üzerinden; Bash tool'dan ssh ÇAĞIRMA)
#    scratchpad/deploy1.sh (tam) veya deploy3.sh (yalnız paket) örnek alınabilir
```

Dosyalar: `/var/www/cryptomind/` — `agi_trader/` (paket), `venv/` (system-site-packages
+ ccxt + yfinance), `web/` (SPA), `runs/` (durum), `ecosystem.config.js`, `daily.sh`.
nginx bloğu `/etc/nginx/sites-enabled/mindcorp` içinde, `location / {` öncesinde.

## Bilinen sınırlar

- Paper portföy sunucudaki `runs/trend_state.json` üzerinden yürür; yerel makinedeki
  Windows görev zamanlayıcısı **ayrı bir kopyayı** ilerletir. Yayımlanan kayıt sunucudakidir.
- Yenileyici döngüsü ~400-450 MB RSS kullanır (ccxt + pandas). `max_memory_restart: 600M`.
- `main` analiz zaman dilimi 4h sabittir; grafik zaman dilimi (1h/4h/1d/1w) isteğe bağlıdır.

## Otopilot katmanı (2026-09-02)

| Uç | Koruma | Ne yapar |
|---|---|---|
| `GET /account/trading/catalog` | — (oturumsuz da çalışır, anahtar içermez) | 10 borsa, anahtar/koşucu durumu, sunucu kapıları, strateji künyesi |
| `GET /account/trading/state?exchange=` | oturum | koşucu: pozisyon, işlem, karar izi, özsermaye eğrisi |
| `POST /account/trading/start` | oturum + CSRF + kapılar | `mode: paper\|testnet\|live` |
| `POST /account/trading/stop\|close_all\|resume\|params\|reset\|remove` | oturum + CSRF | |

Strateji: `agi_trader/strategies/video_scalp.py` (video kurulumu) → `auto/decision_chain.py`
(CryptoMind kapıları) → `auto/live_runner.py` (koşucu) → `execution/broker.py` (ccxt).
Durum `runs/live/runner_<uid>_<borsa>.json`; yeniden başlatmada paper devam eder,
testnet/canlı **yalnız pozisyon yönetimi** ile kalkar (yeni giriş için START).

**Canlı mod şartları (hepsi):** İŞLEM kapsamlı anahtar (para çekme KAPALI) · sunucu operatör
kapısı (`config.yaml execution.mode=live` + `allow_live: true` + `CRYPTOMIND_LIVE_CONFIRM=EVET`
ortamda — `start_api.sh`) · onay cümlesi · paper kanıtı (≥20 sanal işlem, net>0).
Panel paketi (`web/`) hash'li eski dosyalar silinmeden `rsync` ile güncellenir.
Deploy betiği: `C:\Users\Public\cm_deploy.sh` (WSL: `wsl bash /mnt/c/Users/Public/cm_deploy.sh`,
kaynak `C:\Users\Public\cm_stage`).

## Komite stratejisi + 1.000 $ simülatör (2026-09-02, 2. tur)

- `strategies/roles.py` 12 rol (9 oy + 3 VETO: maliyet, risk, denetçi) · `strategies/committee.py`
  (yavaş bağlam = `public_api` önbelleği, hızlı tetikleyici 1 dk) · `strategies/fees.py` (ücret
  tablosu; **MEXC maker %0 / taker %0,05** en ucuz) · `learn/lessons.py` (rol güvenilirliği, ders
  kuralları, veto edilen adayların gölge takibi, günlük `runs/live/GUNLUK_0_mexc.md`) ·
  `auto/simulator.py` (sistem koşucusu, uid 0, açılıştan `CRYPTOMIND_SIM_BOOT_DELAY` (90 sn) sonra).
- Uçlar (GET-only, herkese açık): `/api/simulator`, `/api/simulator/journal`, `/api/simulator/roles`.
- Ortam: `CRYPTOMIND_SIMULATOR=0` kapatır, `CRYPTOMIND_SIM_VENUE=binance` venue'yu zorlar.
- **Bellek:** süreç RSS ~1,0-1,3 GB (nitelendirme taraması). pm2 tavanı 1,4 GB. `start_api.sh`'a
  `MALLOC_ARENA_MAX=2` eklendi; paper modda `load_markets` yapılmaz; sandbox tablosu statik.
  Tavan aşımı = SIGKILL + 502 → panel geçici hatada durumu korur ama simülatör döngüsü sıfırlanır
  (durum `runs/live/runner_0_mexc.json`'dan geri yüklenir).

## 15 parite (2026-09-02, 3. tur)

`public_api.ALLOWED_SYMBOLS` 5 → 15: BTC ETH SOL DOGE AVAX + **LINK SUI NEAR PEPE AAVE UNI LTC DOT FIL BCH**.
Seçim ölçülerek: nitelendirme evreninde · Binance/Bybit/OKX/MEXC dördünde spot · MEXC 1 dk veri
hatasız · spread ≤ 10 bps · ±%2 derinlik ≥ 45 k$ · 1 dk σ ≥ %0,07. ADA/BNB/XRP kalibrasyon
eledi (config `pair_trend_gate: 999`), TRX σ çok düşük, FET/APT spread ~20 bps, ZEC/XMR dört
borsada yok. Ortamdan ezme: `CRYPTOMIND_SYMBOLS`. Simülatör evreni aynı 15; `max_open 4`,
emir tavanı 200 $. 15 paritelik yenileme döngüsü daha uzun (≈3-5 dk); RSS izlenmeli.

## Üst-sağ canlı işlem günlüğü (2026-09-02, 4. tur)

`terminal/src/TradeLog.tsx` — `.cm-toprow` ızgarasında sağ sütun (400 px, yapışkan; dar ekranda en üste
çıkar). Kaynak `GET /api/simulator/feed` (10 sn): durum bandı (`İLK İŞLEM BEKLENİYOR` →
`MAKER EMRİ BEKLİYOR` → `n AÇIK POZİSYON` → `SONRAKİ İŞLEM BEKLENİYOR`; HALT), tetikleyiciye en yakın
pariteler + eksik şartlar (komite `Verdict.fast`), kısa vadeli olasılıklar (mover ≥%1 oynama, 4h
konsensüs LONG/SHORT/NÖTR, ortalama yukarı olasılığı, HMM rejim dağılımı, komite p_win), olay akışı,
kapanan işlemler. Olasılıklar yalnız ölçülmüş modellerden okunur; panel/uç sayı üretmez.

## 5. tur — 40 parite, çoklu tetikleyici, haber tarayıcı, nakit modu (2026-09-02)

- **Katmanlar:** 15 AĞIR (`ALLOWED_SYMBOLS`, orkestratör+formasyon+gösterge+mover) + 25 HAFİF
  (`simulator.LIGHT_SYMBOLS`; `strategies/light_context.py` — 4h/1h rejim+seviye+ATR, 15 dk önbellek).
  Ağır hatta 40 parite bellek tavanını aşardı; hafif katman parite başına ~500 bar tutar.
- **Tetikleyiciler** (`committee.triggers`): haber katalizörü ×0,8 (yalnız hareket DOĞRULANMIŞ) ·
  dip ×1,0 · kırılım ×0,8 (20 bar + hacim ≥1,5×) · geri çekilme ×1,0 · momentum ×0,7 (EMA9×21) ·
  ılımlı dip ×0,6 (z ≤ −1,2, RSI ≤ 42). Öncelik bu sıradadır.
- **Haber/sosyal tarayıcı** `sentiment/news_scanner.py` (betik: `python -m agi_trader.sentiment.news_scanner
  --once --symbols ... --confirm`): RSS ×9 + Google News + Reddit RSS + StockTwits + Binance duyuru;
  kelime-sınırlı sözlük (alt-dize "Binance"→"ban" hatası düzeltildi), yarı-ömür 6 sa, hareketlilik
  doğrulaması (hacim ≥1,5× ve 4 sa ≥ 0,5 ATR). Sunucuda 150 sn sonra başlar, 10 dk'da bir;
  çıktı `runs/news/news_scan.json`. Ortam: `CRYPTOMIND_NEWS=0`, `CRYPTOMIND_NEWS_INTERVAL`.
- **Nakit modu** (`committee.market_risk`): seviye 1 DİKKAT (4h genişlik ≤ −0,3 ya da haber risk-off ≥ 1)
  → yeni giriş yok; seviye 2 NAKİT (genişlik ≤ −0,5 + BTC TREND↓ ya da haber sistemik ≥ 3 puan ve ≥ 2 başlık)
  → bekleyen emirler iptal, pozisyonlar "NAKİT MODU" ile kapatılır; seviye 0'da otomatik çıkış.
- Feed: `risk_mode`, `cash_mode`, `news_market`, `tiers`; her karar `tier` (heavy/light) ve `news` özeti taşır.

## Tur 6 — Quant platform (MASTER PROMPT, 15 faz) — 2026-09-02

Yeni modüller (hepsi LLM'siz, hot-path'te ağ çağrısı yalnız depo üzerinden):
- `data/market_state.py` — MarketStateStore (TTL önbellek + eş zamanlı istek birleştirme + LIVE/DELAYED/STALE tazelik + ucuz özellikler bar-damgasıyla) + RateLimitCoordinator (token bucket + devre kesici; DUVAR SAATİ kullanır, simülasyon saati yalnız veri önbelleğine).
- `strategies/sleeves_fast.py` — 7 yeni sleeve (squeeze_breakout, sweep_reversal, range_edge, vwap_reversion, vwap_continuation, rs_momentum, news_overreaction) + rejim→izinli sleeve seçici (zıt sistemler aynı anda oy kullanmaz) + göreli güç sıralaması.
- `strategies/exit_engine.py` — FIXED_TARGET / PARTIAL_AND_RUN / DYNAMIC_PEAK; hard stop asla kalkmaz; NET yarı-tepe geri-verme (retain 0,35–0,70, varsayılan 0,50, silahlanma eşiği = max(maliyet×2, ATR, min MFE)); chandelier trailing; MODEL_EXIT; EDGE_DECAY; TIME_STOP; peak-capture oranı.
- `strategies/portfolio_mode.py` — RISK_ON / SELECTIVE / DEFENSIVE / CASH (genişlik, korelasyon şoku, BTC rejimi, haber seviyesi, drawdown, sağlık, bayat veri payı).
- `strategies/entry_optimizer.py` — giriş bölgesi / optimal / MAX CHASE; maker-öncelikli ama EV kıyaslı (dolmazsa q=0,85 kovalama). P(fill) SEZGİSEL, kalibre EDİLMEDİ (notta yazılı).
- `execution/fee_adapter.py` (gerçek hesap ücreti, TTL 6 s, yedek statik `verified=False` → güven ×0,85) · `execution/venue_router.py` (tüm-dahil bps kıyası).
- `strategies/lifecycle.py` (IDEA→…→RETIRED; canlı için ≥ LIMITED_LIVE + bilimsel kapılar) · `learn/challenger.py` (ders motoru üretimi DOĞRUDAN değiştirmez; gölgede ≥30 çözülmüş karar + Wilson alt sınır ile terfi).
- `committee.evaluate` yeniden yazıldı: tetikleyiciler ∪ sleeve'ler → her biri için plan+fiş → **EV yarışması** → giriş optimizasyonu → vetolar (NEGATİF EV, MAX CHASE, Tier-3 tek başına katalizör değil).
- `news_scanner`: olay taksonomisi (LISTING…RUMOR), kaynak katmanı 1/2/3, kaynaklar arası dedup (normalize başlık).
- `live_runner` yeniden yazıldı: depo üzerinden tek fetch, Tier-A ilgi taraması → Top-K (kaynak GREEN/YELLOW/RED ile küçülür; RED'de giriş yok, çıkış katmanı sürer), portföy modu, çıkış motoru, challenger döngüsü, `sync_config`, `best_action`, `top_opportunities`.
- Simülatör: geri yüklemede konfig senkronu (15→40 parite bug'ı kapandı), feed'e portföy modu / en iyi eylem / top 3 / kaynak / challenger.
- UI: TradeLog özet şeridi + TOP FIRSAT kartları (bölge/optimal/max chase/stop/hedef/komisyon/risk) + pozisyon kartında tepe/PCR/yarı-tepe seviyesi/chandelier.

Tuzaklar:
- Depo önbelleği simülasyon saati (`now`) ile çalışır; zaman geriye giderse önbellek geçersiz (replay).
- Hız sınırlayıcı simülasyon saatiyle karıştırılırsa RateLimited yağar → duvar saati.
- PARTIAL_AND_RUN'da kalan hedefte KAPANMAZ; yarı-tepe/trailing ile çıkar (eski TP testi buna göre güncellendi).
Testler: 599 geçti (tests/test_quant_platform.py 30 yeni).

## Tur 7 — Kaçırılan-fırsat atıf motoru + Strateji Araştırma Fabrikası (250) — 2026-09-03

- `learn/missed.py` — **MissedEngine**: veto edilen / hiç değerlendirilmeyen (Top-K dışı, açık-pozisyon tavanı, kaynak RED, nakit/portföy, HALT, bayat, rejim seçici) / yürütülemeyen (maker dolmadı, max chase) her aday ufuk boyunca hedef/stop'a karşı izlenir. Çözülünce ATIF: kapı isabeti (Wilson), kapının hesaba katmadığı destekleyici özellikler (lift), bilgi varlığı/yokluğu (ağır bağlam, haber, defter, doğrulanmış ücret, nitelendirme, rol verisi), uyaranlar; Türkçe anlatı `runs/live/KACIRILANLAR_0_mexc.md` + JSONL; kanıtlı (n≥20, Wilson üst<0,5, 12 sa soğuma) öneri → challenger; kapasite (top_k/max_open) doğrudan ama LOGLU + audit.
- **Bulunan gerçek kusur:** rejim seçici 4h YUKARI trendde `dip/dip_moderate/sweep_reversal` sleeve'lerini kapatıyordu → RSI 15'lik trend-içi dip hiç değerlendirilmiyordu (video kurulumunun ta kendisi). Düzeltildi; ayrıca susturulan tetikleyiciler `REJİM_SEÇİCİ` kapısıyla gölgeye alınıyor.
- `learn/allocator.py` — meta-tahsisçi: (sleeve, rejim) Beta güvenilirliği (n≥10), boyut çarpanı [0,5·1,2], Thompson sıralaması yalnız araştırma önceliği; `sleeve_reliability` SEÇİCİ portföy moduna bağlandı.
- `research/library.py` — 250 kayıtlı hipotez (A–S aileleri), dürüst durumlar: IMPLEMENTED / SHADOW / RESEARCH / DATA_NOT_WIRED (OI, likidasyon, işlem akışı) / NO_DATA (zincir-üstü, DEX, sosyal hacim); huni; QUALIFIED = 0 (lifecycle kapıları geçilmedi).
- `research/harvester.py` — açık kaynak (Freqtrade IStrategy) dosyasını AST ile ÇALIŞTIRMADAN hipoteze çevirir: lisans → statik inceleme (exec/subprocess/ağ yasak) → indikatör/koşul çıkarımı → özellik ailesi eşlemesi → `runs/research/inbox/`. Üretime otomatik geçmez.
- Gölge araştırma modülleri (emir YOK): `pairs.py` (EG kointegrasyon ÜÇ kapı: ADF t<−3,5 + getiri korelasyonu ≥0,3 + Kalman/OLS β tutarlılığı; OU yarı-ömür; z≥2/≤0,5/≥4), `carry.py` (funding EV = funding − 2 bacak ücret − spread/kayma − rezerv; tipik %0,03/8s 3 günde NEGATİF), `triangular.py` (bid/ask + ücret + rezerv; Bellman-Ford), `market_making.py` (A–S bps ölçeğinde, kalibre edilmedi), `lab.py` orkestratörü (yalnız GREEN'de; carry opt-in `CRYPTOMIND_RESEARCH_CARRY=1`).
- 6 yeni komite sleeve'i: adaptive_trend (çok-ufuk trend skoru + 0,5–1,5 ATR geri çekilme + defter onayı), donchian_breakout, bos_retest, failed_breakdown (LONG), failed_breakout (SHORT, yalnız vadeli), obi_momentum (defter dengesi + mikro-fiyat). Toplam 20 sleeve EV yarışmasında.
- Venue kıyası: her çağrıda yeni ccxt örneği (load_markets → bellek) yerine registry'de paylaşımlı public istemci + parite başına 30 dk.
- Uçlar: `/api/simulator/missed`, `/api/research` (GET). UI: "KAÇIRILANLAR · NEDEN YAPILMADI" ve "ARAŞTIRMA FABRİKASI (250)" sekmeleri; TradeLog şeridinde kaçırılan/kaçınılan sayacı.
- Testler: 621 geçti (test_research_factory.py 22 yeni).
- **Canlı karşı-olgusal kanıtla ikinci düzeltme:** ders motorunun veto sayacı DENETÇİ kapısında 24 engelleme → 14 hedefe ulaştı / 0 stop. Sebep: "yapısal hedef < 1,2×stop" vetosu. Artık VETO değil: hedef rr×stop'a çekilir, **ilk direnç kısmi-TP** olur, boyut ×0,7 (`build_plan.size_penalty`, `partial_tp_near`). Missed motoru bunu ölçmeye devam eder.
- SAĞLIK RED/UNKNOWN kapısı da kör nokta olarak gölgeleniyor (yeniden başlatma sonrası sağlık UNKNOWN penceresi).
- Hasatçı canlıda 29 gerçek Freqtrade stratejisi denedi: 24 RESEARCH_INBOX (MIT lisansı depo kökünden), 5 FETCH_FAILED (yol farklı) — hiçbiri üretime geçmedi.
- Bellek: deploy sonrası tepe 1175 MB (YELLOW eşiği 1050) → 1009–1017 MB plato; RED'e girmedi, restart yok. Döngü 12–26 sn (YELLOW'da Top-K yarıya iner).
- Tuzak: `store.cheap_features` NaN üretebilir (σ=0) → missed kaydı JSON'da patlıyordu ("Out of range float values") → `_clean` ile NaN→None.

## Tur 8 — "Gece hiç işlem açılmadı" kök nedenleri + veto incelemesi — 2026-09-03

Canlı veriden (döngü 1069, 84 gölge kaydı) bulunan kök nedenler ve düzeltmeler:
1. **SEÇİCİ mod kilidi**: `reliable_only` ölçülmemiş sleeve'leri de kapatıyordu → hiç işlem yok → hiç ölçüm yok → hiç sleeve yok. Artık yalnız ÖLÇÜLMÜŞ ve güvenilmez (<0,5) sleeve'ler kapanır.
2. **Portföy modu zıplaması**: SAVUNMA↔SEÇİCİ↔RİSK AÇIK her dakika değişiyordu; SAVUNMA kapısı 3 kaçırılan / 0 kaçınılan. Eşikler RISK_ON ≤0 · SELECTIVE 1–2 · DEFENSIVE 3–4 · CASH ≥5 + koşucuda 3-döngü histerezis (CASH anında).
3. **Hafif katmanda oy yapısal olarak sıfır**: roller veri vermeyince ağırlıklı oy ~0 → OY vetosu. Tetikleyen sleeve'in kanaati `sleeve_sinyali` rolü (ağırlık 0,70, güven 0,6) olarak oya katıldı; güvenilirliği ders motoru ölçer. Denetçi yalnız HİÇ bağlam yoksa (age None) veto eder.
4. **Devam eden 1 dk barda titreşen sinyal**: 30 sn örnekleme "yeşil bar"ı kaçırıyordu. `closed_bar_fallback`: son KAPANMIŞ bar tetiklediyse ve fiyat ≤0,5 ATR kaçtıysa sinyal geçerli.
5. **Veto incelemesi** (`strategies/veto_review.py`): veto alan her aday için formasyon varlıkları (ağır konsensüs / yapısal bayraklar), indikatör al/sat/nötr sayımı (ağır 300'lük aile ya da hafif 8 gösterge), strateji uyumu, haber etiketi. Yalnız yumuşak vetolar (OY/GÜVEN) bütünsel kanıtla ×0,6 aşılır; roller net karşı oy verdiyse ya da sert veto (komisyon/EV/denetçi/risk/sağlık/haber riski/max chase) varsa asla. Panelde "İNCELEME · …" satırı.
6. Kaçırılan motoru: **yol** ("hedefe 23 dk'da ulaştı; önce aleyhe %0,4 gitti (stop'un %35'i)"), **kazanan profili** (medyan süre, MAE/stop, stop'a yaklaşan pay, kurulum/kapı dağılımı, DERS) ve **stop profili** (MFE/hedef, "hedefe yaklaşıp döndü" payı, uyaranlar, DERS); Tier-A vekil ağırlıkları kaçırılan/kaçınılan isabetinden öğrenilir (n≥8, [0,5·1,5]) — canlıda kırılım vekili 0/10 → ×0,5.
Testler: 632 geçti.
- **NaN 500 kaynağı bulundu:** `/api/risk` ve `/api/trend` (eski uçlar) NaN döndürüp 500 veriyordu; panel her 5 sn'de bir hata logluyordu (55→69). `server/safe_json.py` SafeJSONResponse tüm uygulamalarda varsayılan: NaN/inf→null, numpy skaler→Python.
- **Boyut tabanı:** yığılan haircut'lar (sleeve ×0,6 · inceleme ×0,6 · portföy ×0,7 · rol çarpanları) TIA'da 4,76 $ üretti → "asgari 10 $" reddi. `_floor_notional`: karar AÇ ise ve oda/nakit varsa asgariye tamamlanır.
- Haber tarayıcı açılış gecikmesi 150→240 sn (simülatör boot'uyla çakışan bellek tepesi 1233 MB) + tarama sonrası gc.
- **Venue kıyası bellek tuzağı:** ilk maker emrinde `_compare_venues` üç borsanın (binance/bybit/okx) market tablosunu yükledi → RSS 1078→1389 MB (tavan 1400), kaynak RED, girişler durdu. Artık opt-in (`CRYPTOMIND_VENUE_COMPARE=1`), varsayılan KAPALI.
- **p_win öncülü:** nitelendirme hücresinin `p_model_live` değeri yalnız hücre QUALIFIED ise kullanılır (sıfır QUALIFIED → hep 0,5). Aksi hâlde yönsel edge'i olmayan modelin p=0,3'ü heavy paritelerde EV'yi negatife çekip (SUI: 99 al / 21 sat göstergeye rağmen −%0,72) vetoluyordu.
- İlk maker emirleri girildi (TRUMP 32,40 $ @ 2,221 dip_moderate; STRK) — ikisi de max chase aşıldığı için dolmadı; kaçırılan motoru "yürütme" kaydı olarak izliyor.

## Tur 9 — "+%0,10" teşhisi ve kâr düzeltmeleri — 2026-09-03

Gün sonu ölçümü: 30 kapanan işlem, kazanma %53, kâr faktörü 2,1, ama net +0,98 $. Sebepler ve düzeltmeler:
1. **Günlük tavan 30** (simülatör varsayılanı): 16:00'dan sonra 85 aday engellendi, 13'ü hedefe ulaştı / 0 stop (TRUMP +%3,8, PEPE +%3,3, OP +%2,5). → 200; kaçırılan motoru `max_trades_per_day` için +40 öneri üretebilir.
2. **Boyut 30 $**: sleeve ×0,6 · inceleme ×0,6 · portföy ×0,7 · rol çarpanları çarpılınca %3'lük pozisyon. → `size_floor` 0,5 + emir tavanı 200 $.
3. **13/30 çıkış zaman-stopu, çoğu kârdayken eksiye döndü** (OP +%0,19 tepe → −%0,19; TIA +%0,33 → −%0,41): sabit-hedef modunda başabaş yoktu. → **BE kilidi** (tepe net ≥ 1,5×maliyet → stop başabaş+maliyet), **zaman kilidi** (ufkun %60'ında kârdaysa stop başabaş), silahlanma eşiği 3,0→2,0×maliyet / 0,30→0,20.
4. **dip_moderate 45 dk zaman-stopu = kazananların medyan hedef süresi (44,8 dk)** → 90 dk; dip 90; obi 45.
5. **Stop öngörüsü**: 23 stop'un 0'ı hedefin yarısına ulaştı, 30 kazananın 0'ı stop'un %60'ına kadar aleyhe gitti → **erken iptal** (ufkun ilk yarısında MAE ≥ 0,6×stop ve tepe ≤ 0) + giriş anında **stop-risk skoru** (uyaranlar: aşırı uzama, ters trend, hacim yok, geniş spread, ters kırılım, haber riski; ≥2 → ×0,6, ≥3 → veto).
6. **Bellek**: her 10 dk'lık haber taraması +200 MB → RED → girişler duruyordu. Aralık 20 dk + RED için 2-okuma histerezisi.
7. Simülatör geri yüklemede `exit`/`params` de senkronlanıyor (eski arm 3,0 diskte kalıyordu).
Testler: 639 geçti.

## Tur 10 — Kurumsal katman: replay/bilimsel kabul, kalibrasyon+Monte Carlo, uyarı, defter zinciri, CVD — 2026-09-03

- `auto/replay.py` + `scripts/cm_replay.py`: gerçek geçmiş 1m/1h/4h veriyle (ccxt, CSV önbelleği) komite bar bar oynatılır (lookahead yok, imleç), sleeve başına beklenti/bootstrap CI/maliyet×2/alt-dönem/PSR/DSR/CSCV-PBO → lifecycle kanıtı ve kapılar. Sınırlar dürüst: ağır bağlam yok, dolum bar low/high, haber yok.
- `learn/calibration.py` + `/api/simulator/calibration`: fiş p_win kalibrasyonu (Brier, güvenilirlik kovaları, beceri) + Monte Carlo bootstrap (P5/P50/P95, maks DD, günlük limit aşma, iflas olasılığı).
- `notify/alerts.py`: Telegram/webhook (yalnız ortam değişkeni: CRYPTOMIND_TG_TOKEN/CHAT, CRYPTOMIND_ALERT_WEBHOOK), anahtar başına 60 sn, arka planda; açılış/kapanış/HALT/NAKİT/gün sonu/hata fırtınası.
- Defter zinciri: her kapanan işlem sha256(prev+kayıt) taşır; `verify_ledger` + `scripts/cm_ledger_verify.py`; `full_state.ledger`.
- CVD: Top-K adayları için `fetch_trades` → taker akışı (cvd_ratio/burst) → obi_momentum ve adaptive_trend onayı, veto incelemesinde satır (`CRYPTOMIND_CVD=1`).
- Aciliyet: SLEEVE_URGENCY — kırılım/katalizör/momentum anında taker (0 bar), dip/geri çekilme 2 bar maker.
- Strateji çeşitlendirmesi: `max_open_per_sleeve` (2) + korelasyon bütçesi (|ρ|≥0,7 → boyut ×0,5).
- Tazelik: tarama ve kararlarda LIVE/DELAYED/STALE; panelde GECİKMELİ/BAYAT rozeti; hata sayacı (10 dk).
- `scripts/cm_selfcheck.py` (PASS/FAIL tablosu), `scripts/cm_daily_report.py` (cron 00:05 UTC → runs/reports/), `scripts/cm_ledger_verify.py`.
- Testler: 650 geçti.

## Tur 11 — Ultimate master prompt işlemesi: devre kesici, rotasyon, gün-içi tepe, swing, evren — 2026-09-04

Kanıt (gün sonu): özsermaye tepesi 19:22'de +3,99 $ → gece +0,0 $. Kaybettiren sleeve'ler: dip_moderate 22 işlemde −1,16 $, obi_momentum 12'de −0,67 $, vwap_continuation 2'de −0,72 $. Kazananlar: dip +1,48 $, failed_breakdown +0,67 $, catalyst +0,56 $. Replay (1 gün, 5 parite, gerçek MEXC): 55 işlem, +%0,47, beklenti %0,11 CI[0,02–0,21], maliyet×2 %0,054, PSR 0,99, DSR 0,99 (n_trials 20), **PBO 0,67**, maks DD %0,22, ücret/brüt %34.
- `learn/allocator.py` **sleeve devre kesici**: son ≥12 işlemde net<0 ve (Wilson üst<0,55 ya da t<−1) → 6 sa duraklat; komite duraklatılanı kullanmaz, adaylar `SLEEVE_DURAKLATILDI` kapısıyla gölgelenir.
- `exit_engine.continuation_probability` (sezgisel, kalibre edilmedi) + koşucuda **kalan EV** (`remaining_ev_pct`) → `_maybe_rotate`: EV_B − 2×maliyet − 0,15 > kalan EV_A (silahlı ve devam p≥0,5 olan kazanan rotasyona verilmez); günde ≤6; tavan doluyken aday `_probe_candidate` ile emir vermeden değerlendirilir.
- **Gün-içi tepe geri-verme (portföy)**: gün kazancı ≥ %0,25 sermaye ve %60'ı geri verildiyse SAVUNMA (yeni giriş yok, stoplar başabaşa).
- **swing_trend** sleeve: 4h trend + 1h EMA20 geri çekilmesi, 3 gün ufuk, DYNAMIC_PEAK, chandelier 4h ATR (`atr_hint`).
- `/api/simulator/universe` + panel EVREN sekmesi (tazelik, ilgi, rejim, z/RSI/trend/CVD/OBI, tetik, EV, stop-risk, pozisyon, inceleme).
- `scripts/cm_pair_scout.py`: ölçülmüş ek evren (hacim/derinlik/hareket/spread); MEXC'te bugün yalnız **4 uygun** (PUMP, BNB, TRX, ZEC) — 10 istenmişti, likidite eşiğini geçen yok; deploy'da her seferinde yeniden ölçülür (`runs/live/universe_extra.json`, `CRYPTOMIND_EXTRA_SYMBOLS`).
- Deneme kaydı `runs/research/trials.jsonl` (silinmez) + `VALIDATION_REPORT.json`; DSR n_trials kayıttan.
- `SYSTEM_MAP.md` (EXISTS/PARTIAL/UNVERIFIED/MISSING) ve `EXPERT_REVIEW.md` (8 disiplinli öz-hakemlik; gerçek dış hakem DEĞİL).
- **Tuzak:** `open(path, "w", newline="\n")` ValueError verir ama dosyayı sıfırlar → cm_deploy.sh boşaldı, yeniden yazıldı. Yama betiklerinde `newline=chr(10)`.
- Testler: 663 geçti.

## Tur 12 (2026-09-04) — "5 $ zarar" kök nedenleri + kanıt kapıları + tam günlük/özsermaye

**Ölçüm (85 kapanan işlem, MEXC sanal 1.000 $):** brüt −0,60 $ · komisyon 4,18 $ · net −4,78 $ · kazanma %50,6 ama
ortalama kazanç 0,14 $ / ortalama kayıp 0,26 $ (ödeme oranı 0,55 → %50 kazanma ile negatif beklenti).
- **Taker girişler bütün zarar:** taker 43 işlem −6,02 $ (komisyon 3,10), maker 42 işlem +1,24 $. obi_momentum taker −2,48 $ / maker +0,52 $.
- **Kanıtsız sleeve'ler tam boyutta:** dip_moderate 28 işlem −2,71 $ (kazanma %50 → Beta güvenilirliği 0,5 → ağırlık 1,0!),
  obi_momentum −1,95 $, momentum −0,93 $ (2/2 kayıp), vwap_continuation −0,72 $. Bu dördü hariç: +1,54 $.
- **Boyut 4× büyüdü, kenar kanıtlanmadan:** sabah 19–24 $ (+0,98 $ / 30 işlem) → akşam 77–92 $ (−4,90 $ / 32 işlem, 20–08 UTC).
- **Çıkış nedenleri:** EARLY_ABORT 9/9 kayıp −5,72 $ (hepsi 19:45–04:31 UTC, ort. 82 $), TIME_STOP 17 işlem −3,03 $, STOP −1,52 $;
  kazananlar TRAIL +3,18 / GIVEBACK +2,40. Sabit hedefe 85 işlemin **1'i** ulaştı (hedef medyanı %2,02, kazanan MFE medyanı %0,55).
- **Piyasa betası:** işlem net% ↔ BTC sonraki 15 dk getirisi korelasyonu 0,35; ama erken-iptal işlemleri BTC ±%0,1–0,3 bandında →
  kötü GİRİŞ, piyasa çöküşü değil.

**Karşı-olgusallar (gerçek 1 dk MEXC klines ile, aynı 85 işlem):**
| kural | net | karar |
|---|---|---|
| stop = 0,8× / 0,6× mevcut | +0,08 / −0,19 $ | RED — kazanan MAE/stop q95 = 0,76; daraltma kazananı keser |
| maker çıkış (limit @ çıkış, 2 bar) | dolum 60/85, kom. −1,95 $ ama dolmayan 25'te kayma −3,40 $ | RED |
| **kanıt tavanı 25 $** (kanıtsız sleeve) | **−0,92 $** | UYGULANDI |
| + aciliyet-0 sleeve'de taker girişi yok | +0,08 $ | UYGULANDI |
| + son 20 net<0 → ×0,5 | +0,40 $ | UYGULANDI |
| + seans t≤−1,5 → ×0,5 | +0,56 $ | UYGULANDI (kendini ölçümle günceller) |
| 10 dk'da ≤2 giriş | −3,18 $ (60 işlem) | RED — zayıf |
| zaman-stop 60 dk uzatma | 8/12 başabaşa döndü, 3'ü stop | UYGULANMADI — n küçük |

**Kod (kanıt kapıları):** `learn/allocator.py` sleeve durum makinesi UNPROVEN/PROVEN(n≥20, t≥1)/PAUSED(6→12→24→48 sa,
kanıt SIFIRLANMAZ)/PROBATION(8 işlem) + `notional_cap` · `committee.py` `probe_notional_usdt=25`, `taker_requires_proof`
(PROVEN değilse taker→maker 1 bar, `no_chase`), `achievable_target_pct` (fişte `ev_achievable_pct`, veto DEĞİL) ·
`live_runner.py` `_derisk` (son 20 net<0 ×0,5), `_session_gate` (4 sa UTC bloğu, 14 gün, n≥15: t≤−1,5 ×0,5 / t≤−2,5 kapalı),
`governance()` (panel KANIT KAPILARI kartı), `equity_history` (son 6 sa 30 sn, öncesi 5 dk kova, kalıcı; eski başlangıç
kapanan işlem defterinden `src:ledger` ile tamamlanır), işlem listesi 500→3000 kalıcı.
**Uçlar:** `GET /api/simulator/trades?page=&per_page=` (seq = kronolojik sıra, sayfa 1 = en yeni) ·
`GET /api/simulator/equity?max_points=` (başlangıçtan bugüne, kova-min/maks indirgeme, işlem işaretleri).
**Panel:** `EquityChart.tsx` (zaman ekseni, Brush pencere, sürükle-pan, tekerlek zoom, BAŞLANGIÇ düğmesi) ·
`TradeTable.tsx` (sayfalı, #seq) · GÜNLÜK/İŞLEMLER sekmeleri · KANIT KAPILARI kartı.
Testler: `tests/test_tur12.py` (15). Staging: `bash /c/Users/Public/cm_stage_py.sh` (Git Bash) → `MSYS_NO_PATHCONV=1 wsl bash /mnt/c/Users/Public/cm_deploy.sh`.
- **Deploy 2026-09-04 09:32 UTC** doğrulandı: `/api/simulator` governance (tüm sleeve'ler UNPROVEN → 25 $), `/api/simulator/equity`
  start_ts = ilk işlem (09-03 06:58, 34 defter noktası), `/api/simulator/trades?page=3` → #37–#13, RSS 896 MB GREEN.
- **Tuzak:** `cm_deploy.sh` adım 0'daki `pm2 describe` TTY'siz uzak `bash -s` içinde süresiz takıldı → pm2 çağrıları `timeout` ile
  sarıldı (yedek `cm_deploy.sh.bak-tur12`). Deploy çıktısını `| tail`e değil `tee`ye ver; aksi hâlde takılma görünmez.

## Tur 13 (2026-09-04) — YouTube kurulumları: iddialar değil, mekanik çekirdek

**Kaynak taraması:** yt-dlp ile 8 arama (TR+EN), 90+ video listelendi, **21 videonun transkripti** çıkarıldı
(`--write-auto-subs`; İngilizce altyazıda 429 sık, TR sorunsuz). 5 paralel ajanla kurulum çıkarımı yapıldı.

**Kanıt tablosu (dürüst):** hiçbir videoda üçüncü parti denetimli (myfxbook/broker onaylı) günlük %10 kanıtı YOK.
En iyi kanıtlar: Craig Percoco n=303 geri test (kazanma **%36**, R/R 1:4), TrippaTrading 7 günlük elle test
(31 işlem, %51,6, R/R 1:2, komisyon hariç), Data Trader 6-10 işlemlik mini testler ("yetersiz" itirafıyla).
Kalanı ekran görüntüsü / TradingView replay / sözlü beyan. Karşı görüş (TraderNick): günlük %1 bile bileşik
olarak 5 yılda "Elon Musk'ın 3-4 katı" eder; profesyonel hedef **aylık %1-5**.

**Alınan 10 kurulum** (`strategies/sleeves_video.py`, kaynak künyesiyle): `fvg_fill` (adil değer boşluğu dolumu,
4 kanalda ortak) · `ifvg_reclaim` (ters çevrilen boşluk) · `range_reclaim` (4h aralık dışına kapanış + geri dönüş,
2 kanal, en mekanik) · `manipulation_candle` (`low[t]<low[t-1]` VE `close[t]>high[t-1]`, tam tanımlı) ·
`opening_range` · `ema_engulf` (Trippa; TR videolar içinde en kodlanabiliri) · `poc_reversion` (hacim profili
yaklaşık, tik verisi yok) · `order_block` (MSS öncesi son zıt mum) · `stoch_cross_back` · `bb_lower_band`.

**ALINMAYANLAR** (`NOT_IMPLEMENTED`, panelde görünür): kapalı kaynak göstergeler (MT4 Glass, "Bull Trading",
"AI trading", "Manipulation X") — formülü yayımlanmamış, yeniden üretilemez · elle çizilen trend çizgisi /
"bariz likidite seviyesi" (proxy: swing-pivot + ATR eşiği) · 50-100x kaldıraç ve işlem başına %10 risk
(iflas olasılığı) · stopsuz tutma / ortalama düşürme (martingale) · günlük %10 hedefinin kendisi.

**ÖLÇÜM — İKİ AŞAMA, İKİNCİSİ BİRİNCİYİ ÇÜRÜTTÜ (yöntem: gerçek 1 dk veri, ileriye bakış yok, maliyet %0,14
gidiş-dönüş düşülmüş, komite oylaması/vetoları OLMADAN ham sinyal):**

1) **1 günlük** (33 parite, 868 aday): toplam −0,093% t −5,09 ama NY_AM (13-16 UTC) **+0,141% t +3,38**,
   ASYA −0,241% t −5,72, seans dışı −0,145% t −6,32. NY_AM'de 9 sleeve'in 7'si pozitif → videoların ortak
   "kill zone" iddiası doğrulanmış göründü.
2) **7 günlük** (aynı 33 parite, **7.343 aday**, aynı yöntem): toplam **−0,129% t −20,37**, kazanma %37.
   **NY_AM −0,017% t −0,90** — pozitif DEĞİL. Bütün seanslar negatif (LONDRA −0,244 t −14,5 en kötü).
   Sleeve bazında en iyisi range_reclaim −0,074 t −2,73; hepsi negatif.
   NY_AM içinde 5 kurulum pozitif hücre veriyor (manipulation_candle +0,118 t +1,95 · opening_range +0,112 ·
   range_reclaim +0,090 · bb_lower_band +0,052 · poc_reversion +0,033) ama **50 hücre denendiği için
   çoklu-test eşiğinin (t≈3) altında.**

**KARAR: hiçbiri canlıya çıkmaz.** 10 kurulum `lifecycle.SHADOW_SLEEVES` ile **GÖLGEDE** doğar: sinyal
üretir, **emir VERMEZ** (paper dahil), kaçırılan-fırsat motorunda `silenced` olarak ölçülmeye devam eder.
Kanıt pozitife dönerse `scripts/cm_replay.py --evidence` kapılarından geçip PAPER'a terfi eder; o zaman
`allocator` kanıt tavanı (25 $) devreye girer. Tek günlük ölçüm `MEASURED["superseded_1day"]` olarak
KAYITLI kalır — bulgunun nasıl değiştiği silinmez.

**Bulunan iki mantık hatası (canlı veri gösterdi, regresyon testi kilitledi):**
1. Ters FVG "aşılmamış" filtreli bölgeyle aranıyordu → "hiç aşılmamış" + "şimdi aşıldı" aynı anda doğru olamaz
   → 1815 pencerede **0 ateşleme**. Ham (filtresiz) bölge kullanılmalı.
2. Emir bloğu yalnız MSS BARINDA aranıyordu; kurulum "MSS oldu, SONRA bloğa dönüldü" der → 1815 pencerede 1
   ateşleme. MSS son 40 barda aranıp bloğu hatırlanacak şekilde düzeltildi.

**Kanıt kapıları aynen geçerli:** 10 kurulum `lifecycle`'da PAPER doğar (simülatörde çalışır, **canlı modda
çalışmaz** — LIMITED_LIVE kanıt ister), `allocator` durum makinesinde UNPROVEN → **25 $ kanıt tavanı**,
n ≥ 20 ve t ≥ 1 olmadan tam boyut yok. Kütüphane 250 → **260** kayıt (251-260 `video_sourced`).
**Uç:** `GET /api/video-sources` (künye + iddia + kanıt tipi + ALINMAYANLAR + ölçüm + canlı durum).
Testler: `tests/test_tur13.py` (17), tam paket 691.

**Bellek olayı (aynı gün):** süreç 1318 MB'de plato yaptı, `MEM_CAP_MB` 1400 → RED eşiği 1190 → **kaynak RED =
hiç giriş yok** (deploy sonrası 0 işlem). Düzeltme: `start_api.sh`'a `CRYPTOMIND_MEM_CAP_MB=1600`, pm2
`--max-memory-restart 1800M`; süreç 822 MB'a indi, GREEN. Ayrıca `governance()` sleeve başına `allocator.status()`
çağırıyordu (30 × tam istatistik) → bir kez hesaplanacak şekilde düzeltildi.

- **Susturma sebebi ayrıştırıldı:** gölgedeki (yaşam döngüsü) aday, panelde "rejim seçici kapattı" diye
  görünüyordu — aynı hata sınıfı: "kurulmadı" ≠ "bozuldu". Yeni kapı `YAŞAM_DÖNGÜSÜ` (`missed.GATE_TR`:
  "gölge aşaması — kanıt yok, emir verilmez, ölçüm sürer"). Kaçırılan-fırsat motoru bu adayları ayrı sayar,
  böylece "gölgede kaç fırsat kaçtı" ölçülebilir ve terfi kararı kanıta dayanır.
- **Doğrulama tuzağı:** `last_decisions` runner state'iyle diskten geri yüklenir; deploy sonrası ESKİ etiketli
  kararlar birkaç dakika görünür. Bir etiket/kapı düzeltmesini doğrularken kararın `ts` yaşına bakılmalı
  (yeni karar = yeni etiket), yoksa çalışan düzeltme "çalışmadı" sanılır.

## Tur 14 (2026-09-04) — "neden günlük %1 yok?" + ÇALIŞAN katmanın onarımı

**Günlük %1 imkânsızlık kanıtı (canlı: 102 işlem / 1,56 gün):** günde 65 işlem, ort boyut 57 $,
sermaye 1000 $. %1/gün = 10 $ → işlem başına 0,153 $ = boyutun %0,266'sı; maliyet %0,14 → gereken
brüt %0,406. Lehte hareketin (tepe = ULAŞILABİLİR ÜST SINIR) medyanı %0,349 → her işlemde hareketin
**%116'sını hiç kaybetmeden** almak gerekir. Üst sınırdan büyük = imkânsız.

**Scalping neden kaybediyor:** brüt −1,53 · komisyon 4,62 · net −6,13 $. Komisyon brütün %302'si.
Ciro sermayenin 3,8 katı/gün → yalnız komisyon günde %0,296. Ödeme oranı 0,54 → başabaş %64,8 kazanma
ister, bizde %49. Çıkışların tamamı taker (komisyonun %64'ü). Tepe yakalama %16,1.

**ÇALIŞAN KATMAN — trend takip (paper, gerçek fiyat + maliyet):** 48 günde 10.000 → **10.428,93
(+%4,29)**, günlük ort +%0,0917, **gerçekleşen Sharpe 2,65**, **azami düşüş %2,02**, sağlık yeşil.

**ARIZA (2 gün fark edilmedi):** Yahoo sunucu IP'sini 429'ladı → 12 kripto-dışı varlık düştü →
`fetch_live_daily` `except: pass` ile sessizce yuttu → `equity *= (1+NaN)` → özsermaye kalıcı NaN,
diske yazıldı, panel `equity: null`. **Düzeltme:** `mark()`/`step()`/`rebalance()` NaN-korumalı,
`load_state()` son sağlam güne KURTARIR, veri hattı üç katmanlı (yfinance → doğrudan Yahoo chart API
→ `runs/price_cache` 10 gün), eksik veri artık uyarı üretir. Testler `tests/test_trend_nan.py` (9),
tam paket **700**. Canlı doğrulandı: `/api/trend` equity 10428.93 · `/api/risk` Sharpe 2,68 · DD %2,0.

## Tur 15 (2026-09-05) — SERMAYE TAHSİSİ: risk ölçülmüş kazanana gider

Kullanıcı isteği: "sermayeyi trend katmanına kaydır". `agi_trader/auto/capital_allocator.py`:
  · `trend_metrics` / `scalp_metrics` → gerçekleşen Sharpe, getiri, azami düşüş (NaN günler atlanır)
  · **scalping Sharpe'ı GÜNLÜK ölçekte** hesaplanır (işlem başına değil; 65 işlem/gün yapan katman
    aksi hâlde yanıltıcı görünür)
  · `allocate`: w ∝ max(0, Sharpe − eşik); ölçülmemiş/eşik altı katman yalnız **ölçüm bütçesi %2**
    (kapatılmaz — kapatılan katman bir daha ölçülemez, rejim değişince fark edilmez)
  · **Kelly KULLANILMADI**: f* = μ/σ² trend için ~21× kaldıraç önerir; dağılım bilinmiyor, kaldıraç serbest değil.
  · `scalp_risk_budget`: koşucu limitleri PAYDAN türetilir (pay %2 iken 200 $ emir açmak payı anlamsız kılar)
  · Uç: `GET /api/allocation`

**Canlı sonuç:** trend pay **%95** (ölçülmüş Sharpe 2,65 · n 48 · getiri +%4,29) · scalp **%2**
(2 gün < 5 gün → ölçülmedi, ölçüm bütçesi) · **NAKİT %3**. Birleşik 11.422,78 / 11.000 → +%3,84.
Scalping limitleri 200 $ → **10 $**, günlük tavan 200 → **10**, risk %1 → **%0,1**, max_open 5 → 2;
kanıt boyutu `min(25, emir tavanı)` = 10 $ (aksi hâlde 25 $'lık kapı 10 $'lık tavanda hiç bağlamazdı).

**İKİ GERÇEK KUSUR:**
1. **Normalize hatası:** ağırlıklar 100'e normalize ediliyordu → iki katman da eşik altındayken
   %2 + %2 = %4 toplamı **%50/%50**'ye dönüşüyordu, yani "ikisi de kaybediyor" durumu "sermayeyi
   ikiye böl" oluyordu. Düzeltme: **NAKİT açık bir kalem**; normalize yalnız toplam > %100 ise.
2. **`risk_per_trade_pct` sync listesinde yoktu** → tahsis diskteki koşucuya hiç uygulanmıyordu
   (kodda 10 $, koşucuda 200 $ kaldı). Boyut hesabının (`_size`) ana çarpanı bu alan.
Testler: `tests/test_capital_allocation.py` (15) + `tests/test_trend_nan.py` (9); tam paket **715**.
