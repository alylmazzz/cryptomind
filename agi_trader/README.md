# 🏛️ AGI Trader — Çok Katmanlı, Çok Ajanlı, Açıklanabilir Kripto Karar Motoru

`prompt.txt` spesifikasyonundaki **"AGI-Level Kripto İşlem Motoru"** tasarımının
çalışan, modüler bir uygulamasıdır. Spec'te tanımlanan tüm uzman rolleri ayrı
birer modül/ajan olarak hayata geçirilmiştir.

> **Güvenlik:** Sistem **varsayılan olarak paper-trading** modunda çalışır ve
> gerçek emir göndermez. Hiçbir API anahtarı koda gömülü değildir. Canlı emir
> göndermek bilinçli yapılandırma + kod aktivasyonu gerektirir (aşağıya bakın).

---

## ⚡ Hızlı Başlangıç

```bash
cd agi_trader

# Çekirdek bağımlılıklar (zaten kuruluysa atlayın)
pip install numpy pandas

# Anında demo — internet/anahtar GEREKMEZ (sentetik veri)
python run.py --source synthetic --symbols BTC/USDT ETH/USDT

# Canlı public veri (ccxt kuruluysa otomatik aktifleşir)
pip install ccxt
python run.py --symbols BTC/USDT --timeframe 4h

# Sonucu JSON'a yaz / günlük özeti
python run.py --json runs/out.json
python run.py --summary
```

### 🖥️ Web Dashboard (neon cyberpunk tema)

**En kolay yol — tek tık:** `start.bat`'e çift tıkla. Tüm kütüphaneleri kurar,
sunucuyu başlatır ve arayüzü **Chrome'da** açar.

```bash
# veya elle:
pip install -r requirements.txt
python serve.py            # → http://127.0.0.1:8000
```

> **Port otomatik seçimi:** 8000 başka bir uygulama (ör. WSL/`wslrelay.exe`)
> tarafından kullanılıyorsa başlatıcı **otomatik olarak 8001/8002…** ilk boş porta
> geçer ve tarayıcıyı o adreste açar. Konsolda açılan gerçek adres
> `Dashboard -> http://127.0.0.1:<port>` satırında görünür. Yani `start.bat` her
> koşulda çalışır — "10048 / port kullanımda" hatası artık başlatmayı durdurmaz.

Arayüz: Orbitron/Share-Tech-Mono fontları, neon ızgara + tarama çizgileri,
glassmorphism kartlar, neon glow sekmeler. Sekmeler: **📊 ANALİZ · 🔭 TARAMA ·
🔀 ARBİTRAJ · 💥 LİKİDASYON · 🧪 BACKTEST · ⚙️ SİSTEM** (ortam durumu, katman
ağırlık grafiği, işlem günlüğü, ağırlık öğrenme).

**🔐 Kimlik bilgileri paneli (sol sütun):** Tüm API anahtarları (Borsa, Twitter,
Whale Alert/Etherscan, Telegram/Discord, Anthropic) panelden kutucuklara
yazılıp **💾 Kaydet** ile girilebilir. Anahtarlar yalnız yerel `.env` dosyasına
yazılır, maskeli gösterilir (sadece son 4 hane), boş bırakılanlar değişmez ve
kaydedince ilgili motor (canlı sosyal/on-chain/bildirim) anında devreye girer.
Hiçbiri zorunlu değildir — public veri anahtarsız çalışır. Her sağlayıcının
yanında **🔑 anahtar al →** tıklanır linki (ücretsiz kayıt sayfası) ve
**ÜCRETSİZ/ÜCRETLİ** rozeti gösterilir. Uçlar: `GET/POST /api/credentials`.

#### ➕ Yeni borsa/sağlayıcı eklemek (tek dosya, hata yapması zor)

Tüm sağlayıcılar **tek doğruluk kaynağında** tanımlıdır:
[`agi_trader/providers.py`](agi_trader/providers.py). Yeni bir ücretsiz-API borsa
aracı eklemek için yalnızca `PROVIDERS` listesine **bir satır** ekle:

```python
# Borsa (ccxt) — canlı veriye OTOMATİK katılır, panelde otomatik görünür:
_ex("bitstamp", "Bitstamp", "bitstamp", 0.7, "https://www.bitstamp.net/account/api-access/"),
# Pasaparola gerektiren borsa: passphrase=True ekle (OKX/KuCoin/Bitget gibi)
```

Bu tek ekleme otomatik olarak şunları besler: kimlik bilgileri paneli (UI),
`.env` anahtar listesi, canlı borsa veri katmanı (ccxt önceliği + pasaparola) ve
`.env.example`. Başka hiçbir dosyayı elle düzenlemek gerekmez.

**Doğrula:** `python -m agi_trader.providers` — katalog özetini yazar ve
yinelenen anahtar/id, eksik `ccxt_id` gibi tutarsızlıkları yakalar.
**`.env.example`'ı yenile:** `python -c "from agi_trader.providers import env_example; open('.env.example','w',encoding='utf-8').write(env_example())"`

#### 📈 Formasyon grafiği (her analiz kartında)

Her parite kartı, seçili zaman diliminin **mum grafiğini** çizer ve grafiğin
üstüne **tespit edilen TÜM formasyonları** görsel olarak bindirir:

- **Yön okları** — boğa formasyonu için ▲ (yeşil), ayı için ▼ (kırmızı), tam
  tamamlanma noktasında.
- **Sayısal oranlar küçük yazı olarak** — her formasyonun adı + kalite (k0.82) +
  birincil oranı (harmonikte Fib `AB/XA`, klasikte `tepe farkı %`, `omuz simetri %`)
  hem grafik üzerinde (çakışma önleyici yerleşim) hem de altındaki çip listesinde.
- **Formasyon şekli** — XABCD/omuz-baş-omuz/çift tepe noktalarını birleştiren çizgi.
- **SMC bölgeleri** — FVG (boğa/ayı yarı saydam) ve Order Block kutuları.
- **Seviyeler** — GİRİŞ / STOP / TP1-3 ve formasyon hedef/iptal çizgileri.

Veri ucu: `GET /api/chart?symbol=BTC/USDT&tf=4h&bars=110` (hafif — derin model
çalıştırmaz, anında gelir). Formasyon geometrisi `server/chart.py`'de üretilir;
tespit `analysis/patterns.py` (artık her formasyon `indices`+`ratios` taşır).

**Gerçek veri + kaynak şeffaflığı:** Grafik GERÇEK piyasa verisi çeker. Kaynak
zinciri (`data/exchange_manager.py` → `fetch_ohlcv_with_meta`):
**ccxt borsalar** (Binance/Bybit/OKX/Kraken — anahtarsız public) → **ücretsiz REST
alternatifleri** (`data/market_data.py`: Binance public REST · CryptoCompare ·
CoinGecko — catalog'daki ücretsiz anahtarları kullanır) → **sentetik** (son çare).
Her grafiğin üstünde **CANLI/SENTETİK rozeti** kaynağı + son mum zamanını gösterir
(ör. `CANLI · binance · 06-28 20:00`), böylece sonucun gerçek veriye mi yoksa
simülasyona mı dayandığı her zaman bellidir. _TradingView'in ücretsiz herkese-açık
grafik-verisi REST API'si yoktur; ücretsiz parçası "Lightweight Charts"
görüntüleyicisidir — bu yüzden veri için yukarıdaki ücretsiz API'ler kullanılır._

#### 📍 Sağ Sinyal Tablosu + Çoklu-Piyasa + Olay Öngörü (2026-06-29)

**📍 SİNYAL TABLOSU (sağ panel):** Arayüzün sağında, her parite için **sayısal**
eşik kartı: **Şu an · 🟢 AL eşiği · 🔴 SAT eşiği · ▲ MAKS · ▼ MİN** + giriş→TP1 ·
stop·R/R · al/sat baskı barı · 6-tier **sinyal sınıfı** (KESİN AL/ZAYIF AL/NÖTR/
ZAYIF SAT/KESİN SAT/ACİL ÇIKIŞ, renk-kodlu) · güven/momentum/volatilite/korelasyon.
AL/SAT eşikleri pivot+Fibonacci+swing seviyelerinden türetilir (`decision_engine
_buy_sell_thresholds`); sinyal alanları `signal_class/momentum_score/volatility/
correlation_badge` doldurulur.

**🌐 Çoklu-piyasa (normal borsa + kripto eş zamanlı):** Kripto pariteleri ccxt/
ücretsiz-REST'ten; **hisse/endeks/forex/emtia** (`AAPL`, `THYAO.IS` BIST, `^GSPC`
S&P500, `EURUSD=X`, `GC=F` altın) **yfinance**'ten (`data/market_data.fetch_yfinance`).
Sembolde `/` yoksa otomatik yfinance'e yönlenir — aynı pipeline (analiz/sinyal/
grafik/board/otonom) tüm varlık sınıflarında çalışır.

**📅 Olay öngörü + zamanlama:** `macro/events.py` — yaklaşan yüksek-etkili makro
olayların tahmini takvimi (CPI/NFP/FOMC/OPEX, kalan gün + etki + not) ve
**contagion**: piyasayı süren varlık (driver) + **sürücüye-göreli korelasyon**
matrisiyle takip edecekler + zamanlama (eşzamanlı/gecikmeli). Negatif korelasyon
beklenen yönü ters çevirir (altın SHORT → ters-korele varlık LONG). `GET /api/events`.

**🎲 Monte Carlo backtest:** Backtest işlem getirilerini 1000× bootstrap yeniden
örnekler → sıralamadan bağımsız sağlamlık: kâr olasılığı, beklenen getiri (medyan),
%90 güven aralığı, en kötü drawdown (p95) ve **SAĞLAM/ORTA/KIRILGAN** kararı. Tek
bir backtest sonucunun şansa mı yoksa gerçek edge'e mi dayandığını gösterir
(`backtest/engine.monte_carlo_backtest`; backtest sekmesinde gösterilir).

**🔧 Parametre optimizasyonu (grid search):** Stop/TP/eşik ızgarasını (4×4×3=48
kombinasyon) tarar, her birini **risk-ayarlı skorla** (getiri/drawdown × profit
factor, işlem-sayısı kapılı) puanlar, en iyiyi **Monte Carlo ile doğrular** ve
mevcut config ile kıyaslar. `GET /api/optimize`; backtest sekmesinde "🔧 PARAMETRE
OPTİMİZE ET" — örn. BTC/USDT 4h'de varsayılan %9.9 getiriyi %21.7'ye çıkaran
(ATR×1.5/TP-SL 3/eşik 0.5) config'i, daha düşük drawdown'la bulur. **"✅ UYGULA"**
butonu en iyi config'i (stop çarpanı + TP merdiveni) **çalışan risk motoruna anında
yazar** (`POST /api/optimize/apply`).

**🔁 Online / artımlı öğrenme (kapalı-döngü self-improvement):** `learn/online.py
OnlineLearner` — otonom motor bir işlemi KAPATTIĞINDA, o işlemi açan sinyalin
katman katkıları + gerçek sonuç (kâr/zarar) işlenir. Her katman için "kârlı yönü
doğru işaret etti mi" ağırlıklı biriktirilir; her N işlemde katman ağırlıkları
isabete göre yeniden hesaplanır, karar motoruna **canlı uygulanır** ve
`runs/learned_weights.json`'a yazılır (yeniden başlatmada yüklenir). Mevcut
`/api/learn` fiyat-proxy manuel öğrenmedir; bu modül GERÇEK işlem sonuçlarıyla
otomatik öğrenir. Komuta sekmesinde "🧠 Online Öğrenme" paneli: işlem sayısı,
katman isabet barları, en isabetli katman.

**🧠 HMM piyasa rejimi + dinamik pozisyon boyutlama:** `analysis/regime.py` —
Gizli Markov Modeli (hmmlearn) getiri serisine 3 durum uydurur → **TREND YUKARI /
TREND AŞAĞI / RANGE / VOLATİL** (hmmlearn yoksa ADX+BB sezgiseline düşer). Rejim,
otonom motorun **pozisyon büyüklüğünü ölçekler**: trend-uyumlu ×1.1 · range ×0.6 ·
volatil ×0.4 · **trende-karşı ×0.5** (`engine._size → regime.position_multiplier`).
Rejim, sağ sinyal tablosunda ve Komuta OVERALL grid'inde gösterilir.

#### 🤖 RL v2 — PPO Sıralı İşlem Ajanı (multi-asset) (2026-06-29)

Eski tek-bar fitted-Q yerine **gerçek sıralı RL**: `ai/rl_env.py TradingEnv`
(Gymnasium — ajan pozisyon tutar: flat/long/short, ödül = pozisyon × sonraki-bar
getirisi − işlem maliyeti) + `ai/rl_agent.py RLAgent` (**PPO**, stable-baselines3).
Model symbol+timeframe başına diske önbelleklenir (`runs/models/ppo_*.zip`, 24s
tazelik → her barda yeniden eğitmez; oturumda bir kez). Politika dağılımından
**kalibre p(up)** + aksiyon (long/short/flat) çıkarır ve **`ai_ensemble` katmanına
`rl_ppo(PPO)` backend'i** olarak girer (deep + XGB/LGBM + heuristic ile birlikte).
İlk eğitim ~15-20s/parite (sonra cache'ten ~0.6s). **Çoklu-varlık:** `POST
/api/rl/train` tüm pariteler üzerinde TEK paylaşılan PPO eğitir; `GET /api/rl` durum.

#### 🔌 Entegrasyonlar: Webhook · Sektör · Testnet · Bot (2026-06-29)

**📡 TradingView webhook:** `POST /api/webhook/tradingview` — strateji alarmından
gelen `{symbol, action: buy/sell/close, price, secret}` ile otonom motorda **kâğıt
pozisyon açar/kapatır** (equity'nin %5'i). Gizli anahtar doğrulaması
(`TRADINGVIEW_WEBHOOK_SECRET`). `engine.external_signal()`.

**🔄 Sektör rotasyonu:** `macro/sectors.py` — temsili sepeti 9 sektöre ayırır
(L1/L2/DeFi/Meme/AI/Borsa/Ödeme/Oracle/Gaming), 7g/30g momentum hesaplar, sıralar,
**para giren/çıkan** sektörleri + 🔥 ivmelenme işaretler. `GET /api/sectors`; Komuta
sekmesinde panel.

**🟢 Canlı emir (Binance Testnet):** `execution_engine` canlı stub'ı dolduruldu —
ccxt `set_sandbox_mode` ile **varsayılan TESTNET** (sahte para), `mode=live +
allow_live` çift-onay kapısı + emir başına `live_max_order_usdt` tavanı. Mainnet
yalnız `testnet=false` ile bilinçli. Anahtar yoksa zarif `live_error`.

**🤖 Bot komutları:** `bot/commands.py process_command` — `/durum /portfoy
/analiz <SEMBOL> /sektor /pozisyonlar /alarmlar`. `GET /api/command?cmd=` (genel) +
`POST /api/webhook/telegram` (Telegram bot — yanıtı sohbete gönderir).

#### 🎛️ Komuta Merkezi + Otonom İşlem Motoru (2026-06-29)

**Otonom AutoTrader** (`auto/engine.py` + `auto/portfolio.py`): arka-plan döngüsü
sürekli tüm pariteleri tarar, işlem adayı sinyallerde **risk + konviksiyon + maruziyet
tavanı** ile boyutlanmış **kâğıt (paper) pozisyon** açar, açık pozisyonları güncel
fiyatla yönetir (**kademeli TP çıkışı 1/3 + breakeven stop + stop/iptal**), PnL/
özsermaye/kazanma-oranı/drawdown izler, **kill-switch** uygular ve giriş/çıkışlarda
alarm üretir. **GÜVENLİK: yalnız paper** — canlı emir mevcut çift-onay kapısının
arkasında, motor bilinçli olarak canlı açmaz.

**🎛️ KOMUTA sekmesi** — tek ekranda **OVERALL** görünüm: her paritenin anlık
sinyali (yön/güven/al-sat/R-R/formasyon/birleşim/durum), özsermaye eğrisi, açık
pozisyonlar (canlı PnL), son işlemler, motor olayları ve başlat/durdur/kapat/sıfırla
kontrolleri. Uçlar: `GET /api/auto`, `POST /api/auto/{start,stop,reset,close_all}`.

#### 📐 Grafik üstü göstergeler + yeni veri katmanları (2026-06-29)

**Grafik üzerinde gözlemlenebilir göstergeler** (üstteki "📐 GÖSTERGELER" çubuğundan
aç/kapa): **Bollinger Bantları · Ichimoku Bulutu (tenkan/kijun + span A/B dolgu) ·
EMA 20/50/200 · Supertrend (yön renkli) · VWAP · Fibonacci retracement seviyeleri ·
Pivot (P/R1/S1/R2/S2) · Hacim Profili (VPVR + POC)**. Hepsi mevcut `indicators.py`'den
seri olarak türetilir (`server/chart.py` → `_overlays`), `/api/chart` payload'ında
`overlays`/`volume_profile`/`regime` olarak gelir, dashboard'da özel Chart.js
plugin'leriyle çizilir. Her grafikte **piyasa rejimi rozeti** (📈/📉/↔️/🌪️ — ADX+BB+vol).

**Yeni ücretsiz veri katmanları** (karar motoruna eklendi, kartta `layer_breakdown`'da
görünür): **Korku & Açgözlülük Endeksi** (`sentiment/fear_greed.py` — alternative.me,
ANAHTARSIZ, kontraryen sinyal) · **Haber sentiment** (`sentiment/news.py` —
CryptoPanic/NewsAPI ücretsiz key) · **Makro** (`extra_layers.py` — FRED API ile
gerçek FED faizi + CPI enflasyon → risk-on/off). Header'da F&G / HABER / MAKRO
durum rozetleri. Yeni katalog sağlayıcıları: FRED, NewsAPI, LunarCrush.

**🎯 Formasyon birleşimi (confluence):** Farklı formasyonlar AYNI fiyat noktasını
işaret ediyorsa grafikte **"🎯 N FORMASYON BİRLEŞİMİ"** bandıyla belirtilir, ilgili
formasyon etiketlerine 🎯 eklenir ve pattern güveni artar (aynı yön = daha güçlü).
`analysis/patterns.py → find_confluence`; payload `confluence`.

**🧠 XGBoost + LightGBM ensemble:** `ai/ensemble.py` artık sklearn (GBM/HistGB/RF/
ExtraTrees/Logistic) **+ XGBoost + LightGBM** topluluğu çalıştırır (kuruluysa otomatik)
ve **özellik önemi** (feature importance) raporlar — `ai_ensemble` katmanı kartta görünür.

**🔔 Alarm sistemi (`notify/alarms.py`):** Her analiz turunda **formasyon birleşimi ·
yeni formasyon · seviye kırılımı (stop/TP) · işlem adayı** alarmları üretir; Telegram/
Discord anahtarı varsa anlık iletir. **Manuel fiyat alarmı** da eklenir (fiyat seviyeyi
geçince tetiklenir). Sistem sekmesinde panel + uçlar: `GET/POST/DELETE /api/alarms`.

**🔗 Korelasyon sekmesi:** pariteler arası getiri korelasyon ısı-matrisi +
yoğunlaşma (çeşitlendirme riski) uyarısı (`GET /api/correlation`).
**📤 Dışa aktarım** (Sistem sekmesi): sinyaller JSON / işlem günlüğü CSV
(`GET /api/export?what=signals|journal&fmt=json|csv`).

Panel sekmeleri: **📊 Analiz** (canlı kartlar: yön, güven, alış/satış %,
formasyon grafiği + göstergeler, rejim, sonraki maks/min tahmini, katman kırılımı + canlı fiyat akışı),
**🔭 Tarama + Sosyal** (çoklu-parite tarama + Twitter sosyal ısı),
**🔀 Arbitraj** (borsalar arası spread), **💥 Likidasyon/CVD** (likidasyon
**ısı haritası** + order flow), **🧪 Backtest** (walk-forward + equity curve),
ve **🧠 Ağırlık Öğren** (self-improvement). Tüm motorları tek ekranda toplar.

### 🔭 Çoklu-Parite Tarama + Twitter Sosyal Entegrasyon (`scan/scanner.py`)
Likit pariteleri (hacim filtreli, stablecoin'ler hariç) toplu tarar; her parite
için teknik + SMC + formasyon + alış/satış baskısı + (opsiyonel) whale akışı
skorunu hesaplar. **Twitter takip süreci tek seferde çalışır**
(`scan_social_heat`): kritik 201 hesabın tweet'lerinden coin başına sosyal ısı
(mention sayısı + nedensel yön) çıkarılır ve her paritenin temel coin'ine
eşlenerek tarama skoruna katılır. `TWITTER_BEARER_TOKEN` yoksa sosyal kısım
zarifçe 0 olur, grafik taraması çalışmaya devam eder. Uç: `/api/scan`.

### 💥 Likidasyon Isı Haritası (`onchain/liquidation.py` + panel)
Mevcut fiyat ve yaygın kaldıraç seviyeleri (5x–100x) ile long/short likidasyon
bölgeleri hesaplanır; panelde **fiyat-merdiveni ısı haritası** olarak çizilir
(long 🔴 altta, short 🟢 üstte; renk yoğunluğu = bölge yoğunluğu) + CVD/order flow.

Çekirdek **yalnızca numpy + pandas** ile uçtan uca çalışır. `ccxt`,
`scikit-learn`, `tweepy`, `transformers` **opsiyoneldir** — kuruluysa sistem
otomatik olarak güçlenir, kurulu değilse zarif (graceful) bir şekilde alternatife
geçer ve çalışmaya devam eder.

---

## 🧠 Spec rolleri → Modül eşlemesi

prompt.txt'deki her "zorunlu rol" bir modüle karşılık gelir:

| Spec rolü | Modül |
|---|---|
| Kantitatif Ekonomist / Quant | `decision/decision_engine.py`, `risk/risk_engine.py` |
| Derin Öğrenme / AI Araştırmacısı | `ai/deep_models.py` (**gerçek PyTorch Transformer + LSTM + RL**) + `ai/ensemble.py` |
| Yazılım Mühendisi / Sistem Mimarı | `agents/orchestrator.py`, `data/exchange_manager.py` |
| Veri Mühendisi | `data/exchange_manager.py`, `data/synthetic.py` |
| Blockchain / On-chain Analist | `onchain/flow_engine.py` (**gerçek whale/shark akışı + funding + OI**) |
| NLP / Sentiment Mühendisi | `sentiment/twitter_intelligence.py`, `accounts.py` (201 hesap), `causal.py` (nedensel motor) |
| Risk Yönetim Uzmanı | `risk/risk_engine.py` (Kelly, VaR, CVaR, Monte Carlo) |
| Teknik Analiz & Formasyon Uzmanı | `analysis/indicators.py` (127 gösterge), `analysis/patterns.py`, `analysis/smc.py` |
| Multi-Agent Sistem Mimarisi | `agents/orchestrator.py` |
| Backtest / Strateji Doğrulama | `journal/trade_journal.py` (self-improvement temeli) |
| Frontend / Dashboard | `report.py` (açıklanabilir metin raporu; UI'ya bağlanabilir) |
| Algoritmik Trading / Execution | `execution/execution_engine.py` (paper + kill-switch) |

---

## 🔬 Karar Akışı (pipeline)

Her parite için, her katman bir **uzman ajan** gibi `LayerVote` üretir
(skor −1..+1, kendi güveni, gerekçeler):

```
Veri (çok zaman dilimli OHLCV)
   │
   ├─ technical        → 127 indikatör (EMA/RSI/MACD/ADX/Ichimoku/Supertrend/...)
   ├─ pattern          → Harmonik (Gartley/Bat/Butterfly/Crab/Cypher/Shark) + klasik
   ├─ smc              → BOS/CHOCH/FVG/Order Block + HH-HL yapısı
   ├─ multi_timeframe  → 15m…1M confluence (yüksek TF daha ağır)
   ├─ ai_ensemble      → çok modelli yön olasılığı p(up)
   ├─ sentiment        → kritik Twitter hesapları (ağırlıklı, manipülasyon filtreli)
   ├─ onchain          → akış proxy (OBV/hacim); API ile gerçek veriye genişler
   └─ macro            → FED/CPI/ETF (kaynak bağlıysa)
   │
   ▼
DecisionEngine  → dinamik ağırlıklı birleşik skor + hizalanma + güven
   │   (spec kuralı: güven < %90 VEYA tek sinyal ise → İŞLEM YOK)
   ▼
RiskEngine      → Kelly position size, ATR stop, TP1/2/3, VaR/CVaR, Monte Carlo
   ▼
ExecutionEngine → paper-trade (varsayılan) + kill-switch
   ▼
TradeJournal    → JSONL kayıt (öğrenme/backtest temeli)
   ▼
report.py       → AÇIKLANABİLİR rapor: her kararın hangi katmandan, hangi
                  gerekçeyle, ne ağırlıkla geldiğini gösterir
```

### Sinyal çıktısı (spec'teki tüm alanlar + ek)
LONG/SHORT, Entry, Stop-Loss, TP1/TP2/TP3, Risk/Reward, başarı olasılığı,
beklenen getiri/kayıp, iptal seviyesi, alternatif senaryo, **Confidence Score**
ve katman bazlı **gerekçe raporu**.

Ek olarak (`analysis/forecast.py`):
- **Alış/Satış baskısı %** — katman skoru + hacim akışı + mum gövdesinden
  "🟢 alıcı %X vs 🔴 satıcı %Y" oranı.
- **Sonraki periyot maks/min tahmini** — bir sonraki mumun beklenen
  MAKS/MİN/kapanış değeri + %68 ve %95 güven aralıkları (volatilite-temelli
  beklenti; kesin tahmin değildir).

---

## 🐋 Gerçek On-chain / Whale & Shark Motoru (`onchain/flow_engine.py`)

Whale ve shark hareketlerini GERÇEK, ücretsiz kaynaklardan işler (anahtar
gerekmez):
- **Büyük işlem akışı** (ccxt `fetch_trades`, 1000 işlem): shark ($50K+),
  whale ($250K+), mega-whale ($1M+) tier sınıflandırması + whale CVD (büyük
  taker alım vs satım → net akış).
- **Funding rate** — aşırı funding → long/short sıkışma riski.
- **Open Interest** — OI + fiyat ilişkisi → trend gücü/zayıflığı.
- **Order book imbalance** + büyük likidite duvarları.
- **BTC mempool** (mempool.space) → ağ aktivitesi.
- **Opsiyonel** (anahtar varsa): Whale Alert + Etherscan ile gerçek
  zincir-üstü cüzdan→borsa transferleri.

## 🐦 Twitter İstihbarat + Nedensel Etki Motoru

- `sentiment/accounts.py` — **201 küratörlü kritik hesap** (146 kripto/exchange
  + 55 siyasi/makro: Fed, SEC, Hazine, Beyaz Saray, ECB...). Ağırlık + kategori
  + etkilenen varlık. Yapı 400+200'e genişletilebilir.
- `sentiment/causal.py` — **nedensel motor**: bir tweet'in NEDEN artış/azalışa
  yol açacağını belirler. 21 olay türü (ETF onayı/reddi, hack, listeleme,
  delist, faiz artışı/indirimi, regülasyon, token unlock, depeg, iflas, whale
  birikimi/dağıtımı...) → yön + büyüklük + düz-dil gerekçe.
  Canlı tweet çekimi `TWITTER_BEARER_TOKEN` ile aktifleşir.

## 🤖 Gerçek Derin Öğrenme (`ai/deep_models.py`)

PyTorch ile **gerçek** modeller, çekilen OHLCV üzerinde self-supervised eğitilir
ve ağırlıkları `runs/models/` altına önbelleklenir:
- **TransformerForecaster** — `nn.TransformerEncoder` dizi → p(up)
- **LSTMForecaster** — LSTM dizi → p(up)
- **DQNAgent** — short/flat/long pozisyon politikası (fitted-Q)

Üçü + sklearn + heuristik topluluk olarak `ai_ensemble` katmanında birleşir.
Not: ham fiyat tahmininde doğrulama isabeti ~%52–55 civarındadır — bu piyasanın
doğası gereği gerçekçidir; "her zaman doğru tahmin" mümkün değildir.

---

## ⚙️ Yapılandırma

`config.yaml` (PyYAML kuruluysa) veya koddaki güvenli varsayılanlar
(`agi_trader/config.py`). API anahtarları `.env` üzerinden okunur
(`.env.example` dosyasını `.env` olarak kopyalayın).

Önemli ayarlar:
- `decision.min_confidence` (vars. `0.90`) — spec gereği işlem eşiği
- `decision.layer_weights` — katman ağırlıkları (dinamik normalize edilir)
- `risk.*` — portföy değeri, risk yüzdeleri, ATR çarpanı, TP R-katları
- `execution.mode` / `execution.allow_live` — canlı emir kilidi

---

## 🛡️ Canlı işlem ve güvenlik

Canlı emir göndermek **bilinçli** olarak devre dışıdır. Etkinleştirmek için:
1. `config.yaml` → `execution.mode: live` **ve** `execution.allow_live: true`
2. `.env` içine ilgili borsanın `*_API_KEY` / `*_SECRET` değerleri
3. `execution/execution_engine.py` içindeki `live_disabled` STUB bloğunu kendi
   borsanızın emir çağrısıyla doldurun ve **önce küçük miktarla test edin**.

Kill-switch: portföy drawdown'ı `execution.kill_switch_drawdown` eşiğini aşarsa
tüm yeni işlemler otomatik durur.

> Kripto türev işlemleri yüksek risklidir. Bu yazılım bir **karar-destek**
> aracıdır, yatırım tavsiyesi değildir. Sorumluluk kullanıcıya aittir.

---

## 🚀 Üretime doğru genişletme

Bu depo, spec'teki kurumsal mimarinin **çalışan çekirdeğidir**. Spec'te anılan
ileri bileşenler şu uçlardan genişletilebilir:
- **WebSocket akışı / HFT**: `data/exchange_manager.py` (`stream_supported`) →
  ccxt.pro veya `websockets`
- **Gerçek on-chain**: `agents/extra_layers.py` → Glassnode/CryptoQuant/Nansen/Arkham API
- **Transformer/LSTM/TFT/RL**: `ai/ensemble.py` → PyTorch modelleri (ONNX inference)
- **Kafka/Redis/TimescaleDB/K8s**: `orchestrator` etrafına mikroservis sarmalama
- **Dashboard**: `report.py` JSON çıktısı → React/Next.js + TradingView charts

---

## 📁 Proje yapısı

```
agi_trader/
├── run.py                         # CLI
├── config.yaml / .env.example
├── requirements.txt
└── agi_trader/
    ├── config.py                  # yapılandırma + secret yönetimi
    ├── core/models.py             # dataclass'lar & enum'lar
    ├── data/                      # exchange_manager + synthetic
    ├── analysis/                  # indicators(127) + patterns + smc + multi_timeframe
    ├── ai/ensemble.py             # çok modelli yön topluluğu
    ├── sentiment/                 # twitter_intelligence + accounts
    ├── risk/risk_engine.py        # Kelly / VaR / Monte Carlo
    ├── decision/decision_engine.py# ağırlıklı, ≥%90 güvenli karar
    ├── execution/                 # paper-trade + kill-switch
    ├── journal/                   # JSONL işlem günlüğü
    ├── onchain/                   # whale/shark akışı (flow_engine) + likidasyon/CVD
    ├── backtest/                  # event-driven backtest + walk-forward
    ├── notify/                    # Telegram / Discord bildirim
    ├── learn/                     # otomatik ağırlık öğrenme (self-improvement)
    ├── server/                    # FastAPI dashboard + canlı akış (SSE)
    ├── agents/                    # orchestrator + extra_layers
    └── report.py                  # açıklanabilir rapor
```

## 🧩 Eklenen ileri özellikler (1–8)

| # | Özellik | Modül | Durum |
|---|---|---|---|
| — | **Web Dashboard** | `server/` (`serve.py`) | ✅ canlı |
| 1 | Canlı fiyat akışı (SSE polling) | `server/stream.py` | ✅ |
| 2 | Likidasyon haritası + CVD/Order Flow | `onchain/liquidation.py` | ✅ |
| 3 | Backtest + walk-forward + equity curve | `backtest/engine.py` | ✅ |
| 4 | Bildirim (Telegram/Discord) | `notify/notifier.py` | ✅ (anahtar gated) |
| 6 | Canlı Twitter akışı | `sentiment/twitter_intelligence.py` | ✅ (bearer gated) |
| 7 | Otomatik ağırlık öğrenme | `learn/weight_optimizer.py` | ✅ |
| 8 | Çoklu-borsa füzyon + arbitraj | `data/arbitrage.py` | ✅ |

`/api/*` uçları: `analyze`, `env`, `stream`, `arbitrage`, `liquidations`,
`backtest`, `learn`. Dashboard hepsini görsel olarak kullanır.
