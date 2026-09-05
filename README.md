# CryptoMind

Ölçüm-öncelikli kripto/çoklu-varlık işlem araştırma sistemi. **Tamamı kâğıt (paper) modundadır;
canlı emir göndermez.** Amacı para kazanmak değil, bir stratejinin kenarı olup olmadığını
*kanıtlayabilmektir* — ve çoğu zaman cevabın "hayır" olduğunu dürüstçe raporlamaktır.

## Neden bu depo bu şekilde yazıldı

Sistemin tasarım ilkesi tek cümlede: **ölçülmemiş bir şey canlıya çıkmaz.** Buradaki neredeyse
her kapı, bir varsayımın gerçek veride çürütülmesiyle eklendi:

- YouTube'dan çıkarılan 10 "kâr eden" kurulum kodlandı, 7 günlük gerçek veride ölçüldü
  (7.343 aday, t = −20,4) → **hiçbiri canlıya alınmadı**, gölge modunda ölçülmeye devam ediyor.
- 1 günlük ölçüm "New York seansı kazandırıyor" dedi (t = +3,4); 7 günlük ölçüm bunu sildi
  (t = −0,9). Tek günle karar verilseydi kaybettiren bir kurulum ailesi canlıya alınmış olacaktı.
- Scalping katmanı 116 işlemde net −%7,3 üretti; 11 farklı hedef eşiğinde karşı-olgusal
  çalıştırıldı, **hiçbirinde pozitife dönmedi**. Bu yüzden sermayesi %2 ölçüm bütçesine indirildi.
- Günlük %1 hedefinin imkânsızlığı ölçülerek gösterildi: mevcut ciroda her işlemde lehte
  hareketin %116'sını, hiç kaybetmeden almak gerekiyordu.

Kaybeden ölçümler silinmedi; `MEASURED_*` kayıtlarında ve `runs/` raporlarında duruyor.

## Katmanlar

| Katman | Durum | Ölçüm |
|---|---|---|
| **Trend takip** (günlük yeniden dengeleme, 17 varlık) | ÇALIŞIYOR | 48 gerçek gün, Sharpe 2,61, maks düşüş %2,08 |
| **Scalping** (1 dk, 12 rol komite, 31 sleeve) | ÖLÇÜM BÜTÇESİ | 116 işlem, PF 0,52 — kenar yok |
| **Video kurulumları** (10 adet) | GÖLGE | emir yok, ölçüm sürüyor |
| **Araştırma** (çift/carry/üçgen/piyasa yapıcı) | GÖLGE | emir yok |

### Risk rayları (kaldıraç)

Aynı trend sinyali, dört farklı risk iştahı — hepsi paralel, hepsi kâğıt. Kaldıraç Sharpe'ı
**artırmaz**; ortalamayı da sapmayı da aynı katsayıyla büyütür. Bu yüzden panel getiriyi hep
düşüşle yan yana gösterir.

| Ray | Hedef vol | Maruziyet tavanı | 48 günde ölçülen |
|---|---|---|---|
| `base` | %15 | 3,0× | %38,9/yıl · DD %2,2 |
| `aggressive` | %45 | 3,4× | %108/yıl · DD %7,1 |
| `extreme` | %75 | 5,7× | %192/yıl · DD %12,0 |
| `max` | %225 | 17,0× | %1/gün hedefi · DD ~%35 · **tasfiye mümkün** |

Her rayda düşüşe bağlı kaldıraç kısıcısı (yumuşak → tam, sert → yarı, kill → nakit + histerezis
kilidi) ve 1×'in üstündeki maruziyet için günlük finansman gideri vardır.

## Katkı — herkes strateji önerebilir

Bu depo katkıya açıktır. Bir alım-satım kurulumu, algoritma ya da strateji modeliniz varsa
ekleyebilirsiniz. Tek kural:

> **Katkı SHADOW doğar.** Sinyal üretir, **emir vermez** — paper modda bile.
> PAPER'a terfi bir kod incelemesi kararı değildir; **ölçüm** kararıdır.

```bash
cp agi_trader/agi_trader/strategies/contrib/SABLON.py    agi_trader/agi_trader/strategies/contrib/benim_kurulumum.py
# META künyesini ve fire() fonksiyonunu doldurun, sonra KENDİ ölçümünüzü çalıştırın:
cd agi_trader && python scripts/cm_verify_contribution.py --sleeve benim_kurulumum --days 7
```

Doğrulayıcı dört aşama uygular: **yükleme** (şema + imza + ad çakışması) → **statik denetim**
(ağ erişimi, ileriye bakış `shift(-n)`, tohumsuz rastgelelik) → **gerçek veride ateşleme oranı**
(%0'a yakın ya da %15 üstü reddedilir; ikisi de mantık hatasının ilk işaretidir) → **kenar**
(kurulumun kendi stop/hedefiyle ileri test, maliyet düşülmüş, t-istatistiği + bootstrap CI +
alt-dönem tutarlılığı).

Terfi kapısı `lifecycle.gates()`'tir: OOS beklenti > 0, CI alt sınırı > 0, 2× maliyette pozitif,
alt-dönem tutarlı, DSR > 0, PBO < 0,5, n ≥ 30. **Ölçülmemiş = geçilmedi.**

**Kanıtınız yoksa sorun değil** — `claim_evidence` alanına `"YOK"` yazın. Ölçümde kaybeden
katkı da birleşebilir; gölgede ölçülmeye devam eder ve rejim değişirse fark edilir.
Reddettiren şey kanıtsızlık değil, **abartılmış iddia** ve **öznel tanım**dır.

### Ölçülmüş açık kaynak stratejiler

**Yedi** açık kaynak strateji ([freqtrade/freqtrade-strategies](https://github.com/freqtrade/freqtrade-strategies),
GPL-3.0) katkı hattından geçirildi. Kurallar kaynaktan doğrulandı, **kod kopyalanmadı** — yalnız
kural, bu deponun kendi araçlarıyla bağımsız yazıldı. Her biri **60 gün · Binance · 1 dk**
verisinde, **iki ayrı parite grubunda** ölçüldü:

- **Büyük:** BTC, ETH, SOL, DOGE, AVAX — ort. %2,00 günlük oynaklık, saatlik $10,8M hacim
- **Küçük/oynak:** BONK, ORDI, PYTH, ARB, PEPE — ort. %3,88 oynaklık, saatlik $176K hacim
  (60 günlük geçmişi tam 16 aday arasından oynaklığa göre seçildi)

| Kurulum | Aile | Büyük pariteler | Küçük/oynak pariteler |
|---|---|---|---|
| BbandRsi | BB + RSI ortalamaya dönüş | −%0,138 · t −19,71 · **GÖLGE** | −%0,067 · t −2,65 · GÖLGE |
| Strategy001 | Heikin-Ashi + EMA kesişimi | −%0,193 · t −5,43 · GÖLGE | −%0,210 · t −4,34 · GÖLGE |
| HLHB | RSI/EMA kesişimi + ADX | −%0,095 · t −2,23 · GÖLGE | −%0,220 · t −3,90 · GÖLGE |
| ADXMomentum | ADX + DI± + momentum | oran %17,7 → REDDEDİLDİ | −%0,148 · t −13,08 · GÖLGE |
| Supertrend (üçlü) | ATR trend bantları | oran %20,6 → REDDEDİLDİ | oran %15,3 → REDDEDİLDİ |
| ClucMay72018 | derin BB sapması | n 6 → ÖLÇÜLEMEDİ | n 6 → ÖLÇÜLEMEDİ |
| UniversalMACD | dar hyperopt bandı | n 3 → ÖLÇÜLEMEDİ | n 15 → ÖLÇÜLEMEDİ |

**Yedisinden hiçbiri, hiçbir grupta kenar kanıtlayamadı.** Ölçülebilenlerin hepsi negatif;
ikisi seçicilik kapısında elendi (piyasanın beşte birini "giriş" sayıyorlar); ikisi de
n < 30 ile ölçülemedi.

Üç ayrı sebeple başarısız oluyorlar ve bu ayrım önemli:

1. **Ölçüldü, kenar yok** (BbandRsi, Strategy001, HLHB, ADXMomentum) — hepsi 1 saatlik ya da
   5 dakikalık barlar için yazılmış; 1 dakikada kesişim ve aşırılık sinyalleri gürültüye
   dönüşüyor. HLHB'nin zararı küçük paritelerde iki katına çıkıyor (−%0,095 → −%0,220).
2. **Seçici değil** (Supertrend, ADXMomentum-büyük) — koşul o kadar sık sağlanıyor ki
   "kurulum" piyasanın kendisi oluyor. Kenar ölçülmeden kapıda eleniyorlar.
3. **Ölçülemeyecek kadar nadir** (ClucMay, UniversalMACD) — UniversalMACD'nin hyperopt ile
   bulunmuş 0,0024 genişliğindeki bandı büyük paritelerde **54.000 barda bir** oluşuyor.
   Buradaki bulgu kâr/zarar değil **taşınabilirlik**: optimize edildiği veriden çıkınca
   neredeyse hiç tetiklenmeyen bir parametre bandı, bir piyasa mekanizmasını değil o verinin
   gürültüsünü tarif ediyordur.

Maliyet varsayımı (%0,14) küçük paritelerin ince defterleri için iyimserdir — bu yüzden
**negatif sonuçlar sağlam**; pozitif bir sonuç çıksaydı gerçekçi maliyetle yeniden ölçülürdü.

**Etkin örneklem:** doğrulayıcı her `--step` barda pencere açıp `time_stop_min` boyunca ileri
test eder; ardışık ateşlemeler ileri pencereyi paylaşırsa aynı ticaret defalarca sayılır ve
`|t|` şişer. İstatistik ve kapılar bu yüzden **örtüşmeyen** alt kümede hesaplanır.

**Nadir kurulum uyarısı:** ~10.000 barda bir ateşleyen bir kurulumu adım örneklemesi yapısal
olarak kaçırır. ClucMay ve UniversalMACD bu yüzden tam taramayla ölçüldü.

Ayrıntı: [CONTRIBUTING.md](CONTRIBUTING.md) · Öneri için
[strateji şablonu](../../issues/new?template=strateji-onerisi.yml)

## Kurulum

```bash
cd agi_trader
python -m venv venv && venv/bin/pip install -r requirements.txt
cp .env.example .env          # kendi API anahtarlarınızı girin
python cryptomind_serve.py --host 127.0.0.1 --port 8210
```

Panel:

```bash
cd terminal && npm install && npm run dev
```

Trend katmanı (günlük, cron'a uygun):

```bash
python trend_daemon.py --tracks base,aggressive,extreme,max
```

Testler:

```bash
cd agi_trader && python -m pytest tests/ -q      # 785 test
```

## Güvenlik notları

- **`agi_trader/server/app.py` internete AÇILAMAZ.** Kimlik doğrulaması olmayan
  `POST /api/credentials` (dosyaya yazar), `/api/auto/start`, `/api/learn` uçları içerir.
  Halka açık panel ayrı bir dosyadır: `server/public_api.py` — yalnız GET, tehlikeli uçlar
  orada hiç *tanımlı değildir*.
- Canlı emir için üç bağımsız anahtar aynı anda açık olmalıdır (konfig + bayrak + ortam
  değişkeni). Varsayılan kapalıdır ve hiçbir tek ayar bunu açamaz.
- Para çekme izni olan borsa anahtarları her kapsamda reddedilir.
- `.env`, `runs/` ve tüm `*.db` dosyaları `.gitignore`dadır — anahtarlarınızı depoya koymayın.

## Yasal

Yatırım tavsiyesi değildir. Eğitim ve araştırma amaçlıdır. Kâğıt modunda üretilen sonuçlar
gerçek işlem sonuçları değildir ve gelecekteki performansı garanti etmez.
