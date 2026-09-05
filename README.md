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

Dört açık kaynak strateji ([freqtrade/freqtrade-strategies](https://github.com/freqtrade/freqtrade-strategies),
GPL-3.0) katkı hattından geçirildi. Kurallar kaynaktan doğrulandı, **kod kopyalanmadı** —
yalnız kural, bu deponun kendi araçlarıyla bağımsız yazıldı. Hepsi **aynı 7 günlük pencerede,
aynı 5 paritede, aynı maliyetle** ölçüldü.

| Kurulum | Ateşleme | Oran | Ort. net | t | Verdikt |
|---|---:|---:|---:|---:|---|
| BbandRsi | 91 | %0,95 | −%0,138 | −10,63 | **GÖLGE** |
| ADXMomentum | 1699 | %17,5 | — | — | REDDEDİLDİ |
| Supertrend (üçlü) | 1967 | %20,4 | — | — | REDDEDİLDİ |
| ClucMay72018 | 0 | %0,00 | — | — | REDDEDİLDİ |

**Hiçbiri kenar kanıtlayamadı.** Üçü ateşleme kapısında elendi — kenar ölçülmeden önce:
ADXMomentum ve Supertrend piyasanın beşte birini "giriş" sayacak kadar seçicisiz; ClucMay
hiç ateşlemedi. BbandRsi ölçüldü ve **t = −10,63** ile güvenilir biçimde negatif çıktı.

Bunlar "bu stratejiler kötü" demek değildir. Hepsi 1 saatlik ya da 5 dakikalık barlar için
yazılmış; burada 1 dakikalık barlarda ve bu maliyet yapısında ölçüldüler. Her dosyanın
başındaki **SAPMALAR** notu farkı açıkça yazar, `MEASURED` bloğu da sonucu kalıcı tutar —
çürütülen ölçüm bu depoda silinmez.

ClucMay'in sıfır ateşlemesi ayrıca ölçülerek açıklandı: ilk hipotez ("zaman dilimi değişti")
**çürütüldü** — `close < 0,985×bb_lower` koşulu 5 dakikalık barlarda da %0,000 sıklıkta.
Sebep ölçek: Bollinger yarım genişliği %0,14–0,52 iken koşul bandın %1,5 altını istiyor.

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
