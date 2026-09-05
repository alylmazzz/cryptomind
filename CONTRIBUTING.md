# Katkı Rehberi

Bu depo **herkesin strateji önerebileceği** bir yerdir. Tek bir kural vardır ve istisnası yoktur:

> **Katkı SHADOW doğar.** Sinyal üretir, **emir vermez** — paper modda bile.
> PAPER'a terfi bir kod incelemesi kararı **değildir**; ölçüm kararıdır.

Bu katılık kişisel değil, ölçülmüş bir derse dayanıyor. Bu depoda:

- 21 YouTube videosundan çıkarılan 10 "kâr eden" kurulum kodlandı. 7 günlük gerçek veride
  7.343 aday üzerinden **t = −20,4** çıktı. Hiçbiri canlıya alınmadı.
- Aynı kurulumlar *1 günlük* veride "New York seansı kazandırıyor" dedi (t = +3,4).
  7 günlük veri bunu sildi (t = −0,9). **Tek günle karar verilseydi kaybettiren bir kurulum
  ailesi canlıya alınmış olacaktı.**
- Scalping katmanı 116 işlemde net −%7,3 üretti. 11 farklı hedef eşiğinde karşı-olgusal
  çalıştırıldı; **hiçbirinde pozitife dönmedi.**

Yani: bir fikrin mantıklı, iyi yazılmış ve hatta zekice olması kenarı olduğunu göstermez.
Yalnız ölçüm gösterir. Sizin katkınız da, deponun kendi kurulumları da aynı kapıdan geçer.

---

## Hangi katkılar kabul edilir

| Kabul edilir | Kabul edilmez |
|---|---|
| Mekanik olarak tanımlı kurulum (her koşul koda çevrilebilir) | "Bariz destek", "güçlü momentum" gibi öznel ifadeler |
| Kaynağı yazılmış fikir (kitap / makale / video / kendi fikriniz) | Kaynağı gizlenen ya da formülü yayımlanmamış kapalı gösterge |
| Kanıtı **olmayan** ama dürüstçe "kanıt YOK" diyen öneri | Doğrulanamaz iddia ("%90 kazanma", "günlük %10") |
| Negatif çıkan ölçüm sonucu | Ölçüm sonucunu saklamak / seçerek raporlamak |
| Risk yönetimi, yürütme, ölçüm altyapısı iyileştirmeleri | Kaldıraç/risk kapılarını gevşeten değişiklikler |

**Çürütülen ölçüm de bilgidir ve bu depoda silinmez.** Kurulumunuz ölçümde kaybederse
katkınız yine birleşebilir — gölgede ölçülmeye devam eder ve rejim değişince fark edilir.

---

## Adım adım

### 1. Şablonu kopyalayın

```bash
cp agi_trader/agi_trader/strategies/contrib/SABLON.py \
   agi_trader/agi_trader/strategies/contrib/benim_kurulumum.py
```

`META` künyesini doldurun. **Beş alan zorunludur ve boş bırakılamaz:** `author`, `source`,
`claim`, `claim_evidence`, `mechanism`. Kanıtınız yoksa `claim_evidence` alanına `"YOK"`
yazın — bu katkınızı reddettirmez. Abartılmış iddia reddettirir.

### 2. `fire` fonksiyonunu yazın

```python
def fire(f, p, price, atr_abs):
    if not f.get("trend_up"):
        return None
    return {"direction": "LONG", "size": 0.6,
            "stop_hint": price - 1.2 * atr_abs,
            "target_hint": price + 2.4 * atr_abs,
            "note": "neden tetiklendi"}
```

`f`, komitenin özellik sözlüğüdür (92 alan; listesi `SABLON.py` içindedir).

**`f`'te olmayan bir gösterge gerekiyorsa** `fire`'a beşinci parametre olarak `df` ekleyin —
özellikleri üreten aynı bar çerçevesini alır ve göstergenizi kendiniz hesaplarsınız:

```python
def fire(f, p, price, atr_abs, df):
    adx = ...        # df["high"], df["low"], df["close"], df["volume"]
```

Bu, gerçek açık kaynak stratejilerin çoğu için zorunludur: DI±, MACD, SAR, Supertrend, mum
formasyonları `f`'te yoktur. Çerçeve, sistemin geri kalanının gördüğünün aynısıdır — yani
`df` fazladan bilgi VERMEZ, yalnız kendi göstergenizi hesaplamanıza izin verir.
**Yeni veri çekilmez**; ağ ve dosya erişimi yasaktır.

Yükleyici şunları zorlar: ağ/dosya erişimi yok, ileriye bakış yok (`shift(-n)`), tohumsuz
rastgelelik yok, global durum değişimi yok, spot'ta SHORT yok, `size` 0–1 arası.

### 3. Kendi ölçümünüzü çalıştırın

```bash
cd agi_trader
python scripts/cm_verify_contribution.py --sleeve benim_kurulumum --days 7
```

Doğrulayıcı dört aşamayı sırayla uygular:

1. **Yükleme** — META şeması, `fire` imzası, ad çakışması.
2. **Statik** — ölçümü geçersiz kılacak şeyler (ağ, ileriye bakış, rastgelelik).
3. **Ateşleme oranı** — gerçek 1 dk veride. **Sıfıra yakın ya da %15 üstü oran reddedilir**:
   ikisi de mantık hatasının ilk işaretidir. (Bu depoda ters-FVG kurulumu "hiç aşılmamış +
   şimdi aşıldı" çelişkisi yüzünden 1815 pencerede **0 kez** ateşledi; gevşetilince %9,5
   ateşleyip saf gürültü üretti.)
4. **Kenar** — kurulumun kendi stop/hedefiyle ileri test, gidiş-dönüş maliyet düşülmüş.
   Beklenti, t-istatistiği, bootstrap CI, alt-dönem tutarlılığı, 2× maliyette dayanıklılık.

**ETKİN ÖRNEKLEM (önemli):** doğrulayıcı her `--step` barda bir pencere açar ve ateşlerse
`time_stop_min` boyunca ileri test eder. Adım 5, ufuk 240 iken ardışık iki ateşleme ileri
pencerenin **%98'ini paylaşır** — yani aynı ticaret onlarca kez sayılır, nominal `n` şişer ve
`|t|` olduğundan büyük çıkar. Bu yüzden istatistik ve kapılar **örtüşmeyen** alt kümede
hesaplanır: bir işlem, aynı paritede bir öncekinin ufku bittikten sonra sayılır. Çıktıda
`nominal → ETKİN (şişme ×N)` satırı bunu gösterir.

**Veri penceresi:** MEXC 1 dakikalık geçmişi ~30 günle sınırlıdır (35 günde boş döner).
Daha uzun pencere için `--venue binance` (ya da kucoin/bybit) kullanın:

```bash
python scripts/cm_verify_contribution.py --sleeve hepsi --days 60 --venue binance --step 15
```

Çıktıyı **olduğu gibi** PR açıklamasına yapıştırın. Negatif çıktıysa da yapıştırın.

### 4. PR açın

`.github/pull_request_template.md` doldurulur. Ölçüm çıktısı olmayan strateji PR'ları
incelenmez — çünkü incelenecek bir şey yoktur.

---

## Terfi nasıl olur

`cm_verify_contribution.py` **terfi ettirmez**, kanıt üretir. Terfi kapısı
`agi_trader/strategies/lifecycle.py` içindeki `gates()`'tir ve şunların **hepsini** ister:

| Kapı | Eşik |
|---|---|
| OOS beklenti | > 0 |
| Bootstrap CI alt sınırı | > 0 |
| 2× maliyette beklenti | > 0 |
| Alt-dönem tutarlılığı | ilk yarı ve ikinci yarı ikisi de pozitif |
| DSR (deflated Sharpe) | > 0 |
| PBO (aşırı-uydurma olasılığı) | < 0,5 |
| İşlem sayısı | ≥ 30 |

DSR ve PBO `scripts/cm_replay.py --evidence` ile üretilir. **Ölçülmemiş = geçilmedi**;
eksik bir metrik "nötr" sayılmaz.

Kapıları geçen bir katkı PAPER'a terfi eder ve **25 $ kanıt tavanıyla**, küçük boyutla
işlem görmeye başlar. Boyut ancak kanıt biriktikçe büyür.

---

## Hata bildirimi ve diğer katkılar

Strateji dışı katkılar (hata düzeltmesi, ölçüm altyapısı, dokümantasyon, panel) normal PR
sürecinden geçer. Kural:

- Bir hata bildiriyorsanız **nasıl tekrarlanacağını** yazın.
- Bir davranışı değiştiriyorsanız **testle kilitleyin.** Bu depoda her düzeltme bir regresyon
  testiyle birlikte gelir; testsiz düzeltme aynı hatanın geri gelmesini engellemez.
- `except: pass` eklemeyin. Bu depoda sessiz yutma iki kez üretim arızasına yol açtı
  (biri 45 günlük paper kaydını NaN yaptı, diğeri bütün video kurulumlarını görünmez şekilde
  kapattı). Hata yutulacaksa **kaydedilerek** yutulur.

## Yerel kontrol

```bash
cd agi_trader
python -m pytest tests/ -q          # tam paket
python -c "import ast,glob;[ast.parse(open(f,encoding='utf-8').read()) for f in glob.glob('agi_trader/**/*.py',recursive=True)]"
```

CI aynılarını çalıştırır ve katkı paketinin yükleme hatalarını raporlar.

## Davranış

Ölçüme saygı gösterin, kişiye değil fikre itiraz edin. Bir ölçümü beğenmiyorsanız daha iyi
bir ölçüm önerin — bu depoda bir iddiayı çürütmenin yolu daha iyi veridir, daha yüksek sestir değil.
