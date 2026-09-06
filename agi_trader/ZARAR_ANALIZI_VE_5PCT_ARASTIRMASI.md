# CryptoMind — Zarar Kök-Neden Analizi ve "%5 Hareket" Araştırması

**Tarih:** 2026-09-06 · **Veri:** 218 canlı kâğıt işlem (09-03 → 09-06) + 41 parite × 14 gün × 1 dk bar

---

## BÖLÜM 1 — Zarar tam olarak nereden geliyor

### 1.1 Tek denklem

```
NET −7,185 $  =  BRÜT −1,553 $  −  KOMİSYON 5,632 $

brüt beklenti  = +0,0031 %/işlem      ← girişlerin ürettiği kenar
maliyet        =  0,0796 %/işlem      ← ödenen gidiş-dönüş
```

**Sistemin başabaşa gelmesi için brüt kenarın 26 KATINA çıkması gerekiyor.**
Kayıp bir "kötü çıkış" ya da "kötü stop" sorunu değil: **girişlerde kenar yok, kalan
her şey komisyon.**

Günlük kırılım bunu doğruluyor:

| gün | işlem | brüt | komisyon | net | komisyon / |net| |
|---|---:|---:|---:|---:|---:|
| 09-03 | 74 | +0,558 $ | 3,421 $ | −2,863 $ | **%119** |
| 09-04 | 39 | −2,011 $ | 1,336 $ | −3,347 $ | %40 |
| 09-05 | 41 | +0,017 $ | 0,335 $ | −0,318 $ | **%105** |
| 09-06 | 64 | −0,116 $ | 0,540 $ | −0,657 $ | **%82** |

Üç günde dört, brüt sıfırın etrafında ve **zararın tamamı komisyon.**

### 1.2 Tarihsel kalem: büyük pozisyonlar (kapatıldı)

EARLY_ABORT kovası tüm defterde −7,33 $ — yani net zararın tamamı. Ama ayrıştırınca:

| | n | net | ort | ort notional |
|---|---:|---:|---:|---:|
| notional > 30 $ | 10 | **−6,330 $** | −0,633 $ | 80,4 $ |
| notional ≤ 30 $ | 13 | −0,998 $ | −0,077 $ | 11,1 $ |

Büyük zararların **%86'sı 09-03/09-04'teki 80–110 $'lık pozisyonlardan.** Kanıt tavanı
(25 $) devreye girdikten sonra bu kalem pratikte kapandı. **Bugünün sorunu bu değil.**

### 1.3 Bugünün rejimi (son 100 işlem, hepsi 10 $)

```
NET −0,858 $ · BRÜT −0,018 $ · KOMİSYON 0,840 $
```

Brüt **tam olarak sıfır**. Zarar = komisyon. Günde ~65 işlem × 0,008 $ = **~0,5 $/gün
sızıntı** = 1000 $ hesapta aylık ~%1,5.

### 1.4 Sleeve düzeyi — ve devre kesici ZATEN çalışıyor

Son 120 işlem, sleeve başına (çıkış modu sleeve ile **tam karışık** olduğu için mod
ayrı bir suçlu olarak okunamaz — bunu ayırt etmek için aynı sleeve'i iki modda görmek
gerekirdi, öyle bir örnek yok):

| sleeve | çıkış modu | n | net | ort % | t |
|---|---|---:|---:|---:|---:|
| dip | PARTIAL_AND_RUN | 22 | −0,488 | −0,173 | −1,39 |
| dip_moderate | FIXED_TARGET | 14 | −0,389 | −0,283 | **−2,62** |
| failed_breakdown | PARTIAL_AND_RUN | 17 | −0,359 | −0,225 | **−2,25** |
| obi_momentum | FIXED_TARGET | 21 | −0,202 | −0,125 | −1,28 |
| donchian_breakout | DYNAMIC_PEAK | 12 | +0,199 | +0,166 | +1,47 |
| catalyst | DYNAMIC_PEAK | 13 | +0,538 | +0,414 | +1,42 |

**Devre kesici bunları görmüş ve duraklatmış:** 8 kesici olayı, süre 6→12→24 saat
katlanarak. Şu an `obi_momentum` (09-07 12:08'e kadar), `dip` ve `dip_moderate`
(09-07 00:5x'e kadar) DURAKLATILMIŞ durumda. Yani güvenlik mekanizması çalışıyor —
eklenecek yeni bir kesiciye gerek yok.

### 1.5 Stop mesafesi — dar stop pahalıya mal oluyor

| stop | n | ort % | t |
|---|---:|---:|---:|
| %0,6–1,0 | 5 | −0,399 | −2,13 |
| %1,0–1,5 | 52 | −0,100 | −1,87 |
| %1,5–2,5 | 12 | −0,313 | −1,72 |
| **%2,5+** | **148** | **−0,038** | −0,76 |

Geniş stoplu işlemler neredeyse başabaş; dar stoplular belirgin kaybettiriyor.

### 1.6 Maliyet kapısını SIKMAK işe yaramıyor (ölçüldü)

"Hedef/maliyet oranını yükseltirsek daha iyi işlem seçeriz" hipotezi **yanlış çıktı**:

| hedef/maliyet | n | ort % | t |
|---|---:|---:|---:|
| 0–10 | 6 | −0,019 | −0,13 |
| 10–20 | 54 | −0,000 | −0,00 |
| **20–30** | 72 | **−0,127** | **−2,20** |
| 30–45 | 44 | −0,080 | −1,04 |
| 45+ | 42 | −0,090 | −0,67 |

En iyi kova **en DÜŞÜK** orandaki kova. Kapıyı sıkmak daha iyi işlemleri eleyip
daha kötülerini bırakırdı. **Bu yol kapalı.**

---

## BÖLÜM 2 — "Günde %5 yapan pariteler" araştırması

### 2.1 Tasarım

- 41 parite × 14 gün × 1 dk bar (MEXC), 139.880 karar noktası (5 dk adım)
- Etiket: **ilk-geçiş** — girişten sonra 24 saat içinde **+%5 mi −%2 mi önce gelir?**
- Aynı bar içinde ikisi de görülürse KAYBEDEN sayılır (bar içi sıra gözlenemez)
- Tüm özellikler yalnız GEÇMİŞE bakar

### 2.2 Taban oran — kötü haber burada

```
+%5 hedefe ulaşan : %20,26
−%2 stop yiyen    : %55,55
zaman aşımı       : %24,19

BAŞABAŞ için gereken isabet: %28,6      ŞU ANKİ: %20,26
```

**Rastgele "al ve %5'te sat" kaybettirir.** Bir sinyalin işe yaraması için isabeti
%20'den %29'un üstüne, yani **1,4 kat** çıkarması gerekir.

### 2.3 Parite başına +%5 sıklığı (14 gün)

En sık: UNI %45,2 · PENDLE %31,0 · STRK %30,8 · JUP %29,8 · BONK %29,7 · NEAR %29,6
En seyrek: AVAX %3,9 · LTC %5,8 · ATOM %5,9 · ETH %7,3 · BTC %7,8

Beklendiği gibi: **oynaklık yüksekse %5 daha sık geliyor — ama stop da daha sık geliyor.**

### 2.4 Öncesinde ayırt edici sinyal var mı? — HAYIR

Kural, KENDİ döneminin taban oranına göre ölçüldü (kaldırma). İşe yaraması için
**her iki yarıda da > 1,4** olmalı:

| kural | 1. yarı | 2. yarı | karar |
|---|---:|---:|---|
| hacim patlaması (üst %10) | 0,97× | 0,98× | ❌ |
| sıkışma (alt %20) | 0,84× | 0,97× | ❌ |
| yüksek ATR (üst %20) | 1,28× | 1,02× | ❌ |
| ATR + hacim | 1,15× | 1,07× | ❌ |
| 24sa getiri alt %10 (dip) | 1,07× | 1,71× | ❌ |
| **24sa getiri üst %10** | **1,40×** | **0,61×** | ❌ **tam ters döndü** |
| aralık üst %20 | 1,01× | 0,78× | ❌ |
| saat 17–23 UTC | 1,44× | 1,06× | ❌ |
| **saat 17–23 + ATR üst %20** | **2,33×** | **0,91×** | ❌ **çöktü** |

**Hiçbiri tutarlı değil.** En parlak görünen ikisi (17-23+ATR: 2,33× ve 24sa üst %10:
1,40×) ikinci yarıda tamamen çöküyor. Bu, kenar değil **gürültü/rejim** imzasıdır.

Desil analizinde ayrışan tek şey oynaklık (D1 %9,7 → D10 %22,7) ve ATR (D1 %11,0 →
D10 %21,5) — ama bunlar **totolojik**: oynak varlık %5'i daha kolay görür, %2 stopu da
daha kolay görür. Net kazanç yok.

### 2.5 Seans analizi — profil TEKRARLAMIYOR

Tüm örneklemde 17–23 UTC en iyi (isabet %22,9–25,5), 11–12 UTC en kötü (%14,4–14,8)
görünüyor. Ama iki yarıya bölünce:

- 1. yarı: saat 20 kaldırma **1,87×**, saat 12 **0,53×** (belirgin profil)
- 2. yarı: saat 20 kaldırma **0,97×**, saat 12 **0,80×** (düz)
- **İki yarı arasında sıra korelasyonu (Spearman) = 0,316 → TUTARSIZ**

Ve **24 saatin HİÇBİRİ başabaş eşiğinin (%28,6) üstünde değil** (en iyi: saat 20, %25,5).

> Seans kapısı koymak, ölçülen veriye göre **gürültüye uyum sağlamak** olurdu.

### 2.6 Beklentinin işareti neyi izliyor? — PİYASA YÖNÜNÜ

Hedef +%5 sabit tutulup stop tarandı (maliyet düşülmüş):

| stop | başabaş isabet | 1. yarı beklenti | 2. yarı beklenti |
|---|---:|---:|---:|
| %1,0 | %16,7 | **−0,308%** | **+0,513%** |
| %2,0 | %28,6 | **−0,514%** | **+0,737%** |
| %5,0 | %50,0 | **−0,886%** | **+0,871%** |

**Her stop değerinde işaret aynı yönde değişiyor.** Sebep:

```
1. yarı (23 Ağu – 30 Ağu): ortalama parite getirisi  −1,03 %
2. yarı (30 Ağu – 6 Eyl) : ortalama parite getirisi +12,38 %
```

**"Al ve %5'te sat" bir kenar değil, uzun-beta bahsidir.** Piyasa yükselirken kazanır,
yatay/düşerken kaybeder. Kuralın kendisinin bilgi içeriği yok.

---

## BÖLÜM 3 — Sonuç ve ne yapılabilir

### Ölçümün REDDETTİĞİ fikirler (yapılmadı, sebebiyle)

| fikir | neden reddedildi |
|---|---|
| "+%5 hedefli al-sat botu" | taban %20,3 < başabaş %28,6; işaret piyasa yönünü izliyor |
| "yüksek getirili seansları seç" | saat profili iki yarı arasında tekrarlamıyor (ρ=0,32) |
| "%5 öncesi sinyal yakala" | 15 kuralın hiçbiri iki yarıda birden kaldırma > 1,4 vermedi |
| "maliyet kapısını sık" | en iyi kova EN DÜŞÜK oranlı kova; sıkmak iyileri elerdi |
| "yeni devre kesici ekle" | mevcut kesici zaten çalışıyor (8 olay, 3 sleeve duraklatılmış) |

### Ölçümün DESTEKLEDİĞİ tek aday

`catalyst` + `donchian_breakout` (ikisi de DYNAMIC_PEAK):

```
n = 26 · net +1,301 $ · ort +0,3051 %/işlem · t = +1,94 · kazanma %92,3
```

**t = 1,94 → henüz kanıt DEĞİL** (eşik 2,0). Ama defterdeki tek pozitif alt küme ve
kazanma oranı %92,3. Doğru hareket: örneklem 40–50'ye çıkana kadar bunları izlemek,
sonra kanıt kapısından geçerse boyut vermek. Mevcut `allocator` altyapısı bunu zaten
yapıyor — elle müdahale gerekmiyor, **sabır gerekiyor**.

### Asıl karar kullanıcıya ait

Ölçüm şunu söylüyor: **hiçbir sleeve'in kanıtlanmış kenarı yok, ve sistem kanıt
toplamak için sürekli işlem açtığından günde ~0,5 $ komisyon sızdırıyor.**

İki seçenek var, ikisi de savunulabilir:

1. **Kanıt toplamayı sürdür** — günde ~0,5 $ (%0,05) ödeyip catalyst/donchian
   örneklemini büyüt. ~3 haftada t > 2'ye ulaşılabilir.
2. **Kanıt bulunana kadar NAKİT** — yalnız gölge (emirsiz) modda ölç. Sızıntı sıfır,
   ama kanıt çok daha yavaş birikir (gölge işlemlerin dolum gerçekliği yoktur).

Bu bir ölçüm sorusu değil, **risk iştahı sorusudur** — bu yüzden koda tek taraflı
yazılmadı.
