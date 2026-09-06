# CryptoMind — Kâr Koruma v2

**Tarih:** 2026-09-06 · **Kapsam:** çıkış motoru, yeniden giriş kapısı, maliyet/likidite dürüstlüğü

---

## 1. Önce teşhis: para tam olarak nerede kayboluyordu

Canlı kâğıt defterinden (MEXC, 200 kapanmış işlem, 3,1 gün, 1000 $ sermaye) ölçülenler:

| Ölçüm | Değer |
|---|---|
| Net P&L | **−6,95 $** |
| Brüt P&L | **−1,48 $** |
| Komisyon | **5,47 $** |
| Profit factor | 0,54 |
| Kazanma oranı | %51,0 |
| Ortalama kazanç / ortalama kayıp | +0,0796 $ / −0,1537 $ → **ödeme oranı 0,52** |
| Ortalama TEPE net | **+%0,316** |
| Ortalama YAKALANAN net | **+%0,10** |
| **Peak Capture Ratio** | **0,103** |

İki cümlelik teşhis:

1. **Giriş kenarı sıfır.** Brüt −1,48 $ (200 işlemde). Girişler para kazandırmıyor,
   kaybettirmiyor da. Kaybı yaratan komisyon (5,47 $).
2. **Ortalama işlem GERÇEKTEN kâra geçiyordu, kârın %90'ı geri veriliyordu.**
   Ortalama tepe +%0,316 net; gerçekleşen +%0,10. Yani asıl kayıp "yanlış giriş"
   değil, **kazanılmış kârın elde tutulamaması**.

### Çıkış sebebine göre kırılım (ölçülmüş)

| Sebep | n | net $ | ort. net % | tepe net % | PCR | ort. tutma |
|---|---:|---:|---:|---:|---:|---:|
| EARLY_ABORT | 21 | −7,14 | −0,705 | 0,000 | — | 13,0 dk |
| TIME_STOP | 20 | −3,37 | −0,280 | +0,183 | 0,086 | 51,4 dk |
| EDGE_DECAY | 30 | −1,43 | −0,271 | +0,048 | −0,298 | 32,2 dk |
| STOP | 20 | −1,93 | −0,285 | +0,183 | −0,606 | 26,3 dk |
| ROTATION | 14 | −0,21 | −0,204 | +0,185 | 0,374 | 22,0 dk |
| **BE_LOCK** | **38** | **+0,01** | **+0,012** | **+0,311** | **0,042** | **12,5 dk** |
| GIVEBACK | 31 | +2,89 | +0,212 | +0,529 | 0,391 | 25,1 dk |
| TRAIL | 23 | +4,23 | +0,706 | +1,016 | 0,652 | 28,2 dk |

**BE_LOCK satırı sistemin en pahalı kusurudur:** 38 işlem ortalama +%0,311 net tepeye
çıktı, +%0,0115'te kapandı. Tepenin **%4'ü** alındı.

---

## 2. Kök nedenler (üçü de kodda, üçü de ölçüldü)

### (1) "Başabaş kilidi" kârı SIFIRLIYORDU

`exit_engine.decide_exit` iki ayrı koruma taşıyordu ve **eşikleri farklıydı**:

| Koruma | Eşik (maliyet %0,2 için) | Ne yapıyordu |
|---|---|---|
| Başabaş kilidi | tepe net ≥ 1,5×maliyet = **%0,30** | stop → **başabaş** (net 0) |
| Yarı-tepe koruması | tepe net ≥ max(2×maliyet, %0,20, 0,5×ATR) = **%0,40** | net tepenin %50'sini korur |

**%0,30 ile %0,40 arasındaki bantta yalnız başabaş kilidi etkindi.** BE_LOCK kovasındaki
38 işlemin ortalama tepesi **%0,311** — tam bu bandın içinde. Kâr, tasarım gereği
tamamen geri veriliyordu.

**Ayrıca `live_runner` iki yerde stop'u doğrudan `pos.entry`'ye çekiyordu**
(`_partial_close` ve savunma modundaki `tighten_stops`). Giriş fiyatı net başabaş
DEĞİLDİR — gidiş-dönüş komisyonu kadar eksiktedir. Defterde bunun **8 kurbanı** var:
çıkış fiyatı girişe ±%0,02 içinde, hepsi tam komisyon kadar zarar (−0,20 $).

**v2:** kilit başabaşa değil **tepe × retain** oranına çekilir ve tepeyle birlikte
YUKARI yürür (ratchet). Tabanı net başabaştır; asla aşağı inmez.

### (2) Asgari tutma süresi YALNIZ kârı engelliyordu

`if age < p.min_hold_sec: return None` satırı hard stop ve erken iptalin ARDINDAN,
GIVEBACK/TRAIL'den ÖNCE duruyordu. Sonuç asimetri:

> İlk 15 dakikada **zarar kapanabiliyor, kâr korunamıyordu.**

Ölçüm: 15 dakikadan kısa 52 işlem → net **−4,01 $** (15 dk+ olanlar −2,94 $).
BE_LOCK işlemlerinin medyan tutma süresi **7,5 dakika** — yani koruma kapısı onlar için
hiç açılmamıştı. Bu 52 işlemin 30'u eşiğin üstüne çıkıp tepesinin yarısının altında
kapandı; tepede %50 korunsaydı **+1,79 $**.

**v2:** kâr KORUYUCU çıkışlar (GIVEBACK / TRAIL / merdiven) asgari tutmaya bakmaz.
TAKDİRE dayalı çıkışlar (sabit TP, EDGE_DECAY, TIME_STOP, MODEL_EXIT) eskisi gibi bekler.

### (3) Kısmi kâr alımı FİZİKSEL olarak imkânsızdı

Tek `partial_tp` vardı ve 200 işlemin yalnız **7'sinde** çalıştı. İki sebep:
- kısmi kapı da asgari tutmanın arkasındaydı (BE_LOCK medyanı 7,5 dk);
- **emir boyutu = borsa asgari emri.** Kâğıt modda `min_notional` 10 $, canlı kâğıt
  koşumunda `max_order_usdt` da 10 $. %50'lik bir dilim 5 $ eder — borsanın asgarisinin
  altında. Canlıda bu emir **reddedilir**; kâğıt broker sessizce geçiriyordu.

**v2:** T1…T6 kâr merdiveni + merdivenin **emir boyutuna uydurulması**. Dilimler
asgariyi geçmiyorsa basamaklar BİRLEŞTİRİLİR; hiçbiri sığmıyorsa merdiven kurulmaz
ve **sebebi loglanır** (sessizce çalışmayan özellik, olmayan özellikten kötüdür).

---

## 3. Yapılan değişiklikler

### 3.1 `strategies/exit_engine.py` — v2

| Yeni | Ne yapar |
|---|---|
| `lock_mode: "profit"` | Kilit = tepe × retain (v1: `"breakeven"` — geri alma yolu) |
| `PositionTrack.lock_level()` | Ratchet seviyesi; tabanı net başabaş, tepe ile yukarı yürür |
| `retain_eff()` | Her merdiven basamağından sonra retain +0,07 (koşucu daha sıkı korunur) |
| `protect_before_min_hold` | Koruyucu çıkışlar asgari tutmayı beklemez (varsayılan `True`) |
| `build_ladder()` | T1…T6: R katları + yapısal seviyeler; her basamak NET ≥ maliyet × 2 |
| `LADDER_TP` çıkışı | `partial: True` döner → pozisyon KAPANMAZ, ölçekli çıkış yapılır |

**Merdiven neden "altı tepe tahmini" değil:** kimse gelecekteki altı tepeyi bilemez.
Basamaklar stop mesafesinin (R) katlarıdır ve yakınlarında gerçek yapısal seviye
(swing high / direnç / planın hedefi) varsa ONA oturtulur. Her basamak yalnız net kârı
maliyetin katını aşarsa kurulur — **"borsa ücretinin üstünde miktarlarda oyna" kuralı
koda geçirildi.**

### 3.2 `auto/live_runner.py`

- `Position`: `ladder`, `levels_hit`, `lock_price`, `locked_net_pct`, `realized_net_pct`,
  `reentry_count` alanları; `track()`/`absorb()` bunları taşır.
- `_partial_close(pos, price, now, fraction, reason)`: **`stop = entry` HATASI DÜZELTİLDİ**
  → stop kâr kilidine çekilir. Dilim ve KALAN borsa asgarisini geçmiyorsa ölçekli çıkış
  yapılmaz (kapatılamayacak artık bırakılmaz).
- `_update_portfolio_mode` → `tighten_stops`: aynı `entry` hatası düzeltildi.
- `_fit_ladder_to_size()`: merdiveni emir boyutuna uydurur (yukarıda).
- `_fees(symbol)`: **parite BAZINDA** komisyon. Eskiden `symbols[0]`'ın oranı tüm
  pariteler için kullanılıyordu.
- `_tca()` + `_tca_report()`: **TCA'ya her dolum yazılır** (giriş, kısmi, çıkış).
  `execution/tca.py` yazılmıştı ama **hiçbir yerden çağrılmıyordu** — ölü koddu.
  Artık `full_state()["tca"]` gerçek kayma/maker payı/ödenen maliyeti gösterir.
- `_reentry_gate()` + `exit_state`: yeniden giriş kapısı (aşağıda).
- `_close()` kaydına `stop_pct`, `target_pct`, `cost_pct_roundtrip`, `atr_pct`,
  `levels_hit`, `ladder`, `locked_net_pct` eklendi — **çıkış A/B'si için gerekliydi ve
  defterde YOKTU** (200 işlemlik kayıttan çıkış motorunu yeniden oynatmak bu yüzden
  mümkün değildi).

### 3.3 `strategies/reentry.py` — YENİ

Aynı kapı iki işi birden yapar:

- **İYİ tarafı açar:** kâr merdiveninden/koruyucu çıkıştan sonra trend sürüyorsa
  harekete geri girilebilir (hareketin ikinci ve üçüncü bacağı).
- **KÖTÜ tarafı kapatır:** stop yiyen bir fikre aynı döngüde yeniden girilebiliyordu.
  Hiçbir soğuma yoktu. EARLY_ABORT + STOP kovaları 41 işlemde −9,07 $ üretti.

| Kural | Varsayılan |
|---|---|
| Beklenen salınım ≥ maliyet × k (**ÖLÇÜLMÜŞ** ulaşılabilir hedefle) | k = 2,0 |
| Zararla çıkıştan sonra aynı yöne dönüş yasağı | 30 dk |
| Kârla çıkıştan sonra soğuma | 5 dk |
| Ters yöne dönüş (kamçı önlemi) | 15 dk |
| Aynı harekette en çok yeniden giriş | 3 |

Kapı yalnız kısıtlar; aday hâlâ komitenin bütün kapılarından geçer. Engellediği her
aday `full_state()["reentry"]["recent_blocks"]` altında kaydedilir — **kapının haklı
olup olmadığı sonradan ölçülebilsin diye.**

### 3.4 Sessiz iyimserlikler kapatıldı

**`data/market_state.get_book()`** hata yolunda `spread_bps: 0.0` dönüyordu. Sıfır
spread, maliyet modelinde "bedava yürütme" demektir: **defter alınamadığında işlem
gerçekte olduğundan UCUZ görünüyor ve `max_spread_bps` kapısı otomatik geçiyordu.**
Yani veri kaybı, kapıyı sıkmak yerine GEVŞETİYORDU. Artık `ok: False` bayrağı taşınır;
bayat ama ölçülmüş defter varsa o kullanılır ve `stale` işaretlenir.

**`strategies/roles.role_cost_execution`**: `book_ok=False` → **VETO**. Derinlik alanı
yoksa boyut ×0,5, defter bayatsa ×0,5.

**`auto/decision_chain`**: derinlik ölçülmediğinde notional'ın **50 katı** derinlik
varsayılıp kapı geçiriliyordu. Artık veto (`veto_on_assumed_depth=True`; geriye dönüş
için `False` yapılabilir).

**`execution/broker.MIN_NOTIONAL_FALLBACK`**: `CRYPTOMIND_PAPER_MIN_NOTIONAL` ortam
değişkeniyle borsanın gerçek asgarisi verilebilir. **Varsayılan değişmedi** (10 $) —
bilinçli bir karar olmadan gevşemesin.

---

## 4. Ölçüm araçları (yeni)

| Betik | Ne ölçer |
|---|---|
| `scripts/cm_exit_bench.py` | v1/v2 çıkış motorlarını **aynı fiyat yollarında, aynı girişlerle** kıyaslar; eşleştirilmiş fark testi + %95 GA |
| `scripts/cm_exit_ab.py` | Tam replay A/B (komite dahil) — gerçekçi ama örneklemi zayıf |

`cm_exit_bench` neden ayrı: tam replay komiteyi de çalıştırır ve ağır bağlam replay'de
olmadığı için 2 günde ~10 işlem üretir — çıkış motorundaki farkı ölçmek için çok zayıf.
Tezgâh girişi TARAFSIZ (ya da `--entry dip/breakout` ile canlıya benzer bir aileden)
üretir ve **yalnız çıkış mantığını** değiştirir.

**Tezgâhın sınırı, raporda aynen kalmalı:** girişler sinyal değildir. Tezgâh
"sistem kârlı olur" demez; yalnız "yeni çıkış motoru kârın daha büyük bölümünü tutuyor
mu?" sorusuna cevap verir.

---

## 4.1 ÖLÇÜM SONUÇLARI

### A) Çıkış tezgâhı — 37.905 EŞLEŞTİRİLMİŞ giriş
40 parite × 14 gün × 1 dk bar · dip ailesi girişler · 5 stop mesafesi ×
2 yön · maliyet %0,07 · ufuk 60 dk · her ek kısmi çıkışa 1 bps ceza

| kol | beklenti% | kazanma | öd. oranı | PF | PCR |
|---|---:|---:|---:|---:|---:|
| v1 (mevcut) | −0,06758 | 0,479 | 0,711 | 0,654 | −4,405 |
| **v2 (kilit + merdiven)** | −0,06031 | 0,661 | 0,355 | 0,691 | −4,181 |
| **v2-lock (kilit, merdiven YOK)** | **−0,05878** | 0,660 | 0,359 | **0,699** | −4,182 |
| v2-ladder (yalnız merdiven) | −0,06911 | 0,487 | 0,680 | 0,646 | −4,407 |
| v2-nomin (yalnız asgari-tutma) | −0,07178 | 0,522 | 0,579 | 0,633 | −4,377 |

**Eşleştirilmiş fark (kol − v1)** — aynı giriş, aynı yol; piyasa gürültüsü elenir:

| kol | ort. fark (puan) | t | %95 GA | iyi / kötü / aynı | karar |
|---|---:|---:|---|---|---|
| v2 | +0,00727 | 7,14 | (+0,00528, +0,00927) | 17.815 / 4.460 / 15.630 | ✅ |
| **v2-lock** | **+0,00881** | **8,84** | **(+0,00685, +0,01076)** | 17.311 / 3.709 / 16.885 | ✅ **en iyi** |
| v2-ladder | −0,00153 | −4,79 | (−0,00215, −0,00090) | 2.277 / 1.497 / 34.131 | ❌ **ZARARLI** |
| v2-nomin | −0,00419 | −5,87 | (−0,00559, −0,00279) | 4.102 / 1.687 / 32.116 | ❌ **ZARARLI** |

Stop mesafesinin HER kademesinde v2 > v1 (tutarlılık kontrolü):

| stop | n | v1 | v2 |
|---|---:|---:|---:|
| %0,4 | 7.581 | −0,06993 | −0,06592 |
| %0,7 | 7.581 | −0,07489 | −0,06965 |
| %1,0 | 7.581 | −0,07296 | −0,06552 |
| %1,5 | 7.581 | −0,06467 | −0,05544 |
| %2,5 | 7.581 | −0,05547 | −0,04502 |

### B) Tam replay A/B — komite dahil, gerçek MEXC 1 dk verisi
5 parite × 2 gün · aynı geçmiş, aynı girişler, YALNIZ çıkış parametreleri farklı

| | v1 | v2 |
|---|---:|---:|
| işlem | 30 | 30 |
| net $ | −0,170 | **−0,071** |
| profit factor | 0,473 | **0,746** |
| kazanma oranı | %46,7 | **%60,0** |
| beklenti% | −0,0565 | **−0,0237** |
| **peak capture (PCR)** | **0,068** | **0,502** |

**PCR 0,068 → 0,502 — tepe yakalama 7,4 KAT.** Canlıda ölçülen 0,103'lük PCR'ın
doğrudan hedeflediği metrik budur. (n = 30, GA alt sınırı hâlâ negatif: bu DESTEKLEYİCİ
kanıttır, tek başına kanıt değil. Asıl kanıt yukarıdaki 37.905 örneklemli tezgâhtır.)

### C) Kararlar — ÖLÇÜME göre

| Özellik | Karar | Gerekçe |
|---|---|---|
| Kâr kilidi (`lock_mode="profit"`) | **AÇIK** | +0,0088 puan/işlem, t = 8,84, GA tamamen 0'ın üstünde |
| Koruma asgari tutmadan muaf | **AÇIK** (kilitle birlikte) | v2-lock, kilidin tek başına verdiğinden daha iyi |
| `stop = entry` düzeltmeleri | **AÇIK** | Mantık hatası; defterde 8 kurbanı var |
| Kâr merdiveni (T1…T6) | **KAPALI (opt-in)** | −0,0015 puan, t = −4,79 → **ölçüldü, REDDEDİLDİ** |
| Asgari tutma muafiyeti TEK BAŞINA | **KAPALI** | −0,0042 puan, t = −5,87 |

**Merdiven neden zarar veriyor:** ölçekli çıkış kazananı kırpar, kaybedeni etkilemez.
Ödeme oranı 0,711 → 0,355'e düşüyor; kazanma oranındaki artış bunu karşılamıyor.
Kod ve testler duruyor; `exit.ladder_enabled=true` ile açılabilir ve yeni ölçümle
yeniden değerlendirilebilir.

### D) İkinci ızgara — bağımsız veri penceresi, 37.185 eşleştirilmiş giriş

Aynı tezgâh, farklı (kayan) 14 günlük pencere. **Tekrarlanabilirlik kontrolü.**

| kol | ort. fark (puan) | t | %95 GA | yorum |
|---|---:|---:|---|---|
| **v2-lock** (retain 0,50 + koruma muafiyeti) | **+0,00685** | 6,53 | (+0,00480, +0,00891) | **CANLIYA GİDEN** |
| lock-only (retain 0,50, muafiyet yok) | +0,00570 | 5,62 | (+0,00371, +0,00768) | muafiyet +0,0012 katıyor |
| lock-r35 (retain 0,35) | +0,00437 | 5,19 | (+0,00272, +0,00603) | |
| lock-r65 (retain 0,65) | +0,00825 | 6,95 | (+0,00592, +0,01057) | aday |
| lock-r70 (retain 0,70) | +0,00861 | 6,92 | (+0,00617, +0,01105) | aday |
| ladder-late (2R + 3,5R, 2 dilim) | +0,00615 | 5,81 | (+0,00408, +0,00823) | v2-lock'un ALTINDA |
| ladder-1lvl (2,5R, tek %33 dilim) | +0,00653 | 6,19 | (+0,00446, +0,00859) | v2-lock'un ALTINDA |

**Üç bağımsız sonuç:**
1. **Kâr kilidi iki ayrı veri penceresinde de kazandı** (+0,0088 ve +0,0069; her ikisinde
   de GA tamamen 0'ın üstünde). Sonuç tek pencereye ait değil.
2. **Asgari-tutma muafiyeti kilitle BİRLİKTE katkı yapıyor** (+0,00570 → +0,00685), ama
   TEK BAŞINA zararlı (−0,00419). İkisi ayrılamaz.
3. **Merdivenin HER varyantı v2-lock'un altında kaldı** — 1R (zararlı), 2R+3,5R ve tek
   2,5R dilimi (pozitif ama kilit-yalnızdan düşük). Ölçekli çıkış her biçimde eksiltiyor.

**Retain oranı (kârın ne kadarı korunacak):** ızgara tekdüze artıyor
(0,35 → 0,50 → 0,65 → 0,70). Buna rağmen **canlıya 0,50 gidiyor**, çünkü:
- Fark dolar cinsinden küçük: 0,65'e geçiş ~65 işlem/gün × 0,0014 puan × ~30 $ ≈ **+0,03 $/gün**;
  kilidin kendisi ise ≈ **+0,13 $/gün**.
- `retain_fraction` canlıda strateji parametresi `giveback`'ten gelir ve challenger/ders
  motorunun terfi hattındadır. Kanıtla oraya girmesi, elle sabitlenmesinden doğrudur.
- Bu popülasyon dip ailesidir (sürüklenmesiz). Trendli bir ailede yüksek retain koşucuyu
  keser; 0,70'i tek pencereye bakarak sabitlemek tam da bu deponun kaçındığı hatadır.

---

## 5. Geri alma (rollback)

Tek satırlık config ile v1 davranışına dönülür:

```json
{"exit": {"lock_mode": "breakeven", "protect_before_min_hold": false, "ladder_enabled": false},
 "reentry": {"enabled": false},
 "chain": {"veto_on_assumed_depth": false}}
```

`tests/test_kar_koruma_v2.py::test_kilit_v1_moduna_geri_alinabilir` bu yolu doğrular.

---

## 6. Yapılmayanlar ve NEDEN (dürüstlük bölümü)

- **Günlük %10 hedefi konmadı.** Ortalama işlem tepesi net %0,32, gidiş-dönüş maliyet
  ~%0,07–0,20. Günde %10 için sermayenin tamamıyla, maliyet sonrası, günde onlarca
  mükemmel işlem gerekir. YouTube'daki tek gün örnekleri kaldıraç, seçilmiş dönem ve
  hayatta kalma yanlılığı taşır. Hedef, **bulunan kenarı olabildiğince yakalamak.**
- **Altı tepe TAHMİN EDİLMİYOR.** Olasılıksal "T1 %82, T2 %69…" tabloları üretmek
  kolaydır ama bu sayılar kalibre edilmedikçe sahte kesinliktir. Bunun yerine ölçekli
  çıkış basamakları kondu; olasılık iddiası yok.
- **`continuation_probability` hâlâ kalibre DEĞİL** ve bu yazılı. Bu yüzden tek başına
  çıkış üretmesine izin verilmiyor: yalnız DYNAMIC_PEAK modunda, pozisyon silahlandıktan
  (kâr zaten korunuyorken) sonra devrede.
- **Gösterge konsensüsü ağırlığı değiştirilmedi.** Geçmiş ölçümde 4h korelasyonu
  −0,164 çıkmıştı, ama ağırlığı ölçmeden değiştirmek tam da bu deponun kaçındığı hata
  olur. Ölçüm hattı (replay + DSR + PBO) mevcut; karar oraya bırakıldı.
- **Giriş kenarı bu turda ARTIRILMADI.** Brüt ≈ 0. Çıkış iyileştirmesi kayıp kârı
  kurtarır ama yoktan kenar yaratmaz. Bir sonraki turun konusu budur.
