# CryptoMind — Hakem İncelemesi (2026-09-04)

**Dürüstlük notu:** Bu belge, dünyaca ünlü akademisyenler ya da ücretli bir hakem heyeti tarafından yazılmadı. Aşağıdaki değerlendirme, ilgili disiplinlerin yayımlanmış standartlarına (López de Prado *Advances in Financial ML*, Bailey–López de Prado DSR/PBO, Avellaneda–Stoikov, Almgren–Chriss, Kissell TCA, Basel model-risk yönergeleri) göre yapılmış, her bulgusu bu depodaki koda ve ölçümlere bağlanmış bir öz-inceleme raporudur. Gerçek dış hakemlik istenirse aşağıdaki maddeler doğrudan gönderilebilecek biçimde yazılmıştır.

## 1. Principal Quantitative Researcher
- **Bulgu:** Sleeve seçiminde PBO 0,67 (replay, 1 gün, 5 parite). En iyi sleeve'i örneklem-içi seçmek örneklem-dışı tutmuyor. **Aksiyon:** meta-tahsisçi Beta(2,2) büzülmesi korunuyor; sleeve ağırlığı [0,5·1,2] bandında; EV yarışması "en iyi" değil "pozitif ve dayanıklı" olanı seçiyor. Kabul: hiçbir sleeve QUALIFIED sayılmadı.
- **Bulgu:** Kazanma oranı ikincil metrik (madde 133). Panelde ana metrikler kâr faktörü, beklenti, PCR, fee drag. Kabul.
- **Eksik:** Alt-dönem/rejim başına performans (madde 87) yalnız yarı-yarı; ay/rejim kırılımı replay'e eklenmeli.

## 2. Market Microstructure Researcher
- **Bulgu:** Defter yalnız üst seviye (bid/ask + ±%2 derinlik); OBI çok seviyeli değil, kuyruk pozisyonu modellenmiyor. P(fill) sezgisel. **Risk:** maker dolum tahmini iyimser olabilir → maker-miss kayıpları missed motorunda ölçülüyor (MAX_CHASE/MAKER_DOLMADI kapıları). **Aksiyon önerisi:** L2 5/10/20 seviye OBI ve kuyruk-bazlı P(fill) (madde 33, 62) — WebSocket olmadan maliyetli; P1.
- **Bulgu:** CVD 100 işlemlik pencereyle hesaplanıyor; 5s/15s/30s değişimleri yok. Kabul edilebilir başlangıç.

## 3. Execution / OMS Engineer
- **Bulgu:** Emir durum makinesi eksik (UNKNOWN/RECONCILIATION_REQUIRED). Paper'da zararsız; canlı öncesi zorunlu (madde 66). **Aksiyon:** canlı kapılarından biri olarak eklenmeli.
- **Bulgu:** clientOrderId idempotent (sha1). Kabul.
- **Bulgu:** TCA modülü var ama koşucuya bağlı değil; fill sonrası ters seçim ölçümü yalnız MM gölgesinde. P1.

## 4. Portfolio & Risk Engineer
- **Bulgu:** Korelasyon bütçesi (|ρ|≥0,7 → ×0,5) ve sleeve tavanı var; portföy optimizasyonu (madde 57) yok. Küçük sermayede kabul edilebilir.
- **Bulgu:** Gün-içi tepe geri-verme portföy düzeyinde eklendi (19:22'de +3,99 $ → +0,0 $ gözlemi). Doğru yön: kazancı eriten yeni girişleri kısıyor, mevcut pozisyonların stoplarını başabaşa çekiyor.
- **Bulgu:** Devre kesici (t-stat/Wilson) sleeve düzeyinde; venue/strateji kill-switch (madde 69) kısmi.

## 5. ML / Validation Engineer
- **Bulgu:** DSR gerçek deneme kaydıyla deflate ediliyor (trials.jsonl). PBO CSCV kendi bölmesi (4 grup). Kabul.
- **Bulgu:** Kalibrasyon Brier ile ölçülüyor; n küçük (bugün 3 fişli işlem). Sonuç "bilgisiz/kötümser" dürüst.
- **Eksik:** Overlapping label etkin örneklem düzeltmesi (madde 80) replay'de yok; işlem sayısı n olarak kullanılıyor. P1.
- **Eksik:** Lookahead testi otomatik değil; ReplayExchange imleç tasarımı lookahead'i yapısal olarak engelliyor, test var.

## 6. SRE / Performance
- **Bulgu:** RSS 0,9–1,15 GB (tavan 1,4). Haber taraması ve venue kıyası bellek tepesi yaptı; kıyas kapatıldı, haber aralığı 20 dk. Döngü 12–19 sn (30 sn periyot). **Risk:** 48 parite ile döngü 20+ sn.
- **Eksik:** CPU/event-loop metrikleri; WebSocket yok (madde 6). P1.

## 7. Application Security
- **Bulgu:** Para çekme izinli anahtar reddi, kasa, GET-only panel, sırlar frontend'e gitmiyor. Kabul. **Eksik:** IP allowlist, anahtar rotasyonu (eski not), okuma/işlem anahtarı ayrımı kısmen var.

## 8. Model-Risk Validator
- **Karar:** Sistem **PAPER/SHADOW** aşamasında kalmalı. Canlı için: ≥30 işlem/sleeve OOS, DSR>0, PBO<0,5, kalibrasyon n≥50, OMS durum makinesi, mutabakat kanıtı. Bugün bunların hiçbiri sağlanmadı; bu bir kusur değil, doğru sınıflandırma.

## Öncelikli aksiyon listesi (bu turda yapılanlar ✓)
1. ✓ Sleeve devre kesici (kanıt: dip_moderate −1,16 $, obi_momentum −0,67 $).
2. ✓ Gün-içi tepe geri-verme (portföy).
3. ✓ Devam olasılığı + kalan EV + rotasyon (madde 49–51).
4. ✓ Swing sleeve (1–3 gün, 4h/1h).
5. ✓ Evren görünürlüğü + ölçülmüş ek pariteler.
6. ✓ Deneme kaydı + VALIDATION_REPORT + n_trials ile DSR.
7. ☐ OMS durum makinesi + TCA bağlantısı (canlı öncesi).
8. ☐ Çok seviyeli OBI / kuyruk P(fill) (WebSocket gerektirir).
9. ☐ Rejim/ay kırılımlı replay raporu + etkin örneklem düzeltmesi.
