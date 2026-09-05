## Ne değişti

<!-- Bir cümle. -->

## Tür

- [ ] Strateji / kurulum katkısı (`contrib/`)
- [ ] Hata düzeltmesi
- [ ] Ölçüm / altyapı
- [ ] Dokümantasyon

---

## Strateji katkısıysa — ÖLÇÜM ZORUNLU

Ölçüm çıktısı olmayan strateji PR'ları incelenmez; incelenecek bir şey yoktur.
**Negatif sonuç da geçerli bir sonuçtur** — katkı gölgede birleşebilir.

```
python scripts/cm_verify_contribution.py --sleeve <ad> --days 7
```

<details>
<summary>Doğrulayıcı çıktısı (olduğu gibi yapıştırın)</summary>

```
buraya
```
</details>

- **Kaynak:**
- **İddia:**
- **İddianın kanıtı:** <!-- yoksa "YOK" yazın; bu reddettirmez -->
- **Ateşleme oranı:** <!-- %0–15 dışı reddedilir -->
- **Maliyet düşülmüş beklenti / t:**
- **Verdikt:** <!-- REDDEDİLDİ / GÖLGE / KANIT VAR -->

---

## Kontrol listesi

- [ ] `python -m pytest tests/ -q` yerelde geçiyor
- [ ] Davranış değiştiyse **regresyon testi** ekledim (bu depoda testsiz düzeltme kabul edilmez)
- [ ] `except: pass` eklemedim — hata yutulacaksa **kaydedilerek** yutulur
- [ ] Sır eklemedim (`.env`, anahtar, token, sunucu adresi)
- [ ] Risk/kaldıraç kapılarını gevşetmedim

## Ölçümü değiştiriyorsa

- [ ] Eski ölçümü **silmedim**; çürütülen ölçüm kayıtta kalır (`MEASURED_*`, `runs/`)
- [ ] Yeni ölçümün örneklem büyüklüğünü ve dönemini yazdım
