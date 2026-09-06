#!/bin/bash
# Saatlik kanıt + EV kalibrasyonu. Ağ erişimi yok, işlem akışına dokunmaz.
# crontab:  7 * * * * /var/www/cryptomind/scripts/cm_evidence_cron.sh
set -e
cd /var/www/cryptomind
PY=./venv/bin/python

$PY scripts/cm_evidence.py --runs runs --tag 0_mexc \
  --out runs/live/evidence_report.json > runs/live/evidence_report.txt 2>&1

$PY scripts/cm_ev_calib.py --runs runs --tag 0_mexc \
  --out runs/live/ev_calib.json > runs/live/ev_calib.txt 2>&1

# ── HABER VERME: ölçüm hazır olduğu AN kalıcı bir sonuç dosyası yaz ────────────
# Uzun süren bir bekleyiciye (oturuma/sürece) bağlı kalmamak için: verdikt oluşunca
# tek seferlik `EV_KALIBRASYON_SONUC.md` üretilir ve bir daha üzerine yazılmaz.
# Böylece cevap, kimse beklemese de kendiliğinden ortaya çıkar ve kaybolmaz.
$PY - <<'PY'
import json, pathlib, time
p = pathlib.Path("runs/live/ev_calib.json")
son = pathlib.Path("runs/live/EV_KALIBRASYON_SONUC.md")
if not p.exists() or son.exists():
    raise SystemExit
d = json.loads(p.read_text(encoding="utf-8"))
if not d.get("hazir"):
    raise SystemExit
k = d.get("eslestirilmis_hata_farki") or {}
ev, eva = d.get("ev_pct") or {}, d.get("ev_achievable") or {}
son.write_text(f"""# EV KALİBRASYON SONUCU
Üretildi: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}
Örneklem: iki EV'si birden kayıtlı **{d.get('n_ikisi')}** işlem

## VERDİKT: {k.get('karar', '—')}
eşleştirilmiş |hata| farkı {k.get('ort_fark')} · t {k.get('t')} · %95 GA {k.get('ci95')}
eva daha az yanılan: {k.get('eva_daha_iyi_sayisi')}/{k.get('n')}

| | ev_pct (plan hedefi) | ev_achievable (ölçülmüş) |
|---|---|---|
| n | {ev.get('n')} | {eva.get('n')} |
| Pearson r | {ev.get('pearson')} GA{ev.get('pearson_ci95')} | {eva.get('pearson')} GA{eva.get('pearson_ci95')} |
| kalibrasyon eğimi | {ev.get('kalibrasyon_egimi')} | {eva.get('kalibrasyon_egimi')} |
| yanlılık | {ev.get('yanlilik')} | {eva.get('yanlilik')} |
| beşli tekdüze | {ev.get('tekduze')} | {eva.get('tekduze')} |
| bilgi var mı | {ev.get('bilgi_var')} | {eva.get('bilgi_var')} |
| seçim kazancı | {ev.get('secim_kazanci')} | {eva.get('secim_kazanci')} |

## SIRADAKİ KARAR
`_maybe_rotate` şu an `ticket["ev_pct"]` kullanıyor. Bu tablo hangisinin
kullanılacağını belirler. UYARI: eğim ≈ 0,2 gibi bir değer "şişik ama BİLGİLİ"
demektir — o durumda ev_pct atılmaz, ÖLÇEKLENİR.
""", encoding="utf-8")
print("EV_KALIBRASYON_SONUC.md yazildi")
PY

# kanıt defteri tavanı (200k satır ≈ 8 yıl) — pratikte tetiklenmez, sınır yine de yazılı
$PY - <<'PY'
import sys; sys.path.insert(0,'.')
from agi_trader.learn import evidence as EV
n = EV.dondur("runs", tag="0_mexc")
print(f"arşive taşınan: {n}" if n else "döndürme gerekmedi")
PY
