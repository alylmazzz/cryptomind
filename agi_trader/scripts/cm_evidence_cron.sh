#!/bin/bash
# Saatlik kanıt raporu — panel ve operatör için. Ağ erişimi yok, işlem akışına dokunmaz.
# crontab:  7 * * * * /var/www/cryptomind/scripts/cm_evidence_cron.sh
set -e
cd /var/www/cryptomind
./venv/bin/python scripts/cm_evidence.py --runs runs --tag 0_mexc \
  --out runs/live/evidence_report.json > runs/live/evidence_report.txt 2>&1
# kanıt defteri tavanı (200k satır ≈ 8 yıl) — pratikte tetiklenmez, sınır yine de yazılı
./venv/bin/python - <<'PY'
import sys; sys.path.insert(0,'.')
from agi_trader.learn import evidence as EV
n = EV.dondur("runs", tag="0_mexc")
print(f"arşive taşınan: {n}" if n else "döndürme gerekmedi")
PY
