#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# CryptoMind — KÂR KORUMA v2 kurulumu (kâğıt sistem)
#
# Kod ZATEN sunucuda hazırlık alanına kopyalandı: /root/cm_v2_stage
# Bu betik yalnız: yedek al → dosyaları koy → sözdizimi/import doğrula →
#                  pm2'yi yeniden başlat → doğrula
#
# Sunucuda çalıştır:   bash /root/cm_v2_stage/deploy.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
STAMP=$(date +%Y%m%d-%H%M%S)
SRC=/root/cm_v2_stage
DST=/var/www/cryptomind

echo "=== 1) YEDEK ==="
cp -a $DST/agi_trader $DST/agi_trader.bak-karkoruma-$STAMP
echo "yedek: $DST/agi_trader.bak-karkoruma-$STAMP"

echo "=== 2) DEĞİŞEN DOSYALAR ==="
rsync -ai --exclude '__pycache__' --exclude '*.pyc' $SRC/agi_trader/ $DST/agi_trader/ | head -40

echo "=== 3) SÖZDİZİMİ ==="
cd $DST && ./venv/bin/python -m compileall -q agi_trader >/dev/null && echo "compileall OK"

echo "=== 4) İMPORT KAPANIŞI + ETKİN VARSAYILANLAR ==="
cd $DST && ./venv/bin/python -c "
import sys; sys.path.insert(0,'.')
from agi_trader.auto import live_runner as LR
from agi_trader.strategies import exit_engine as XE, reentry as RE
from agi_trader.execution import tca as TCA
xp = XE.ExitParams().validated()
print('exit :', {k: getattr(xp,k) for k in ('lock_mode','ladder_enabled','protect_before_min_hold','retain_fraction','be_lock_cost_mult')})
print('reentry:', RE.ReentryParams().validated().to_dict())
print('import OK')
"

echo "=== 5) YENİDEN BAŞLAT ==="
pm2 restart cryptomind --update-env
sleep 15
pm2 list | grep -E 'cryptomind ' || true

echo "=== 6) DOĞRULAMA ==="
sleep 10
echo "--- hata günlüğü (son 15) ---"
tail -n 15 /root/.pm2/logs/cryptomind-error.log 2>/dev/null || true
echo "--- panel ucu ---"
curl -s -m 25 http://127.0.0.1:8210/api/simulator 2>/dev/null | head -c 600 || echo "(uc yanit vermedi)"
echo
echo "=== BİTTİ ==="
echo "GERİ ALMA (tek satır):"
echo "  rm -rf $DST/agi_trader && mv $DST/agi_trader.bak-karkoruma-$STAMP $DST/agi_trader && pm2 restart cryptomind"
echo
echo "DAVRANIŞI GERİ ALMA (kodu bırakıp v1 gibi davranması için) — panel/API config:"
echo '  exit.lock_mode="breakeven", exit.protect_before_min_hold=false, reentry.enabled=false, chain.veto_on_assumed_depth=false'
