@echo off
chcp 65001 >nul
title AGI TRADER // Neural Decision Grid
cd /d "%~dp0"
color 0B

echo.
echo  ============================================================
echo     A G I   T R A D E R   //   neural decision grid
echo  ============================================================
echo.

REM --- Python kontrol ---
python --version >nul 2>&1
if errorlevel 1 (
  echo  [HATA] Python bulunamadi.
  echo  https://www.python.org/downloads/ adresinden Python 3.10+ kurun
  echo  ve kurulumda "Add Python to PATH" secenegini isaretleyin.
  echo.
  echo [TEST] pause atlandi
  exit /b 1
)
for /f "delims=" %%v in ('python --version') do echo  [OK] %%v

echo.
echo  [1/2] Gerekli kutuphaneler kuruluyor (ilk seferde birkac dakika surebilir)...
echo [TEST] pip upgrade atlandi
echo [TEST] pip install atlandi
if errorlevel 1 (
  echo  [UYARI] Bazi opsiyonel paketler kurulamadi; cekirdek ile devam ediliyor.
)

echo.
echo  [2/2] Sunucu baslatiliyor. Arayuz hazir olunca Chrome OTOMATIK acilacak.
echo        ( http://127.0.0.1:8000 - port doluysa otomatik 8001/8002... secilir )
echo        Acilan adres asagida "Dashboard -^> http://..." satirinda yazar.
echo        Durdurmak icin bu pencerede Ctrl+C yapin veya pencereyi kapatin.
echo.

echo [TEST] serve.py atlandi

echo.
echo  Sunucu durduruldu.
echo [TEST] pause atlandi
