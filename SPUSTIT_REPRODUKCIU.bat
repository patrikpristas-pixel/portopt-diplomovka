@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

if not exist ".venv\Scripts\python.exe" (
  echo.
  echo [CHYBA] Chyba virtualne prostredie .venv
  echo Najprv spusti SPUSTIT_WEB.bat, ktory vytvori .venv a nainstaluje zavislosti.
  echo.
  pause
  exit /b 1
)

echo ================================================================
echo   REPRODUKCIA STATISTICKYCH VYSLEDKOV DIPLOMOVEJ PRACE
echo   Kazdy skript cita ulozene data a generuje cisla uvedene v praci.
echo ================================================================

echo.
echo ----------------------------------------------------------------
echo  [1/4] predspracovanie dat
echo ----------------------------------------------------------------
echo C:\portopt_diplomovka^> python scripts\02_preprocess.py
.venv\Scripts\python.exe scripts\02_preprocess.py

echo.
echo ----------------------------------------------------------------
echo  [2/4] Probability of Backtest Overfit (PBO)
echo ----------------------------------------------------------------
echo C:\portopt_diplomovka^> python scripts\03_compute_pbo.py
.venv\Scripts\python.exe scripts\03_compute_pbo.py

echo.
echo ----------------------------------------------------------------
echo  [3/4] Deflated Sharpe Ratio (DSR)
echo ----------------------------------------------------------------
echo C:\portopt_diplomovka^> python scripts\04_compute_dsr.py
.venv\Scripts\python.exe scripts\04_compute_dsr.py

echo.
echo ----------------------------------------------------------------
echo  [4/4] Diebold-Mariano test (DM)
echo ----------------------------------------------------------------
echo C:\portopt_diplomovka^> python scripts\05_compute_dm.py
.venv\Scripts\python.exe scripts\05_compute_dm.py

echo.
echo ================================================================
echo   HOTOVO. Vsetky cisla vyssie zodpovedaju hodnotam v praci.
echo ================================================================
echo.
pause
