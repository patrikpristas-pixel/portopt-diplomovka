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

set PY=.venv\Scripts\python.exe

echo ================================================================
echo   REPRODUKCIA VSETKYCH VYSLEDKOV DIPLOMOVEJ PRACE
echo   Skripty citaju ulozene Optuna trialy a generuju cisla z prace.
echo   Tento beh netrenuje 810 trialov nanovo.
echo ================================================================

echo.
echo ----------------------------------------------------------------
echo  [1/6] predspracovanie dat
echo ----------------------------------------------------------------
echo C:\portopt_diplomovka^> python scripts\02_preprocess.py
%PY% scripts\02_preprocess.py

echo.
echo ----------------------------------------------------------------
echo  [2/6] prepocet tabuliek a grafov z ulozenych trialov
echo ----------------------------------------------------------------
echo C:\portopt_diplomovka^> python scripts\06_generate_thesis_outputs.py
%PY% scripts\06_generate_thesis_outputs.py

echo.
echo ----------------------------------------------------------------
echo  [3/6] Probability of Backtest Overfit (PBO)
echo ----------------------------------------------------------------
echo C:\portopt_diplomovka^> python scripts\03_compute_pbo.py
%PY% scripts\03_compute_pbo.py

echo.
echo ----------------------------------------------------------------
echo  [4/6] Deflated Sharpe Ratio (DSR)
echo ----------------------------------------------------------------
echo C:\portopt_diplomovka^> python scripts\04_compute_dsr.py
%PY% scripts\04_compute_dsr.py

echo.
echo ----------------------------------------------------------------
echo  [5/6] Diebold-Mariano test (DM)
echo ----------------------------------------------------------------
echo C:\portopt_diplomovka^> python scripts\05_compute_dm.py
%PY% scripts\05_compute_dm.py

echo.
echo ----------------------------------------------------------------
echo  [6/6] VYPIS VSETKYCH TABULIEK VYSLEDKOV (Tab 3 az 12)
echo ----------------------------------------------------------------
echo C:\portopt_diplomovka^> python scripts\06_print_word_results.py
%PY% scripts\06_print_word_results.py

echo.
echo ================================================================
echo   HOTOVO. Vsetky cisla vyssie zodpovedaju tabulkam v praci.
echo ================================================================
echo.
pause
