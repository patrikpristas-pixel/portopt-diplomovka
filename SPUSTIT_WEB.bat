@echo off
REM ============================================================
REM  Portfolio AI (Diplomovka) - setup + spustenie web aplikacie
REM  Prve spustenie: vytvori .venv a nainstaluje zavislosti.
REM  Dalsie spustenia: rovno zapne Streamlit web.
REM ============================================================

chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

if exist ".venv\Scripts\streamlit.exe" goto run

echo.
echo ================================================================
echo   PRVE SPUSTENIE - pripravujem prostredie
echo   Toto moze trvat 10-20 minut, stahuje sa PyTorch a kniznice.
echo   Staci raz; dalsie spustenia uz budu okamzite.
echo ================================================================
echo.

if exist ".venv\Scripts\python.exe" goto install

echo [1/2] Vytvaram virtualne prostredie .venv ...
py -3.11 -m venv .venv 2>nul
if not exist ".venv\Scripts\python.exe" python -m venv .venv
if exist ".venv\Scripts\python.exe" goto install

echo.
echo [CHYBA] Nepodarilo sa vytvorit virtualne prostredie.
echo Projekt vyzaduje Python 3.11 - nainstaluj ho z python.org a spusti znova.
echo.
pause
exit /b 1

:install
echo [2/2] Instalujem zavislosti, prikaz pip install -e . ...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e .
if exist ".venv\Scripts\streamlit.exe" goto run

echo.
echo [CHYBA] Instalacia zavislosti zlyhala. Skontroluj vystup vyssie.
echo.
pause
exit /b 1

:run
echo ================================================================
echo   PORTFOLIO AI - Streamlit web
echo   Otvaram prehliadac na http://localhost:8501
echo   Pre vypnutie zatvor toto okno alebo stlac Ctrl+C
echo ================================================================
echo.
.venv\Scripts\streamlit.exe run app.py

echo.
echo Web bol zastaveny.
pause
