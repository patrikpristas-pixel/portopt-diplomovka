@echo off
REM ============================================================
REM  Spustac webovej aplikacie - Portfolio AI (Diplomovka)
REM  Dvojklik => spusti Streamlit, otvori prehliadac automaticky
REM ============================================================

chcp 65001 >nul
cd /d "%~dp0"

set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

echo.
echo ================================================================
echo   PORTFOLIO AI - Streamlit web
echo ================================================================
echo.
echo   Otvaram prehliadac na http://localhost:8501
echo   Pre vypnutie zatvor toto okno alebo stlac Ctrl+C
echo.
echo ================================================================
echo.

REM Skontroluj ze existuje venv
if not exist ".venv\Scripts\streamlit.exe" (
    echo [CHYBA] Nenajdeny .venv\Scripts\streamlit.exe
    echo Musis najprv nainstalovat venv:
    echo     python -m venv .venv
    echo     .venv\Scripts\pip install -e .
    pause
    exit /b 1
)

REM Spusti Streamlit (automaticky otvori prehliadac)
.venv\Scripts\streamlit.exe run app.py

echo.
echo Web bol zastaveny.
pause
