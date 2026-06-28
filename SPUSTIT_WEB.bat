@echo off
REM ============================================================
REM  Portfolio AI (Diplomovka) - setup + spustenie web aplikacie
REM  Prve spustenie: vytvori .venv a nainstaluje zavislosti.
REM  Dalsie spustenia: rovno zapne Streamlit web.
REM  Dvojklik => spusti vsetko automaticky.
REM ============================================================

chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

REM --- Ak prostredie este nie je pripravene, vytvor ho a nainstaluj ---
if not exist ".venv\Scripts\streamlit.exe" (
    echo.
    echo ================================================================
    echo   PRVE SPUSTENIE - pripravujem prostredie
    echo   Toto moze trvat 10-20 minut (stahuje sa PyTorch a kniznice).
    echo   Staci raz; dalsie spustenia uz budu okamzite.
    echo ================================================================
    echo.

    if not exist ".venv\Scripts\python.exe" (
        echo [1/2] Vytvaram virtualne prostredie .venv ...
        REM Skus Python 3.11 cez launcher, inak fallback na python
        py -3.11 -m venv .venv 2>nul
        if not exist ".venv\Scripts\python.exe" python -m venv .venv
        if not exist ".venv\Scripts\python.exe" (
            echo.
            echo [CHYBA] Nepodarilo sa vytvorit .venv.
            echo Projekt vyzaduje Python 3.11. Nainstaluj ho z python.org
            echo a spusti tento subor znova.
            echo.
            pause
            exit /b 1
        )
    )

    echo [2/2] Instalujem zavislosti (pip install -e .) ...
    .venv\Scripts\python.exe -m pip install --upgrade pip
    .venv\Scripts\python.exe -m pip install -e .
    if not exist ".venv\Scripts\streamlit.exe" (
        echo.
        echo [CHYBA] Instalacia zavislosti zlyhala. Skontroluj vystup vyssie.
        echo.
        pause
        exit /b 1
    )
    echo.
    echo Prostredie je pripravene.
    echo.
)

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
