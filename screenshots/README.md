# Spustenie kódu v príkazovom riadku a reprodukcia výsledkov

Tento priečinok obsahuje screenshoty dokumentujúce spustenie kódu v príkazovom
riadku (cmd) a generovanie číselných výsledkov uvedených v diplomovej práci.

## Postup spustenia

1. `SPUSTIT_WEB.bat` — postaví virtuálne prostredie `.venv` a nainštaluje
   závislosti (`pip install -e .`), následne spustí Streamlit dashboard.
2. `SPUSTIT_REPRODUKCIU.bat` — spustí štatistické skripty a vypíše ich výstupy.

## Čo screenshoty ukazujú

## Čerstvé command-line screenshoty z 29. 6. 2026

Nasledujúce obrázky sú pomenované podľa skriptov a ukazujú priamo príkaz,
čas spustenia, výstup skriptu, `EXIT_CODE: 0` a trvanie výpočtu:

| Súbor | Skript | Dôkaz |
|---|---|---|
| `command_line_02_preprocess_2026-06-29.png` | `scripts/02_preprocess.py` | načítanie raw cien, zarovnanie 26 tickerov, uloženie `prices.parquet`, `log_returns.parquet`, `coverage.csv` |
| `command_line_03_compute_pbo_2026-06-29.png` | `scripts/03_compute_pbo.py` | PBO hodnoty 91,9 / 74,1 / 71,0 / 87,3 % a uloženie do `reports/pbo_results/` |
| `command_line_04_compute_dsr_2026-06-29.png` | `scripts/04_compute_dsr.py` | DSR search/holdout hodnoty pre všetky štyri portfóliá a uloženie do `reports/dsr_results/` |
| `command_line_05_compute_dm_2026-06-29.png` | `scripts/05_compute_dm.py` | Dieboldov-Mariano test, p-hodnoty a uloženie do `reports/dm_results/` |

**Inštalácia prostredia** — vytvorenie `.venv` a inštalácia knižníc
(numpy, pandas, torch, optuna, PyPortfolioOpt a ďalšie), zakončené
`Successfully installed ... portopt-0.1`.

**Reprodukcia štatistických výsledkov** (`SPUSTIT_REPRODUKCIU.bat`):

| Skript | Výstup | Hodnoty |
|---|---|---|
| `02_preprocess.py` | predspracovanie dát | 26 tickerov, 3174 dní, 0 chýbajúcich |
| `03_compute_pbo.py` | Probability of Backtest Overfit | 91,9 / 74,1 / 71,0 / 87,3 % |
| `04_compute_dsr.py` | Deflated Sharpe Ratio (search/holdout) | 99,9/77,0 · 98,8/59,0 · 98,7/73,6 · 99,3/67,3 |
| `05_compute_dm.py` | Diebold-Mariano test | dm_stat 2,543 / 2,424 / 2,448 / 2,664 / 3,667 |

Tieto čísla zodpovedajú hodnotám uvedeným v práci.

## Skripty, ktoré nie sú na screenshotoch (a prečo)

- `01_download_data.py` — sťahuje denné ceny z Yahoo Finance (`yfinance`).
  Vyžaduje sieťové pripojenie a výsledok môže závisieť od aktuálnej dostupnosti
  Yahoo Finance. Dátový snapshot použitý v práci je preto už priložený v `data/`
  a reprodukcia výsledkov začína deterministickým krokom `02_preprocess.py`.
- `auto_search_v2.py` — Optuna TPE tréning neurónovej siete (spolu 810 trialov
  naprieč portfóliami). Ide o viachodinový výpočet; jeho výstupy (trialy, NAV,
  váhy) sú uložené v `portfolios/` a skripty `03`–`05` z nich čítajú.
- `export_vix_results.py` — VIX ablačná štúdia. VIX vstup bol po ablácii
  odstránený z finálneho modelu (sekcia 4.6 práce), preto sa tento skript
  v aktuálnej verzii nereprodukuje; jeho výsledky sú uložené
  v `reports/vix_results/`.
