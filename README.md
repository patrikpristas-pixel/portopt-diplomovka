# Optimalizácia portfólií finančných aktív pomocou umelej inteligencie

Zdrojový kód, dáta a výpočtové výstupy k diplomovej práci.
Autor: **Bc. Patrik Prístaš**, Fakulta managementu UK v Bratislave.

Repozitár slúži na **overenie a reprodukciu** všetkých číselných výsledkov
uvedených v práci. Obsahuje kompletný kód, zamknutý dátový snapshot a výstupy
experimentov. Natrénované modely (`.pt`, ~6,4 GB) nie sú priložené, pretože sú
deterministicky reprodukovateľné z kódu a pevného seedu.

---

## Štruktúra

```
src/portopt/            jadro projektu (inštalovateľný balík)
  data/                 sťahovanie a predspracovanie dát
  models/               softmax policy neurónová sieť (PyTorch)
  strategies/           baseliny: Markowitz, Black-Litterman, 1/N, Momentum, SPY
  backtest/             walk-forward engine, NAV, transakčné náklady, splity
  evaluation/           metriky + inferenčná štatistika (PSR, DSR, DM, PBO, bootstrap)
scripts/
  01_download_data.py   stiahnutie denných dát (yfinance)
  02_preprocess.py      log-výnosy, engineered príznaky
  auto_search_v2.py     Optuna TPE hľadanie hyperparametrov
  export_vix_results.py VIX ablačná štúdia
  03_compute_pbo.py     výpočet PBO (CSCV) — reprodukuje čísla v práci
data/                   dátový snapshot (parquet) — 2013-2022
portfolios/             výsledky per portfólio (trials, baseliny, váhy, výnosy)
reports/                výstupy: VIX ablation, PBO (summary + logits + histogram)
app.py                  Streamlit dashboard (interaktívne spustenie + grafy)
```

## Inštalácia

**Požiadavka: Python 3.11** (projekt je viazaný na `>=3.11,<3.12`).

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
pip install -e .
```

## Reprodukcia kľúčových výsledkov

```bash
# PBO (Probability of Backtest Overfitting) — deterministické, netrénuje nič
python scripts/03_compute_pbo.py
# -> reports/pbo_results/pbo_summary.csv + lambda histogram

# Interaktívny dashboard (tabuľky, NAV krivky, štatistika, per-trial detail)
streamlit run app.py
```

Na Windowse stačí dvojklik na `SPUSTIT_WEB.bat`. Pri prvom spustení sám
vytvorí virtuálne prostredie, nainštaluje závislosti a spustí dashboard;
ďalšie spustenia sú okamžité. Vyžaduje nainštalovaný Python 3.11.

## Metodológia (zhrnutie)

- **Model:** MLP softmax policy network, ktorá priamo generuje nezáporné
  portfóliové váhy so súčtom 1 (negeneruje predikciu výnosov).
- **Walk-forward:** 12-mesačné okná, hľadanie 2013-2019, **TRUE OOS holdout**
  2020-2022 (zamknutý pred Optunou).
- **Hyperparametre:** Optuna TPE (100-400 trialov podľa portfólia).
- **Inferenčná štatistika:** stationary bootstrap CI, PSR, DSR (korekcia na
  multiple testing), Diebold-Mariano test, **PBO cez CSCV**.
- **Baseliny:** Markowitz, Black-Litterman, 1/N, Momentum (126 dní), SPY.

## Hlavný empirický záver

Rozsiahlo ladená neurónová sieť v **skutočne nevidenom holdout období
(2020-2022) neprekonala jednoduché benchmarky** v žiadnom zo štyroch portfólií
a vykázala ~52-70 % pokles Sharpeho pomeru oproti hľadaciemu obdobiu.

PBO (71-92 %) a generalization gap nezávisle **potvrdzujú silné pretrénovanie
počas hľadania**. Kontribúciou práce nie je výkonný model, ale **metodologický
rámec, ktorý toto pretrénovanie korektne diagnostikoval**, čo je v súlade
s literatúrou o nedostatočnej výhode zložitých ML modelov oproti naivnej
diverzifikácii.

## Poznámka k použitým nástrojom

Výskumný kód v tomto repozitári (pipeline, model, baseliny, štatistika)
predstavuje vlastnú implementáciu. Dátový zdroj: Yahoo Finance (`yfinance`),
obdobie 2013-2022.
