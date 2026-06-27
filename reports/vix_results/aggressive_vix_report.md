# VIX vysledky pre diplomovu pracu

Portfolio: **aggressive**

Tento export je ulozeny mimo aplikacie, aby VIX nemusel ostat v aktivnej logike a nepredlzoval dalsie testovanie.

## Scenare
- VIX ON: scenar `03dba391`, pokusy 100, vyhry 91, najlepsi trial #55
- VIX OFF: scenar `891866cb`, pokusy 100, vyhry 93, najlepsi trial #71

## Najlepsi vysledok kazdej vetvy
### VIX ON
- Search NAV: 30,145 EUR
- Search Sharpe: 1.51099
- Holdout NAV: 50,307 EUR
- Holdout Sharpe: 0.54306
### VIX OFF
- Search NAV: 29,989 EUR
- Search Sharpe: 1.48190
- Holdout NAV: 46,796 EUR
- Holdout Sharpe: 0.46459

## Rychly zaver
- VIX ON mal vyssi najlepsi holdout NAV o 3,511 EUR.

## Parove porovnanie rovnakych trial cisel
- VIX ON vyhral v 54 paroch.
- VIX OFF vyhral v 46 paroch.
- Remiza: 0.

## Ulozene subory
- `aggressive_vix_summary.csv`
- `aggressive_vix_on_trials.csv`
- `aggressive_vix_off_trials.csv`
- `aggressive_vix_paired_trials.csv`