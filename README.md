# Forecasting Philippine Food Prices Before, During, and After COVID-19
Time-series analysis of WFP/HDX retail food prices (Jan 2016 to Jun 2026).

## Where things are
- notebooks/ analysis notebooks 01-15 and helper scripts (study_config.py holds period definitions)
- outputs/tables/ and outputs/figures/ all result tables and figures
  - outputs/tables/15_dashboard_data_contract.csv - **the data contract for the dashboard**
  - outputs/tables/15_results_register.csv, 15_traceability.csv - headline results and question-to-evidence map
- data/processed/ analysis datasets (raw data not included; download from HDX)

## Important conventions
- Period labels: Pre-pandemic (Jan 2016-May 2019), Pandemic (May 2020-May 2023), Post-pandemic (Jun 2023-Jun 2026). Jun 2019-Apr 2020 is excluded.
- Prices are nominal PHP. Period comparisons use matched commodity-market series.
- Forecasts are indicative: SARIMA beats Seasonal Naive but is no better than plain Naive.

## Setup
python -m venv .venv, activate it, then pip install -r requirements.txt (add streamlit for the dashboard).
Data source: WFP food prices via HDX (check the dataset page for license and attribution).
