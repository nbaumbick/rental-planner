# Rental-Planner
A machine learning tool that predicts 12-month rent trends by ZIP code using XGBoost trained on Zillow ZORI and HUD Small Area Fair Market Rate data. Provides YoY rent change forecasts for Tier 1 markets and historical seasonality profiles showing intra-year price patterns to help renters time and plan their move.

Live demo: [huggingface.co/spaces/NBaumbick/Rental-Planner](https://huggingface.co/spaces/NBaumbick/Rental-Planner)

---

## What it does

Enter a ZIP code and bedroom count to get:

- A 12-month rent forecast with projected year-over-year change
- A monthly seasonality profile showing which months are historically cheapest and most expensive to sign a lease
- A data confidence tier indicating how much history is available for that market

---

## How it works

### Data

Two public datasets are combined to build the training set:

- **Zillow Observed Rent Index (ZORI)** — monthly ZIP-level rent indices from January 2015 through April 2026, covering ~8,300 ZIP codes across the United States
- **HUD Small Area Fair Market Rents (SAFMR)** — annual bedroom-segmented rent benchmarks by ZIP code from FY2016 through FY2026

ZORI provides the monthly time series signal. HUD SAFMR provides the bedroom-specific pricing ratios used to convert aggregate forecasts into bedroom-level estimates.

### Model

An XGBoost regression model trained to predict 12-month forward year-over-year rent change for a given ZIP code.

**Features include:**
- ZORI lag values at 1, 3, 6, 12, 24, and 36 months
- Rolling mean and standard deviation over 6 and 12 month windows
- Month-over-month, year-over-year, and 2-year cumulative rate of change
- HUD FMR values and year-over-year changes for studio, 1BR, and 2BR units
- Calendar features including month, quarter, and year
- Metro and state target encodings
- Luxury market flag (top 10% by median ZORI)

**Training details:**
- 181,312 training rows across 2,336 Tier 1 ZIP codes
- Group-based train/test split by ZIP code to prevent data leakage
- Target clipped at [-15%, +25.4%] to reduce COVID-era outlier influence
- Test MAE: 2.07% | Test R²: 0.77 | Median per-ZIP R²: 0.73

### Seasonality profiles

Separate from the forecast model, historical monthly deviation from annual mean rent is computed directly from ZORI data for each ZIP code. No ML — just observed historical patterns aggregated by month.

### Data tiers

ZIP codes are assigned to one of three tiers based on available ZORI history:

| Tier | Requirement | Output |
|------|-------------|--------|
| 1 | 60+ months of ZORI data | Full forecast + seasonality profile |
| 2 | 36-59 months of ZORI data | Seasonality profile only |
| 3 | Under 36 months | Metro-level seasonality fallback |

---

## Known limitations

- The model uses calendar year as a feature to capture macro-economic regimes (COVID rent surge, post-pandemic normalization). This means it may not extrapolate reliably to years well beyond its training range.
- Luxury markets and ZIP codes with idiosyncratic demand drivers (resort towns, oil-dependent markets) tend to have higher error rates.
- HUD Fair Market Rents are set at the 40th percentile of market rents, not the median. Bedroom-specific estimates in high-cost markets may be conservative.
- ZORI coverage is limited to ~8,300 ZIP codes. Many rural and low-density markets fall back to metro-level estimates.

---

## Data sources

- [Zillow Research Data](https://www.zillow.com/research/data/)
- [HUD Small Area Fair Market Rents](https://www.huduser.gov/portal/datasets/fmr/smallarea/index.html)

---

## Tech stack

Python, XGBoost, pandas, SHAP, Streamlit, Plotly, Hugging Face Spaces
