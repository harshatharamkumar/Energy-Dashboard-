# ⚡ Industrial Renewable Energy & Production Intelligence

A Streamlit dashboard that forecasts renewable energy generation, optimises industrial energy allocation, and provides actionable cost-reduction recommendations for factory operations.

---

## 📸 Overview

This dashboard helps industrial facilities:
- Forecast solar/renewable energy generation using a SARIMA model
- Allocate energy between renewable sources and the grid
- Track per-clutch production efficiency (energy, diesel, cost)
- Receive tiered AI recommendations based on renewable shortfall or surplus

---

## 🗂️ Project Structure

```
energy/
├── app.py                        # Entry point — UI wiring, sidebar, tabs
├── config.py                     # Constants: file paths, column names
├── data_loader.py                # Data loading, KPI derivation, date filtering
├── forecasting.py                # Forecast logic, energy allocation, cost calculation
├── requirements.txt
├── data/
│   └── final_energy_dataset.csv  # Historical energy & production data
├── model/
│   └── sarima_model.pkl          # Pre-trained SARIMA forecasting model
├── notebooks/                    # EDA and model training notebooks
└── components/
    ├── __init__.py
    ├── tab_overview.py           # Executive Overview tab
    ├── tab_historical.py         # Historical Analysis tab
    ├── tab_forecast.py           # Forecast & Planning tab
    └── tab_production.py         # Production Intelligence + AI Recommendations tab
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Bharathwaaj18/Energy-Dashboard-.git
cd your-repo-name
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

---

## 📊 Dashboard Tabs

| Tab | Description |
|-----|-------------|
| **Executive Overview** | KPI metrics + stacked bar chart of grid vs renewable cost |
| **Historical Analysis** | Line charts for renewable generation and energy per clutch |
| **Forecast & Planning** | Actual vs forecast chart + daily energy allocation table |
| **Production Intelligence** | Per-clutch efficiency metrics + AI energy recommendations |

---

## 🤖 AI Recommendation Engine

The recommendation engine in the Production Intelligence tab analyses the renewable forecast against projected industrial demand and responds in three ways:

- ✅ **Surplus** — suggests grid export or battery charging
- 🟡🟠 **Minor / Moderate Shortfall** — recommends load shifting and TOD tariff negotiation
- 🔴 **Critical Shortfall** — flags battery storage sizing, grid procurement strategy, and renewable capacity expansion

---

## ⚙️ Configuration

All settings are adjustable from the sidebar at runtime:

| Setting | Default | Description |
|---------|---------|-------------|
| Forecast Days | 30 | Number of days to forecast ahead |
| Grid Cost / kWh | ₹8.00 | Current grid electricity rate |
| Renewable Cost / kWh | ₹2.00 | Cost of self-generated renewable energy |
| Historical Date Range | Full range | Filter historical charts by date |

---

## 📦 Requirements

Key dependencies (see `requirements.txt` for full list):

```
streamlit
pandas
numpy
plotly
joblib
statsmodels
```

---

## 📁 Data Format

The CSV at `data/final_energy_dataset.csv` must contain the following columns:

| Column | Description |
|--------|-------------|
| `Date` | Date of record (parseable by pandas) |
| `E-Today (KWH)` | Renewable energy generated that day |
| `Daily Energy consumption in units` | Total industrial energy demand |
| `Clutch Produced` | Number of clutches manufactured |
| `Diesel Consumed in ltrs` | Diesel used that day |
| `Diesel cost / Day in INR` | Cost of diesel for that day |

---

## 🧠 Model

The SARIMA model (`model/sarima_model.pkl`) is pre-trained on historical solar generation data. If the model file is missing or fails to load, the app falls back to the historical mean as the forecast value.

To retrain the model, refer to the notebooks in the `notebooks/` directory.

---

## 📄 License

MIT License — free to use, modify, and distribute.
