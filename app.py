# app.py — Main entry point (UI wiring only)
import streamlit as st
from config import DATA_PATH, MODEL_PATH
from data_loader import load_data, load_model, add_kpis, filter_by_date
from forecasting import build_forecast
from components.tab_overview   import render as render_overview
from components.tab_historical import render as render_historical
from components.tab_forecast   import render as render_forecast
from components.tab_production import render as render_production

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Industrial Energy Intelligence", layout="wide")
st.title("Industrial Renewable Energy & Production Intelligence")
st.caption("Predicts renewable energy generation and optimises industrial energy allocation.")

# ── LOAD DATA & MODEL ─────────────────────────────────────────────────────────
df    = add_kpis(load_data(DATA_PATH))
model = load_model(MODEL_PATH)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    forecast_days  = st.slider("Forecast Days", 7, 60, 30)
    grid_cost      = st.number_input("Grid Cost / kWh (INR)",      value=8.0, step=0.5)
    renewable_cost = st.number_input("Renewable Cost / kWh (INR)", value=2.0, step=0.5)

    min_d, max_d = df["Date"].min().date(), df["Date"].max().date()
    date_range   = st.date_input(
        "Historical Range",
        value=(min_d, max_d),
        min_value=min_d,
        max_value=max_d,
    )

# ── FILTER & FORECAST ─────────────────────────────────────────────────────────
df_view = filter_by_date(df, date_range)
fdf     = build_forecast(df, model, forecast_days, grid_cost, renewable_cost)

# ── TABS ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4 = st.tabs([
    "Executive Overview",
    "Historical Analysis",
    "Forecast & Planning",
    "Production Intelligence",
])

with t1: render_overview(fdf)
with t2: render_historical(df_view)
with t3: render_forecast(df_view, fdf)
with t4: render_production(df_view, df, fdf)
