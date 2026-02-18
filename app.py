import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Industrial Energy & Production Intelligence",
    layout="wide",
)

st.title("Industrial Renewable Energy & Production Intelligence")
st.markdown("### Project Objective")
st.write("This dashboard predicts renewable energy generation and optimizes industrial energy allocation.")

# --------------------------------------------------
# PATHS
# --------------------------------------------------
DATA_PATH = Path("data/final_energy_dataset.csv")
MODEL_PATH = Path("model/sarima_model.pkl")
RENEWABLE_COL = "E-Today (KWH)"
DEMAND_COL = "Daily Energy consumption in units"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    return df

@st.cache_resource
def load_model(path):
    if path.exists():
        try:
            return joblib.load(path)
        except:
            return None
    return None

df = load_data(DATA_PATH)
model = load_model(MODEL_PATH)

# --------------------------------------------------
# SAFE DIVIDE
# --------------------------------------------------
def safe_divide(n, d):
    n = np.asarray(n, dtype=float)
    d = np.asarray(d, dtype=float)
    result = np.zeros_like(n)
    mask = (d != 0) & (~np.isnan(d))
    result[mask] = n[mask] / d[mask]
    return result

# --------------------------------------------------
# CREATE KPIs
# --------------------------------------------------
df["Units_per_Clutch"] = safe_divide(
    df["Daily Energy consumption in units"],
    df["Clutch Produced"]
)

df["Diesel_per_Clutch"] = safe_divide(
    df["Diesel Consumed in ltrs"],
    df["Clutch Produced"]
)

df["Energy_Cost_Per_Clutch"] = safe_divide(
    df["Diesel cost / Day in INR"],
    df["Clutch Produced"]
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:

    st.header("System Configuration")

    forecast_days = st.slider("Forecast Days", 7, 60, 30)

    grid_cost = st.number_input("Grid Cost per kWh (INR)", value=8.0)
    renewable_cost = st.number_input("Renewable Cost per kWh (INR)", value=2.0)

    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()

    date_range = st.date_input(
        "Historical Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

# --------------------------------------------------
# DATE FILTERING
# --------------------------------------------------
if not isinstance(date_range, (list, tuple)):
    date_range = (date_range, date_range)

start_ts = pd.to_datetime(date_range[0])
end_ts = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

df_filtered = df[(df["Date"] >= start_ts) & (df["Date"] <= end_ts)].copy()
df_view = df_filtered if not df_filtered.empty else df

# --------------------------------------------------
# FORECAST RENEWABLE
# --------------------------------------------------
solar_series = df.set_index("Date")[RENEWABLE_COL]

if model:
    try:
        renewable_forecast = model.forecast(steps=forecast_days)
    except:
        renewable_forecast = np.repeat(solar_series.mean(), forecast_days)
else:
    renewable_forecast = np.repeat(solar_series.mean(), forecast_days)

forecast_dates = pd.date_range(
    start=solar_series.index[-1],
    periods=forecast_days + 1
)[1:]

forecast_df = pd.DataFrame({
    "Date": forecast_dates,
    "Renewable_Forecast": renewable_forecast
})

# --------------------------------------------------
# DEMAND MODEL (SEPARATE FROM RENEWABLE)
# Using historical rolling average for realism
# --------------------------------------------------
energy_series = df.set_index("Date")[DEMAND_COL]

rolling_mean_demand = energy_series.rolling(30).mean().iloc[-1]

if np.isnan(rolling_mean_demand):
    rolling_mean_demand = energy_series.mean()

forecast_df["Industrial_Demand"] = np.repeat(
    rolling_mean_demand,
    forecast_days
)

# --------------------------------------------------
# CLEAN ENERGY ALLOCATION LOGIC
# --------------------------------------------------
forecast_df["Renewable_Forecast"] = np.maximum(0, forecast_df["Renewable_Forecast"])
forecast_df["Industrial_Demand"] = np.maximum(0, forecast_df["Industrial_Demand"])

forecast_df["Renewable_Used"] = np.minimum(
    forecast_df["Industrial_Demand"],
    forecast_df["Renewable_Forecast"]
)

forecast_df["Grid_Required"] = (
    forecast_df["Industrial_Demand"] -
    forecast_df["Renewable_Used"]
)

forecast_df["Grid_Required"] = np.maximum(0, forecast_df["Grid_Required"])

forecast_df["Renewable_%"] = np.where(
    forecast_df["Industrial_Demand"] > 0,
    (forecast_df["Renewable_Used"] /
     forecast_df["Industrial_Demand"]) * 100,
    0
)

# --------------------------------------------------
# COST CALCULATION
# --------------------------------------------------
forecast_df["Grid_Cost"] = forecast_df["Grid_Required"] * grid_cost
forecast_df["Renewable_Cost"] = forecast_df["Renewable_Used"] * renewable_cost
forecast_df["Total_Energy_Cost"] = (
    forecast_df["Grid_Cost"] +
    forecast_df["Renewable_Cost"]
)

forecast_df = forecast_df.round(2)

# --------------------------------------------------
# TABS
# --------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Overview",
    "Historical Analysis",
    "Forecast & Planning",
    "Production Intelligence"
])

# --------------------------------------------------
# TAB 1
# --------------------------------------------------
with tab1:

    st.subheader("Energy & Cost Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Renewable Used (kWh)",
              f"{forecast_df['Renewable_Used'].sum():.0f}")

    c2.metric("Total Grid Required (kWh)",
              f"{forecast_df['Grid_Required'].sum():.0f}")

    c3.metric("Avg Renewable Utilization %",
              f"{forecast_df['Renewable_%'].mean():.2f}%")

    c4.metric("Total Energy Cost (INR)",
              f"{forecast_df['Total_Energy_Cost'].sum():.0f}")

    fig_cost = go.Figure()

    fig_cost.add_trace(go.Bar(
        x=forecast_df['Date'],
        y=forecast_df['Grid_Cost'],
        name="Grid Cost"
    ))

    fig_cost.add_trace(go.Bar(
        x=forecast_df['Date'],
        y=forecast_df['Renewable_Cost'],
        name="Renewable Cost"
    ))

    fig_cost.update_layout(barmode="stack", height=350)

    st.plotly_chart(fig_cost, use_container_width=True)
    

# --------------------------------------------------
# TAB 2
# --------------------------------------------------
with tab2:

    fig_hist = px.line(df_view, x="Date", y=RENEWABLE_COL)
    st.plotly_chart(fig_hist, use_container_width=True)

    fig_units = px.line(df_view, x="Date", y="Units_per_Clutch")
    st.plotly_chart(fig_units, use_container_width=True)

# --------------------------------------------------
# TAB 3
# --------------------------------------------------
with tab3:

    actual_series = df_view.set_index("Date")[RENEWABLE_COL]

    fig_fore = go.Figure()

    fig_fore.add_trace(go.Scatter(
        x=actual_series.index,
        y=actual_series,
        mode="lines",
        name="Actual"
    ))

    fig_fore.add_trace(go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Renewable_Forecast"],
        mode="lines",
        name="Forecast"
    ))

    st.plotly_chart(fig_fore, use_container_width=True)

    st.subheader("Energy Allocation Plan")

    st.dataframe(
        forecast_df.style.format({
            "Renewable_%": "{:.1f}%",
            "Total_Energy_Cost": "₹{:,.0f}"
        })
    )

# --------------------------------------------------
# TAB 4
# --------------------------------------------------
with tab4:

    p1, p2, p3 = st.columns(3)

    p1.metric("Avg Units / Clutch",
              f"{df_view['Units_per_Clutch'].mean():.2f}")

    p2.metric("Avg Diesel / Clutch (L)",
              f"{df_view['Diesel_per_Clutch'].mean():.2f}")

    p3.metric("Avg Energy Cost / Clutch (INR)",
              f"{df_view['Energy_Cost_Per_Clutch'].mean():.2f}")

    st.subheader("AI Recommendation")

    avg_forecast = forecast_df["Renewable_Forecast"].mean()
    avg_demand = forecast_df["Industrial_Demand"].mean()

    if avg_forecast >= avg_demand:
        st.success("Renewable fully supports projected demand.")
    else:
        reduction_needed = (
            (avg_demand - avg_forecast) /
            df["Units_per_Clutch"].mean()
        )
        st.warning(
            f"Reduce production by approx {int(reduction_needed)} clutches/day to avoid grid usage."
        )
