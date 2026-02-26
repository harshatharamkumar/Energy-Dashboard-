import pandas as pd
import numpy as np
from config import RENEWABLE_COL, DEMAND_COL


def build_forecast(
    df: pd.DataFrame,
    model,
    forecast_days: int,
    grid_cost: float,
    renewable_cost: float,
) -> pd.DataFrame:
    """
    Returns a DataFrame with daily renewable forecast, demand estimate,
    energy allocation split, and cost breakdown.
    """
    solar  = df.set_index("Date")[RENEWABLE_COL]
    demand = df.set_index("Date")[DEMAND_COL]

    # Renewable forecast
    if model:
        try:
            renewable_vals = np.asarray(model.forecast(steps=forecast_days))
        except Exception:
            renewable_vals = np.full(forecast_days, solar.mean())
    else:
        renewable_vals = np.full(forecast_days, solar.mean())

    # Demand estimate
    demand_val = demand.rolling(30).mean().iloc[-1]
    if np.isnan(demand_val):
        demand_val = demand.mean()

    # Build base DataFrame
    dates = pd.date_range(start=solar.index[-1], periods=forecast_days + 1)[1:]
    fdf = pd.DataFrame({
        "Date":               dates,
        "Renewable_Forecast": np.maximum(0, renewable_vals),
        "Industrial_Demand":  demand_val,
    })

    # Allocation 
    fdf["Renewable_Used"] = np.minimum(fdf["Industrial_Demand"], fdf["Renewable_Forecast"])
    fdf["Grid_Required"]  = np.maximum(0, fdf["Industrial_Demand"] - fdf["Renewable_Used"])
    fdf["Renewable_%"]    = np.where(
        fdf["Industrial_Demand"] > 0,
        fdf["Renewable_Used"] / fdf["Industrial_Demand"] * 100,
        0,
    )

    # Costs
    fdf["Grid_Cost"]         = fdf["Grid_Required"]  * grid_cost
    fdf["Renewable_Cost"]    = fdf["Renewable_Used"] * renewable_cost
    fdf["Total_Energy_Cost"] = fdf["Grid_Cost"] + fdf["Renewable_Cost"]

    return fdf.round(2)
