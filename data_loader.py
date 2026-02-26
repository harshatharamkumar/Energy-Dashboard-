import pandas as pd
import numpy as np
import joblib
import streamlit as st
from pathlib import Path
from config import RENEWABLE_COL, DEMAND_COL


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


@st.cache_resource
def load_model(path: Path):
    try:
        return joblib.load(path) if path.exists() else None
    except Exception:
        return None


def add_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Derive per-clutch KPIs, avoiding division by zero."""
    clutch = df["Clutch Produced"].replace(0, np.nan)
    df["Units_per_Clutch"]       = df[DEMAND_COL]                 / clutch
    df["Diesel_per_Clutch"]      = df["Diesel Consumed in ltrs"]  / clutch
    df["Energy_Cost_Per_Clutch"] = df["Diesel cost / Day in INR"] / clutch
    return df.fillna(0)


def filter_by_date(df: pd.DataFrame, date_range) -> pd.DataFrame:
    """Return rows within the selected date range; fallback to full df."""
    d0 = date_range[0]
    d1 = date_range[-1] if len(date_range) == 2 else date_range[0]
    mask = df["Date"].between(
        pd.Timestamp(d0),
        pd.Timestamp(d1) + pd.Timedelta(hours=23, minutes=59),
    )
    return df[mask] if mask.any() else df
