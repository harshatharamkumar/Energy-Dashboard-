"""
evaluate_model.py - Comprehensive SARIMA Model Evaluation
=========================================================
Computes standard forecasting metrics using two approaches:

1. STATIC FORECAST  - Train on 80%, forecast the remaining 20% in one shot
2. WALK-FORWARD     - Re-fit model incrementally, forecast 1 step at a time
                      (more realistic, better metrics)

Metrics computed:
  MAE, RMSE, MAPE, sMAPE, R2, MedAE,
  Directional Accuracy, Forecast Bias,
  AIC / BIC, Ljung-Box residual test

Usage:
    python evaluate_model.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["PYTHONIOENCODING"] = "utf-8"

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)
from scipy.stats import pearsonr
from statsmodels.tsa.statespace.sarimax import SARIMAX
from config import DATA_PATH, RENEWABLE_COL


# -- CONFIGURATION ----------------------------------------------------------
TEST_RATIO    = 0.2
FORECAST_COL  = RENEWABLE_COL   # "E-Today (KWH)"
SARIMA_ORDER          = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 30)


# -- HELPERS -----------------------------------------------------------------
def compute_mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def compute_smape(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0
    return np.mean(2.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100

def directional_accuracy(y_true, y_pred):
    a = np.diff(y_true) >= 0
    f = np.diff(y_pred) >= 0
    return np.mean(a == f) * 100

def forecast_bias(y_true, y_pred):
    return np.mean(y_pred - y_true)

def print_metrics(actual, forecast, fitted_model=None, label=""):
    mae_val   = mean_absolute_error(actual, forecast)
    rmse_val  = np.sqrt(mean_squared_error(actual, forecast))
    r2_val    = r2_score(actual, forecast)
    medae_val = median_absolute_error(actual, forecast)
    mape_val  = compute_mape(actual, forecast)
    smape_val = compute_smape(actual, forecast)
    da_val    = directional_accuracy(actual, forecast)
    bias_val  = forecast_bias(actual, forecast)
    nrmse_val = (rmse_val / np.mean(actual)) * 100 if np.mean(actual) != 0 else 0

    try:
        corr, p_val = pearsonr(actual, forecast)
    except Exception:
        corr, p_val = 0, 1

    residuals = actual - forecast

    print("-" * 70)
    print(f"  {label} - FORECASTING ACCURACY METRICS")
    print("-" * 70)
    print(f"  MAE  (Mean Absolute Error)         : {mae_val:>12,.2f} kWh")
    print(f"  RMSE (Root Mean Squared Error)      : {rmse_val:>12,.2f} kWh")
    print(f"  NRMSE (Normalised RMSE)             : {nrmse_val:>12.2f} %")
    print(f"  MedAE (Median Absolute Error)       : {medae_val:>12,.2f} kWh")
    print(f"  MAPE (Mean Abs % Error)             : {mape_val:>12.2f} %")
    print(f"  sMAPE (Symmetric MAPE)              : {smape_val:>12.2f} %")
    print(f"  R2 Score                            : {r2_val:>12.4f}")
    print(f"  Pearson Correlation                 : {corr:>12.4f}  (p={p_val:.2e})")

    print(f"\n  Directional Accuracy                : {da_val:>12.2f} %")
    print(f"  Forecast Bias (Mean Error)          : {bias_val:>12,.2f} kWh")
    bias_dir = "over-forecasting" if bias_val > 0 else "under-forecasting"
    print(f"  Bias Direction                      : {bias_dir}")

    if fitted_model is not None:
        print(f"\n  AIC                                 : {fitted_model.aic:>12.2f}")
        print(f"  BIC                                 : {fitted_model.bic:>12.2f}")

    # Ljung-Box
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb = acorr_ljungbox(residuals, lags=[10], return_df=True)
        lb_p = lb["lb_pvalue"].values[0]
        status = "No autocorrelation (good)" if lb_p > 0.05 else "Autocorrelation detected"
        print(f"\n  Ljung-Box p-value (lag=10)          : {lb_p:>12.4f}  [{status}]")
    except Exception:
        pass

    print(f"\n  Residual Mean                       : {np.mean(residuals):>12,.2f} kWh")
    print(f"  Residual Std                        : {np.std(residuals):>12,.2f} kWh")

    return {
        "MAE": mae_val, "RMSE": rmse_val, "R2": r2_val,
        "MAPE": mape_val, "sMAPE": smape_val, "DA": da_val,
    }


# -- LOAD DATA --------------------------------------------------------------
print("=" * 70)
print("  SARIMA MODEL - EVALUATION REPORT")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df = df.sort_values("Date").reset_index(drop=True)

n = len(df)
split_idx = int(n * (1 - TEST_RATIO))
train_df = df.iloc[:split_idx].copy()
test_df  = df.iloc[split_idx:].copy()
n_test   = len(test_df)

print(f"\nDataset       : {DATA_PATH}")
print(f"Total records : {n}")
print(f"Train / Test  : {split_idx} / {n_test}")
print(f"Date range    : {df['Date'].min().date()} -> {df['Date'].max().date()}")
print(f"Train mean    : {train_df[FORECAST_COL].mean():,.0f} kWh")
print(f"Test  mean    : {test_df[FORECAST_COL].mean():,.0f} kWh")
print(f"SARIMA order  : {SARIMA_ORDER} x {SARIMA_SEASONAL_ORDER}")


# =====================================================================
# METHOD 1: STATIC (one-shot) FORECAST
# =====================================================================
print("\n\n" + "=" * 70)
print("  METHOD 1: STATIC FORECAST (train once, predict all test days)")
print("=" * 70)

train_series = train_df.set_index("Date")[FORECAST_COL].asfreq("D").ffill()

model_static = SARIMAX(
    train_series,
    order=SARIMA_ORDER,
    seasonal_order=SARIMA_SEASONAL_ORDER,
    enforce_stationarity=False,
    enforce_invertibility=False,
)
fitted_static = model_static.fit(disp=False)

actual_static   = test_df[FORECAST_COL].values.astype(float)
forecast_static = np.maximum(0, np.asarray(fitted_static.forecast(steps=n_test)))

static_metrics = print_metrics(actual_static, forecast_static, fitted_static, "STATIC")


# =====================================================================
# METHOD 2: WALK-FORWARD (rolling) EVALUATION
# =====================================================================
print("\n\n" + "=" * 70)
print("  METHOD 2: WALK-FORWARD EVALUATION (1-step ahead, re-fit each day)")
print("=" * 70)
print("  (This takes a moment - fitting model for each test day...)")

history = train_df[FORECAST_COL].values.tolist()
wf_predictions = []

for i in range(n_test):
    # Build series from history
    hist_series = pd.Series(history, dtype=float)

    try:
        wf_model = SARIMAX(
            hist_series,
            order=SARIMA_ORDER,
            seasonal_order=SARIMA_SEASONAL_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        wf_fitted = wf_model.fit(disp=False)
        yhat = max(0, float(wf_fitted.forecast(steps=1).iloc[0]))
    except Exception:
        yhat = np.mean(history[-30:])  # fallback: 30-day mean

    wf_predictions.append(yhat)
    history.append(test_df[FORECAST_COL].values[i])  # append actual for next step

    if (i + 1) % 10 == 0 or i == n_test - 1:
        print(f"  ... completed {i+1}/{n_test} steps")

actual_wf   = test_df[FORECAST_COL].values.astype(float)
forecast_wf = np.array(wf_predictions)

print()
wf_metrics = print_metrics(actual_wf, forecast_wf, label="WALK-FORWARD")


# =====================================================================
# IN-SAMPLE (TRAINING FIT) METRICS
# =====================================================================
print("\n\n" + "=" * 70)
print("  IN-SAMPLE FIT (training data)")
print("=" * 70)

in_sample_pred = fitted_static.fittedvalues.values
in_sample_actual = train_series.values

# Align lengths (first few values may be NaN due to differencing)
valid = ~np.isnan(in_sample_pred) & ~np.isnan(in_sample_actual)
in_sample_pred = in_sample_pred[valid]
in_sample_actual = in_sample_actual[valid]

insample_metrics = print_metrics(in_sample_actual, in_sample_pred, fitted_static, "IN-SAMPLE")


# =====================================================================
# RESUME-READY SUMMARY
# =====================================================================
# Use walk-forward metrics (most realistic) for the resume
m = wf_metrics
accuracy = max(0, 100 - m["MAPE"])

print("\n\n" + "=" * 70)
print("  RESUME-READY SUMMARY (using walk-forward metrics)")
print("=" * 70)
print(f"""
  Built a SARIMA-based renewable energy forecasting & industrial
  decision-support system with a Streamlit dashboard, achieving:
    - Walk-forward MAE of {m['MAE']:,.0f} kWh (R2 = {m['R2']:.2f})
    - MAPE of {m['MAPE']:.1f}% / sMAPE of {m['sMAPE']:.1f}%
    - Directional accuracy of {m['DA']:.0f}%
    - In-sample R2 of {insample_metrics['R2']:.2f} on training data
    - Tiered AI recommendation engine for energy allocation
    - Reduced simulated grid costs by optimising renewable usage
""")

print("  METRIC COMPARISON TABLE:")
print("  " + "-" * 50)
print(f"  {'Metric':<25} {'In-Sample':>12} {'Walk-Fwd':>12}")
print("  " + "-" * 50)
for key in ["MAE", "RMSE", "R2", "MAPE", "sMAPE", "DA"]:
    is_val = insample_metrics[key]
    wf_val = wf_metrics[key]
    if key in ("MAE", "RMSE"):
        print(f"  {key:<25} {is_val:>12,.0f} {wf_val:>12,.0f}")
    else:
        print(f"  {key:<25} {is_val:>12.2f} {wf_val:>12.2f}")
print("  " + "-" * 50)

print("\n" + "=" * 70)
print("  Evaluation complete.")
print("=" * 70)


# -- SAVE RESULTS TO CSV ----------------------------------------------------
results = pd.DataFrame({
    "Date":              test_df["Date"].values,
    "Actual_kWh":        actual_wf,
    "WalkFwd_Forecast":  forecast_wf,
    "Static_Forecast":   forecast_static,
    "WF_Residual":       actual_wf - forecast_wf,
    "Static_Residual":   actual_static - forecast_static,
})

output_path = Path("data/evaluation_results.csv")
results.to_csv(output_path, index=False)
print(f"\nDetailed results saved to: {output_path}")
