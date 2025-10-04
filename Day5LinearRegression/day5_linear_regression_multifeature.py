# day5_linear_regression_multifeature.py
# Compare single-feature vs multi-feature Linear Regression on California housing dataset

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

print("📊 Loading dataset...")
housing = fetch_california_housing(as_frame=True)
df = housing.frame
print("Dataset shape:", df.shape)
print(df.head())

# --- Single-feature model (MedInc) -----------------------------------------
X_single = df[['MedInc']]
y = df['MedHouseVal']

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_single, y, test_size=0.2, random_state=42
)

model_single = LinearRegression()
model_single.fit(X_train_s, y_train_s)
y_pred_s = model_single.predict(X_test_s)

mse_s = mean_squared_error(y_test_s, y_pred_s)
r2_s = r2_score(y_test_s, y_pred_s)

print("\nSingle-feature model (MedInc) results:")
print(f"  MSE: {mse_s:.6f}")
print(f"  R²:  {r2_s:.6f}")
print(f"  Coef: {model_single.coef_}, Intercept: {model_single.intercept_}")

# --- Multi-feature model ---------------------------------------------------
# Choose a few sensible features
features = ['MedInc', 'HouseAge', 'AveRooms', 'Population', 'AveOccup']
X_multi = df[features]

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y, test_size=0.2, random_state=42
)

# Use a pipeline with scaling (helps some models; harmless here)
model_multi = make_pipeline(StandardScaler(), LinearRegression())
model_multi.fit(X_train_m, y_train_m)
y_pred_m = model_multi.predict(X_test_m)

mse_m = mean_squared_error(y_test_m, y_pred_m)
r2_m = r2_score(y_test_m, y_pred_m)

# get coefficients from the linear model inside the pipeline
lin = model_multi.named_steps['linearregression']
coefs = lin.coef_
intercept = lin.intercept_

print("\nMulti-feature model results (features = " + ", ".join(features) + "):")
print(f"  MSE: {mse_m:.6f}")
print(f"  R²:  {r2_m:.6f}")
print(f"  Coefs: {dict(zip(features, coefs))}")
print(f"  Intercept: {intercept}")

# --- Compare predictions visually -----------------------------------------
# We'll plot actual vs predicted for multi-feature model (scatter + y=x line)
plt.figure(figsize=(7,7))
plt.scatter(y_test_m, y_pred_m, alpha=0.4, s=10)
lims = [min(y_test_m.min(), y_pred_m.min()), max(y_test_m.max(), y_pred_m.max())]
plt.plot(lims, lims, '--', color='grey')  # perfect prediction line
plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Median House Value")
plt.title("Actual vs Predicted (Multi-feature Linear Regression)")
plt.xlim(lims)
plt.ylim(lims)
plt.grid(alpha=0.2)

outfile = os.path.join(OUT_DIR, "day5_multi_feature_actual_vs_pred.png")
plt.savefig(outfile, dpi=150, bbox_inches='tight')
plt.show()

print(f"\n🎉 Plotted and saved: {outfile}")

# --- Quick summary output for README / logging --------------------------------
summary = {
    "single": {"mse": float(mse_s), "r2": float(r2_s)},
    "multi": {"mse": float(mse_m), "r2": float(r2_m), "features": features}
}
print("\nSummary (dict):", summary)
