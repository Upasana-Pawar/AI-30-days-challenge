# day8_linear_regression.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# --- prepare output folder
os.makedirs("figures", exist_ok=True)

# --- 1) Load dataset
data = fetch_california_housing(as_frame=True)
df = data.frame.copy()
df["MedHouseVal"] = data.target  # target (median house value)
print("Loaded dataset shape:", df.shape)

# --- 2) Quick EDA (print head and simple stats)
print("\nHead:")
print(df.head())
print("\nSummary statistics:")
print(df.describe())

# optional: correlation heatmap (save)
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="RdBu_r")
plt.title("Correlation matrix")
plt.tight_layout()
plt.savefig("figures/day8_corr_matrix.png")
plt.close()

# --- 3) Features & target
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

# --- 4) Train/Test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# --- 5) Scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# --- 6) Baseline Linear Regression
lr = LinearRegression()
lr.fit(X_train_s, y_train)
y_pred = lr.predict(X_test_s)

# metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nBaseline LinearRegression metrics:")
print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

# cross-validation (5-fold) using neg MSE -> convert to RMSE
cv_scores = cross_val_score(lr, scaler.transform(X), y, scoring="neg_mean_squared_error", cv=5)
cv_rmse = np.sqrt(-cv_scores)
print("CV RMSE (5-fold):", np.round(cv_rmse, 4))
print("CV RMSE mean:", np.round(np.mean(cv_rmse), 4))

# --- 7) Plot Predicted vs Actual
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.4, s=12)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linewidth=2)
plt.xlabel("Actual Median Value")
plt.ylabel("Predicted Median Value")
plt.title("Actual vs Predicted - Linear Regression")
plt.tight_layout()
plt.savefig("figures/day8_pred_vs_actual.png")
plt.close()

# --- 8) Residuals histogram
residuals = y_test - y_pred
plt.figure(figsize=(6,4))
sns.histplot(residuals, bins=40, kde=True)
plt.title("Residuals distribution")
plt.tight_layout()
plt.savefig("figures/day8_residuals.png")
plt.close()

# --- 9) Try a simple regularized model - Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_s, y_train)
y_pred_ridge = ridge.predict(X_test_s)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
r2_ridge = r2_score(y_test, y_pred_ridge)
print("\nRidge (alpha=1.0) metrics:")
print(f"RMSE: {rmse_ridge:.4f}, R2: {r2_ridge:.4f}")

# --- 10) Optional: polynomial features small test (degree=2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_p = poly.fit_transform(X_train_s)
X_test_p = poly.transform(X_test_s)
lr_poly = LinearRegression()
lr_poly.fit(X_train_p, y_train)
y_pred_poly = lr_poly.predict(X_test_p)
rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))
print("\nPoly deg=2 LinearRegression metrics:")
print(f"RMSE: {rmse_poly:.4f}")

# --- 11) Save model + scaler
joblib.dump(lr, "day8_lr_model.joblib")
joblib.dump(scaler, "day8_scaler.joblib")
print("\nModels saved: day8_lr_model.joblib, day8_scaler.joblib")

print("\nFinished. Figures saved to ./figures/")
