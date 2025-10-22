"""
Day23: Uncertainty Estimation & Bayesian Methods

- Classification: Deep Ensembles using RandomForest (different random seeds).
  Compute mean predicted probability and std (uncertainty) across ensemble members.
  Analyze relationship between uncertainty and prediction correctness.

- Regression: BayesianRidge on diabetes dataset.
  Use BayesianRidge.predict(X, return_std=True) to obtain predictive mean and std.
  Plot predicted mean with +/- 2*std band.

- Optional: MC Dropout classification demo if tensorflow/keras is available.
  The code checks for tensorflow and runs MC Dropout only when present.

Outputs:
- classification ensemble plot (uncertainty vs sample)
- regression plot (mean +/- uncertainty)
- saved models and metrics summary
"""

import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

# Optional: test for tensorflow for MC Dropout demo
TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# --- Paths ---
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "day23_artifacts"
MODELS_DIR = OUT_DIR / "day23_models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_PLOT = OUT_DIR / "day23_classification_ensemble_uncertainty.png"
REG_PLOT = OUT_DIR / "day23_regression_bayesianridge.png"
MC_DROP_PLOT = OUT_DIR / "day23_mc_dropout.png"
METRICS_FILE = OUT_DIR / "day23_metrics.txt"

# --- 1) Classification: Deep Ensemble uncertainty (using RandomForest as base) ---
print("Loading breast_cancer dataset for classification ensemble demo...")
bc = load_breast_cancer(as_frame=True)
Xc = bc.data
yc = bc.target

Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.25, random_state=42, stratify=yc)

# Standardize features for ensembles to be comparable (RF doesn't strictly need it but it's fine)
scaler_c = StandardScaler().fit(Xc_train)
Xc_train_s = scaler_c.transform(Xc_train)
Xc_test_s = scaler_c.transform(Xc_test)

n_members = 10
ensemble = []
member_probas = []

print(f"Training ensemble of {n_members} RandomForest members (different random seeds)...")
for i in range(n_members):
    seed = 100 + i
    clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    clf.fit(Xc_train_s, yc_train)
    joblib.dump(clf, MODELS_DIR / f"rf_ensemble_member_{i+1}.joblib")
    ensemble.append(clf)
    # get predicted probability for positive class
    p = clf.predict_proba(Xc_test_s)[:, 1]
    member_probas.append(p)

member_probas = np.vstack(member_probas)  # shape: (n_members, n_samples)
mean_proba = member_probas.mean(axis=0)
std_proba = member_probas.std(axis=0)  # epistemic uncertainty proxy

# Evaluate ensemble predictive mean
y_pred_mean = (mean_proba >= 0.5).astype(int)
acc_mean = accuracy_score(yc_test, y_pred_mean)
roc_mean = roc_auc_score(yc_test, mean_proba)
brier_mean = brier_score_loss(yc_test, mean_proba)

# Short analysis: uncertainty for correct vs incorrect predictions
correct_mask = y_pred_mean == yc_test.values
uncertainty_correct = std_proba[correct_mask]
uncertainty_incorrect = std_proba[~correct_mask]

# Save classification metrics and summary
with open(METRICS_FILE, "w", encoding="utf-8") as f:
    f.write("Day23 - Uncertainty & Bayesian Methods Metrics Summary\n")
    f.write("="*64 + "\n\n")
    f.write("Classification (Ensemble)\n")
    f.write(f"  Ensemble members : {n_members}\n")
    f.write(f"  Test accuracy (mean proba threshold 0.5) : {acc_mean:.4f}\n")
    f.write(f"  Test ROC AUC (ensemble mean)            : {roc_mean:.4f}\n")
    f.write(f"  Test Brier score (ensemble mean)        : {brier_mean:.4f}\n\n")

    f.write("Uncertainty analysis (std of member probabilities):\n")
    f.write(f"  Mean std (correct predictions)  : {np.mean(uncertainty_correct):.5f}\n")
    f.write(f"  Mean std (incorrect predictions): {np.mean(uncertainty_incorrect):.5f}\n")
    f.write("\nNote: higher std -> higher epistemic uncertainty (members disagree more).\n\n")

# Plot classification uncertainties: sorted by uncertainty, color by correctness
order = np.argsort(std_proba)[::-1]  # highest uncertainty first
sorted_std = std_proba[order]
sorted_mean = mean_proba[order]
sorted_labels = yc_test.values[order]
sorted_pred = y_pred_mean[order]
sorted_idx = np.arange(len(sorted_std))

plt.figure(figsize=(12,5))
# scatter: mean prob with errorbar = std
plt.errorbar(sorted_idx, sorted_mean, yerr=sorted_std, fmt='o', alpha=0.8, label='predicted mean ± std')
# color markers by correctness
for i in range(len(sorted_idx)):
    c = 'green' if sorted_pred[i] == sorted_labels[i] else 'red'
    plt.scatter(sorted_idx[i], sorted_mean[i], color=c, s=20)
plt.axhline(0.5, linestyle='--', color='grey')
plt.title("Day23 — Ensemble predictive mean ± std (sorted by uncertainty)\n(green = correct, red = incorrect)")
plt.xlabel("Test sample (sorted by descending std)")
plt.ylabel("Predicted probability (positive class)")
plt.tight_layout()
plt.savefig(CLASS_PLOT, dpi=150)
plt.close()
print(f"Saved classification ensemble uncertainty plot to {CLASS_PLOT}")

# Append more metrics for regression below file (so file already exists)
with open(METRICS_FILE, "a", encoding="utf-8") as f:
    f.write("Saved classification ensemble plot: {}\n\n".format(CLASS_PLOT))

# --- 2) Regression: BayesianRidge predictive mean & std (diabetes dataset) ---
print("Loading diabetes dataset for BayesianRidge regression demo...")
db = load_diabetes(as_frame=True)
Xr = db.data
yr = db.target

# Use a single feature for clear plotting (choose bmi or a high-correlated feature)
feat_name = "bmi" if "bmi" in Xr.columns else Xr.columns[0]
Xr_feat = Xr[[feat_name]]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr_feat, yr, test_size=0.25, random_state=42)

scaler_r = StandardScaler().fit(Xr_train)
Xr_train_s = scaler_r.transform(Xr_train)
Xr_test_s = scaler_r.transform(Xr_test)

br = BayesianRidge()
br.fit(Xr_train_s, yr_train)
# predict with std
y_mean, y_std = br.predict(Xr_test_s, return_std=True)

# Save BayesianRidge model
joblib.dump(br, MODELS_DIR / "bayesian_ridge.joblib")

# Evaluate regression
mse = np.mean((y_mean - yr_test.values)**2)
with open(METRICS_FILE, "a", encoding="utf-8") as f:
    f.write("Regression (BayesianRidge on diabetes - single feature '{}')\n".format(feat_name))
    f.write(f"  Test MSE (BayesianRidge mean) : {mse:.4f}\n")
    f.write(f"  Mean predictive std            : {np.mean(y_std):.4f}\n\n")

# Plot regression predictions with uncertainty bands
# prepare sorted by feature value for nice line plot
order_r = np.argsort(Xr_test[feat_name].values.flatten())
x_plot = Xr_test[feat_name].values.flatten()[order_r]
y_mean_plot = y_mean[order_r]
y_std_plot = y_std[order_r]
y_true_plot = yr_test.values.flatten()[order_r]

plt.figure(figsize=(8,6))
plt.plot(x_plot, y_true_plot, 'o', alpha=0.6, label='true')
plt.plot(x_plot, y_mean_plot, '-', label='predicted mean')
plt.fill_between(x_plot, y_mean_plot - 2*y_std_plot, y_mean_plot + 2*y_std_plot, alpha=0.2, label='±2 std (95% approx)')
plt.xlabel(feat_name)
plt.ylabel("target")
plt.title("Day23 — BayesianRidge predictions with uncertainty (diabetes dataset)")
plt.legend()
plt.tight_layout()
plt.savefig(REG_PLOT, dpi=150)
plt.close()
print(f"Saved regression BayesianRidge uncertainty plot to {REG_PLOT}")

with open(METRICS_FILE, "a", encoding="utf-8") as f:
    f.write("Saved regression plot: {}\n\n".format(REG_PLOT))

# --- 3) Optional: MC Dropout (classification) if TF/Keras is available ---
if TF_AVAILABLE:
    try:
        print("TensorFlow detected. Running small MC Dropout demo (this is optional and fast for small data).")
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        # Prepare small neural net on breast_cancer (using scaled features)
        Xtr = Xc_train_s
        Xte = Xc_test_s
        ytr = yc_train.values
        yte = yc_test.values

        def build_mc_model(input_dim, dropout_rate=0.5):
            model = Sequential([
                Dense(64, activation='relu', input_shape=(input_dim,)),
                Dropout(dropout_rate),
                Dense(32, activation='relu'),
                Dropout(dropout_rate),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
            return model

        mc_model = build_mc_model(Xtr.shape[1], dropout_rate=0.3)
        # train quickly (few epochs)
        mc_model.fit(Xtr, ytr, epochs=15, batch_size=32, verbose=0)

        # MC predictions: run model.predict with training=True many times to keep dropout active
        T = 50
        preds = []
        for t in range(T):
            p = mc_model(Xte, training=True).numpy().flatten()
            preds.append(p)
        preds = np.vstack(preds)  # T x n_samples
        mc_mean = preds.mean(axis=0)
        mc_std = preds.std(axis=0)

        # simple plot: mean ± std for first 80 sorted by std
        ord_mc = np.argsort(mc_std)[::-1][:80]
        plt.figure(figsize=(12,5))
        plt.errorbar(np.arange(len(ord_mc)), mc_mean[ord_mc], yerr=mc_std[ord_mc], fmt='o', alpha=0.8)
        plt.title("Day23 — MC Dropout predictive mean ± std (subset sorted by uncertainty)")
        plt.xlabel("sample (subset)")
        plt.ylabel("predicted probability")
        plt.tight_layout()
        plt.savefig(MC_DROP_PLOT, dpi=150)
        plt.close()
        joblib.dump(mc_model, MODELS_DIR / "mc_dropout_keras_model.joblib")
        with open(METRICS_FILE, "a", encoding="utf-8") as f:
            f.write("MC Dropout demo run and plot saved: {}\n".format(MC_DROP_PLOT))
        print(f"Saved MC Dropout plot to {MC_DROP_PLOT}")
    except Exception as e:
        print("MC Dropout demo failed:", e)
else:
    print("TensorFlow/Keras not available — skipping MC Dropout demo (optional).")

print("\nDay23 complete. Artifacts saved to:", OUT_DIR)
