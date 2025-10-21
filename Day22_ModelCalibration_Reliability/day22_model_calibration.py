"""
Day22: Model Calibration & Reliability

- Trains a few classifiers (RandomForest, GradientBoosting, Stacking)
- Calibrates each using Platt scaling (sigmoid) and Isotonic regression via CalibratedClassifierCV
- Evaluates using:
    - Reliability diagram (calibration curves)
    - Brier score
    - Expected Calibration Error (ECE)
- Saves calibrated models, plots and summary metrics.

Run: PowerShell with your .venv activated (see README)
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

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, accuracy_score, roc_auc_score

# --- Paths ---
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "day22_artifacts"
MODELS_DIR = OUT_DIR / "day22_models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RELIABILITY_PLOT = OUT_DIR / "day22_reliability_diagram.png"
Brier_ECE_CSV = OUT_DIR / "day22_brier_ece.csv"
METRICS_FILE = OUT_DIR / "day22_metrics.txt"

# --- Helper: ECE calculation ---
def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Compute Expected Calibration Error (ECE) with n_bins equal-width bins.
    y_true: binary true labels (0/1)
    y_prob: predicted probability for positive class
    """
    bins = np.linspace(0.0, 1.0, n_bins+1)
    bin_idx = np.digitize(y_prob, bins, right=True) - 1  # bin indices 0..n_bins-1
    ece = 0.0
    for i in range(n_bins):
        bin_mask = bin_idx == i
        if np.sum(bin_mask) == 0:
            continue
        bin_size = np.sum(bin_mask)
        avg_prob = np.mean(y_prob[bin_mask])
        avg_true = np.mean(y_true[bin_mask])
        ece += (bin_size / len(y_prob)) * abs(avg_prob - avg_true)
    return ece

# --- Load data ---
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# --- Define base models (pipelines w/ scaler) ---
rf_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
])

gb_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("gb", GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42))
])

# Stacking as another base (RF + GB -> LR meta)
estimators = [
    ("rf_base", RandomForestClassifier(n_estimators=100, random_state=42)),
    ("gb_base", GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, random_state=42))
]
stack_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("stack", StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000), cv=5, n_jobs=-1, passthrough=False))
])

models = {
    "RandomForest": rf_pipe,
    "GradientBoosting": gb_pipe,
    "Stacking": stack_pipe
}

# --- Train uncalibrated models and save ---
trained_models = {}
metrics_summary = []

for name, pipeline in models.items():
    print(f"\nTraining uncalibrated {name} ...")
    pipeline.fit(X_train, y_train)
    # save uncalibrated
    joblib.dump(pipeline, MODELS_DIR / f"{name.lower()}_uncal.joblib")
    trained_models[f"{name}_uncal"] = pipeline

# --- Calibrate each model with two methods: 'sigmoid' (Platt) and 'isotonic' ---
calibrated_models = {}
methods = ["sigmoid", "isotonic"]

# For isotonic, sklearn requires sufficient samples; we'll attempt it and fallback gracefully on failure.
for base_name, pipeline in list(trained_models.items()):
    for method in methods:
        label = f"{base_name.split('_')[0]}_{method}"
        print(f"Calibrating {base_name} with {method} ...")
        try:
            # CalibratedClassifierCV expects an estimator; pipeline is fine.
            # Use cv=5 to hold-out folds for calibration (not the test set)
            calib = CalibratedClassifierCV(base_estimator=pipeline, method=method, cv=5)
            calib.fit(X_train, y_train)
            calibrated_models[label] = calib
            joblib.dump(calib, MODELS_DIR / f"{label.lower()}.joblib")
            print(f"Saved calibrated model: {label}")
        except Exception as e:
            print(f"Calibration with method={method} failed for {base_name}: {e}")

# --- Evaluate: Brier score, ECE, Accuracy, ROC-AUC --- 
# We'll evaluate uncalibrated and calibrated models on the test set.
eval_rows = []
probs_for_plot = {}  # name -> (y_prob, y_true)

for name, pipeline in trained_models.items():
    model_label = name  # e.g., "RandomForest_uncal"
    try:
        y_pred = pipeline.predict(X_test)
        try:
            y_prob = pipeline.predict_proba(X_test)[:,1]
        except Exception:
            # fallback
            y_prob = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        ece = expected_calibration_error(y_test.values if hasattr(y_test, "values") else y_test, np.array(y_prob), n_bins=10)
        eval_rows.append({
            "model": model_label,
            "method": "uncalibrated",
            "accuracy": acc,
            "roc_auc": roc,
            "brier": brier,
            "ece": ece
        })
        probs_for_plot[model_label] = (y_prob, y_test)
        print(f"{model_label} -> acc: {acc:.4f}, roc_auc: {roc:.4f}, brier: {brier:.4f}, ece: {ece:.4f}")
    except Exception as e:
        print(f"Failed evaluation for {model_label}: {e}")

for name, calib in calibrated_models.items():
    try:
        y_prob = calib.predict_proba(X_test)[:,1]
        y_pred = (y_prob >= 0.5).astype(int)
        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        ece = expected_calibration_error(y_test.values if hasattr(y_test, "values") else y_test, np.array(y_prob), n_bins=10)
        eval_rows.append({
            "model": name,
            "method": "calibrated",
            "accuracy": acc,
            "roc_auc": roc,
            "brier": brier,
            "ece": ece
        })
        probs_for_plot[name] = (y_prob, y_test)
        print(f"{name} (calibrated) -> acc: {acc:.4f}, roc_auc: {roc:.4f}, brier: {brier:.4f}, ece: {ece:.4f}")
    except Exception as e:
        print(f"Failed evaluation for calibrated model {name}: {e}")

# --- Save metrics CSV & text summary ---
df_eval = pd.DataFrame(eval_rows).sort_values(["model", "method"])
df_eval.to_csv(Brier_ECE_CSV, index=False)

with open(METRICS_FILE, "w", encoding="utf-8") as f:
    f.write("Day22 - Model Calibration Metrics Summary\n")
    f.write("="*56 + "\n\n")
    for _, row in df_eval.iterrows():
        f.write(f"{row['model']} ({row['method']}):\n")
        f.write(f"  Accuracy : {row['accuracy']:.4f}\n")
        f.write(f"  ROC AUC  : {row['roc_auc']:.4f}\n")
        f.write(f"  Brier    : {row['brier']:.4f}\n")
        f.write(f"  ECE      : {row['ece']:.4f}\n\n")

print(f"\nSaved metrics to {Brier_ECE_CSV} and text summary to {METRICS_FILE}")

# --- Reliability diagram (calibration curves) --- 
plt.figure(figsize=(10,7))
n_bins = 10
for label, (y_prob, y_true) in probs_for_plot.items():
    try:
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
        plt.plot(mean_pred, frac_pos, marker='o', label=label)
    except Exception as e:
        print(f"Could not plot calibration curve for {label}: {e}")

plt.plot([0,1],[0,1], linestyle='--', color='grey', label='Perfectly calibrated')
plt.title("Day22 — Reliability Diagram (calibration curves)")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.legend(fontsize='small', loc='lower right')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(RELIABILITY_PLOT, dpi=150)
plt.close()
print(f"Reliability diagram saved to {RELIABILITY_PLOT}")

print("\nArtifacts saved in:", OUT_DIR)
print("Calibrated model files are in:", MODELS_DIR)
print("\nDone.")
