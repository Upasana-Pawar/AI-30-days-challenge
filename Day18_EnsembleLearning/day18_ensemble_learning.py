"""
Day18: Ensemble Learning (Bagging, Boosting, Stacking)
Run on Windows PowerShell with your .venv activated.

What the script does:
- Loads sklearn's breast_cancer dataset (binary classification)
- Trains:
    - RandomForestClassifier (bagging-like)
    - AdaBoostClassifier (boosting)
    - GradientBoostingClassifier (boosting)
    - StackingClassifier (stacking of logistic/rf/gb with meta-learner)
- Evaluates using accuracy and ROC AUC
- Plots accuracy comparison and ROC curves
- Saves models (joblib) and plots to disk
- Writes a short metrics summary to day18_metrics.txt
"""

import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, classification_report

# --- Config / Paths ---
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "day18_artifacts"
MODELS_DIR = OUT_DIR / "day18_models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_FILE = OUT_DIR / "day18_metrics.txt"
ACCURACY_PLOT = OUT_DIR / "day18_accuracy_comparison.png"
ROC_PLOT = OUT_DIR / "day18_roc_comparison.png"

# --- Data load & split ---
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target
feature_names = X.columns.tolist()

# Train / test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# --- Define models (with pipelines for scaling where helpful) ---
# Random Forest (bagging-like)
rf_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
])

# AdaBoost (boosting)
ada_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("ada", AdaBoostClassifier(n_estimators=100, random_state=42))
])

# Gradient Boosting (boosting)
gb_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("gb", GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42))
])

# Stacking: base learners -> meta learner
estimators = [
    ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
    ("gb", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)),
    ("lr", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42))
]
stack = Pipeline([
    ("scaler", StandardScaler()),
    ("stack", StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=5, n_jobs=-1, passthrough=False))
])

models = {
    "RandomForest": rf_pipe,
    "AdaBoost": ada_pipe,
    "GradientBoosting": gb_pipe,
    "Stacking": stack
}

# --- Train models and evaluate ---
results = {}
y_test_probas = {}

for name, model in models.items():
    print(f"\nTraining {name} ...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # For ROC AUC: use predict_proba when available; else use decision_function fallback
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # Some estimators may have decision_function
        try:
            y_proba = model.decision_function(X_test)
            # If decision_function returns shape (n_samples,), convert via sigmoid-ish approx (but roc_auc works on scores)
        except Exception:
            # fallback: convert predictions to 0/1
            y_proba = y_pred

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    results[name] = {"accuracy": acc, "roc_auc": roc_auc}
    y_test_probas[name] = y_proba

    # save model
    joblib.dump(model, MODELS_DIR / f"{name.lower()}.joblib")
    print(f"{name} -> acc: {acc:.4f}, roc_auc: {roc_auc:.4f}. Saved to {MODELS_DIR / (name.lower() + '.joblib')}")

# --- Save metrics summary ---
with open(METRICS_FILE, "w", encoding="utf-8") as f:
    f.write("Day18 - Ensemble Learning Metrics Summary\n")
    f.write("="*48 + "\n\n")
    for name, mets in results.items():
        f.write(f"{name}:\n")
        f.write(f"  Accuracy : {mets['accuracy']:.4f}\n")
        f.write(f"  ROC AUC  : {mets['roc_auc']:.4f}\n\n")

print(f"\nMetrics saved to {METRICS_FILE}")

# --- Plot accuracy comparison ---
plt.figure(figsize=(8,5))
sns.barplot(x=[k for k in results.keys()], y=[results[k]["accuracy"] for k in results.keys()])
plt.ylim(0.8, 1.0)
plt.title("Day18 — Accuracy Comparison (test set)")
plt.ylabel("Accuracy")
plt.xlabel("")
plt.tight_layout()
plt.savefig(ACCURACY_PLOT, dpi=150)
plt.close()
print(f"Accuracy plot saved to {ACCURACY_PLOT}")

# --- Plot ROC curves ---
plt.figure(figsize=(8,6))
for name, probs in y_test_probas.items():
    # handle constant predictions
    try:
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_val:.3f})")
    except ValueError:
        # if only one class present in y_test for predictions - skip plotting
        print(f"Skipping ROC for {name} (unable to compute ROC curve).")

plt.plot([0,1],[0,1], linestyle="--", color="grey", linewidth=1)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Day18 — ROC Comparison (test set)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(ROC_PLOT, dpi=150)
plt.close()
print(f"ROC plot saved to {ROC_PLOT}")

# --- Extra: feature importances for tree-based models (RF & GB) ---
fi_file = OUT_DIR / "day18_feature_importances.csv"
fi_rows = []
for name in ["RandomForest", "GradientBoosting"]:
    model = joblib.load(MODELS_DIR / f"{name.lower()}.joblib")
    # model may be pipeline -> get last step
    if isinstance(model, Pipeline):
        core = model.named_steps[list(model.named_steps.keys())[-1]]
    else:
        core = model
    if hasattr(core, "feature_importances_"):
        importances = core.feature_importances_
        for feat, imp in zip(feature_names, importances):
            fi_rows.append({"model": name, "feature": feat, "importance": imp})

if fi_rows:
    df_fi = pd.DataFrame(fi_rows).sort_values(["model","importance"], ascending=[True, False])
    df_fi.to_csv(fi_file, index=False)
    print(f"Feature importances saved to {fi_file}")

print("\nDone.")
