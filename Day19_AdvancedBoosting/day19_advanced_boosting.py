"""
Day19: Advanced Boosting (XGBoost, LightGBM, CatBoost)

- Trains and compares XGBoost, LightGBM, CatBoost (if installed).
- Falls back to sklearn's HistGradientBoostingClassifier if optional libs are missing.
- Uses sklearn breast_cancer dataset for reproducible results.
- Saves models, plots, and metrics to disk.
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

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc

# Attempt imports for optional high-performance boosters
available_boosters = {}
try:
    import xgboost as xgb  # pip install xgboost
    available_boosters['xgboost'] = xgb
except Exception:
    print("XGBoost not available in this environment (skipping).")

try:
    import lightgbm as lgb  # pip install lightgbm
    available_boosters['lightgbm'] = lgb
except Exception:
    print("LightGBM not available in this environment (skipping).")

try:
    import catboost as cat  # pip install catboost
    available_boosters['catboost'] = cat
except Exception:
    print("CatBoost not available in this environment (skipping).")

from sklearn.ensemble import HistGradientBoostingClassifier

# --- Paths ---
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "day19_artifacts"
MODELS_DIR = OUT_DIR / "day19_models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

METRICS_FILE = OUT_DIR / "day19_metrics.txt"
ACCURACY_PLOT = OUT_DIR / "day19_accuracy_comparison.png"
ROC_PLOT = OUT_DIR / "day19_roc_comparison.png"
FI_FILE = OUT_DIR / "day19_feature_importances.csv"

# --- Load data ---
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# --- Model definitions (with reasonable default hyperparams) ---
models = {}

# HistGradientBoosting (sklearn) — always available
models['HistGB'] = Pipeline([
    ("scaler", StandardScaler()),
    ("hgb", HistGradientBoostingClassifier(max_iter=200, random_state=42))
])

# XGBoost (if available)
if 'xgboost' in available_boosters:
    # use scikit-learn wrapper
    XGBClassifier = available_boosters['xgboost'].XGBClassifier
    models['XGBoost'] = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs= -1))
    ])

# LightGBM (if available)
if 'lightgbm' in available_boosters:
    LGBMClassifier = available_boosters['lightgbm'].LGBMClassifier
    models['LightGBM'] = Pipeline([
        ("scaler", StandardScaler()),
        ("lgbm", LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=42, n_jobs=-1))
    ])

# CatBoost (if available)
if 'catboost' in available_boosters:
    CatBoostClassifier = available_boosters['catboost'].CatBoostClassifier
    # CatBoost handles scaling internally but we keep a scaler for consistent pipelines
    models['CatBoost'] = Pipeline([
        ("scaler", StandardScaler()),
        ("cat", CatBoostClassifier(iterations=300, learning_rate=0.05, verbose=0, random_seed=42))
    ])

if not models:
    raise RuntimeError("No models configured. This should not happen (HistGB always added).")

# --- Train & evaluate ---
results = {}
probas = {}

for name, pipeline in models.items():
    print(f"\nTraining {name} ...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    # get probability or fallback to decision_function/predictions
    if hasattr(pipeline, "predict_proba"):
        try:
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        except Exception:
            # sometimes pipeline doesn't expose predict_proba; fetch from final estimator
            core = pipeline.named_steps[list(pipeline.named_steps.keys())[-1]]
            if hasattr(core, "predict_proba"):
                y_proba = core.predict_proba(pipeline.named_steps.get('scaler', lambda x: x).transform(X_test))[:,1]
            else:
                y_proba = y_pred
    else:
        # fallback
        try:
            core = pipeline.named_steps[list(pipeline.named_steps.keys())[-1]]
            if hasattr(core, "predict_proba"):
                y_proba = core.predict_proba(X_test)[:, 1]
            else:
                y_proba = y_pred
        except Exception:
            y_proba = y_pred

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    results[name] = {"accuracy": acc, "roc_auc": roc_auc}
    probas[name] = y_proba

    # save model artifact
    joblib.dump(pipeline, MODELS_DIR / f"{name.lower()}.joblib")
    print(f"{name} trained. Accuracy: {acc:.4f}, ROC AUC: {roc_auc:.4f}. Model saved to {MODELS_DIR / (name.lower() + '.joblib')}")

# --- Save metrics ---
with open(METRICS_FILE, "w", encoding="utf-8") as f:
    f.write("Day19 - Advanced Boosting Metrics Summary\n")
    f.write("="*56 + "\n\n")
    for name,m in results.items():
        f.write(f"{name}:\n")
        f.write(f"  Accuracy : {m['accuracy']:.4f}\n")
        f.write(f"  ROC AUC  : {m['roc_auc']:.4f}\n\n")
print(f"\nMetrics written to {METRICS_FILE}")

# --- Accuracy bar plot ---
plt.figure(figsize=(8,5))
sns.barplot(x=list(results.keys()), y=[results[k]['accuracy'] for k in results.keys()])
plt.ylim(0.8, 1.0)
plt.title("Day19 — Accuracy Comparison (test set)")
plt.ylabel("Accuracy")
plt.xlabel("")
plt.tight_layout()
plt.savefig(ACCURACY_PLOT, dpi=150)
plt.close()
print(f"Accuracy plot saved to {ACCURACY_PLOT}")

# --- ROC curves ---
plt.figure(figsize=(8,6))
for name, y_proba in probas.items():
    try:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_val:.3f})")
    except Exception as e:
        print(f"Could not compute ROC for {name}: {e}")

plt.plot([0,1],[0,1], linestyle="--", linewidth=1)
plt.xlim([0.0,1.0]); plt.ylim([0.0,1.05])
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("Day19 — ROC Comparison (test set)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(ROC_PLOT, dpi=150)
plt.close()
print(f"ROC plot saved to {ROC_PLOT}")

# --- Optional: Feature importances for models that provide them ---
fi_rows = []
for name in models.keys():
    try:
        pipeline = joblib.load(MODELS_DIR / f"{name.lower()}.joblib")
        core = pipeline.named_steps[list(pipeline.named_steps.keys())[-1]]
        if hasattr(core, "feature_importances_"):
            importances = core.feature_importances_
            for feat, imp in zip(feature_names, importances):
                fi_rows.append({"model": name, "feature": feat, "importance": imp})
        elif hasattr(core, "get_feature_importance"):  # CatBoost
            vals = core.get_feature_importance()
            for feat, imp in zip(feature_names, vals):
                fi_rows.append({"model": name, "feature": feat, "importance": imp})
    except Exception:
        continue

if fi_rows:
    df_fi = pd.DataFrame(fi_rows).sort_values(["model","importance"], ascending=[True, False])
    df_fi.to_csv(FI_FILE, index=False)
    print(f"Feature importances saved to {FI_FILE}")

print("\nDay19 complete.")
