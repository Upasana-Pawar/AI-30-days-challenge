"""
Day21: Stacked Ensemble of Tuned Boosters

- Builds a stacked ensemble using tuned boosters (if tuned artifacts from Day20 exist),
  else uses sensible default/tuned-like params for available boosters.
- Uses sklearn breast_cancer dataset for reproducible results.
- Evaluates individual models and stacked meta-model (accuracy, ROC-AUC).
- Saves artifacts: models, plots, metrics, and base-level predictions (meta-features).
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc

# --- Paths ---
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "day21_artifacts"
MODELS_DIR = OUT_DIR / "day21_models"
DAY20_MODELS_DIR = Path(__file__).resolve().parent.parent / "Day20_HyperparameterTuning_Boosters" / "day20_artifacts" / "day20_models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

METRICS_FILE = OUT_DIR / "day21_metrics.txt"
ACCURACY_PLOT = OUT_DIR / "day21_accuracy_comparison.png"
ROC_PLOT = OUT_DIR / "day21_roc_comparison.png"
BASE_PREDS_CSV = OUT_DIR / "day21_base_predictions.csv"

# --- Detect available boosters & try to load tuned models from Day20 if present ---
available = {}
try:
    import xgboost as xgb
    available['xgboost'] = xgb
except Exception:
    pass
try:
    import lightgbm as lgb
    available['lightgbm'] = lgb
except Exception:
    pass
try:
    import catboost as cat
    available['catboost'] = cat
except Exception:
    pass

# --- Load data ---
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# --- Utility to attempt to load a tuned model artifact from Day20 ---
def try_load_day20_model(name_key):
    """
    Given a model key like 'xgboost', try to load tuned artifact from Day20 folder.
    Expected filenames in Day20: '{Name}_tuned.joblib' where Name is e.g. 'XGBoost' or 'HistGB'
    """
    mapping = {
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "catboost": "CatBoost",
        "histgb": "HistGB"
    }
    model_name = mapping.get(name_key.lower())
    if not model_name:
        return None
    filepath = DAY20_MODELS_DIR / f"{model_name.lower()}_tuned.joblib"
    if filepath.exists():
        try:
            obj = joblib.load(filepath)
            print(f"Loaded tuned model from Day20: {filepath}")
            return obj
        except Exception as e:
            print(f"Failed to load {filepath}: {e}")
            return None
    return None

# --- Prepare base estimators list for stacking ---
estimators = []  # list of tuples (name, estimator)

# Helper: wrap an estimator in pipeline with scaler
def wrap_estimator(estimator, name):
    return (name, Pipeline([("scaler", StandardScaler()), ("clf", estimator)]))

# 1) Try to include tuned XGBoost
if 'xgboost' in available:
    # attempt to load tuned pipeline from Day20
    loaded = try_load_day20_model("xgboost")
    if loaded is not None:
        # If it's a pipeline with a 'clf' step or similar, use as-is
        # StackingClassifier expects estimators as (str, estimator). If loaded is pipeline, fine.
        estimators.append(("xgb_tuned", loaded))
        joblib.dump(loaded, MODELS_DIR / "xgb_tuned_used.joblib")
    else:
        # create a reasonable XGB default pipeline
        XGB = available['xgboost'].XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
        estimators.append(wrap_estimator(XGB, "xgb_default"))
        joblib.dump(wrap_estimator(XGB, "xgb_default")[1], MODELS_DIR / "xgb_default_used.joblib")

# 2) LightGBM
if 'lightgbm' in available:
    loaded = try_load_day20_model("lightgbm")
    if loaded is not None:
        estimators.append(("lgbm_tuned", loaded))
        joblib.dump(loaded, MODELS_DIR / "lgbm_tuned_used.joblib")
    else:
        LGB = available['lightgbm'].LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=42, n_jobs=-1)
        estimators.append(wrap_estimator(LGB, "lgbm_default"))
        joblib.dump(wrap_estimator(LGB, "lgbm_default")[1], MODELS_DIR / "lgbm_default_used.joblib")

# 3) CatBoost
if 'catboost' in available:
    loaded = try_load_day20_model("catboost")
    if loaded is not None:
        estimators.append(("cat_tuned", loaded))
        joblib.dump(loaded, MODELS_DIR / "cat_tuned_used.joblib")
    else:
        CAT = available['catboost'].CatBoostClassifier(iterations=300, learning_rate=0.05, verbose=0, random_seed=42)
        estimators.append(wrap_estimator(CAT, "cat_default"))
        joblib.dump(wrap_estimator(CAT, "cat_default")[1], MODELS_DIR / "cat_default_used.joblib")

# 4) Always include HistGradientBoosting (either tuned artifact or default)
loaded_hgb = try_load_day20_model("histgb")
if loaded_hgb is not None:
    estimators.append(("hgb_tuned", loaded_hgb))
    joblib.dump(loaded_hgb, MODELS_DIR / "hgb_tuned_used.joblib")
else:
    HGB = HistGradientBoostingClassifier(max_iter=200, random_state=42)
    estimators.append(wrap_estimator(HGB, "hgb_default"))
    joblib.dump(wrap_estimator(HGB, "hgb_default")[1], MODELS_DIR / "hgb_default_used.joblib")

# If for some reason there are duplicate pipelines (because loaded objects already include scaler),
# ensure the estimator objects are estimators (not tuples). StackingClassifier accepts estimators as (name, estimator)
# We'll normalize: if loaded object is pipeline, use it directly; if wrapper produced a tuple, use estimator.
normalized_estimators = []
for name, est in estimators:
    # if est is a tuple or list (from wrap_estimator mistake), fix it
    if isinstance(est, tuple) and len(est) == 2 and isinstance(est[1], Pipeline):
        normalized_estimators.append((name, est[1]))
    else:
        normalized_estimators.append((name, est))

# Keep only unique names (avoid duplicates)
seen = set()
final_estimators = []
for n,e in normalized_estimators:
    if n in seen:
        continue
    seen.add(n)
    final_estimators.append((n,e))

print("\nFinal base estimators for stacking:")
for n,_ in final_estimators:
    print(" -", n)

# --- If only one base estimator present, stacking isn't useful; just save that model as stacking fallback ---
if len(final_estimators) < 2:
    print("Warning: fewer than 2 base estimators available. Stacking requires multiple base models. Will skip stacking and evaluate available models.")
    stacking = None
else:
    # Build StackingClassifier with LogisticRegression meta-learner
    meta = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
    # pass through False to only use base predictions as meta-features
    stacking = StackingClassifier(estimators=final_estimators, final_estimator=meta, cv=5, n_jobs=-1, passthrough=False)

# --- Train/evaluate individual base estimators and stacking ---
results = {}
base_probas = {}

# Train & evaluate each base estimator (if not already fitted)
for name, est in final_estimators:
    print(f"\nFitting & evaluating base estimator: {name}")
    try:
        # if estimator is a Pipeline that may already be fitted (if loaded from joblib), check
        # We'll fit regardless to ensure consistency
        est.fit(X_train, y_train)
        y_pred = est.predict(X_test)
        try:
            y_proba = est.predict_proba(X_test)[:,1]
        except Exception:
            y_proba = est.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)
        results[name] = {"accuracy": acc, "roc_auc": roc}
        base_probas[name] = y_proba
        # save (overwrite) the used base model
        joblib.dump(est, MODELS_DIR / f"{name}_final.joblib")
        print(f"{name} -> acc: {acc:.4f}, roc_auc: {roc:.4f}")
    except Exception as e:
        print(f"Failed to fit/evaluate {name}: {e}")

# Train stacking if possible
if stacking is not None:
    print("\nTraining stacking classifier (meta-learner: LogisticRegression)...")
    stacking.fit(X_train, y_train)
    y_pred_stack = stacking.predict(X_test)
    y_proba_stack = stacking.predict_proba(X_test)[:,1]
    acc_stack = accuracy_score(y_test, y_pred_stack)
    roc_stack = roc_auc_score(y_test, y_proba_stack)
    results["stacking_tuned"] = {"accuracy": acc_stack, "roc_auc": roc_stack}
    base_probas["stacking_tuned"] = y_proba_stack
    # save stacking model
    joblib.dump(stacking, MODELS_DIR / "stacking_tuned.joblib")
    print(f"Stacking -> acc: {acc_stack:.4f}, roc_auc: {roc_stack:.4f}")

# --- Save base-level predictions for meta-inspection (columns: y_true, <base1>, <base2>, ...) ---
df_preds = pd.DataFrame({"y_true": y_test})
for name, probs in base_probas.items():
    df_preds[f"proba_{name}"] = probs
df_preds.to_csv(BASE_PREDS_CSV, index=False)
print(f"\nBase-level predictions saved to {BASE_PREDS_CSV}")

# --- Save metrics summary ---
with open(METRICS_FILE, "w", encoding="utf-8") as f:
    f.write("Day21 - Stacked Ensemble Metrics Summary\n")
    f.write("="*48 + "\n\n")
    for name, mets in results.items():
        f.write(f"{name}:\n")
        f.write(f"  Accuracy : {mets['accuracy']:.4f}\n")
        f.write(f"  ROC AUC  : {mets['roc_auc']:.4f}\n\n")
print(f"\nMetrics written to {METRICS_FILE}")

# --- Accuracy comparison plot ---
plt.figure(figsize=(10,5))
names = list(results.keys())
accs = [results[n]["accuracy"] for n in names]
sns.barplot(x=names, y=accs)
plt.ylim(0.8,1.0)
plt.title("Day21 — Accuracy Comparison (test set)")
plt.ylabel("Accuracy")
plt.xlabel("")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(ACCURACY_PLOT, dpi=150)
plt.close()
print(f"Accuracy plot saved to {ACCURACY_PLOT}")

# --- ROC curves plot ---
plt.figure(figsize=(8,6))
for name, probs in base_probas.items():
    try:
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_val:.3f})")
    except Exception:
        continue
plt.plot([0,1],[0,1], linestyle="--", color="grey", linewidth=1)
plt.xlim([0.0,1.0]); plt.ylim([0.0,1.05])
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("Day21 — ROC Comparison (test set)")
plt.legend(loc="lower right", fontsize="small")
plt.tight_layout()
plt.savefig(ROC_PLOT, dpi=150)
plt.close()
print(f"ROC plot saved to {ROC_PLOT}")

print("\nDone. All artifacts are in:", OUT_DIR)
