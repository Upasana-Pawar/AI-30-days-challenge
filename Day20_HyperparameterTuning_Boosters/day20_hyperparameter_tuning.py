"""
Day20: Hyperparameter Tuning for Boosters

- Tries to use Optuna for tuning (recommended).
- Falls back to RandomizedSearchCV if Optuna isn't installed.
- Tunes XGBoost, LightGBM, CatBoost, and sklearn HistGradientBoosting where available.
- Uses sklearn breast_cancer dataset for reproducibility.
- Saves tuned models, comparison plots, metrics, and Optuna study (if used).

Notes:
- This script is defensive: it will run even if optional libs (xgboost/lightgbm/catboost/optuna) are missing.
- For speed on local runs, Optuna is limited to `n_trials=25` by default; increase if you want deeper tuning.
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
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc

# --- Optional libraries detection ---
optuna_available = False
xgb_available = False
lgb_available = False
cat_available = False

try:
    import optuna
    optuna_available = True
except Exception:
    print("Optuna not installed — RandomizedSearchCV fallback will be used.")

try:
    import xgboost as xgb
    xgb_available = True
except Exception:
    print("XGBoost not installed — skipping XGBoost.")

try:
    import lightgbm as lgb
    lgb_available = True
except Exception:
    print("LightGBM not installed — skipping LightGBM.")

try:
    import catboost as cat
    cat_available = True
except Exception:
    print("CatBoost not installed — skipping CatBoost.")

from sklearn.ensemble import HistGradientBoostingClassifier

# --- Paths ---
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "day20_artifacts"
MODELS_DIR = OUT_DIR / "day20_models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

METRICS_FILE = OUT_DIR / "day20_metrics.txt"
ACCURACY_PLOT = OUT_DIR / "day20_accuracy_comparison.png"
ROC_PLOT = OUT_DIR / "day20_roc_comparison.png"
OPTUNA_STUDY_FILE = OUT_DIR / "day20_optuna_study.pkl"
OPTUNA_HISTORY_PLOT = OUT_DIR / "day20_optuna_history.png"

# --- Data ---
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# --- Model wrappers & hyperparameter search spaces ---
models_config = {}

# 1) HistGradientBoosting (always available)
from sklearn.ensemble import HistGradientBoostingClassifier
models_config['HistGB'] = {
    "constructor": lambda params=None: HistGradientBoostingClassifier(**(params or {"max_iter":200, "random_state":42})),
    "space": {
        "max_iter": [100, 150, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [None, 3, 5, 8],
        "min_samples_leaf": [1, 5, 10]
    }
}

# 2) XGBoost (if available)
if xgb_available:
    XGBClassifier = xgb.XGBClassifier
    models_config['XGBoost'] = {
        "constructor": lambda params=None: XGBClassifier(use_label_encoder=False, eval_metric='logloss', **(params or {"n_estimators":200, "random_state":42, "n_jobs":-1})),
        "space": {
            "n_estimators": [100, 150, 200, 300],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "max_depth": [3, 4, 6, 8],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.5, 0.7, 1.0]
        }
    }

# 3) LightGBM
if lgb_available:
    LGBMClassifier = lgb.LGBMClassifier
    models_config['LightGBM'] = {
        "constructor": lambda params=None: LGBMClassifier(**(params or {"n_estimators":200, "random_state":42, "n_jobs":-1})),
        "space": {
            "n_estimators": [100, 150, 200, 300],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "num_leaves": [15, 31, 63],
            "subsample": [0.6, 0.8, 1.0]
        }
    }

# 4) CatBoost
if cat_available:
    CatBoostClassifier = cat.CatBoostClassifier
    models_config['CatBoost'] = {
        "constructor": lambda params=None: CatBoostClassifier(verbose=0, **(params or {"iterations":300, "learning_rate":0.05, "random_seed":42})),
        "space": {
            "iterations": [200, 300, 400],
            "learning_rate": [0.01, 0.03, 0.05],
            "depth": [4,6,8]
        }
    }

if not models_config:
    raise RuntimeError("No models configured to tune. At least HistGB should be available.")

# --- Utility functions ---
def random_param_sample(space, n_iter=10, random_state=42):
    """Create a list of random parameter dicts sampled from the given 'space' dict.
       space values are lists of candidates."""
    rng = np.random.RandomState(random_state)
    keys = list(space.keys())
    samples = []
    for _ in range(n_iter):
        d = {}
        for k in keys:
            vals = space[k]
            d[k] = rng.choice(vals)
        samples.append(d)
    return samples

def evaluate_model_pipeline(pipeline, X_test, y_test):
    """Return accuracy, roc_auc and proba vector (for ROC)."""
    y_pred = pipeline.predict(X_test)
    # try predict_proba
    try:
        y_proba = pipeline.predict_proba(X_test)[:,1]
    except Exception:
        try:
            # try final estimator on scaled data if pipeline
            core = pipeline.named_steps[list(pipeline.named_steps.keys())[-1]]
            if hasattr(core, "predict_proba"):
                scaler = pipeline.named_steps.get('scaler', None)
                Xs = scaler.transform(X_test) if scaler else X_test
                y_proba = core.predict_proba(Xs)[:,1]
            else:
                y_proba = y_pred
        except Exception:
            y_proba = y_pred

    acc = accuracy_score(y_test, y_pred)
    try:
        roc = roc_auc_score(y_test, y_proba)
    except Exception:
        roc = float("nan")
    return acc, roc, y_proba

# --- Tuning logic ---
results = {}  # store default and tuned metrics
optuna_study = None
optuna_used = False
start_time = time.time()

for name, cfg in models_config.items():
    print(f"\n--- Processing {name} ---")
    # default model pipeline (scaler + estimator)
    default_estimator = cfg["constructor"]()
    default_pipe = Pipeline([("scaler", StandardScaler()), ("clf", default_estimator)])
    # train default
    default_pipe.fit(X_train, y_train)
    d_acc, d_roc, d_proba = evaluate_model_pipeline(default_pipe, X_test, y_test)
    print(f"{name} default -> acc: {d_acc:.4f}, roc_auc: {d_roc:.4f}")
    # save default model
    joblib.dump(default_pipe, MODELS_DIR / f"{name.lower()}_default.joblib")

    # tuning
    tuned_pipe = None
    tuned_best_params = None

    if optuna_available:
        # Use Optuna for tuning (small trial budget by default)
        optuna_used = True
        def objective(trial):
            # construct param dict by sampling from cfg["space"]
            params = {}
            for k, vals in cfg["space"].items():
                # try to infer type (int vs float vs categorical)
                # if vals look like floats -> sample uniform/log; else choose categorical from list
                if all(isinstance(v, int) for v in vals):
                    params[k] = trial.suggest_categorical(k, vals)
                elif all(isinstance(v, float) for v in vals):
                    # use log scale if range spans several orders
                    low, high = min(vals), max(vals)
                    if low > 0 and high / low >= 100:
                        params[k] = trial.suggest_loguniform(k, low, high)
                    else:
                        params[k] = trial.suggest_categorical(k, vals)
                else:
                    # mixed types -> categorical
                    params[k] = trial.suggest_categorical(k, vals)

            # build pipeline and cross-validate quickly using small cv
            estimator = cfg["constructor"](params)
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
            # Fast CV (3 folds)
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            rocs = []
            for train_idx, val_idx in cv.split(X_train, y_train):
                Xtr, Xval = X_train.iloc[train_idx], X_train.iloc[val_idx]
                ytr, yval = y_train.iloc[train_idx], y_train.iloc[val_idx]
                try:
                    pipe.fit(Xtr, ytr)
                    # get proba
                    try:
                        p = pipe.predict_proba(Xval)[:,1]
                    except Exception:
                        p = pipe.predict(Xval)
                    rocs.append(roc_auc_score(yval, p))
                except Exception:
                    return 0.0  # if fails, return poor score
            return float(np.mean(rocs))

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        n_trials = 25  # modest for local runs; increase if you want deeper tuning
        print(f"Running Optuna for {name} ({n_trials} trials)...")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best = study.best_params
        tuned_best_params = best
        print(f"Optuna best params for {name}: {best}")
        # fit tuned model on full train
        tuned_estimator = cfg["constructor"](best)
        tuned_pipe = Pipeline([("scaler", StandardScaler()), ("clf", tuned_estimator)])
        tuned_pipe.fit(X_train, y_train)
        # save study
        try:
            joblib.dump(study, OPTUNA_STUDY_FILE)
        except Exception:
            pass
        optuna_study = study

    else:
        # RandomizedSearchCV fallback (short search)
        print(f"Optuna not available — running RandomizedSearchCV for {name}")
        # create param distributions for RandomizedSearchCV
        param_dist = {}
        for k, vals in cfg["space"].items():
            param_dist[f"clf__{k}"] = vals
        # pipeline wrapper
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", cfg["constructor"]())])
        rs = RandomizedSearchCV(pipe, param_dist, n_iter=12, cv=3, scoring="roc_auc", random_state=42, n_jobs=-1)
        rs.fit(X_train, y_train)
        tuned_pipe = rs.best_estimator_
        # extract best params (strip clf__)
        tuned_best_params = {k.replace("clf__", ""): v for k, v in rs.best_params_.items()}
        print(f"RandomizedSearchCV best params for {name}: {tuned_best_params}")

    # Evaluate tuned model
    if tuned_pipe is not None:
        t_acc, t_roc, t_proba = evaluate_model_pipeline(tuned_pipe, X_test, y_test)
        print(f"{name} tuned -> acc: {t_acc:.4f}, roc_auc: {t_roc:.4f}")
        # save tuned model + record metrics
        joblib.dump(tuned_pipe, MODELS_DIR / f"{name.lower()}_tuned.joblib")
    else:
        t_acc, t_roc, t_proba = d_acc, d_roc, d_proba  # if tuning failed, use default

    # store results
    results[name] = {
        "default": {"accuracy": d_acc, "roc_auc": d_roc},
        "tuned": {"accuracy": t_acc, "roc_auc": t_roc},
        "tuned_params": tuned_best_params
    }
    # store probas for ROC plotting (use tuned if available)
    results[name]["proba"] = t_proba if tuned_pipe is not None else d_proba

end_time = time.time()
print(f"\nTuning & evaluation completed in {(end_time - start_time)/60:.2f} minutes")

# --- Save metrics summary ---
with open(METRICS_FILE, "w", encoding="utf-8") as f:
    f.write("Day20 - Hyperparameter Tuning Metrics Summary\n")
    f.write("="*56 + "\n\n")
    for name, info in results.items():
        f.write(f"{name}:\n")
        f.write(f"  Default Accuracy : {info['default']['accuracy']:.4f}\n")
        f.write(f"  Default ROC AUC  : {info['default']['roc_auc']:.4f}\n")
        f.write(f"  Tuned  Accuracy  : {info['tuned']['accuracy']:.4f}\n")
        f.write(f"  Tuned  ROC AUC   : {info['tuned']['roc_auc']:.4f}\n")
        f.write(f"  Tuned Params     : {info['tuned_params']}\n\n")

print(f"Metrics written to {METRICS_FILE}")

# --- Accuracy comparison plot (default vs tuned) ---
model_names = []
default_accs = []
tuned_accs = []
for name, info in results.items():
    model_names.append(name)
    default_accs.append(info['default']['accuracy'])
    tuned_accs.append(info['tuned']['accuracy'])

x = np.arange(len(model_names))
width = 0.35

plt.figure(figsize=(10,5))
plt.bar(x - width/2, default_accs, width, label='Default')
plt.bar(x + width/2, tuned_accs, width, label='Tuned')
plt.xticks(x, model_names)
plt.ylim(0.8, 1.0)
plt.ylabel("Accuracy")
plt.title("Day20 — Default vs Tuned Accuracy (test set)")
plt.legend()
plt.tight_layout()
plt.savefig(ACCURACY_PLOT, dpi=150)
plt.close()
print(f"Accuracy comparison plot saved to {ACCURACY_PLOT}")

# --- ROC curves for tuned models ---
plt.figure(figsize=(8,6))
for name, info in results.items():
    proba = info.get("proba", None)
    if proba is None:
        continue
    try:
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_val:.3f})")
    except Exception:
        continue

plt.plot([0,1],[0,1], linestyle="--", color="grey", linewidth=1)
plt.xlim([0.0,1.0]); plt.ylim([0.0,1.05])
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("Day20 — Tuned Models ROC Comparison (test set)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(ROC_PLOT, dpi=150)
plt.close()
print(f"ROC comparison plot saved to {ROC_PLOT}")

# --- Optional: Optuna study visualization (if optuna used) ---
if optuna_available and optuna_study is not None:
    try:
        import matplotlib.pyplot as plt
        fig = optuna.visualization.plot_optimization_history(optuna_study)
        # optuna returns a plotly figure — save static PNG via kaleido if available
        try:
            fig.write_image(str(OPTUNA_HISTORY_PLOT))
            print(f"Optuna history saved to {OPTUNA_HISTORY_PLOT}")
        except Exception:
            # fallback: save study as pickle only
            pass

    except Exception:
        pass

print("\nDay20 complete.")
