"""
Day25: Interpretable Probabilistic Models & Decision Thresholding (Updated for sklearn compatibility)

- Train RandomForest classifier on breast_cancer dataset.
- Calibrate probabilities with Platt scaling (CalibratedClassifierCV).
- Show:
    1) Cost-aware threshold selection.
    2) Expected-value decision rule.
    3) Selective prediction (reject/abstain).
    4) Reliability diagram.
    5) Simple human-readable rules.

Compatible with scikit-learn versions using either 'estimator' or 'base_estimator' param.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import inspect
warnings.filterwarnings("ignore")

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, brier_score_loss

# --- Paths ---
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "day25_artifacts"
MODELS_DIR = OUT_DIR / "day25_models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

COST_PLOT = OUT_DIR / "day25_cost_vs_threshold.png"
REJECT_PLOT = OUT_DIR / "day25_reject_coverage_accuracy.png"
RELIABILITY_PLOT = OUT_DIR / "day25_reliability_diagram.png"
METRICS_FILE = OUT_DIR / "day25_metrics.txt"
RULES_FILE = OUT_DIR / "day25_decision_rules.txt"

# --- Data ---
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# --- Model pipeline ---
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
])

# ✅ Compatible CalibratedClassifierCV initialization (estimator vs base_estimator)
calib_kwargs = {"method": "sigmoid", "cv": 5}
sig = inspect.signature(CalibratedClassifierCV)
params = sig.parameters

if "estimator" in params:
    calib = CalibratedClassifierCV(estimator=pipeline, **calib_kwargs)
elif "base_estimator" in params:
    calib = CalibratedClassifierCV(base_estimator=pipeline, **calib_kwargs)
else:
    try:
        calib = CalibratedClassifierCV(estimator=pipeline, **calib_kwargs)
    except TypeError:
        calib = CalibratedClassifierCV(base_estimator=pipeline, **calib_kwargs)

# --- Train calibrated model ---
calib.fit(X_train, y_train)
joblib.dump(calib, MODELS_DIR / "rf_calibrated.joblib")

# --- Predict probabilities ---
probs = calib.predict_proba(X_test)[:, 1]
y_pred_default = (probs >= 0.5).astype(int)

acc_default = accuracy_score(y_test, y_pred_default)
roc_default = roc_auc_score(y_test, probs)
brier_default = brier_score_loss(y_test, probs)

# --- Reliability diagram ---
plt.figure(figsize=(8, 6))
frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10, strategy="uniform")
plt.plot(mean_pred, frac_pos, "o-", label="Calibrated RF")
plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Day25 — Reliability Diagram")
plt.legend()
plt.tight_layout()
plt.savefig(RELIABILITY_PLOT, dpi=150)
plt.close()

# --- Cost-aware threshold selection ---
cost_fp, cost_fn = 1.0, 5.0
thresholds = np.linspace(0, 1, 201)
costs, metrics = [], []

for t in thresholds:
    preds = (probs >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    total_cost = fp * cost_fp + fn * cost_fn
    costs.append(total_cost / len(y_test))
    acc = (tp + tn) / (tp + tn + fp + fn)
    metrics.append({
        "threshold": t, "avg_cost": total_cost / len(y_test), "accuracy": acc,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    })

df_metrics = pd.DataFrame(metrics)
min_idx = np.argmin(costs)
best_thresh = thresholds[min_idx]

plt.figure(figsize=(8, 5))
plt.plot(thresholds, costs, label=f"avg cost per sample (FP={cost_fp}, FN={cost_fn})")
plt.axvline(best_thresh, ls="--", color="red", label=f"min cost t={best_thresh:.3f}")
plt.xlabel("Decision threshold")
plt.ylabel("Avg cost per sample")
plt.title("Day25 — Cost vs Decision Threshold")
plt.legend()
plt.tight_layout()
plt.savefig(COST_PLOT, dpi=150)
plt.close()

# --- Expected value decision rule ---
B, C = 10.0, 2.0
ev_threshold = C / (B + C)
ev_rule_text = f"If P(positive) >= {ev_threshold:.3f}, take action. (B={B}, C={C})"

# --- Selective prediction (reject/abstain) ---
conf_thresholds = np.linspace(0.5, 0.995, 50)
coverage, accepted_accuracy = [], []

for ct in conf_thresholds:
    accept_mask = np.maximum(probs, 1 - probs) >= ct
    accepted = probs[accept_mask]
    y_acc = y_test.values[accept_mask]
    acc = np.nan if len(y_acc) == 0 else accuracy_score(y_acc, (accepted >= 0.5).astype(int))
    coverage.append(accept_mask.mean())
    accepted_accuracy.append(acc)

plt.figure(figsize=(8, 5))
plt.plot(coverage, accepted_accuracy, "o-")
plt.xlabel("Coverage (fraction accepted)")
plt.ylabel("Accuracy on accepted samples")
plt.title("Day25 — Selective Prediction: Coverage vs Accuracy")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(REJECT_PLOT, dpi=150)
plt.close()

# --- Human-readable rules ---
try:
    rf_model = pipeline.named_steps["rf"]
    importances = rf_model.feature_importances_
    top_idx = np.argsort(importances)[-3:][::-1]
    top_feats = [(feature_names[i], importances[i]) for i in top_idx]

    with open(RULES_FILE, "w", encoding="utf-8") as f:
        f.write("Day25 — Simple decision rules (median splits from top features)\n")
        f.write("=" * 70 + "\n\n")
        for feat, imp in top_feats:
            thr = X_train[feat].median()
            f.write(f"If {feat} >= {thr:.4f}, increase suspicion (importance={imp:.4f})\n")
        f.write("\nExpected-value rule:\n" + ev_rule_text + "\n")
except Exception as e:
    with open(RULES_FILE, "w", encoding="utf-8") as f:
        f.write(f"Failed to extract rules: {e}\n")

# --- Save summary metrics ---
with open(METRICS_FILE, "w", encoding="utf-8") as f:
    f.write("Day25 - Interpretable Probabilistic Models & Decision Thresholding\n")
    f.write("=" * 72 + "\n\n")
    f.write(f"Default threshold (0.5) accuracy : {acc_default:.4f}\n")
    f.write(f"Default ROC AUC                 : {roc_default:.4f}\n")
    f.write(f"Default Brier score             : {brier_default:.4f}\n\n")
    f.write(f"Cost-aware threshold (FP={cost_fp}, FN={cost_fn}):\n")
    f.write(f"  Best threshold: {best_thresh:.3f}\n")
    f.write(f"  Min avg cost  : {costs[min_idx]:.5f}\n\n")
    f.write(f"Expected-value rule: {ev_rule_text}\n\n")
    f.write("Selective prediction (coverage → acc):\n")
    for cov, acc in zip(coverage[::5], accepted_accuracy[::5]):
        f.write(f"  coverage={cov:.3f} -> acc={acc if not np.isnan(acc) else 'NA'}\n")
    f.write(f"\nRules saved to: {RULES_FILE}\n")

df_metrics.to_csv(OUT_DIR / "day25_threshold_metrics.csv", index=False)
print("\n✅ Done! All artifacts saved in:", OUT_DIR)
