"""
Day29: Human-in-the-Loop Simulation & Audit Dashboard Metrics
============================================================

Simulate a human review workflow on top of a calibrated classifier:

Workflow:
- Train calibrated RandomForest classifier
- Produce predicted probabilities on test set
- Divide predictions into confidence buckets
- Human "reviews" uncertain predictions (threshold-based)
- Evaluate:
    • model accuracy before intervention
    • expected accuracy after intervention
    • how many samples a human must inspect
    • error reduction
    • hit-rate of human reviews (efficiency)
- Plot:
    • accuracy by probability bin
    • error rate by probability bin
    • cumulative error reduction curve

Artifacts are saved in ./day29_artifacts
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix

# ------------------- Paths -------------------
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "day29_artifacts"
MODELS_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ACCURACY_PLOT = OUT_DIR / "day29_accuracy_by_confidence.png"
ERROR_PLOT = OUT_DIR / "day29_error_rate_by_confidence.png"
CUM_PLOT = OUT_DIR / "day29_cumulative_error_reduction.png"
METRICS_FILE = OUT_DIR / "day29_metrics.txt"
REVIEW_CSV = OUT_DIR / "day29_review_table.csv"

CONFIDENCE_BUCKETS = np.linspace(0,1,11)  # 10 bins

# ------------------- Load Data -------------------
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# ------------------- Train Calibrated RF -------------------
rf = RandomForestClassifier(n_estimators=200, random_state=42)
pipe = Pipeline([("scaler", StandardScaler()), ("rf", rf)])
calib = CalibratedClassifierCV(pipe, method="sigmoid", cv=5)
calib.fit(X_train, y_train)
joblib.dump(calib, MODELS_DIR / "rf_calibrated.joblib")

probs = calib.predict_proba(X_test)[:,1]
preds = (probs >= 0.5).astype(int)
baseline_acc = accuracy_score(y_test, preds)

# ------------------- Confidence Bucketing -------------------
bucket_idx = np.digitize(probs, CONFIDENCE_BUCKETS) - 1
df = pd.DataFrame({
    "true": y_test.values,
    "pred": preds,
    "prob": probs,
    "bucket": bucket_idx
})

# Error column
df["is_error"] = (df["true"] != df["pred"]).astype(int)

# ------------------- Human Review Rule -------------------
# Human checks low-confidence (middle) predictions:
LOW = 0.45
HIGH = 0.55

df["review"] = (df["prob"] >= LOW) & (df["prob"] <= HIGH)

# Human "fixes" reviewed errors perfectly
df["corrected_pred"] = df["pred"]
mask = df["review"] & (df["is_error"] == 1)
df.loc[mask, "corrected_pred"] = df.loc[mask,"true"]

# ------------------- Metrics -------------------
post_acc = accuracy_score(df["true"], df["corrected_pred"])
review_rate = df["review"].mean()
errors_before = df["is_error"].sum()
errors_after = (df["true"] != df["corrected_pred"]).sum()
errors_reduced = errors_before - errors_after

# ------------------- Bucket stats -------------------
bucket_stats = df.groupby("bucket").agg(
    count=("true","size"),
    error_rate=("is_error","mean"),
    accuracy=("is_error",lambda x: 1-x.mean()),
    review_rate=("review","mean")
).reset_index()

bucket_stats.to_csv(REVIEW_CSV, index=False)

# ------------------- Plots -------------------
# Accuracy by confidence
plt.figure(figsize=(8,4))
sns.barplot(x="bucket", y="accuracy", data=bucket_stats)
plt.title("Accuracy by Confidence Bucket (Day29)")
plt.tight_layout()
plt.savefig(ACCURACY_PLOT, dpi=150)
plt.close()

# Error rate by confidence
plt.figure(figsize=(8,4))
sns.barplot(x="bucket", y="error_rate", data=bucket_stats, color="red")
plt.title("Error Rate by Confidence Bucket (Day29)")
plt.tight_layout()
plt.savefig(ERROR_PLOT, dpi=150)
plt.close()

# Cumulative error reduction vs review %
df_sorted = df.sort_values("prob").reset_index(drop=True)
df_sorted["cumu_review"] = df_sorted["review"].cumsum() / len(df_sorted)
df_sorted["cumu_errors_fixed"] = (df_sorted["review"] & df_sorted["is_error"].astype(bool)).cumsum()

plt.figure(figsize=(8,4))
plt.plot(df_sorted["cumu_review"], df_sorted["cumu_errors_fixed"], marker="o")
plt.xlabel("Fraction Reviewed (cumulative)")
plt.ylabel("Errors Fixed")
plt.title("Cumulative Error Reduction (Day29)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(CUM_PLOT, dpi=150)
plt.close()

# ------------------- Save Metrics -------------------
with open(METRICS_FILE,"w") as f:
    f.write("Day29 - Human-in-the-loop Simulation\n")
    f.write("="*60 + "\n\n")
    f.write(f"Baseline accuracy: {baseline_acc:.4f}\n")
    f.write(f"Post-review accuracy: {post_acc:.4f}\n")
    f.write(f"Review rate (human effort): {review_rate:.4f}\n")
    f.write(f"Errors before: {errors_before}\n")
    f.write(f"Errors after: {errors_after}\n")
    f.write(f"Errors reduced: {errors_reduced}\n")

print("\nDone. Artifacts saved to:", OUT_DIR)
