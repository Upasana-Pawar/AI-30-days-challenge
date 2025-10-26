"""
Day27: Rule-based Model Compression & Global Surrogate Evaluation

- Train a RandomForest black-box and a shallow DecisionTree surrogate (or reuse Day26 models if present).
- Extract rules programmatically from a DecisionTree (paths -> conditions).
- For each rule:
    - compute coverage = fraction of samples (test set) the rule applies to
    - compute precision = fraction of positive class among covered samples
    - compute fidelity = fraction where rule label == black-box label (for samples rule covers)
- Prune/compress rules by thresholds (min_coverage, min_precision).
- Evaluate rule-set:
    - overall coverage (fraction of samples any rule covers)
    - accuracy of rule-set on covered samples (vs true labels)
    - fidelity of rule-set vs black-box predictions
- Evaluate coverage & fidelity across simple slices (subpopulations) e.g. by quartiles of top feature.
- Save CSVs, plots, and human-readable rules.

Run (PowerShell + .venv).
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
from sklearn.tree import DecisionTreeClassifier, _tree, export_text
from sklearn.metrics import accuracy_score

# --- Paths ---
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "day27_artifacts"
MODELS_DIR = OUT_DIR / "day27_models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RULES_CSV = OUT_DIR / "day27_rules.csv"
RULES_TXT = OUT_DIR / "day27_rules.txt"
RULE_METRICS_PNG = OUT_DIR / "day27_rule_metrics.png"
SLICES_CSV = OUT_DIR / "day27_fidelity_coverage_by_slice.csv"
METRICS_FILE = OUT_DIR / "day27_metrics.txt"

# --- Config / pruning thresholds ---
MIN_COVERAGE = 0.02   # keep rules that cover at least 2% of test set
MIN_PRECISION = 0.7   # keep rules with precision >= 70%

# --- Load data (breast_cancer) ---
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target
feature_names = X.columns.tolist()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# --- Train black-box RandomForest (or reuse existing file if present) ---
blackbox_path = MODELS_DIR / "rf_blackbox.joblib"
if blackbox_path.exists():
    pipe_rf = joblib.load(blackbox_path)
    print("Loaded existing black-box RF from", blackbox_path)
else:
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    pipe_rf = Pipeline([("scaler", StandardScaler()), ("rf", rf)])
    pipe_rf.fit(X_train, y_train)
    joblib.dump(pipe_rf, blackbox_path)
    print("Trained & saved black-box RF to", blackbox_path)

# Black-box predictions (used for fidelity computation)
y_pred_rf_test = pipe_rf.predict(X_test)
y_pred_rf_train = pipe_rf.predict(X_train)

# --- Train surrogate DecisionTree to mimic RF (shallow for interpretability) ---
surrogate_path = MODELS_DIR / "surrogate_chosen.joblib"
if surrogate_path.exists():
    sur_pipe = joblib.load(surrogate_path)
    print("Loaded existing surrogate from", surrogate_path)
else:
    # train on RF labels on train set (distillation)
    rf_labels_train = y_pred_rf_train
    dt = DecisionTreeClassifier(max_depth=4, random_state=42)  # shallow tree for rules
    sur_pipe = Pipeline([("scaler", StandardScaler()), ("dt", dt)])
    sur_pipe.fit(X_train, rf_labels_train)
    joblib.dump(sur_pipe, surrogate_path)
    print("Trained & saved surrogate to", surrogate_path)

# Surrogate predictions on test
y_pred_sur_test = sur_pipe.predict(X_test)

# Save models paths for reference
joblib.dump(pipe_rf, MODELS_DIR / "rf_blackbox.joblib")
joblib.dump(sur_pipe, MODELS_DIR / "surrogate_chosen.joblib")

# --- Helper: extract rules from DecisionTreeClassifier (core) ---
def tree_rules_list(tree: DecisionTreeClassifier, feature_names):
    """
    Extract rules from sklearn DecisionTreeClassifier (uses tree_.structure).
    Returns a list of dicts: {'conditions': [(feature, op, threshold), ...], 'value': predicted_class}
    """
    tree_ = tree.tree_
    feature = tree_.feature
    threshold = tree_.threshold

    paths = []

    def recurse(node, conditions):
        # If leaf
        if tree_.feature[node] == _tree.TREE_UNDEFINED:
            # predicted class: argmax of value
            value = tree_.value[node][0]
            pred_class = int(np.argmax(value))
            paths.append({'conditions': list(conditions), 'prediction': pred_class, 'n_node_samples': int(tree_.n_node_samples[node])})
            return
        # left child: feature <= threshold
        f_name = feature_names[feature[node]]
        thresh = threshold[node]
        # left
        conditions.append((f_name, "<=", float(thresh)))
        recurse(tree_.children_left[node], conditions)
        conditions.pop()
        # right
        conditions.append((f_name, ">", float(thresh)))
        recurse(tree_.children_right[node], conditions)
        conditions.pop()

    recurse(0, [])
    return paths

# Unwrap pipeline to get tree core
dt_core = sur_pipe.named_steps['dt']
rules = tree_rules_list(dt_core, feature_names)
print(f"Extracted {len(rules)} raw rules from surrogate tree.")

# --- Evaluate each rule on test set (coverage, precision, fidelity) ---
def rule_mask_from_conditions(df: pd.DataFrame, conditions):
    """
    Given DataFrame df and a list of conditions [(feat, op, thr), ...],
    return boolean mask where all conditions hold.
    """
    mask = np.ones(len(df), dtype=bool)
    for feat, op, thr in conditions:
        if op == "<=":
            mask &= (df[feat].values <= thr)
        else:
            mask &= (df[feat].values > thr)
    return mask

rule_rows = []
n_test = len(X_test)
X_test_df = X_test.reset_index(drop=True)
y_test_arr = y_test.reset_index(drop=True).values
y_rf_test = pd.Series(y_pred_rf_test).reset_index(drop=True)

for i, r in enumerate(rules, 1):
    conds = r['conditions']
    pred = r['prediction']
    mask = rule_mask_from_conditions(X_test_df, conds)
    covered = mask.sum()
    coverage = covered / n_test
    if covered == 0:
        precision = np.nan
        fidelity = np.nan
        acc_vs_true = np.nan
    else:
        # precision: fraction of true positives among covered when rule predicts positive (pred==1)
        # But rule may predict class 0 or 1. We define precision relative to predicting positive class.
        covered_true = y_test_arr[mask]  # true labels in covered samples
        # For precision relative to predicted class:
        precision = np.mean(covered_true == pred)
        # fidelity: fraction where rule prediction == black-box prediction (on covered)
        fidelity = np.mean(y_rf_test[mask].values == pred)
        # accuracy of rule vs ground truth on covered samples
        acc_vs_true = np.mean(covered_true == pred)

    rule_rows.append({
        "rule_id": i,
        "prediction": int(pred),
        "coverage": coverage,
        "covered_count": int(covered),
        "precision": float(precision) if not np.isnan(precision) else np.nan,
        "fidelity": float(fidelity) if not np.isnan(fidelity) else np.nan,
        "accuracy_vs_true": float(acc_vs_true) if not np.isnan(acc_vs_true) else np.nan,
        "conditions": " AND ".join([f"{f} {op} {thr:.4f}" for (f,op,thr) in conds])
    })

df_rules = pd.DataFrame(rule_rows)
df_rules.to_csv(RULES_CSV, index=False)

# --- Prune / compress rules ---
pruned = df_rules[(df_rules.coverage >= MIN_COVERAGE) & (df_rules.precision >= MIN_PRECISION)]
pruned = pruned.sort_values(['precision', 'coverage'], ascending=[False, False]).reset_index(drop=True)

# Evaluate rule-set ensemble (apply pruned rules in order: first-match)
def apply_rule_set(df_X: pd.DataFrame, rules_df: pd.DataFrame):
    """
    Apply rules sequentially (first-match wins). Returns:
      - mask_covered (bool array)
      - preds (array with -1 for uncovered samples else predicted class)
    """
    n = len(df_X)
    preds = np.full(n, -1, dtype=int)  # -1 -> uncovered
    covered_mask = np.zeros(n, dtype=bool)
    for _, row in rules_df.iterrows():
        cond_str = row['conditions']
        # parse conditions back to list for evaluation (we already have them in df_rules)
        # Instead re-use earlier masks by recomputing from string
        # Simple parser:
        parts = [p.strip() for p in cond_str.split(" AND ")] if cond_str else []
        conds = []
        for p in parts:
            if "<=" in p:
                f,thr = p.split("<=")
                conds.append((f.strip(), "<=", float(thr)))
            elif ">" in p:
                f,thr = p.split(">")
                conds.append((f.strip(), ">", float(thr)))
        mask = rule_mask_from_conditions(df_X, conds)
        # apply to samples not yet covered
        new_apply = mask & (~covered_mask)
        preds[new_apply] = int(row['prediction'])
        covered_mask |= new_apply
    return covered_mask, preds

covered_mask, preds_by_rules = apply_rule_set(X_test_df, pruned)

# Rule-set metrics
coverage_ruleset = covered_mask.mean()
if covered_mask.sum() > 0:
    accuracy_ruleset_vs_true = np.mean(y_test_arr[covered_mask] == preds_by_rules[covered_mask])
    fidelity_ruleset_vs_rf = np.mean(y_pred_rf_test[covered_mask] == preds_by_rules[covered_mask])
else:
    accuracy_ruleset_vs_true = np.nan
    fidelity_ruleset_vs_rf = np.nan

# Save pruned rules text
with open(RULES_TXT, "w", encoding="utf-8") as f:
    f.write("Day27 — Extracted & Pruned Rules (first-match applies)\n")
    f.write("="*80 + "\n\n")
    f.write(f"Pruning thresholds: min_coverage={MIN_COVERAGE}, min_precision={MIN_PRECISION}\n\n")
    if pruned.empty:
        f.write("No rules passed pruning thresholds.\n")
    else:
        for idx, row in pruned.iterrows():
            f.write(f"Rule {idx+1}: IF {row['conditions']} THEN predict={int(row['prediction'])}\n")
            f.write(f"  coverage={row['coverage']:.3f}, precision={row['precision']:.3f}, fidelity={row['fidelity']:.3f}\n\n")
    f.write("\nRule-set metrics:\n")
    f.write(f"  overall coverage: {coverage_ruleset:.3f}\n")
    f.write(f"  accuracy on covered (vs true): {accuracy_ruleset_vs_true:.4f}\n")
    f.write(f"  fidelity on covered (vs black-box): {fidelity_ruleset_vs_rf:.4f}\n")

# --- Slice analysis: evaluate coverage & fidelity across subpopulations ---
# We'll pick the top feature by RF importance and slice into quartiles; compute metrics per slice
# Get top feature
rf_core = pipe_rf.named_steps['rf']
fi = rf_core.feature_importances_
top_feat_idx = int(np.argmax(fi))
top_feat = feature_names[top_feat_idx]
vals = X_test_df[top_feat].values
# quartile bins
bins = np.quantile(vals, [0, 0.25, 0.5, 0.75, 1.0])
slice_labels = []
slice_rows = []
for i in range(len(bins)-1):
    low, high = bins[i], bins[i+1]
    mask = (vals >= low) & (vals <= high) if i == len(bins)-2 else (vals >= low) & (vals < high)
    if mask.sum() == 0:
        continue
    cov = covered_mask[mask].mean()
    if mask.sum() > 0:
        if covered_mask[mask].sum() > 0:
            fidelity_slice = np.mean(y_pred_rf_test[mask][covered_mask[mask]] == preds_by_rules[mask][covered_mask[mask]])
            acc_slice = np.mean(y_test_arr[mask][covered_mask[mask]] == preds_by_rules[mask][covered_mask[mask]])
        else:
            fidelity_slice = np.nan
            acc_slice = np.nan
    else:
        fidelity_slice = np.nan
        acc_slice = np.nan
    slice_rows.append({
        "slice_idx": i,
        "feat": top_feat,
        "range_low": float(low),
        "range_high": float(high),
        "n_samples": int(mask.sum()),
        "coverage": float(cov),
        "fidelity_on_covered": float(fidelity_slice) if not np.isnan(fidelity_slice) else np.nan,
        "accuracy_on_covered": float(acc_slice) if not np.isnan(acc_slice) else np.nan
    })

df_slices = pd.DataFrame(slice_rows)
df_slices.to_csv(SLICES_CSV, index=False)

# --- Plot rule metrics (coverage vs precision) ---
plt.figure(figsize=(8,6))
plt.scatter(df_rules.coverage, df_rules.precision, s=df_rules.covered_count, alpha=0.7)
plt.axvline(MIN_COVERAGE, color='red', linestyle='--', label=f"min_cov={MIN_COVERAGE}")
plt.axhline(MIN_PRECISION, color='green', linestyle='--', label=f"min_prec={MIN_PRECISION}")
plt.xlabel("Coverage (test fraction)")
plt.ylabel("Precision (vs true label)")
plt.title("Day27 — Rule metrics (size ~ covered_count)")
plt.legend()
plt.tight_layout()
plt.savefig(RULE_METRICS_PNG, dpi=150)
plt.close()

# --- Save summary metrics ---
with open(METRICS_FILE, "w", encoding="utf-8") as f:
    f.write("Day27 - Rule-based Model Compression & Global Surrogate Evaluation\n")
    f.write("="*80 + "\n\n")
    f.write(f"Number of raw rules extracted: {len(df_rules)}\n")
    f.write(f"Number of pruned rules: {len(pruned)}\n\n")
    f.write(f"Rule-set overall coverage (test): {coverage_ruleset:.4f}\n")
    f.write(f"Rule-set accuracy on covered (vs true): {accuracy_ruleset_vs_true:.4f}\n")
    f.write(f"Rule-set fidelity on covered (vs black-box): {fidelity_ruleset_vs_rf:.4f}\n\n")
    f.write(f"Rules CSV: {RULES_CSV}\n")
    f.write(f"Pruned rules TXT: {RULES_TXT}\n")
    f.write(f"Rule metrics plot: {RULE_METRICS_PNG}\n")
    f.write(f"Slice metrics CSV: {SLICES_CSV}\n")

print("\nDone. Artifacts saved to:", OUT_DIR)
