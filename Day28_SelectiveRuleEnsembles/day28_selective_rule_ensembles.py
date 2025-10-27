"""
Day28: Selective Rule Ensembles & Coverage Maximization

- Train / load a black-box RandomForest and a DecisionTree surrogate.
- Extract rules from surrogate.
- Greedy rule selection:
    - Iteratively select the rule that gives the largest increase in uncovered coverage
      while maintaining precision >= PRECISION_THRESHOLD on its covered set.
    - Stop when no rule can be added or desired coverage reached.
- Produce coverage vs precision diagnostic plot, save selected rules and metrics.

Designed to run in Windows PowerShell with your .venv (no extra packages beyond sklearn, pandas, matplotlib).
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Paths ---
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "day28_artifacts"
MODELS_DIR = OUT_DIR / "day28_models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

COVERAGE_PREC_PLOT = OUT_DIR / "day28_coverage_precision_curve.png"
SELECTED_RULES_CSV = OUT_DIR / "day28_selected_rules.csv"
SELECTED_RULES_TXT = OUT_DIR / "day28_rules.txt"
METRICS_FILE = OUT_DIR / "day28_rule_set_metrics.txt"

# --- Config ---
PRECISION_THRESHOLD = 0.80   # per-rule precision threshold to allow a rule into ensemble
TARGET_COVERAGE = 0.85       # optional target coverage (stop early if reached)
MAX_RULES = 20               # upper bound on number of selected rules

# --- Utilities: tree -> rules (re-used concept from Day27) ---
def tree_rules_list(tree: DecisionTreeClassifier, feature_names):
    tree_ = tree.tree_
    feature = tree_.feature
    threshold = tree_.threshold
    paths = []

    def recurse(node, conditions):
        if tree_.feature[node] == _tree.TREE_UNDEFINED:
            value = tree_.value[node][0]
            pred_class = int(np.argmax(value))
            paths.append({'conditions': list(conditions), 'prediction': pred_class, 'n_node_samples': int(tree_.n_node_samples[node])})
            return
        f_name = feature_names[feature[node]]
        thr = threshold[node]
        # left
        conditions.append((f_name, "<=", float(thr)))
        recurse(tree_.children_left[node], conditions)
        conditions.pop()
        # right
        conditions.append((f_name, ">", float(thr)))
        recurse(tree_.children_right[node], conditions)
        conditions.pop()
    recurse(0, [])
    return paths

def rule_mask_from_conditions(df: pd.DataFrame, conditions):
    mask = np.ones(len(df), dtype=bool)
    for feat, op, thr in conditions:
        if op == "<=":
            mask &= (df[feat].values <= thr)
        else:
            mask &= (df[feat].values > thr)
    return mask

# --- Load dataset and split (use same dataset as previous days) ---
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
X_test_df = X_test.reset_index(drop=True)
y_test_arr = y_test.reset_index(drop=True).values

# --- Train or load black-box RF (for fidelity baseline & optional distillation) ---
rf_path = MODELS_DIR / "rf_blackbox.joblib"
if rf_path.exists():
    pipe_rf = joblib.load(rf_path)
else:
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    pipe_rf = Pipeline([("scaler", StandardScaler()), ("rf", rf)])
    pipe_rf.fit(X_train, y_train)
    joblib.dump(pipe_rf, rf_path)
# black-box predictions on test
y_pred_rf_test = pipe_rf.predict(X_test)

# --- Train or load surrogate tree (shallow for rule extraction) ---
sur_path = MODELS_DIR / "surrogate_chosen.joblib"
if sur_path.exists():
    sur_pipe = joblib.load(sur_path)
else:
    # distill rf -> train tree on rf labels
    rf_train_labels = pipe_rf.predict(X_train)
    dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    sur_pipe = Pipeline([("scaler", StandardScaler()), ("dt", dt)])
    sur_pipe.fit(X_train, rf_train_labels)
    joblib.dump(sur_pipe, sur_path)

# Extract rules from surrogate
dt_core = sur_pipe.named_steps['dt']
raw_rules = tree_rules_list(dt_core, feature_names)
print(f"Extracted {len(raw_rules)} raw rules from surrogate.")

# Evaluate all raw rules on test set (compute coverage, precision vs true label, fidelity vs rf)
n_test = len(X_test_df)
rule_rows = []
for i, r in enumerate(raw_rules, start=1):
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
        covered_true = y_test_arr[mask]
        precision = np.mean(covered_true == pred)
        fidelity = np.mean(y_pred_rf_test[mask] == pred)
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
df_rules.sort_values(["precision", "coverage"], ascending=[False, False], inplace=True)

# Save raw rule table
df_rules.to_csv(OUT_DIR / "day28_all_rules.csv", index=False)

# --- Greedy rule selection ---
selected = []            # indices of selected rules (rule_id)
covered_mask = np.zeros(n_test, dtype=bool)
current_coverage = 0.0
selection_history = []   # list of dicts: rule_id, new_covered, cumulative_coverage, rule_precision, rule_fidelity

candidates = df_rules.copy()

for step in range(MAX_RULES):
    best_rule = None
    best_new_cov = 0.0
    # evaluate each candidate not yet selected
    for _, row in candidates.iterrows():
        rid = int(row['rule_id'])
        conds = row['conditions']
        # compute mask
        mask = rule_mask_from_conditions(X_test_df, [tuple(s.strip().split(" ",2)) if isinstance(s, str) else s for s in []]) if False else None
        # easier: reuse row conditions string parsing
        parts = [p.strip() for p in row['conditions'].split(" AND ")] if row['conditions'] else []
        conds_parsed = []
        for p in parts:
            if "<=" in p:
                f, thr = p.split("<=")
                conds_parsed.append((f.strip(), "<=", float(thr)))
            elif ">" in p:
                f, thr = p.split(">")
                conds_parsed.append((f.strip(), ">", float(thr)))
        mask = rule_mask_from_conditions(X_test_df, conds_parsed)
        # new coverage if we add this rule (only on previously uncovered samples)
        new_apply = mask & (~covered_mask)
        new_covered = new_apply.sum()
        if new_covered == 0:
            continue
        # compute precision on the rule's covered set (the rule's own precision, not only new_apply)
        rule_precision = row['precision']
        # enforce per-rule precision threshold
        if np.isnan(rule_precision) or rule_precision < PRECISION_THRESHOLD:
            continue
        # objective: maximize new_covered (could also use weighted objective)
        if new_covered > best_new_cov:
            best_new_cov = new_covered
            best_rule = {"row": row, "conds": conds_parsed, "new_apply": new_apply}
    if best_rule is None:
        # no candidate passes precision or increases coverage
        break
    # select the best_rule
    row = best_rule['row']
    new_apply = best_rule['new_apply']
    covered_mask = covered_mask | new_apply
    cumulative_cov = covered_mask.mean()
    selected.append(int(row['rule_id']))
    selection_history.append({
        "rule_id": int(row['rule_id']),
        "conditions": row['conditions'],
        "prediction": int(row['prediction']),
        "rule_precision": float(row['precision']),
        "rule_fidelity": float(row['fidelity']) if not np.isnan(row['fidelity']) else np.nan,
        "new_covered_count": int(best_new_cov),
        "cumulative_coverage": float(cumulative_cov)
    })
    # remove selected rule from candidates
    candidates = candidates[candidates['rule_id'] != row['rule_id']]
    # optional early stop if reach target coverage
    if cumulative_cov >= TARGET_COVERAGE:
        break

# Build final rule-set preds (first-match applies in selection order)
final_preds = np.full(n_test, -1, dtype=int)  # -1 uncovered
covered_mask_final = np.zeros(n_test, dtype=bool)
for sel in selection_history:
    # parse conds
    parts = [p.strip() for p in sel['conditions'].split(" AND ")] if sel['conditions'] else []
    conds_parsed = []
    for p in parts:
        if "<=" in p:
            f, thr = p.split("<=")
            conds_parsed.append((f.strip(), "<=", float(thr)))
        elif ">" in p:
            f, thr = p.split(">")
            conds_parsed.append((f.strip(), ">", float(thr)))
    mask = rule_mask_from_conditions(X_test_df, conds_parsed)
    new_apply = mask & (~covered_mask_final)
    final_preds[new_apply] = int(sel['prediction'])
    covered_mask_final |= new_apply

overall_coverage = covered_mask_final.mean()
if covered_mask_final.sum() > 0:
    accuracy_on_covered = np.mean(y_test_arr[covered_mask_final] == final_preds[covered_mask_final])
    fidelity_on_covered = np.mean(y_pred_rf_test[covered_mask_final] == final_preds[covered_mask_final])
else:
    accuracy_on_covered = np.nan
    fidelity_on_covered = np.nan

# --- Save selected rules + history ---
df_selected = pd.DataFrame(selection_history)
df_selected.to_csv(SELECTED_RULES_CSV, index=False)

with open(SELECTED_RULES_TXT, "w", encoding="utf-8") as f:
    f.write("Day28 — Selected Rule Ensemble (greedy)\n")
    f.write("="*80 + "\n\n")
    f.write(f"Precision threshold per rule: {PRECISION_THRESHOLD}\n")
    f.write(f"Target coverage: {TARGET_COVERAGE}\n\n")
    if df_selected.empty:
        f.write("No rules selected under current thresholds.\n")
    else:
        for i, r in df_selected.iterrows():
            f.write(f"Rule {i+1} (id={r['rule_id']}): IF {r['conditions']} THEN predict={r['prediction']}\n")
            f.write(f"  precision={r['rule_precision']:.3f}, fidelity={r['rule_fidelity']:.3f}, new_covered={r['new_covered_count']}, cumulative_cov={r['cumulative_coverage']:.3f}\n\n")
    f.write("\nFinal rule-set metrics:\n")
    f.write(f"  overall_coverage = {overall_coverage:.4f}\n")
    f.write(f"  accuracy_on_covered = {accuracy_on_covered:.4f}\n")
    f.write(f"  fidelity_on_covered = {fidelity_on_covered:.4f}\n")

# --- Diagnostic plot: incremental coverage & per-rule precision ---
if not df_selected.empty:
    covs = df_selected['cumulative_coverage'].astype(float).values
    per_rule_prec = df_selected['rule_precision'].astype(float).values
    plt.figure(figsize=(8,5))
    ax1 = plt.gca()
    ax1.plot(range(1, len(covs)+1), covs, marker='o', label='cumulative coverage')
    ax1.set_xlabel("selected rule count")
    ax1.set_ylabel("coverage (fraction)", color='tab:blue')
    ax2 = ax1.twinx()
    ax2.plot(range(1, len(per_rule_prec)+1), per_rule_prec, marker='s', color='tab:orange', label='per-rule precision')
    ax2.set_ylabel("per-rule precision", color='tab:orange')
    ax1.set_title("Day28 — Selected rule ensemble: coverage (left) vs per-rule precision (right)")
    ax1.grid(alpha=0.2)
    ax1.set_ylim(0,1)
    ax2.set_ylim(0,1)
    plt.tight_layout()
    plt.savefig(COVERAGE_PREC_PLOT, dpi=150)
    plt.close()

# --- Save metrics summary ---
with open(METRICS_FILE, "w", encoding="utf-8") as f:
    f.write("Day28 - Selective Rule Ensembles & Coverage Maximization\n")
    f.write("="*80 + "\n\n")
    f.write(f"Number of raw rules available: {len(df_rules)}\n")
    f.write(f"Number of rules selected: {len(df_selected)}\n\n")
    f.write(f"Per-rule precision threshold (min): {PRECISION_THRESHOLD}\n")
    f.write(f"Target coverage: {TARGET_COVERAGE}\n")
    f.write(f"Overall coverage achieved: {overall_coverage:.4f}\n")
    f.write(f"Accuracy on covered samples (vs true): {accuracy_on_covered:.4f}\n")
    f.write(f"Fidelity on covered samples (vs RF): {fidelity_on_covered:.4f}\n\n")
    f.write(f"Selected rules CSV: {SELECTED_RULES_CSV}\n")
    f.write(f"Selected rules TXT: {SELECTED_RULES_TXT}\n")
    f.write(f"Coverage/precision plot: {COVERAGE_PREC_PLOT}\n")

print("Done. Artifacts saved to:", OUT_DIR)
