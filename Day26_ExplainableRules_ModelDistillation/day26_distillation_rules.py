"""
Day26: Explainable Decision Rules & Model Distillation

- Train a black-box RandomForest classifier on breast_cancer.
- Distill / approximate it with DecisionTree surrogates of varying depth.
- Measure:
    - fidelity: how often surrogate predicts same label as black-box (on test set)
    - accuracy_surrogate: surrogate vs true labels
    - accuracy_blackbox: black-box vs true labels
- Export:
    - feature importance plot (black-box)
    - surrogate decision tree PNG (using sklearn.tree.plot_tree)
    - rules text (export_text)
    - metrics summary and models saved as joblib

Run in your .venv on Windows PowerShell.
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
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

# --- Paths ---
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "day26_artifacts"
MODELS_DIR = OUT_DIR / "day26_models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FI_PLOT = OUT_DIR / "day26_rf_feature_importance.png"
FIDELITY_PLOT = OUT_DIR / "day26_surrogate_depth_vs_fidelity.png"
TREE_PLOT = OUT_DIR / "day26_surrogate_tree.png"
METRICS_FILE = OUT_DIR / "day26_metrics.txt"
RULES_FILE = OUT_DIR / "day26_rules.txt"

# --- Load data ---
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# --- Train black-box model (RandomForest) ---
print("Training RandomForest black-box model...")
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
pipe_rf = Pipeline([("scaler", StandardScaler()), ("rf", rf)])
pipe_rf.fit(X_train, y_train)
joblib.dump(pipe_rf, MODELS_DIR / "rf_blackbox.joblib")

# Evaluate black-box
y_pred_rf = pipe_rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

# Feature importances (from raw RF inside pipeline)
rf_core = pipe_rf.named_steps["rf"]
importances = rf_core.feature_importances_
fi_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)

plt.figure(figsize=(8,6))
sns.barplot(x="importance", y="feature", data=fi_df.head(15))
plt.title("Day26 — RandomForest Feature Importances (top 15)")
plt.tight_layout()
plt.savefig(FI_PLOT, dpi=150)
plt.close()

# --- Distillation: train decision tree surrogates to mimic RF predictions ---
# Surrogate will be trained to predict the black-box's predictions (labels) on train set,
# then evaluated on test set for fidelity and accuracy vs ground truth.

# Create training labels from RF predictions on training set (out-of-fold production would be better; this is demo)
y_train_rf_labels = pipe_rf.predict(X_train)  # black-box labels on train
y_test_rf_labels = pipe_rf.predict(X_test)    # black-box labels on test (for fidelity computation)

max_depths = list(range(1, 9))  # try shallow to moderate depths
fidelities = []
acc_surrogates = []

surrogate_models = {}

for depth in max_depths:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    # pipeline: scale -> dt (scaler to keep input scale consistent)
    from sklearn.pipeline import Pipeline as SKPipeline
    pipe_dt = SKPipeline([("scaler", StandardScaler()), ("dt", dt)])
    # Train surrogate on X_train -> labels predicted by RF
    pipe_dt.fit(X_train, y_train_rf_labels)
    # Save surrogate
    joblib.dump(pipe_dt, MODELS_DIR / f"surrogate_depth{depth}.joblib")
    surrogate_models[depth] = pipe_dt

    # Evaluate fidelity (surrogate vs black-box labels on test)
    surrogate_preds_on_test = pipe_dt.predict(X_test)
    fidelity = np.mean(surrogate_preds_on_test == y_test_rf_labels)
    fidelities.append(fidelity)

    # Evaluate surrogate accuracy vs ground truth
    acc_surr = accuracy_score(y_test, surrogate_preds_on_test)
    acc_surrogates.append(acc_surr)

    print(f"Depth {depth}: fidelity={fidelity:.4f}, surrogate_acc={acc_surr:.4f}")

# Choose a surrogate depth by trading fidelity vs complexity (pick best depth with fidelity >= threshold or inspect plot)
plt.figure(figsize=(8,5))
plt.plot(max_depths, fidelities, marker='o', label='fidelity (surrogate vs RF)')
plt.plot(max_depths, acc_surrogates, marker='s', label='surrogate accuracy (vs true)')
plt.axhline(acc_rf, color='grey', linestyle='--', label='RF accuracy (vs true)')
plt.xlabel("Surrogate tree max_depth")
plt.ylabel("Score")
plt.title("Day26 — Surrogate depth vs fidelity & accuracy")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(FIDELITY_PLOT, dpi=150)
plt.close()

# Pick depth with good fidelity but low complexity — e.g., first depth with fidelity >= 0.95 or the max fidelity
chosen_depth = None
for depth, fid in zip(max_depths, fidelities):
    if fid >= 0.95:
        chosen_depth = depth
        break
if chosen_depth is None:
    # fallback: choose depth with highest fidelity but <= 6 to keep complexity moderate
    best_idx = int(np.argmax(fidelities))
    chosen_depth = max_depths[best_idx] if max_depths[best_idx] <= 6 else 6

chosen_surrogate = surrogate_models[chosen_depth]
joblib.dump(chosen_surrogate, MODELS_DIR / f"surrogate_chosen_depth{chosen_depth}.joblib")

# --- Export human-readable rules from chosen surrogate ---
try:
    dt_core = chosen_surrogate.named_steps["dt"]
    tree_rules = export_text(dt_core, feature_names=feature_names, max_depth=chosen_depth)
    with open(RULES_FILE, "w", encoding="utf-8") as f:
        f.write("Day26 — Surrogate Decision Tree Rules\n")
        f.write("="*72 + "\n\n")
        f.write(f"Chosen surrogate max_depth = {chosen_depth}\n\n")
        f.write(tree_rules)
except Exception as e:
    with open(RULES_FILE, "w", encoding="utf-8") as f:
        f.write("Failed to export rules: {}\n".format(e))

# --- Plot the chosen tree (may be large; we limit feature names and fontsize) ---
plt.figure(figsize=(20,10))
plot_tree(dt_core, feature_names=feature_names, class_names=[str(c) for c in pipe_rf.named_steps["rf"].classes_],
          filled=True, impurity=False, rounded=True, fontsize=8)
plt.title(f"Day26 — Surrogate Decision Tree (max_depth={chosen_depth})")
plt.tight_layout()
plt.savefig(TREE_PLOT, dpi=150)
plt.close()

# --- Save metrics summary ---
with open(METRICS_FILE, "w", encoding="utf-8") as f:
    f.write("Day26 - Explainable Decision Rules & Model Distillation\n")
    f.write("="*72 + "\n\n")
    f.write(f"Black-box RandomForest accuracy (test): {acc_rf:.4f}\n\n")
    f.write("Surrogate results (by depth):\n")
    for depth, fid, acc in zip(max_depths, fidelities, acc_surrogates):
        f.write(f"  depth={depth} -> fidelity={fid:.4f}, surrogate_acc={acc:.4f}\n")
    f.write(f"\nChosen surrogate depth: {chosen_depth}\n")
    f.write(f"Chosen surrogate fidelity: {fidelities[max_depths.index(chosen_depth)]:.4f}\n")
    f.write(f"Chosen surrogate accuracy (vs true): {acc_surrogates[max_depths.index(chosen_depth)]:.4f}\n\n")
    f.write(f"Saved surrogate tree rules to: {RULES_FILE}\n")
    f.write(f"Saved models to: {MODELS_DIR}\n")

print("\nDone. Artifacts saved in:", OUT_DIR)
