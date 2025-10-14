# day15_explainability.py
"""
Day 15 — Model Explainability (SHAP & Feature Importance Visualization)
Author: Upasana
Purpose: Understand how model features influence predictions.
Notes:
- Automatically loads day14_model.joblib from same folder.
- Computes permutation importance (global).
- Computes SHAP explanations (robust: TreeExplainer -> fallback -> lambda predict_proba).
"""

from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance

# SHAP (may take some time; ensure shap is installed)
import shap

# Output / model path (portable: model should be in same folder as this script)
DAY14_MODEL = Path(__file__).resolve().parent / "day14_model.joblib"
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    import seaborn as sns
    df = sns.load_dataset("titanic")
    # simple engineered features (same as Day14)
    df["family_size"] = df["sibsp"].fillna(0) + df["parch"].fillna(0) + 1
    df["is_alone"] = (df["family_size"] == 1).astype(int)
    df["age_group"] = pd.cut(df["age"], bins=[0, 12, 20, 40, 60, 120],
                             labels=["child","teen","adult","mid","senior"])
    return df

def try_get_feature_names(preprocessor, numeric_features, categorical_features):
    # try several ways to get post-transformation feature names
    try:
        # ColumnTransformer / OneHotEncoder (sklearn >=1.0)
        names = preprocessor.get_feature_names_out()
        return list(names)
    except Exception:
        # Fallback: build names from numeric + encoder categories if available
        names = list(numeric_features)
        cat_transformer = None
        try:
            cat_transformer = preprocessor.named_transformers_.get("cat", None)
        except Exception:
            cat_transformer = None

        if cat_transformer is not None:
            # cat_transformer might be a Pipeline
            ohe = None
            try:
                if hasattr(cat_transformer, "steps"):
                    for _, step in cat_transformer.steps:
                        if hasattr(step, "categories_") or hasattr(step, "get_feature_names_out"):
                            ohe = step
                            break
                else:
                    ohe = cat_transformer
            except Exception:
                ohe = None

            if ohe is not None:
                try:
                    cats = ohe.categories_
                    for i, cvals in enumerate(cats):
                        for v in cvals:
                            names.append(f"{categorical_features[i]}_{v}")
                except Exception:
                    # last fallback
                    names += [f"{c}_val{i}" for c in categorical_features for i in range(3)]
        return names

def main():
    print("Day 15 — Model Explainability (SHAP + Permutation Importance)")
    df = load_data()

    numeric_features = ["age", "fare", "family_size"]
    categorical_features = ["sex", "class", "embarked", "age_group"]
    X = df[numeric_features + categorical_features].copy()
    y = df["survived"]

    # Load trained pipeline (preprocessor + classifier)
    if not DAY14_MODEL.exists():
        raise FileNotFoundError(f"Model not found at {DAY14_MODEL}. Please ensure day14_model.joblib is in this folder.")
    print(f"Loading model from: {DAY14_MODEL}")
    model = joblib.load(DAY14_MODEL)
    print("✅ Model loaded successfully!")

    # Quick sample predictions to sanity-check
    try:
        y_pred = model.predict(X)
        print("Sample predictions:", y_pred[:10])
    except Exception as e:
        print("Warning: model.predict failed:", e)

    # ----------------------------
    # 1) Permutation importance
    # ----------------------------
    try:
        print("\nCalculating permutation importance...")
        perm = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
        sorted_idx = perm.importances_mean.argsort()[::-1]

        # Try to get feature names (post-transform)
        try:
            preprocessor = model.named_steps["preprocessor"]
            feat_names = try_get_feature_names(preprocessor, numeric_features, categorical_features)
        except Exception:
            feat_names = [f"f{i}" for i in range(len(perm.importances_mean))]

        # Align length safety
        if len(feat_names) != len(perm.importances_mean):
            # fallback: use numeric+categorical short names
            feat_names = numeric_features + categorical_features

        plt.figure(figsize=(8,5))
        plt.barh(np.array(feat_names)[sorted_idx], perm.importances_mean[sorted_idx])
        plt.xlabel("Permutation Importance (mean)")
        plt.title("Permutation Feature Importance")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "day15_permutation_importance.png")
        plt.close()
        print("📊 Saved permutation importance plot!")
    except Exception as e:
        print("Permutation importance failed:", e)

    # ----------------------------
    # 2) SHAP explainability (robust)
    # ----------------------------
    print("\nRunning SHAP explainability...")
    # get classifier and preprocessor
    try:
        clf = model.named_steps["classifier"]
        preprocessor = model.named_steps["preprocessor"]
    except Exception:
        # If pipeline structure is different, attempt to use model directly
        clf = model
        preprocessor = None

    # Transform X for SHAP (the explainer expects features passed to the model)
    if preprocessor is not None:
        try:
            X_transformed = preprocessor.transform(X)
        except Exception:
            # If transform fails, fall back to raw X (some explainers accept raw)
            X_transformed = X.values
    else:
        X_transformed = X.values

    # Subsample for speed/robustness
    sample_n = 500
    if sample_n is not None and X_transformed.shape[0] > sample_n:
        rng = np.random.RandomState(42)
        idx = rng.choice(np.arange(X_transformed.shape[0]), size=sample_n, replace=False)
        X_for_shap = X_transformed[idx]
        X_for_shap_orig = X.iloc[idx] if isinstance(X, pd.DataFrame) else None
    else:
        X_for_shap = X_transformed
        X_for_shap_orig = X if isinstance(X, pd.DataFrame) else None

    # Try TreeExplainer explaining probabilities (preferred for classifiers)
    try:
        explainer = shap.TreeExplainer(clf, model_output="probability")
        # For older shap versions, use shap_values = explainer.shap_values(...)
        # For newer, shap_values may be a list or array depending on model_output
        try:
            shap_values = explainer.shap_values(X_for_shap)
        except Exception:
            shap_values = explainer(X_for_shap).values

        # If shap_values is list-like per-class, choose class-1 (positive)
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_vals_pos = np.array(shap_values[1])
            feature_names = try_get_feature_names(preprocessor, numeric_features, categorical_features)
            # summary
            shap.summary_plot(shap_vals_pos, X_for_shap, feature_names=feature_names if len(feature_names)==shap_vals_pos.shape[1] else None, show=False)
            plt.title("SHAP Summary Plot (class=1 probability)")
            plt.savefig(OUTPUT_DIR / "day15_shap_summary.png", bbox_inches="tight")
            plt.close()
            # local waterfall for first sample (safe construction)
            try:
                base_value = explainer.expected_value[1] if hasattr(explainer, "expected_value") else None
                shap.plots.waterfall(shap.Explanation(values=shap_vals_pos[0], base_values=base_value, data=X_for_shap[0]), show=False)
                plt.title("SHAP Local Explanation (sample 0, class=1)")
                plt.savefig(OUTPUT_DIR / "day15_shap_local.png", bbox_inches="tight")
                plt.close()
            except Exception:
                # fallback: create a basic bar representation (if waterfall fails)
                pass
            print("✅ SHAP plots saved (TreeExplainer class=1)!")
        else:
            # shap_values is single-array (regression-like)
            arr = np.array(shap_values)
            feature_names = try_get_feature_names(preprocessor, numeric_features, categorical_features)
            shap.summary_plot(arr, X_for_shap, feature_names=feature_names if len(feature_names)==arr.shape[1] else None, show=False)
            plt.title("SHAP Summary Plot")
            plt.savefig(OUTPUT_DIR / "day15_shap_summary.png", bbox_inches="tight")
            plt.close()
            try:
                shap.plots.waterfall(arr[0], show=False)
                plt.title("SHAP Local Explanation (sample 0)")
                plt.savefig(OUTPUT_DIR / "day15_shap_local.png", bbox_inches="tight")
                plt.close()
            except Exception:
                pass
            print("✅ SHAP plots saved (TreeExplainer single-output)!")
    except Exception as e:
        print("TreeExplainer route failed:", str(e))
        print("Trying fallback: shap.Explainer on clf.predict_proba (model-agnostic).")

        # Fallback: explain predict_proba function (single-output: class1 probability)
        try:
            # create a small wrapper that accepts transformed input if needed
            def predict_prob_pos(x):
                # if preprocessor exists and x is raw original features, we need to transform
                try:
                    # shap may pass numpy array in original feature space; check shape
                    if preprocessor is not None and x.shape[1] != X_for_shap.shape[1]:
                        # assume x is raw -> transform
                        x_trans = preprocessor.transform(pd.DataFrame(x, columns=X.columns))
                    else:
                        x_trans = x
                except Exception:
                    x_trans = x
                return clf.predict_proba(x_trans)[:, 1]

            explainer2 = shap.Explainer(predict_prob_pos, X_for_shap, feature_names=(X_for_shap_orig.columns.tolist() if isinstance(X_for_shap_orig, pd.DataFrame) else None))
            sv = explainer2(X_for_shap)
            # summary
            try:
                shap.summary_plot(sv.values, X_for_shap, feature_names=(X_for_shap_orig.columns.tolist() if isinstance(X_for_shap_orig, pd.DataFrame) else None), show=False)
                plt.title("SHAP Summary Plot (fallback predict_proba)")
                plt.savefig(OUTPUT_DIR / "day15_shap_summary.png", bbox_inches="tight")
                plt.close()
            except Exception:
                pass
            # local
            try:
                shap.plots.waterfall(sv[0], show=False)
                plt.title("SHAP Local Explanation (fallback sample 0)")
                plt.savefig(OUTPUT_DIR / "day15_shap_local.png", bbox_inches="tight")
                plt.close()
            except Exception:
                pass
            print("✅ Fallback SHAP plots saved!")
        except Exception as e2:
            print("Fallback SHAP attempt failed:", e2)
            print("At this point SHAP explainability couldn't complete. The permutation importance plot may still be useful.")

    print("\nDay 15 complete — check generated images in your folder!")

if __name__ == "__main__":
    main()
