"""
Day 13 — Feature Selection & Correlation Analysis

What it does:
 - Loads Iris dataset (as supervised example)
 - Standard-scales features
 - Correlation heatmap (and list of high-corr pairs)
 - VarianceThreshold to drop near-constant features
 - SelectKBest (ANOVA F-test) to score top-k features
 - RandomForestClassifier feature importances
 - Combine scores into a single DataFrame and save CSV
 - Save figures and a joblib containing scaler + selectors/models

Outputs (in folder):
 - figures/correlation_heatmap.png
 - figures/variance_distribution.png
 - figures/selectkbest_scores.png
 - figures/rf_feature_importance.png
 - selected_features_summary.csv
 - feature_selectors.joblib
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

RND = 42

# Output paths
OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)
JOBLIB_PATH = OUT_DIR / "feature_selectors.joblib"
CSV_PATH = OUT_DIR / "selected_features_summary.csv"

def load_data():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="target")
    return X, y

def scale_data(X):
    scaler = StandardScaler()
    Xs = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return Xs, scaler

def correlation_analysis(X, threshold=0.8):
    corr = X.corr()
    # plot heatmap
    plt.figure(figsize=(6,5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    # find high-correlation pairs
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            val = corr.iloc[i, j]
            if abs(val) >= threshold:
                pairs.append((cols[i], cols[j], val))
    return corr, pairs

def variance_threshold_analysis(X, threshold=0.0):
    # variance threshold: drop near-constant features
    vt = VarianceThreshold(threshold=threshold)
    vt.fit(X)
    variances = pd.Series(vt.variances_, index=X.columns)
    # plot variances
    plt.figure(figsize=(6,3))
    variances.plot(kind="bar")
    plt.ylabel("Variance")
    plt.title("Feature variances (after scaling)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "variance_distribution.png", dpi=150)
    plt.close()
    selected_mask = vt.get_support()
    return vt, variances, selected_mask

def select_kbest(X, y, k=2):
    skb = SelectKBest(score_func=f_classif, k=k)
    skb.fit(X, y)
    scores = pd.Series(skb.scores_, index=X.columns)
    # plot
    plt.figure(figsize=(6,3))
    scores.sort_values(ascending=True).plot(kind="barh")
    plt.xlabel("ANOVA F-score")
    plt.title(f"SelectKBest (k={k}) scores")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "selectkbest_scores.png", dpi=150)
    plt.close()
    selected_mask = skb.get_support()
    return skb, scores, selected_mask

def rf_feature_importance(X, y, n_estimators=200):
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=RND)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    # plot
    plt.figure(figsize=(6,3))
    importances.sort_values(ascending=True).plot(kind="barh")
    plt.xlabel("Feature importance")
    plt.title("RandomForest feature importances")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rf_feature_importance.png", dpi=150)
    plt.close()
    return rf, importances

def combine_results(variances, skb_scores, rf_importances, vt_mask, skb_mask):
    df = pd.DataFrame({
        "variance": variances,
        "SelectKBest_score": skb_scores,
        "RandomForest_importance": rf_importances,
        "VarianceThreshold_keep": vt_mask,
        "SelectKBest_keep": skb_mask
    })
    # Normalize columns for easier comparison
    df["SelectKBest_score_norm"] = df["SelectKBest_score"] / (df["SelectKBest_score"].abs().max() + 1e-12)
    df["RandomForest_importance_norm"] = df["RandomForest_importance"] / (df["RandomForest_importance"].max() + 1e-12)
    # Simple aggregated score (you can tweak weights)
    df["aggregated_score"] = (df["SelectKBest_score_norm"] * 0.5) + (df["RandomForest_importance_norm"] * 0.5)
    df = df.sort_values("aggregated_score", ascending=False)
    return df

def main(k=2, corr_threshold=0.8, var_threshold=0.0):
    print("Loading data...")
    X, y = load_data()
    print("Scaling data...")
    Xs, scaler = scale_data(X)

    print("Running correlation analysis...")
    corr, high_pairs = correlation_analysis(Xs, threshold=corr_threshold)
    if high_pairs:
        print("Highly correlated pairs (abs >= {:.2f}):".format(corr_threshold))
        for a,b,val in high_pairs:
            print(f"  {a} <-> {b} = {val:.2f}")
    else:
        print("No highly correlated pairs found.")

    print("Variance threshold analysis...")
    vt, variances, vt_mask = variance_threshold_analysis(Xs, threshold=var_threshold)

    print(f"SelectKBest: choosing top {k} features by ANOVA F-test...")
    skb, skb_scores, skb_mask = select_kbest(Xs, y, k=k)

    print("RandomForest feature importances...")
    rf, rf_importances = rf_feature_importance(Xs, y)

    print("Combining results into a summary dataframe...")
    summary = combine_results(variances, skb_scores, rf_importances, vt_mask, skb_mask)
    summary.to_csv(CSV_PATH)
    print(f"Saved summary to {CSV_PATH}")

    # Save selector objects + scaler + rf model
    joblib.dump({
        "scaler": scaler,
        "variance_threshold": vt,
        "select_kbest": skb,
        "random_forest": rf,
        "summary_df": summary
    }, JOBLIB_PATH)
    print(f"Saved selector objects and models to {JOBLIB_PATH}")

    # Optional: quick RF CV score to verify predictive power on full feature set
    scores = cross_val_score(rf, Xs, y, cv=5)
    print(f"RandomForest CV accuracy (5-fold) on scaled features: {scores.mean():.3f} +/- {scores.std():.3f}")

    print("Done. Inspect the figures folder and selected_features_summary.csv for details.")

if __name__ == "__main__":
    main()
