"""
Day 10 — DBSCAN clustering (Iris)
- Creates k-distance plot to help pick eps
- Runs DBSCAN and visualizes results with PCA
- Saves figures, cluster assignment CSV, and joblib of the fitted objects
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# --- configuration ---
OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)
MODEL_PATH = OUT_DIR / "dbscan_model.joblib"

RANDOM_STATE = 42

def load_data():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="true_label")
    return X, y, iris

def preprocess(X):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler

def plot_k_distance(Xs, k, out_path):
    # compute k-nearest distances
    nbrs = NearestNeighbors(n_neighbors=k).fit(Xs)
    distances, _ = nbrs.kneighbors(Xs)
    # distance to k-th neighbor for each point
    k_distances = np.sort(distances[:, k-1])
    plt.figure(figsize=(6,4))
    plt.plot(range(1, len(k_distances)+1), k_distances)
    plt.xlabel("Points (sorted by k-distance)")
    plt.ylabel(f"Distance to {k}th nearest neighbor")
    plt.title(f"k-distance plot (k={k}) — pick eps near the elbow")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def run_dbscan(Xs, eps=0.5, min_samples=5):
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=-1)
    labels = db.fit_predict(Xs)
    return db, labels

def silhouette_for_nonnoise(Xs, labels):
    mask = labels != -1
    unique_labels = np.unique(labels[mask]) if mask.any() else np.array([])
    if mask.sum() > 1 and unique_labels.size >= 2:
        score = silhouette_score(Xs[mask], labels[mask])
        return score
    return None

def plot_clusters_pca(Xs, labels, out_path, true_labels=None):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    proj = pca.fit_transform(Xs)
    df = pd.DataFrame(proj, columns=["PC1","PC2"])
    df["cluster"] = labels.astype(str)
    if true_labels is not None:
        df["true_label"] = true_labels.astype(str)
    plt.figure(figsize=(6,5))
    # noise label will be '-1' string; show it distinctively
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="cluster", palette="tab10", s=60, edgecolor='k')
    plt.title("DBSCAN clusters visualized with PCA (2D)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def main():
    print("Loading data...")
    X, y, iris = load_data()
    print("Scaling features...")
    Xs, scaler = preprocess(X)

    # choose min_samples and produce k-distance plot
    min_samples = 5
    print(f"Creating k-distance plot for k={min_samples} to help choose eps...")
    kdist_png = FIG_DIR / "k_distance.png"
    plot_k_distance(Xs, k=min_samples, out_path=kdist_png)
    print(f"Saved k-distance plot to: {kdist_png}")

    # Heuristic default eps (you should inspect k_distance.png and adjust)
    eps_default = 0.6
    print(f"Running DBSCAN with eps={eps_default}, min_samples={min_samples} (you can change eps after inspecting the k-distance plot)...")
    db, labels = run_dbscan(Xs, eps=eps_default, min_samples=min_samples)

    # Silhouette on non-noise points
    sil_score = silhouette_for_nonnoise(Xs, labels)
    if sil_score is not None:
        print(f"Silhouette score (non-noise): {sil_score:.4f}")
    else:
        print("Silhouette score not available (not enough clusters after removing noise).")

    # Save cluster visualization
    clusters_png = FIG_DIR / "dbscan_clusters_pca.png"
    print(f"Saving cluster PCA visualization to {clusters_png}")
    plot_clusters_pca(Xs, labels, clusters_png, true_labels=y)

    # Save model + scaler
    print(f"Saving DBSCAN model and scaler to {MODEL_PATH}")
    joblib.dump({"dbscan": db, "scaler": scaler, "features": list(X.columns)}, MODEL_PATH)

    # Save assignments CSV
    summary_df = X.copy()
    summary_df["cluster"] = labels
    summary_df["true_label"] = y
    summary_csv = OUT_DIR / "dbscan_cluster_assignments.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved assignments to {summary_csv}")

    print("Done. Inspect the k-distance plot and, if needed, re-run with a different eps value.")

if __name__ == "__main__":
    main()
