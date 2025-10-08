"""
Day 9 — K-Means Clustering (Iris)
Script does:
 - load iris dataset
 - scale features
 - run k-means for a range of k (elbow)
 - compute silhouette score for best K
 - run final KMeans, save model with joblib
 - create plots: elbow, silhouette (k vs score), 2D PCA clusters
 - save plots under ./figures/
"""

import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# --- configuration ---
OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)
MODEL_PATH = OUT_DIR / "kmeans_model.joblib"
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

def find_elbow_and_silhouette(Xs, k_min=2, k_max=10):
    inertias = []
    silhouettes = []
    K_range = list(range(k_min, k_max+1))
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(Xs)
        inertias.append(km.inertia_)
        # silhouette needs at least 2 clusters and <n_samples clusters
        if 1 < k < Xs.shape[0]:
            silhouettes.append(silhouette_score(Xs, labels))
        else:
            silhouettes.append(np.nan)
    return K_range, inertias, silhouettes

def plot_elbow(K_range, inertias, out_path):
    plt.figure(figsize=(6,4))
    plt.plot(K_range, inertias, marker='o')
    plt.xticks(K_range)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia (within-cluster SSE)")
    plt.title("Elbow Method for KMeans")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_silhouette(K_range, silhouettes, out_path):
    plt.figure(figsize=(6,4))
    plt.plot(K_range, silhouettes, marker='o')
    plt.xticks(K_range)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette score vs k")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_clusters_pca(Xs, labels, out_path, true_labels=None):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    proj = pca.fit_transform(Xs)
    df = pd.DataFrame(proj, columns=["PC1","PC2"])
    df["cluster"] = labels.astype(str)
    if true_labels is not None:
        df["true_label"] = true_labels.astype(str)
    plt.figure(figsize=(6,5))
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="cluster", palette="tab10", s=60, edgecolor='k')
    plt.title("KMeans clusters visualized with PCA (2D)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def main():
    print("Loading data...")
    X, y, iris = load_data()
    print(f"Data shape: {X.shape}")

    print("Scaling features...")
    Xs, scaler = preprocess(X)

    print("Computing elbow and silhouette scores...")
    K_range, inertias, silhouettes = find_elbow_and_silhouette(Xs, k_min=2, k_max=8)

    elbow_png = FIG_DIR / "elbow.png"
    silhouette_png = FIG_DIR / "silhouette.png"
    clusters_png = FIG_DIR / "clusters_pca.png"

    print(f"Saving elbow plot to {elbow_png}")
    plot_elbow(K_range, inertias, elbow_png)

    print(f"Saving silhouette plot to {silhouette_png}")
    plot_silhouette(K_range, silhouettes, silhouette_png)

    # heuristic: choose k with highest silhouette (>=2)
    valid_pairs = [(k,s) for k,s in zip(K_range, silhouettes) if not np.isnan(s)]
    best_k = max(valid_pairs, key=lambda t: t[1])[0]
    print(f"Best k by silhouette: {best_k}")

    print(f"Training final KMeans with k={best_k}...")
    final_km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
    final_labels = final_km.fit_predict(Xs)

    print(f"Saving cluster visualization to {clusters_png}")
    plot_clusters_pca(Xs, final_labels, clusters_png, true_labels=y)

    # save model and scaler
    print(f"Saving KMeans model to {MODEL_PATH}")
    joblib.dump({"kmeans": final_km, "scaler": scaler, "features": list(X.columns)}, MODEL_PATH)

    # Save a small summary CSV with cluster assignments
    summary_df = X.copy()
    summary_df["cluster"] = final_labels
    summary_df["true_label"] = y
    summary_csv = OUT_DIR / "cluster_assignments.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved assignments to {summary_csv}")

    print("Done. Figures saved in ./figures, model saved as kmeans_model.joblib")

if __name__ == "__main__":
    main()
