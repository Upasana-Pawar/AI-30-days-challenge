"""
Day 11 — Hierarchical Clustering (Agglomerative)
Uses Iris dataset to demonstrate Ward linkage and dendrogram visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Paths
OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)

MODEL_PATH = OUT_DIR / "hierarchical_model.joblib"

def main():
    print("Loading Iris dataset...")
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="true_label")

    print("Scaling data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Computing linkage matrix for dendrogram...")
    Z = linkage(X_scaled, method="ward")

    print("Saving dendrogram plot...")
    plt.figure(figsize=(8, 5))
    dendrogram(Z)
    plt.title("Hierarchical Clustering Dendrogram (Ward linkage)")
    plt.xlabel("Sample index")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "dendrogram.png", dpi=150)
    plt.close()

    # Choose number of clusters (e.g., 3 for Iris)
    n_clusters = 3
    print(f"Fitting Agglomerative Clustering with {n_clusters} clusters...")
    clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = clusterer.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)
    print(f"Silhouette score: {sil:.4f}")

    # PCA for visualization
    print("Plotting PCA cluster visualization...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df["cluster"] = labels.astype(str)

    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="cluster", palette="tab10", s=60, edgecolor='k')
    plt.title("Agglomerative Clustering (Ward) - PCA Visualization")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pca_clusters.png", dpi=150)
    plt.close()

    # Save outputs
    joblib.dump({"model": clusterer, "scaler": scaler}, MODEL_PATH)
    X["cluster"] = labels
    X["true_label"] = y
    X.to_csv(OUT_DIR / "cluster_assignments.csv", index=False)

    print("All done! Figures and model saved in Day11Hierarchical/figures/")

if __name__ == "__main__":
    main()
