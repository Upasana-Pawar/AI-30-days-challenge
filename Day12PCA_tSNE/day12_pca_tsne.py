"""
Day 12 — Dimensionality Reduction: PCA + t-SNE (updated)
Compatibility fix: TSNE uses `max_iter` in newer scikit-learn, older versions use `n_iter`.
This script will attempt to use `max_iter` first and fall back to `n_iter` if needed.
- Loads Iris dataset
- Standard scales features
- Runs PCA: prints explained variance ratios, saves scree/2D projection
- Runs t-SNE (2D) and saves visualization
- Saves PCA model + scaler in joblib and a CSV with projections
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

RND = 42

# Paths
OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)
MODEL_PATH = OUT_DIR / "pca_tsne_models.joblib"

def load_data():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="true_label")
    target_names = iris.target_names
    return X, y, target_names

def preprocess(X):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler

def run_pca(Xs, n_components=4):
    pca = PCA(n_components=n_components, random_state=RND)
    pcs = pca.fit_transform(Xs)
    return pca, pcs

def plot_explained_variance(pca, outpath):
    ratios = pca.explained_variance_ratio_
    cum = np.cumsum(ratios)
    plt.figure(figsize=(6,4))
    plt.bar(range(1, len(ratios)+1), ratios, alpha=0.7, label='individual')
    plt.step(range(1, len(ratios)+1), cum, where='mid', label='cumulative')
    plt.xticks(range(1, len(ratios)+1))
    plt.xlabel("Principal component")
    plt.ylabel("Explained variance ratio")
    plt.title("PCA explained variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_pca_2d(pcs, y, target_names, outpath):
    df = pd.DataFrame(pcs[:, :2], columns=["PC1", "PC2"])
    df["label"] = y.astype(str)
    plt.figure(figsize=(6,5))
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="label", palette="tab10", s=70, edgecolor='k')
    plt.title("PCA 2D projection")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def run_tsne(Xs, perplexity=30, learning_rate=200, n_iter=1000):
    """
    Attempt to create TSNE using `max_iter` (new scikit-learn) first.
    If that raises a TypeError (old scikit-learn expecting `n_iter`), fall back to `n_iter`.
    """
    tsne = None
    emb = None
    params_common = dict(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        random_state=RND,
        init='pca'
    )

    # Try with max_iter (newer sklearn)
    try:
        tsne = TSNE(**params_common, max_iter=n_iter)
        emb = tsne.fit_transform(Xs)
        return tsne, emb
    except TypeError:
        # fallback: old sklearn uses n_iter
        tsne = TSNE(**params_common, n_iter=n_iter)
        emb = tsne.fit_transform(Xs)
        return tsne, emb

def plot_tsne_2d(emb, y, outpath):
    df = pd.DataFrame(emb, columns=["tSNE1","tSNE2"])
    df["label"] = y.astype(str)
    plt.figure(figsize=(6,5))
    sns.scatterplot(data=df, x="tSNE1", y="tSNE2", hue="label", palette="tab10", s=70, edgecolor='k')
    plt.title("t-SNE 2D projection")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def main():
    print("Loading data...")
    X, y, target_names = load_data()
    print("Scaling features...")
    Xs, scaler = preprocess(X)

    print("Running PCA...")
    pca, pcs = run_pca(Xs, n_components=min(Xs.shape[1], 4))
    print("Explained variance ratios:", np.round(pca.explained_variance_ratio_, 4))
    plot_explained_variance(pca, FIG_DIR / "pca_explained_variance.png")
    plot_pca_2d(pcs, y, target_names, FIG_DIR / "pca_2d.png")

    print("Running t-SNE (this may take a few seconds)...")
    tsne, emb = run_tsne(Xs, perplexity=30, learning_rate=200, n_iter=1000)
    plot_tsne_2d(emb, y, FIG_DIR / "tsne_2d.png")

    # Save models + scaler
    print("Saving PCA model and scaler to", MODEL_PATH)
    joblib.dump({"pca": pca, "tsne": tsne, "scaler": scaler}, MODEL_PATH)

    # Save projections to CSV
    proj_df = X.copy()
    proj_df["PC1"] = pcs[:,0]
    proj_df["PC2"] = pcs[:,1]
    proj_df["tSNE1"] = emb[:,0]
    proj_df["tSNE2"] = emb[:,1]
    proj_df["true_label"] = y
    proj_csv = OUT_DIR / "projections.csv"
    proj_df.to_csv(proj_csv, index=False)
    print("Saved projections to", proj_csv)

    print("Done. Figures saved in ./figures, models saved in pca_tsne_models.joblib")

if __name__ == "__main__":
    main()
