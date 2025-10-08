
# Day 9 — K-Means Clustering (Iris)

## What I did
- Loaded Iris dataset.
- Scaled features using `StandardScaler`.
- Analyzed `k` using **Elbow method** (inertia) and **Silhouette score**.
- Trained final `KMeans` with best `k` (based on silhouette).
- Visualized clusters using PCA (2D).
- Saved plots in `figures/`, model as `kmeans_model.joblib`, and `cluster_assignments.csv`.

## How to run (PowerShell)
1. Activate your venv (PowerShell):
   ```powershell
   .\.venv\Scripts\Activate.ps1
