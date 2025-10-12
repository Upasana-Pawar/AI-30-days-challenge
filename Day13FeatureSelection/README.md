# Day 13 — Feature Selection & Correlation Analysis

## What I did
- Loaded Iris dataset and scaled features.
- Visualized feature correlations and listed high-correlation pairs.
- Used VarianceThreshold to detect near-constant features.
- Used SelectKBest (ANOVA F-test) to choose top-k features.
- Used RandomForest to compute feature importances.
- Combined scores into a single `selected_features_summary.csv`.
- Saved selector objects and models in `feature_selectors.joblib`.

