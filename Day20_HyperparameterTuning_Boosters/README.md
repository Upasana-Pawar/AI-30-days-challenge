# Day 20 — Hyperparameter Tuning for Boosters (Optuna + RandomizedSearchCV fallback)

## Overview
This day covers hyperparameter tuning for advanced boosters:
- Prefers Optuna (Bayesian-style tuning).
- Falls back to RandomizedSearchCV when Optuna isn't installed.
- Tunes: XGBoost, LightGBM, CatBoost (if available), and sklearn's HistGradientBoosting.
- Evaluates default vs tuned models on accuracy and ROC-AUC.

