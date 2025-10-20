# Day 21 — Stacked Ensemble of Tuned Boosters

## Overview
Create a stacked ensemble that blends tuned boosters (XGBoost, LightGBM, CatBoost) and HistGradientBoosting as a fallback. Uses LogisticRegression as the meta-learner. Evaluates models on accuracy and ROC-AUC.

The script will:
- Try to load tuned models from `Day20_HyperparameterTuning_Boosters/day20_artifacts/day20_models/` (if present).
- Otherwise, use sensible defaults for available boosters.
- Train base estimators, train stacking meta-model, evaluate, and save artifacts.

