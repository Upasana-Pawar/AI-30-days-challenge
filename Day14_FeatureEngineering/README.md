# Day 14 — Feature Engineering & Pipelines

## Goal
Demonstrate reproducible feature engineering and preprocessing using scikit-learn `ColumnTransformer` and `Pipeline`. We use the `titanic` dataset (from seaborn) to show:
- missing value imputation
- categorical encoding (OneHot)
- scaling numeric features
- new feature creation (family_size, is_alone, age_group)
- mapping feature importances back to feature names
- saving a pipeline (`joblib`) for later inference

## Files
- `day14_feature_engineering.py` — full script (train, evaluate, save pipeline).
- `requirements.txt` — Python dependencies.
- After running the script, these files will be created:
  - `day14_model.joblib` — saved pipeline (preprocessor + classifier).
  - `day14_report.png` — bar chart of feature importances (if computed).
  - `day14_confusion_matrix.png` — confusion matrix image.

