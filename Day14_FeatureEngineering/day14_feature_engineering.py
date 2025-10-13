# day14_feature_engineering.py
"""
Day 14 — Feature Engineering & Pipelines
Author: Upasana
Purpose: Demo preprocessing (impute, encode, scale), feature creation,
         ColumnTransformer + Pipeline, training, evaluation, and saving the pipeline.
Dataset: seaborn 'titanic' dataset (built-in)
"""

import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, FunctionTransformer, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

OUTPUT_DIR = Path(__file__).resolve().parent
MODEL_PATH = OUTPUT_DIR / "day14_model.joblib"
REPORT_PNG = OUTPUT_DIR / "day14_report.png"

def load_data():
    import seaborn as sns
    df = sns.load_dataset("titanic")
    return df

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    # Create a small engineered feature: family_size = sibsp + parch + 1 (self)
    df = df.copy()
    df["family_size"] = df["sibsp"].fillna(0) + df["parch"].fillna(0) + 1
    # a boolean feature: is_alone
    df["is_alone"] = (df["family_size"] == 1).astype(int)
    # bin age into groups (simple)
    df["age_group"] = pd.cut(df["age"], bins=[0, 12, 20, 40, 60, 120], labels=["child","teen","adult","mid","senior"])
    return df

def get_feature_sets(df: pd.DataFrame):
    # target
    y = df["survived"]

    # choose a compact set of features for the demo
    numeric_features = ["age", "fare", "family_size"]
    categorical_features = ["sex", "class", "embarked", "age_group"]
    # We'll drop columns that are not used
    X = df[numeric_features + categorical_features].copy()
    return X, y, numeric_features, categorical_features

def build_pipeline(numeric_features, categorical_features):
    # Numeric transformer: impute then scale
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical transformer: for low-cardinality use OneHot
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

    # Combine with ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ], remainder="drop", verbose_feature_names_out=False)

    # Full pipeline: preprocess -> classifier
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        # optional: polynomial features (commented out; use only if needed)
        # ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    return clf

def explain_feature_names(preprocessor, numeric_features, categorical_features):
    # Extract transformed feature names for reporting / feature importances
    # numeric names are simple
    num_out = numeric_features
    # categorical names from OneHotEncoder
    cat_transformer = preprocessor.named_transformers_.get("cat")
    if cat_transformer is None:
        cat_out = []
    else:
        # access OneHotEncoder inside pipeline
        ohe = None
        if isinstance(cat_transformer, Pipeline):
            for name, step in cat_transformer.steps:
                if isinstance(step, OneHotEncoder):
                    ohe = step
                    break
        else:
            if isinstance(cat_transformer, OneHotEncoder):
                ohe = cat_transformer

        if ohe is not None:
            # sklearn >=1.0: get_feature_names_out available
            try:
                cat_out = list(ohe.get_feature_names_out(categorical_features))
            except Exception:
                # fallback if API differs
                cat_out = []
                for i, cats in enumerate(ohe.categories_):
                    for cat in cats:
                        cat_out.append(f"{categorical_features[i]}_{cat}")
        else:
            cat_out = []

    feature_names = num_out + cat_out
    return feature_names

def main():
    print("Day 14 — Feature Engineering & Pipelines")
    df = load_data()
    print(f"Raw data shape: {df.shape}")

    df = create_features(df)
    X, y, num_feats, cat_feats = get_feature_sets(df)

    # quick look
    print("\nSample after feature creation:")
    print(X.head())

    # train / test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline = build_pipeline(num_feats, cat_feats)

    # cross-validated score (quick)
    print("\nRunning 3-fold cross-validation (accuracy)...")
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring="accuracy", n_jobs=-1)
    print("CV accuracy scores:", np.round(cv_scores, 3))
    print("CV mean accuracy:", np.round(cv_scores.mean(), 3))

    # fit on full train set
    pipeline.fit(X_train, y_train)

    # predict & evaluate
    y_pred = pipeline.predict(X_test)
    print("\nTest accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3))

    # feature importances (mapped back to names)
    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        feat_names = explain_feature_names(preprocessor, num_feats, cat_feats)
        importances = classifier.feature_importances_
        feat_imp = pd.Series(importances, index=feat_names).sort_values(ascending=False)
        print("\nTop feature importances:")
        print(feat_imp.head(10))

        # plot importances
        plt.figure(figsize=(8, 5))
        sns.barplot(x=feat_imp.values[:12], y=feat_imp.index[:12])
        plt.title("Top feature importances")
        plt.tight_layout()
        plt.savefig(REPORT_PNG)
        print(f"\nSaved report plot to: {REPORT_PNG}")
    except Exception as e:
        print("Could not extract feature importances automatically:", e)

    # confusion matrix heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    # append to same PNG or save separately; we'll save a separate one
    plt.savefig(OUTPUT_DIR / "day14_confusion_matrix.png")
    print(f"Saved confusion matrix to: {OUTPUT_DIR / 'day14_confusion_matrix.png'}")

    # save pipeline (preprocessor + model)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nSaved pipeline (model + preprocessing) to: {MODEL_PATH}")

    print("\nDone. You can use the saved pipeline for inference later using joblib.load()")

if __name__ == "__main__":
    main()
