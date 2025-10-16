# day17_evaluation.py
"""
Day 17 — Model Evaluation & Metrics Comparison
Author: Upasana
Purpose: Train a classifier on the Titanic dataset and generate evaluation metrics & plots.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)

# Output paths
OUTPUT_DIR = Path(__file__).resolve().parent
MODEL_PATH = OUTPUT_DIR / "day17_model.joblib"
CONFUSION_PNG = OUTPUT_DIR / "day17_confusion_matrix.png"
ROC_PNG = OUTPUT_DIR / "day17_roc_curve.png"
PR_PNG = OUTPUT_DIR / "day17_pr_curve.png"
METRICS_TXT = OUTPUT_DIR / "day17_metrics.txt"

def load_and_create_features():
    import seaborn as sns
    df = sns.load_dataset("titanic")
    # Basic feature creation (same pattern as Day14)
    df["family_size"] = df["sibsp"].fillna(0) + df["parch"].fillna(0) + 1
    df["is_alone"] = (df["family_size"] == 1).astype(int)
    df["age_group"] = pd.cut(df["age"], bins=[0, 12, 20, 40, 60, 120],
                             labels=["child","teen","adult","mid","senior"])
    return df

def build_pipeline(numeric_features, categorical_features):
    # numeric transformer
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # OneHotEncoder kwargs compatible across sklearn versions
    skl_version = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    ohe_kwargs = {"handle_unknown": "ignore"}
    if skl_version >= (1, 2):
        ohe_kwargs["sparse_output"] = False
    else:
        ohe_kwargs["sparse"] = False

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(**ohe_kwargs))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ], remainder="drop", verbose_feature_names_out=False)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    return pipeline

def plot_confusion(cm, labels, outpath):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_roc(y_true, y_scores, outpath):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0,1],[0,1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_pr(y_true, y_scores, outpath):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    plt.figure(figsize=(6,5))
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def main():
    print("Day 17 — Model Evaluation & Metrics Comparison")
    df = load_and_create_features()

    numeric_features = ["age", "fare", "family_size"]
    categorical_features = ["sex", "class", "embarked", "age_group"]

    X = df[numeric_features + categorical_features].copy()
    y = df["survived"]

    # Drop rows where target is missing
    mask = y.notna()
    X = X[mask]
    y = y[mask].astype(int)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = build_pipeline(numeric_features, categorical_features)
    pipeline.fit(X_train, y_train)

    # Predictions and probabilities
    y_pred = pipeline.predict(X_test)
    # use predict_proba for positive-class score
    try:
        y_score = pipeline.predict_proba(X_test)[:, 1]
    except Exception:
        # fallback: use decision_function if available, else use predicted labels
        try:
            y_score = pipeline.decision_function(X_test)
        except Exception:
            y_score = y_pred

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_test, y_score)
    except Exception:
        roc_auc = float("nan")
    ap = average_precision_score(y_test, y_score)

    # Save metrics text
    with open(METRICS_TXT, "w") as f:
        f.write("Day 17 — Model Evaluation Metrics\n")
        f.write("===============================\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n")
        f.write(f"Average Precision (AP): {ap:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred, digits=4))
    print("Saved metrics to", METRICS_TXT.name)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion(cm, labels=["Not Survived","Survived"], outpath=CONFUSION_PNG)
    print("Saved confusion matrix to", CONFUSION_PNG.name)

    # ROC curve
    try:
        plot_roc(y_test, y_score, ROC_PNG)
        print("Saved ROC curve to", ROC_PNG.name)
    except Exception as e:
        print("ROC plotting failed:", e)

    # Precision-Recall curve
    try:
        plot_pr(y_test, y_score, PR_PNG)
        print("Saved PR curve to", PR_PNG.name)
    except Exception as e:
        print("PR plotting failed:", e)

    # Save model
    joblib.dump(pipeline, MODEL_PATH)
    print("Saved trained pipeline to", MODEL_PATH.name)

    print("\nDone — check the generated files in the folder.")

if __name__ == "__main__":
    main()
