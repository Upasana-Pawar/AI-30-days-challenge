from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Paths
MODEL_PATH = Path(__file__).resolve().parent / "day16_best_model.joblib"
REPORT_PATH = Path(__file__).resolve().parent / "day16_tuning_report.png"

def main():
    print("\n🎯 Day 16 — Hyperparameter Tuning (GridSearchCV & RandomizedSearchCV)")

    # Load dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Base model
    model = RandomForestClassifier(random_state=42)

    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # GridSearchCV
    print("\n🔎 Running GridSearchCV ...")
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    print(f"\n✅ Best Grid Params: {grid_search.best_params_}")
    print(f"GridSearch Accuracy: {grid_search.best_score_:.3f}")

    # RandomizedSearchCV (faster)
    print("\n⚡ Running RandomizedSearchCV ...")
    random_search = RandomizedSearchCV(
        model, param_distributions=param_grid, n_iter=10, cv=5,
        n_jobs=-1, random_state=42, verbose=1
    )
    random_search.fit(X_train, y_train)
    print(f"\n✅ Best Random Params: {random_search.best_params_}")
    print(f"RandomSearch Accuracy: {random_search.best_score_:.3f}")

    # Choose best
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🏆 Final Model Accuracy: {acc:.3f}")

    # Plot confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix — Best Tuned Model")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(REPORT_PATH)
    print(f"\n📊 Saved report to {REPORT_PATH.name}")

    # Save model
    joblib.dump(best_model, MODEL_PATH)
    print(f"💾 Saved best model as {MODEL_PATH.name}")

if __name__ == "__main__":
    main()
