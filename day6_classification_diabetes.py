# Day 6: Classification – Predict Diabetes using Logistic Regression
# Author: Upasana
# Goal: Learn binary classification with the Pima Indians Diabetes dataset

# ---------- Imports ----------
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ---------- Step 1: Load the dataset ----------
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(url, names=columns)
print("Dataset loaded successfully!")
print("Shape of dataset:", df.shape)
print(df.head())

# ---------- Step 2: Prepare features (X) and target (y) ----------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ---------- Step 3: Split into train/test sets ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nData split complete: ")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ---------- Step 4: Standardize the data ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\nFeature scaling complete.")

# ---------- Step 5: Train the Logistic Regression model ----------
model = LogisticRegression(max_iter=500)
model.fit(X_train_scaled, y_train)
print("\nModel training complete.")

# ---------- Step 6: Make predictions ----------
y_pred = model.predict(X_test_scaled)

# ---------- Step 7: Evaluate the model ----------
print("\nEvaluation Metrics:")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# ---------- Step 8: Confusion Matrix ----------
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Diabetes", "Diabetes"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Diabetes Classification")
plt.tight_layout()

# Create a 'figures' folder if it doesn't exist
import os
os.makedirs("figures", exist_ok=True)

plt.savefig("figures/day6_confusion_matrix.png")
plt.show()

print("\nConfusion matrix saved in 'figures/day6_confusion_matrix.png'")
