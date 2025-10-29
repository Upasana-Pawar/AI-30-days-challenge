# ==============================================
# Day 30 — Heart Disease Capstone
# Step 3: Data Preprocessing & Model Building
# ==============================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==============================
# 1. Load Data
# ==============================
df = pd.read_csv("heart.csv")
print("✅ Data loaded successfully — Shape:", df.shape)

# ==============================
# 2. Basic Cleaning
# ==============================
# Drop duplicates if any
df.drop_duplicates(inplace=True)

# Check target column
if 'target' not in df.columns:
    raise ValueError("❌ No 'target' column found in dataset. Please verify column names.")

# Split features and target
X = df.drop("target", axis=1)
y = df["target"]

# ==============================
# 3. Split Train/Test
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("🔹 Train shape:", X_train.shape, " Test shape:", X_test.shape)

# ==============================
# 4. Feature Scaling
# ==============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# 5. Model Training
# ==============================
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n🚀 Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"✅ {name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

# ==============================
# 6. Compare Models
# ==============================
results_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
print("\n📊 Model Comparison:\n", results_df)

plt.figure(figsize=(6,4))
sns.barplot(data=results_df, x="Model", y="Accuracy", palette="viridis")
plt.title("Model Comparison on Test Set")
plt.ylim(0,1)
plt.tight_layout()

# Save the comparison chart
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/day30_model_comparison.png")
plt.show()

# ==============================
# 7. Save Best Model
# ==============================
best_model_name = results_df.loc[results_df["Accuracy"].idxmax(), "Model"]
best_model = models[best_model_name]
joblib.dump(best_model, f"outputs/best_model_{best_model_name.replace(' ', '_')}.joblib")
print(f"\n💾 Best model saved as: best_model_{best_model_name.replace(' ', '_')}.joblib")

# Save scaler
joblib.dump(scaler, "outputs/scaler.joblib")

# ==============================
# 8. Confusion Matrix for Best Model
# ==============================
y_pred_best = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"{best_model_name} — Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/day30_confusion_matrix.png")
plt.show()

print("\n✅ Step 3 completed successfully — Models trained, evaluated, and saved.")
