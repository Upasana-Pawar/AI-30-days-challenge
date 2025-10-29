# ==============================================
# Day 30 — Heart Disease Capstone
# Step 4: Model Explainability (SHAP + Feature Importance)
# ==============================================

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# ==============================
# 1. Load Saved Artifacts
# ==============================
model_path = [f for f in os.listdir("outputs") if f.startswith("best_model_")][0]
model = joblib.load(f"outputs/{model_path}")
scaler = joblib.load("outputs/scaler.joblib")

print(f"✅ Loaded model: {model_path}")

# ==============================
# 2. Load Dataset
# ==============================
df = pd.read_csv("heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

# Scale features using the same scaler
X_scaled = scaler.transform(X)

# ==============================
# 3. Feature Importance (for tree-based models)
# ==============================
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    feature_imp = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(8,5))
    sns.barplot(data=feature_imp, x="Importance", y="Feature", palette="mako")
    plt.title("Feature Importance (Best Model)")
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/day30_feature_importance.png")
    plt.show()
else:
    print("⚠️ Model does not have built-in feature_importances_ (e.g., Logistic Regression). Skipping plot.")

# ==============================
# 4. SHAP Explainability
# ==============================
print("\n💡 Computing SHAP values... (this might take a minute)")
explainer = shap.Explainer(model, X_scaled)
shap_values = explainer(X_scaled)

# Summary Plot
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("outputs/day30_shap_summary_bar.png")
plt.show()

# Detailed Summary
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig("outputs/day30_shap_summary_scatter.png")
plt.show()

print("\n✅ Step 4 completed — SHAP & feature importance visualizations saved.")
