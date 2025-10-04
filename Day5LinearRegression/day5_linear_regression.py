# Day 5: Linear Regression with Scikit-Learn
# Project: Predict house prices

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load dataset (using sklearn's built-in dataset for simplicity)
from sklearn.datasets import fetch_california_housing

print("📊 Loading dataset...")
housing = fetch_california_housing(as_frame=True)
df = housing.frame

print("Dataset shape:", df.shape)
print(df.head())

# Step 2: Select features (X) and target (y)
X = df[['MedInc']]  # Median income
y = df['MedHouseVal']  # Median house value

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train model
model = LinearRegression()
model.fit(X_train, y_train)

print("✅ Model trained.")

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Step 7: Visualization
plt.scatter(X_test, y_test, color="blue", label="Actual")
plt.scatter(X_test, y_pred, color="red", alpha=0.5, label="Predicted")
plt.xlabel("Median Income")
plt.ylabel("House Value")
plt.legend()
plt.title("Linear Regression: Income vs House Value")
plt.savefig("figures/day5_linear_regression.png")
plt.show()

print("🎉 Plot saved in figures/day5_linear_regression.png")
