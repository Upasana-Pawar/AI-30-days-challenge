import pandas as pd

# Load dataset
df = pd.read_csv("Datasets/titanic_data.csv")
print(df.columns)

# Show first 5 rows
print(df.head())

# Average age
print("Average Age:", df["Age"].mean())

# Survivors by gender
print("Survivors by Gender:\n", df.groupby("Sex")["Survived"].sum())

# Survival rate by class
print("Survival Rate by Class:\n", df.groupby("Pclass")["Survived"].mean())

# Youngest and oldest passenger
print("Youngest Passenger:", df["Age"].min())
print("Oldest Passenger:", df["Age"].max())
