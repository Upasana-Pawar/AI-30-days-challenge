import pandas as pd

df = pd.read_csv("Datasets\movies_data.csv")
print(df.head())
print("Columns:", df.columns)

# Top 5 highest-rated movies
print(df.sort_values("Rating", ascending=False).head(5))

# Average rating by genre
print(df.groupby("Genre")["Rating"].mean())

# Movies released per year
print(df["Year"].value_counts().sort_index())

# Most common genre
print("Most Common Genre:", df["Genre"].value_counts().idxmax())
