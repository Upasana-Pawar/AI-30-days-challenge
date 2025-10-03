# Day 4 - Pandas Practice
# Goal: Clean movies dataset, create decade feature, analyze genres, and save results

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid", context="notebook")
plt.rcParams['figure.dpi'] = 110

#  path to the dataset (relative to this script)
CSV_PATH = r"C:\Users\upasa\Desktop\AI 30 day challenge\Day2Pandas\Datasets\movies_data.csv"


# Output folder for results
OUTDIR = "figures_day4"
os.makedirs(OUTDIR, exist_ok=True)

# Load dataset
df = pd.read_csv(CSV_PATH)
print("Original shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head(5))

# Detect important columns
def find_col(candidates):
    for c in candidates:
        for col in df.columns:
            if col.lower() == c.lower():
                return col
    return None

col_rating = find_col(['rating', 'imdb_rating', 'avg_vote'])
col_genre = find_col(['genre', 'genres', 'category'])
col_year = find_col(['year', 'release_year', 'release date'])
col_title = find_col(['title', 'movie_title', 'name'])

print("Detected:", col_rating, col_genre, col_year, col_title)

# --- Data cleaning ---
# Drop rows without title
if col_title:
    df = df.dropna(subset=[col_title])

# Convert year to integer
if col_year:
    df[col_year] = pd.to_numeric(df[col_year], errors="coerce").astype("Int64")

# Convert rating to numeric
if col_rating:
    df[col_rating] = pd.to_numeric(df[col_rating], errors="coerce")

# Create 'decade' column
if col_year:
    df["decade"] = (df[col_year] // 10) * 10

# Clean genre column
if col_genre:
    df["first_genre"] = df[col_genre].astype(str).str.split("[,|;]").str[0].str.strip()

# --- Grouped analysis ---
if col_rating and col_year:
    avg_rating_decade = (
        df.dropna(subset=[col_rating, "decade"])
          .groupby("decade")[col_rating].mean().reset_index()
    )
    print("\nAverage rating per decade:\n", avg_rating_decade.head())
    avg_rating_decade.to_csv(os.path.join(OUTDIR, "avg_rating_by_decade.csv"), index=False)

if "first_genre" in df.columns:
    genre_counts = df["first_genre"].value_counts().reset_index()
    genre_counts.columns = ["genre", "count"]
    print("\nTop genres:\n", genre_counts.head())
    genre_counts.to_csv(os.path.join(OUTDIR, "genre_counts.csv"), index=False)

# --- Visualizations ---
if col_rating and "decade" in df.columns:
    plt.figure(figsize=(8,4))
    sns.lineplot(data=avg_rating_decade, x="decade", y=col_rating, marker="o")
    plt.title("Average Rating by Decade")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "avg_rating_by_decade.png"))

if "first_genre" in df.columns:
    top10 = genre_counts.head(10)
    plt.figure(figsize=(7,4))
    sns.barplot(data=top10, x="count", y="genre")
    plt.title("Top 10 Genres")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "top10_genres.png"))

# Save cleaned dataset
CLEANED_CSV = "movies_data_cleaned.csv"
df.to_csv(CLEANED_CSV, index=False)
print("\nSaved cleaned dataset to:", CLEANED_CSV)
print("Figures saved in:", OUTDIR)
