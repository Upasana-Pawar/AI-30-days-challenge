
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

# ---------- Styling ----------
sns.set(style="darkgrid", context="notebook")
plt.rcParams['figure.dpi'] = 110

# ---------- Paths ----------
# Determine project root (two levels up from this file: Day3Matplotlib -> project root)
THIS_DIR = os.path.dirname(__file__)                      # .../Day3Matplotlib
PROJECT_ROOT = os.path.dirname(THIS_DIR)                  # .../AI 30 day challenge
OUTDIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(OUTDIR, exist_ok=True)

# ---------- 1) Find a CSV automatically ----------
csv_candidates = glob.glob(os.path.join(PROJECT_ROOT, "**", "*.csv"), recursive=True)
if not csv_candidates:
    raise FileNotFoundError("No CSV files found in project. Put your dataset anywhere under the project folder.")

# Prefer files with 'movie' in the filename (case-insensitive)
csv_selected = None
for p in csv_candidates:
    if "movie" in os.path.basename(p).lower():
        csv_selected = p
        break
if csv_selected is None:
    # fallback: use first CSV found
    csv_selected = csv_candidates[0]

print("Using CSV:", csv_selected)

# ---------- 2) Load CSV ----------
# read CSV with pandas (default options); if encoding issues arise, you may add encoding='utf-8' or 'latin1'
df = pd.read_csv(csv_selected)
print("Rows, cols:", df.shape)

# ---------- 3) Helper: find column by candidate names ----------
def find_col(df, candidates):
    # Exact match first
    for c in candidates:
        if c in df.columns:
            return c
    # Case-insensitive match
    lower_map = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None

# Candidate name groups (expand if your dataset uses other names)
rating_cands = ["rating", "Rating", "imdb_rating", "IMDB_Rating", "avg_vote", "score", "ratingValue"]
genre_cands = ["genre", "genres", "Genre", "category", "Category"]
runtime_cands = ["runtime", "Runtime", "duration", "length"]
revenue_cands = ["revenue", "Revenue", "box_office", "box office", "gross"]
budget_cands = ["budget", "Budget", "cost"]
year_cands = ["year", "Year", "release_year", "releaseDate", "release_date"]

# ---------- 4) Heuristic: coerce numeric-like columns to numeric ----------
# For each column, take a small sample, strip commas/symbols and check fraction numeric-like -> coerce if majority numeric
for col in df.columns:
    try:
        sample = df[col].dropna().astype(str).head(200).str.replace(",", "")
        # remove currency symbols and letters for check
        sample_clean = sample.str.replace(r"[^\d\.\-]", "", regex=True)
        if len(sample_clean) == 0:
            continue
        # fraction of values that are numeric-looking
        num_frac = sample_clean.apply(lambda s: s.replace(".", "", 1).lstrip("-").isdigit()).mean()
        if num_frac > 0.7:
            # coerce to numeric for the whole column
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
    except Exception:
        # If anything goes wrong, skip that column
        pass

# ---------- 5) Detect columns ----------
rating_col = find_col(df, rating_cands)
genre_col = find_col(df, genre_cands)
runtime_col = find_col(df, runtime_cands)
revenue_col = find_col(df, revenue_cands)
budget_col = find_col(df, budget_cands)
year_col = find_col(df, year_cands)

print("Detected columns -> rating:", rating_col, "genre:", genre_col,
      "runtime:", runtime_col, "revenue:", revenue_col, "budget:", budget_col, "year:", year_col)

# ---------- 6) PLOTS ----------
# Each block checks for required columns and data, then saves a PNG into OUTDIR.
# Helpful: use plt.close() after saving so the script doesn't accumulate open figures.

# 6.1 Histogram: Rating distribution (if rating found)
if rating_col and df[rating_col].dropna().shape[0] > 0:
    plt.figure()
    sns.histplot(df[rating_col].dropna(), bins=20)
    plt.title(f"Distribution of {rating_col}")
    plt.xlabel(rating_col)
    plt.ylabel("Count")
    outpath = os.path.join(OUTDIR, "rating_hist.png")
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
    print("✅ Saved:", os.path.basename(outpath))
else:
    print("⚠️ Skipped rating histogram (no rating column found)")

# 6.2 Bar chart: Top categories/genres (if genre found)
if genre_col and df[genre_col].dropna().shape[0] > 0:
    ser = df[genre_col].dropna().astype(str)
    # If entries contain separators like ',' or '|' treat them as multi-genres and explode
    if ser.str.contains(r"[|,;]").any():
        exploded = ser.str.split(r"[|,;]").explode().str.strip()
        top = exploded.value_counts().nlargest(10)
    else:
        top = ser.value_counts().nlargest(10)
    plt.figure(figsize=(8,5))
    sns.barplot(x=top.values, y=top.index)
    plt.title(f"Top 10 categories ({genre_col})")
    plt.xlabel("Count")
    plt.ylabel(genre_col)
    outpath = os.path.join(OUTDIR, "top_genres.png")
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
    print("✅ Saved:", os.path.basename(outpath))
else:
    print("⚠️ Skipped genres bar chart (no genre column found)")

# 6.3 Boxplot: Rating by Genre (useful if rating + genre exist)
if rating_col and genre_col:
    # prepare DataFrame to handle multi-genre cells
    ser = df[[genre_col, rating_col]].dropna()
    if ser[genre_col].astype(str).str.contains(r"[|,;]").any():
        ser = ser.assign(_g=ser[genre_col].astype(str).str.split(r"[|,;]")).explode("_g")
        ser['__genre_exploded'] = ser['_g'].str.strip()
        plot_df = ser[['__genre_exploded', rating_col]].rename(columns={'__genre_exploded': genre_col})
    else:
        plot_df = ser
    top_g = plot_df[genre_col].value_counts().nlargest(6).index.tolist()
    plot_df = plot_df[plot_df[genre_col].isin(top_g)]
    if not plot_df.empty:
        plt.figure(figsize=(10,6))
        sns.boxplot(data=plot_df, x=genre_col, y=rating_col)
        plt.xticks(rotation=30)
        plt.title(f"{rating_col} distribution by {genre_col} (top 6)")
        outpath = os.path.join(OUTDIR, "rating_by_genre_boxplot.png")
        plt.savefig(outpath, bbox_inches="tight")
        plt.close()
        print("✅ Saved:", os.path.basename(outpath))
    else:
        print("⚠️ Skipped rating-by-genre (no data after filtering)")
else:
    print("⚠️ Skipped rating-by-genre (need both rating and genre columns)")

# 6.4 Average rating by year (line plot) -- if rating + year exist
if rating_col and year_col:
    tmp = df[[year_col, rating_col]].dropna().copy()
    # attempt to coerce year to int for grouping (works if year is numeric-like)
    try:
        tmp[year_col] = tmp[year_col].astype(int)
    except Exception:
        pass
    agg = tmp.groupby(year_col)[rating_col].mean().reset_index().sort_values(year_col)
    if not agg.empty:
        plt.figure(figsize=(10,5))
        sns.lineplot(data=agg, x=year_col, y=rating_col, marker="o")
        plt.title(f"Average {rating_col} by {year_col}")
        plt.xlabel(year_col)
        plt.ylabel(f"Avg {rating_col}")
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=12))
        plt.tight_layout()
        outpath = os.path.join(OUTDIR, "avg_rating_by_year.png")
        plt.savefig(outpath, bbox_inches="tight")
        plt.close()
        print("✅ Saved:", os.path.basename(outpath))
    else:
        print("⚠️ Skipped avg rating by year (no aggregated data)")
else:
    print("⚠️ Skipped avg rating by year (need both rating and year columns)")

# 6.5 Movies per year (count)
if year_col:
    cnt = df[year_col].dropna().astype(str).value_counts().sort_index()
    if not cnt.empty:
        plt.figure(figsize=(10,5))
        sns.barplot(x=cnt.index, y=cnt.values)
        plt.xticks(rotation=60)
        plt.title("Number of movies per Year")
        plt.xlabel(year_col)
        plt.ylabel("Count")
        outpath = os.path.join(OUTDIR, "movies_per_year.png")
        plt.savefig(outpath, bbox_inches="tight")
        plt.close()
        print("✅ Saved:", os.path.basename(outpath))
    else:
        print("⚠️ Skipped movies-per-year (no year data)")
else:
    print("⚠️ Skipped movies-per-year (no year column)")

# 6.6 Top directors (optional): try common director column names
dir_candidates = ["director", "Director", "Directors", "Director(s)", "director_name"]
dir_col = None
for c in dir_candidates:
    if c in df.columns:
        dir_col = c
        break
# case-insensitive fallback
if dir_col is None:
    lower_map = {col.lower(): col for col in df.columns}
    for c in dir_candidates:
        if c.lower() in lower_map:
            dir_col = lower_map[c.lower()]
            break

if dir_col:
    top_dirs = df[dir_col].dropna().astype(str).value_counts().nlargest(10)
    if not top_dirs.empty:
        plt.figure(figsize=(8,5))
        sns.barplot(x=top_dirs.values, y=top_dirs.index)
        plt.title("Top 10 Directors (by movie count)")
        plt.xlabel("Count")
        plt.ylabel("Director")
        outpath = os.path.join(OUTDIR, "top_directors.png")
        plt.savefig(outpath, bbox_inches="tight")
        plt.close()
        print("✅ Saved:", os.path.basename(outpath))
    else:
        print("⚠️ Skipped top directors (no director data)")
else:
    print("⚠️ No director column found (skipping top directors)")

# 6.7 Correlation heatmap for numeric columns (always useful)
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
if len(num_cols) > 1:
    corr = df[num_cols].corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("Correlation matrix (numeric columns)")
    outpath = os.path.join(OUTDIR, "correlation_matrix.png")
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
    print("✅ Saved:", os.path.basename(outpath))
else:
    print("⚠️ Skipped correlation heatmap (not enough numeric columns)")

# ---------- Done ----------
print("\n🎉 Done. Check the 'figures' folder at:", OUTDIR)
