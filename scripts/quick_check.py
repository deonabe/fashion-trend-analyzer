import sys

import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/2025-09-25/reddit_posts.parquet"
df = pd.read_parquet(path)

print("Rows, Cols:", df.shape)
print("\nColumns:", list(df.columns))
print("\nDtypes:\n", df.dtypes)

# timestamps & basic ranges
df["created_dt"] = pd.to_datetime(df["created_utc"], unit="s")
print("\nCreated range:", df["created_dt"].min(), "→", df["created_dt"].max())

# per-subreddit counts
print("\nCounts by subreddit:\n", df["subreddit"].value_counts().head())

# duplicates and nulls
print("\nDuplicate IDs:", df["id"].duplicated().sum())
print("\nNull ratio (top 5):\n", df.isna().mean().sort_values(ascending=False).head())

# quick month distribution
m = df["created_dt"].dt.to_period("M")
print("\nCounts by month (last 12):\n", m.value_counts().sort_index().tail(12))

# spot-check rows
print(
    "\nSample rows:\n", df.sample(min(5, len(df)))[["title", "permalink", "created_dt"]]
)
