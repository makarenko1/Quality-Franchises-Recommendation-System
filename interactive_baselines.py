"""
Interactive baselines aligned with evaluate.py.

Baselines:
  - popular: most-rated movies
  - highest_rated: IMDb-style Bayesian weighted rating
  - random: seeded random ranking

Unlike the old version, this reads the combined dataset_ratings_and_tags.csv
instead of only MovieLens 1M, and it uses dataset.csv as the movie universe.
"""

from __future__ import annotations

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd


RATINGS_AND_TAGS_PATH = Path("dataset_ratings_and_tags.csv")
MOVIES_PATH = Path("dataset.csv")
RANDOM_SEED = 42
READ_CHUNK_SIZE = 500_000


def load_movies_metadata(movies_path: Path = MOVIES_PATH) -> pd.DataFrame:
    movies = pd.read_csv(movies_path, low_memory=False)
    movies["MovieID"] = pd.to_numeric(movies["MovieID"], errors="coerce")
    movies = movies.dropna(subset=["MovieID"]).copy()
    movies["MovieID"] = movies["MovieID"].astype(int)

    if "Title" not in movies.columns:
        raise ValueError(f"{movies_path} must contain a Title column")

    if "Year" not in movies.columns:
        movies["Year"] = np.nan

    for i in range(1, 9):
        col = f"Genre{i}"
        if col not in movies.columns:
            movies[col] = np.nan

    return movies.drop_duplicates(subset=["MovieID"], keep="first").reset_index(drop=True)


def load_rating_stats(
    ratings_path: Path = RATINGS_AND_TAGS_PATH,
    allowed_movie_ids: set[int] | None = None,
    chunk_size: int = READ_CHUNK_SIZE,
) -> pd.DataFrame:
    """
    Load item rating statistics from the same combined file used by evaluate.py.
    Tags are ignored; only InteractionType == 'rating' is used.
    """
    sums = Counter()
    counts = Counter()
    source_counts = Counter()

    usecols = ["MovieID", "Rating", "InteractionType", "SourceDataset"]

    for chunk in pd.read_csv(ratings_path, usecols=usecols, chunksize=chunk_size, low_memory=False):
        chunk = chunk[chunk["InteractionType"] == "rating"].copy()
        if chunk.empty:
            continue

        chunk["MovieID"] = pd.to_numeric(chunk["MovieID"], errors="coerce")
        chunk["Rating"] = pd.to_numeric(chunk["Rating"], errors="coerce")
        chunk = chunk.dropna(subset=["MovieID", "Rating"])
        chunk["MovieID"] = chunk["MovieID"].astype(int)

        if allowed_movie_ids is not None:
            chunk = chunk[chunk["MovieID"].isin(allowed_movie_ids)]

        if chunk.empty:
            continue

        for source, n in chunk["SourceDataset"].value_counts().items():
            source_counts[str(source)] += int(n)

        grouped = chunk.groupby("MovieID")["Rating"].agg(["sum", "count"])
        for mid, row in grouped.iterrows():
            sums[int(mid)] += float(row["sum"])
            counts[int(mid)] += int(row["count"])

    if not counts:
        raise ValueError("No rating rows were loaded. Check dataset_ratings_and_tags.csv.")

    stats = pd.DataFrame({
        "MovieID": list(counts.keys()),
        "vote_count": [counts[mid] for mid in counts.keys()],
        "rating_sum": [sums[mid] for mid in counts.keys()],
    })
    stats["vote_average"] = stats["rating_sum"] / stats["vote_count"]
    stats = stats.set_index("MovieID")

    print("Loaded rating rows by source:")
    for source, n in sorted(source_counts.items()):
        print(f"  {source}: {n:,}")

    return stats


def resolve_input_ids(input_values: list[str], movies: pd.DataFrame) -> list[int]:
    """
    Resolve typed MovieIDs or titles to MovieIDs.
    Exact title match is tried first, then contains match.
    """
    input_ids: list[int] = []
    title_lower = movies["Title"].astype(str).str.lower()

    for value in input_values:
        query = str(value).strip()
        if not query:
            continue

        if query.isdigit():
            mid = int(query)
            if mid in set(movies["MovieID"]):
                input_ids.append(mid)
                continue

        q = query.lower()
        match = movies[title_lower == q]
        if match.empty:
            match = movies[title_lower.str.contains(q, regex=False, na=False)]

        if match.empty:
            print(f"Could not resolve input: {query}")
            continue

        input_ids.append(int(match.iloc[0]["MovieID"]))

    return list(dict.fromkeys(input_ids))


def format_results(movies: pd.DataFrame, mids: list[int]) -> list[dict]:
    movie_rows = movies.set_index("MovieID", drop=False)
    results = []

    for mid in mids:
        if mid not in movie_rows.index:
            continue

        row = movie_rows.loc[mid]
        genres = [
            row[f"Genre{i}"]
            for i in range(1, 9)
            if f"Genre{i}" in row and pd.notna(row.get(f"Genre{i}"))
        ]

        results.append({
            "MovieID": int(mid),
            "title": row["Title"],
            "year": int(row["Year"]) if pd.notna(row["Year"]) else None,
            "genres": ", ".join(map(str, genres)),
        })

    return results


def recommend_popular(input_ids: list[int], movies: pd.DataFrame, stats: pd.DataFrame, n: int = 3) -> list[dict]:
    ranked = stats.sort_values("vote_count", ascending=False)
    picks = [int(mid) for mid in ranked.index if int(mid) not in set(input_ids)][:n]
    return format_results(movies, picks)


def recommend_highest_rated(input_ids: list[int], movies: pd.DataFrame, stats: pd.DataFrame, n: int = 3) -> list[dict]:
    """
    Same Bayesian weighted-rating formula used in evaluate.py:
      WR = (v / (v + m)) * R + (m / (v + m)) * C
    """
    C = float(stats["vote_average"].mean())
    m = float(stats["vote_count"].quantile(0.90))

    qualified = stats[stats["vote_count"] >= m].copy()
    v = qualified["vote_count"]
    R = qualified["vote_average"]
    qualified["score"] = (v / (v + m)) * R + (m / (v + m)) * C

    ranked = qualified.sort_values("score", ascending=False)
    picks = [int(mid) for mid in ranked.index if int(mid) not in set(input_ids)][:n]
    return format_results(movies, picks)


def recommend_random(input_ids: list[int], movies: pd.DataFrame, stats: pd.DataFrame, n: int = 3, seed: int = RANDOM_SEED) -> list[dict]:
    pool = np.array([int(mid) for mid in stats.index if int(mid) not in set(input_ids)], dtype=np.int32)
    rng = np.random.default_rng(seed)
    if len(pool) == 0:
        return []
    picks = rng.choice(pool, size=min(n, len(pool)), replace=False)
    return format_results(movies, [int(mid) for mid in picks])


def recommend_baseline(input_ids: list[int], baseline: str, movies: pd.DataFrame, stats: pd.DataFrame, n: int = 3) -> list[dict]:
    if baseline == "popular":
        return recommend_popular(input_ids, movies, stats, n)
    if baseline == "highest_rated":
        return recommend_highest_rated(input_ids, movies, stats, n)
    if baseline == "random":
        return recommend_random(input_ids, movies, stats, n)
    raise ValueError(f"Unknown baseline: {baseline}")


def main() -> None:
    movies = load_movies_metadata()
    allowed_movie_ids = set(movies["MovieID"].astype(int))
    stats = load_rating_stats(allowed_movie_ids=allowed_movie_ids)

    print("Enter 3 movies you like. You can type MovieIDs or titles:")
    values = []
    for i in range(1, 4):
        values.append(input(f"  Movie {i}: ").strip())

    input_ids = resolve_input_ids(values, movies)
    if not input_ids:
        print("No valid input movies found.")
        return

    baseline = input("\nBaseline (popular / highest_rated / random): ").strip()
    print(f"\n--- {baseline} ---")

    for r in recommend_baseline(input_ids, baseline, movies, stats):
        print(f"  {r['MovieID']} | {r['title']} ({r['year']}) - {r['genres']}")


if __name__ == "__main__":
    main()
