"""
Build one wide recommendation table from a cleaned initial-ratings spreadsheet,
using the same three baselines (popular / highest-rated / random) as evaluate.py.

Input:
  - recommendations_initial.xlsx

Expected sheet/columns:
  submission_id, timestamp, respondent, movie_rank, MovieID, Title, Year,
  Rating_Original, RatingScale_Original, Rating_1_5, is_positive_input

Output:
  - recommendations_new.csv

The output has one row per participant/submission and separate columns for:
  our_system_1..3, popular_1..3, highest_rated_1..3, random_1..3

Run with --interactive to query a single baseline from the terminal instead of
building the batch table (type MovieIDs or titles, then pick a baseline).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from recommendations_algorithm import (
    MOVIES_PATH,
    load_movies_metadata,
    load_dialogue_features,
    load_svd_model,
    load_franchise_map,
    load_installment_rating_trend,
    recommend_from_movie_ids,
)


RATINGS_AND_TAGS_PATH = Path("dataset_ratings_and_tags.csv")
INITIAL_RATINGS_PATH = Path("recommendations_initial.xlsx")
OUT_WIDE = Path("recommendations_new.csv")

RANDOM_SEED = 42
READ_CHUNK_SIZE = 500_000
N_PER_METHOD = 3


# ── Baselines shared with evaluate.py ────────────────────────────────────────

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


def recommend_popular(exclude_ids: list[int] | set[int], stats: pd.DataFrame, n: int = 3) -> list[int]:
    excluded = set(int(mid) for mid in exclude_ids)
    ranked = stats.sort_values("vote_count", ascending=False)
    return [int(mid) for mid in ranked.index if int(mid) not in excluded][:n]


def recommend_highest_rated(exclude_ids: list[int] | set[int], stats: pd.DataFrame, n: int = 3) -> list[int]:
    """
    Same Bayesian weighted-rating formula used in evaluate.py:
      WR = (v / (v + m)) * R + (m / (v + m)) * C
    """
    excluded = set(int(mid) for mid in exclude_ids)
    C = float(stats["vote_average"].mean())
    m = float(stats["vote_count"].quantile(0.90))

    qualified = stats[stats["vote_count"] >= m].copy()
    v = qualified["vote_count"]
    R = qualified["vote_average"]
    qualified["score"] = (v / (v + m)) * R + (m / (v + m)) * C

    ranked = qualified.sort_values("score", ascending=False)
    return [int(mid) for mid in ranked.index if int(mid) not in excluded][:n]


def recommend_random(
    exclude_ids: list[int] | set[int],
    stats: pd.DataFrame,
    n: int = 3,
    seed: int = RANDOM_SEED,
) -> list[int]:
    excluded = set(int(mid) for mid in exclude_ids)
    pool = np.array([int(mid) for mid in stats.index if int(mid) not in excluded], dtype=np.int32)
    if len(pool) == 0:
        return []
    rng = np.random.default_rng(seed)
    picks = rng.choice(pool, size=min(n, len(pool)), replace=False)
    return [int(mid) for mid in picks]


def recommend_baseline(exclude_ids: list[int] | set[int], baseline: str, stats: pd.DataFrame, n: int = 3) -> list[int]:
    if baseline == "popular":
        return recommend_popular(exclude_ids, stats, n)
    if baseline == "highest_rated":
        return recommend_highest_rated(exclude_ids, stats, n)
    if baseline == "random":
        return recommend_random(exclude_ids, stats, n)
    raise ValueError(f"Unknown baseline: {baseline}")


# ── Interactive single-query mode (python generate_recommendations.py --interactive) ──

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


def format_baseline_results(movies: pd.DataFrame, mids: list[int]) -> list[dict]:
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


def interactive_main() -> None:
    movies = load_movies_metadata(MOVIES_PATH)
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

    mids = recommend_baseline(input_ids, baseline, stats)
    for r in format_baseline_results(movies, mids):
        print(f"  {r['MovieID']} | {r['title']} ({r['year']}) - {r['genres']}")


def load_initial_ratings(path: Path = INITIAL_RATINGS_PATH) -> pd.DataFrame:
    """
    Load the already-cleaned initial ratings sheet.

    Ratings must already be on the same 1-5 scale in Rating_1_5, so the script
    does not infer scale from dates and does not move columns around.
    """
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name="initial_ratings_clean")
    else:
        df = pd.read_csv(path, low_memory=False)

    required = {
        "submission_id",
        "timestamp",
        "respondent",
        "movie_rank",
        "MovieID",
        "Title",
        "Year",
        "Rating_1_5",
        "is_positive_input",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = df.copy()
    df["MovieID"] = pd.to_numeric(df["MovieID"], errors="coerce")
    df = df.dropna(subset=["MovieID"]).copy()
    df["MovieID"] = df["MovieID"].astype(int)

    df["movie_rank"] = pd.to_numeric(df["movie_rank"], errors="coerce").fillna(999).astype(int)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Rating_1_5"] = pd.to_numeric(df["Rating_1_5"], errors="coerce")

    # Make sure boolean values survive Excel import.
    df["is_positive_input"] = df["is_positive_input"].map(
        lambda x: bool(x) if isinstance(x, (bool, np.bool_)) else str(x).strip().lower() in {"true", "1", "yes", "y"}
    )

    print(f"Submissions loaded: {df['submission_id'].nunique():,}")
    print(f"Movie rows loaded: {len(df):,}")
    return df


def movie_genres(row: pd.Series) -> str:
    genres = [
        str(row.get(f"Genre{i}"))
        for i in range(1, 9)
        if f"Genre{i}" in row and pd.notna(row.get(f"Genre{i}"))
    ]
    return ", ".join(genres)


def format_recommendations(
    mids: list[int],
    movies: pd.DataFrame,
    respondent_row: dict,
    method: str,
    input_titles: str,
) -> list[dict]:
    movie_rows = movies.set_index("MovieID", drop=False)
    output = []

    for rank, mid in enumerate(mids, start=1):
        mid = int(mid)
        if mid not in movie_rows.index:
            continue

        row = movie_rows.loc[mid]
        output.append({
            **respondent_row,
            "method": method,
            "rank_within_method": rank,
            "recommended_MovieID": mid,
            "recommended_Title": row["Title"],
            "recommended_Year": int(row["Year"]) if pd.notna(row["Year"]) else None,
            "recommended_Genres": movie_genres(row),
            "input_movies": input_titles,
        })

    return output


def recommendation_display_value(row: pd.Series) -> str:
    year = row.get("recommended_Year")
    year_text = str(int(year)) if pd.notna(year) else "unknown year"
    return f"{row['recommended_Title']} ({year_text}) [MovieID {int(row['recommended_MovieID'])}]"


def build_wide_output(by_method: pd.DataFrame, initial: pd.DataFrame) -> pd.DataFrame:
    initial_sorted = initial.sort_values(["submission_id", "movie_rank"]).copy()

    input_summary = (
        initial_sorted
        .groupby("submission_id", sort=False)
        .apply(
            lambda group: "; ".join(
                f"{row.Title} ({int(row.Year) if pd.notna(row.Year) else 'unknown'})"
                for row in group.itertuples(index=False)
            )
        )
        .rename("input_movies")
        .reset_index()
    )

    respondent_info = (
        initial_sorted
        .groupby("submission_id", sort=False)
        .agg(
            timestamp=("timestamp", "first"),
            respondent=("respondent", "first"),
        )
        .reset_index()
    )

    wide = respondent_info.merge(input_summary, on="submission_id", how="left")
    method_order = ["our_system", "popular", "highest_rated", "random"]

    for method in method_order:
        method_df = by_method[by_method["method"] == method].copy()
        method_df = method_df.sort_values(["submission_id", "rank_within_method"])

        for rank in range(1, N_PER_METHOD + 1):
            col = f"{method}_{rank}"
            rank_df = method_df[method_df["rank_within_method"] == rank].copy()

            if rank_df.empty:
                wide[col] = ""
                continue

            rank_df[col] = rank_df.apply(recommendation_display_value, axis=1)
            wide = wide.merge(rank_df[["submission_id", col]], on="submission_id", how="left")

    recommendation_cols = [
        f"{method}_{rank}"
        for method in method_order
        for rank in range(1, N_PER_METHOD + 1)
    ]

    for col in recommendation_cols:
        if col not in wide.columns:
            wide[col] = ""
        wide[col] = wide[col].fillna("")

    return wide[["submission_id", "respondent", "timestamp", "input_movies"] + recommendation_cols]


def main() -> None:
    initial = load_initial_ratings()

    movies = load_movies_metadata(MOVIES_PATH)
    metadata_movie_ids = set(movies["MovieID"].astype(int))

    print("Loading recommendation algorithm components...")
    dq_map = load_dialogue_features()
    movie_ids, movie_factors, movie_quality_scores = load_svd_model()
    franchise_map = load_franchise_map()
    installment_trend = load_installment_rating_trend()

    model_movie_ids = set(int(mid) for mid in movie_ids)
    allowed_movie_ids = metadata_movie_ids.intersection(model_movie_ids)

    print(f"Metadata movies: {len(metadata_movie_ids):,}")
    print(f"Model movies: {len(model_movie_ids):,}")
    print(f"Candidate movies: {len(allowed_movie_ids):,}")

    stats = load_rating_stats(allowed_movie_ids=allowed_movie_ids)

    all_rows = []

    for submission_id, group in initial.groupby("submission_id", sort=False):
        group = group.sort_values("movie_rank")

        positive_inputs = group[group["is_positive_input"]]["MovieID"].astype(int).tolist()
        exclude_ids = set(group["MovieID"].astype(int).tolist())

        algorithm_inputs = [mid for mid in positive_inputs if mid in allowed_movie_ids]
        input_titles = "; ".join(
            f"{row.Title} ({int(row.Year) if pd.notna(row.Year) else 'unknown'})"
            for row in group.itertuples(index=False)
        )

        respondent_row = {
            "submission_id": submission_id,
            "timestamp": group["timestamp"].iloc[0],
            "respondent": group["respondent"].iloc[0],
        }

        if not algorithm_inputs:
            print(f"Skipping {submission_id} / {respondent_row['respondent']}: no valid algorithm inputs.")
            continue

        algorithm_mids = recommend_from_movie_ids(
            algorithm_inputs,
            movies,
            list(movie_ids),
            movie_factors,
            movie_quality_scores,
            dq_map,
            franchise_map,
            installment_trend,
            n=N_PER_METHOD,
            exclude_ids=exclude_ids,
        )

        methods = {
            "our_system": algorithm_mids,
            "popular": recommend_popular(exclude_ids, stats, n=N_PER_METHOD),
            "highest_rated": recommend_highest_rated(exclude_ids, stats, n=N_PER_METHOD),
            "random": recommend_random(
                exclude_ids,
                stats,
                n=N_PER_METHOD,
                seed=RANDOM_SEED + int(str(submission_id).replace("S", "")),
            ),
        }

        for method, mids in methods.items():
            all_rows.extend(format_recommendations(mids, movies, respondent_row, method, input_titles))

    by_method = pd.DataFrame(all_rows)
    if by_method.empty:
        raise ValueError("No recommendations were generated.")

    wide = build_wide_output(by_method, initial)
    wide.to_csv(OUT_WIDE, index=False, encoding="utf-8-sig")

    print(f"Saved: {OUT_WIDE}")


if __name__ == "__main__":
    if "--interactive" in sys.argv:
        interactive_main()
    else:
        main()
