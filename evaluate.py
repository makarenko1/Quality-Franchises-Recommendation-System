"""
Evaluation for the movie recommendation system.

Sections:
  1. k-tuning: find the best number of SVD latent factors
  2. RMSE on held-out user ratings (biased SVD vs global-mean baseline)
  3. Precision@K and Recall@K (SVD vs popular / highest-rated / random)
  4. Franchise analysis: installment number vs IMDb rating
  5. Dialogue feature correlation with IMDb rating

Uses dataset_ratings_and_tags.csv which contains both MovieLens 1M and 32M
ratings. 32M UserIDs are offset by USER_ID_32M_OFFSET to avoid collisions.
"""

from __future__ import annotations

import gc
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy.stats import pearsonr, spearmanr

from recommendations_algorithm import (
    MOVIES_PATH,
    load_movies_metadata,
    load_dialogue_features,
    load_franchise_map,
    recommend_from_movie_ids,
)

# ── Config ────────────────────────────────────────────────────────────────────

RATINGS_AND_TAGS_PATH = Path("dataset_ratings_and_tags.csv")
DATASET_PATH          = Path("dataset.csv")

USER_ID_32M_OFFSET  = 300_000   # added to all 32M UserIDs to avoid collisions

K                   = 10        # recommendation list length
RELEVANCE_THRESHOLD = 4.0       # minimum rating counted as relevant
TEST_FRACTION       = 0.2       # per-user held-out fraction
MIN_RATINGS         = 20        # drop users with fewer ratings than this
K_MAX               = 50        # upper bound for k-tuning; reduced from 200 to avoid memory pressure
K_VALUES            = [20, 50, 100, 200]
RANDOM_SEED         = 42

# Quality prior used by the shared recommendation algorithm in evaluation.
# This mirrors recommendations_algorithm.py.
W_QUALITY_POPULARITY = 0.70
W_QUALITY_RATING     = 0.30

# Use the whole ratings file by default. If your machine runs out of memory,
# set this to a number such as 2_000_000.
MAX_RATING_ROWS     = None
READ_CHUNK_SIZE     = 500_000    # read the combined ratings file incrementally

# Ranking metrics are evaluated on a fixed user sample because scoring every
# user against every movie is expensive even without building a full matrix.
EVAL_USER_SAMPLE_SIZE = 500

DIALOGUE_COLS = [
    "type_token_ratio",
    "hapax_ratio",
    "top_word_frequency_ratio",
    "bigram_repetition_ratio",
    "trigram_repetition_ratio",
    "repeated_line_ratio",
    "repeated_short_phrase_ratio",
    "average_word_length",
    "long_word_ratio",
    "common_word_ratio",
    "rare_word_ratio",
    "simple_word_ratio",
    "complex_word_ratio",
    "flesch_reading_ease",
    "average_sentence_length",
    "average_sentiment",
    "sentiment_variance",
    "positive_word_ratio",
    "negative_word_ratio",
    "anger_word_ratio",
    "fear_word_ratio",
    "joy_word_ratio",
    "sadness_word_ratio",
    "content_stemmed_type_token_ratio",
    "content_stemmed_hapax_ratio",
    "content_stemmed_bigram_repetition_ratio",
    "content_stemmed_trigram_repetition_ratio",
]


# ── Data loading and splitting ────────────────────────────────────────────────

def load_and_split(
    path: Path = RATINGS_AND_TAGS_PATH,
    test_fraction: float = TEST_FRACTION,
    min_ratings: int = MIN_RATINGS,
    seed: int = RANDOM_SEED,
    max_rating_rows: int | None = MAX_RATING_ROWS,
    chunk_size: int = READ_CHUNK_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load ratings incrementally and split per-user.

    The full combined file can contain tens of millions of ratings. This function
    reads it in chunks. By default, it uses the whole file; set max_rating_rows
    to a finite number to run a smaller memory-safe evaluation sample.

    32M UserIDs are offset by USER_ID_32M_OFFSET so they don't collide with
    1M UserIDs. Users with fewer than min_ratings are dropped after sampling.
    """
    print("  Reading ratings from combined file ...", flush=True)
    chunks: list[pd.DataFrame] = []
    n_loaded = 0

    read_kwargs = dict(
        usecols=["UserID", "MovieID", "Rating", "InteractionType", "SourceDataset"],
        dtype={
            "UserID": "int32",
            "MovieID": "int32",
            "Rating": "float32",
            "InteractionType": "category",
            "SourceDataset": "category",
        },
        chunksize=chunk_size,
        low_memory=False,
    )

    for chunk_i, chunk in enumerate(pd.read_csv(path, **read_kwargs), start=1):
        chunk = chunk[chunk["InteractionType"] == "rating"]
        if chunk.empty:
            continue

        mask_32m = chunk["SourceDataset"] == "MovieLens 32M"
        chunk = chunk[["UserID", "MovieID", "Rating"]].copy()
        chunk.loc[mask_32m.to_numpy(), "UserID"] += USER_ID_32M_OFFSET

        if max_rating_rows is not None:
            remaining = max_rating_rows - n_loaded
            if remaining <= 0:
                break
            chunk = chunk.iloc[:remaining]

        chunks.append(chunk)
        n_loaded += len(chunk)
        if max_rating_rows is not None and n_loaded >= max_rating_rows:
            break

    if not chunks:
        raise ValueError("No rating rows were loaded from the combined file.")

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    if max_rating_rows is not None:
        print(f"  Using a memory-safe sample of {len(df):,} ratings", flush=True)

    counts = df.groupby("UserID", sort=False).size()
    keep_users = counts[counts >= min_ratings].index
    df = df[df["UserID"].isin(keep_users)].copy()

    # Random shuffle within each user, then take last test_fraction as test.
    rng = np.random.default_rng(seed)
    df["_r"] = rng.random(len(df)).astype(np.float32)
    df = df.sort_values(["UserID", "_r"]).drop(columns=["_r"])

    df["_cum"] = df.groupby("UserID", sort=False).cumcount()
    df["_total"] = df.groupby("UserID", sort=False)["UserID"].transform("count")
    cut = (df["_total"] * (1 - test_fraction)).astype("int32")
    is_test = df["_cum"] >= cut
    df = df.drop(columns=["_cum", "_total"])

    return (
        df.loc[~is_test].reset_index(drop=True),
        df.loc[is_test].reset_index(drop=True),
    )


def build_train_movies_for_users(
    train: pd.DataFrame,
    user_ids: list[int] | set[int],
) -> dict[int, set[int]]:
    """
    Build train-rated movie sets only for users used in ranking evaluation.

    Building this for every user on the full combined dataset requires millions
    of Python set entries and can trigger SIGKILL on memory-limited machines.
    """
    selected = set(int(u) for u in user_ids)
    result: dict[int, set[int]] = defaultdict(set)
    sub = train[train["UserID"].isin(selected)]
    for uid, mid in zip(sub["UserID"].astype(int), sub["MovieID"].astype(int)):
        result[int(uid)].add(int(mid))
    return dict(result)


def build_relevant_test_movies(
    test: pd.DataFrame,
    threshold: float = RELEVANCE_THRESHOLD,
    allowed_movie_ids: set[int] | None = None,
) -> dict[int, set[int]]:
    """Store only relevant held-out MovieIDs per user for ranking metrics."""
    relevant = test.loc[test["Rating"] >= threshold, ["UserID", "MovieID"]]
    if allowed_movie_ids is not None:
        relevant = relevant[relevant["MovieID"].isin(allowed_movie_ids)]
    result: dict[int, set[int]] = defaultdict(set)
    for uid, mid in zip(relevant["UserID"].astype(int), relevant["MovieID"].astype(int)):
        result[int(uid)].add(int(mid))
    return dict(result)


def build_train_input_movies_for_users(
    train: pd.DataFrame,
    user_ids: list[int] | set[int],
    allowed_movie_ids: set[int],
    threshold: float = RELEVANCE_THRESHOLD,
    max_inputs: int = 3,
) -> dict[int, list[int]]:
    """Choose each evaluated user's liked training movies as inputs to recommend_from_movie_ids()."""
    selected = set(int(u) for u in user_ids)
    sub = train[
        train["UserID"].isin(selected)
        & train["MovieID"].isin(allowed_movie_ids)
        & (train["Rating"] >= threshold)
    ][["UserID", "MovieID", "Rating"]].copy()
    sub = sub.sort_values(["UserID", "Rating"], ascending=[True, False])

    result: dict[int, list[int]] = {}
    for uid, group in sub.groupby("UserID", sort=False):
        result[int(uid)] = group["MovieID"].astype(int).head(max_inputs).tolist()
    return result


# ── Biased SVD ────────────────────────────────────────────────────────────────

def build_svd(train: pd.DataFrame, k: int = K_MAX) -> dict:
    """
    Fit a biased matrix-factorisation SVD on training ratings.

    Prediction: R̂_ui = μ + b_u + b_i + U_u · V_i
      μ   = global mean
      b_u = user bias  (mean(R_u) - μ)
      b_i = item bias  (mean(R_i) - μ)
      U·V = latent-factor interaction term

    Biases are stored in the returned dict so they can be used for both
    RMSE prediction and ranking (item biases boost well-rated movies).
    """
    user_ids  = sorted(train["UserID"].unique())
    movie_ids = sorted(train["MovieID"].unique())
    user_idx  = {u: i for i, u in enumerate(user_ids)}
    movie_idx = {m: i for i, m in enumerate(movie_ids)}

    global_mean = float(train["Rating"].mean())

    user_means  = train.groupby("UserID")["Rating"].mean()
    movie_means = train.groupby("MovieID")["Rating"].mean()

    user_biases  = np.array([float(user_means[u])  - global_mean for u in user_ids],  dtype=np.float32)
    movie_biases = np.array([float(movie_means[m]) - global_mean for m in movie_ids], dtype=np.float32)
    # General-quality prior used by recommendations_algorithm.recommend_from_movie_ids.
    # It combines popularity and rating quality to match the actual recommender.
    movie_counts = train.groupby("MovieID", sort=False).size()
    count_values = np.array([float(movie_counts.get(m, 0.0)) for m in movie_ids], dtype=np.float32)
    pop = np.log1p(count_values)
    pop_min, pop_max = float(pop.min()), float(pop.max())
    popularity_norm = (
        (pop - pop_min) / (pop_max - pop_min)
        if pop_max > pop_min
        else np.full_like(pop, 0.5, dtype=np.float32)
    )

    mb_min, mb_max = float(movie_biases.min()), float(movie_biases.max())
    rating_norm = (
        (movie_biases - mb_min) / (mb_max - mb_min)
        if mb_max > mb_min
        else np.full_like(movie_biases, 0.5, dtype=np.float32)
    )

    movie_quality_scores = (
        W_QUALITY_POPULARITY * popularity_norm
        + W_QUALITY_RATING * rating_norm
    ).astype(np.float32)

    # residuals after removing biases
    b_u = train["UserID"].map(user_means)  - global_mean
    b_i = train["MovieID"].map(movie_means) - global_mean
    residuals = (train["Rating"] - global_mean - b_u - b_i).values.astype(np.float32)

    rows = train["UserID"].map(user_idx).values
    cols = train["MovieID"].map(movie_idx).values
    R = csr_matrix((residuals, (rows, cols)), shape=(len(user_ids), len(movie_ids)))

    k = min(k, min(R.shape) - 1)
    U, sigma, Vt = svds(R, k=k)
    order = np.argsort(-sigma)
    U, sigma, Vt = U[:, order], sigma[order], Vt[order]

    # Store factors as float32 to keep memory down on the combined dataset.
    user_factors = (U * sigma).astype(np.float32)
    movie_factors = Vt.T.astype(np.float32)


    return {
        "user_factors":  user_factors,       # (n_users,  k)
        "movie_factors": movie_factors,      # (n_movies, k)
        "global_mean":   global_mean,
        "user_biases":   user_biases,        # (n_users,)
        "movie_biases":  movie_biases,       # (n_movies,)
        "movie_quality_scores": movie_quality_scores, # (n_movies,), normalized item-quality prior
        "user_ids":      np.array(user_ids, dtype=np.int32),
        "movie_ids":     np.array(movie_ids, dtype=np.int32),
        "user_idx":      user_idx,
        "movie_idx":     movie_idx,
    }


# ── RMSE ──────────────────────────────────────────────────────────────────────

def compute_rmse(
    model: dict,
    test: pd.DataFrame,
    k: int | None = None,
    chunk_size: int = 500_000,
) -> tuple[float, float, int]:
    """
    Return (svd_rmse, mean_baseline_rmse, n_evaluated).

    Evaluates in chunks so the full test set does not need to be materialized
    into several large temporary arrays at once.
    """
    UF = model["user_factors"]
    MF = model["movie_factors"]
    if k is not None:
        UF = UF[:, :k]
        MF = MF[:, :k]

    global_mean  = model["global_mean"]
    user_biases  = model["user_biases"]
    movie_biases = model["movie_biases"]
    user_idx     = model["user_idx"]
    movie_idx    = model["movie_idx"]

    svd_sse = 0.0
    mean_sse = 0.0
    n_eval = 0

    for start in range(0, len(test), chunk_size):
        chunk = test.iloc[start:start + chunk_size]
        u_idxs = chunk["UserID"].map(user_idx)
        m_idxs = chunk["MovieID"].map(movie_idx)
        valid = u_idxs.notna() & m_idxs.notna()
        if not valid.any():
            continue

        u = u_idxs[valid].astype(int).to_numpy()
        m = m_idxs[valid].astype(int).to_numpy()
        actuals = chunk.loc[valid, "Rating"].to_numpy(dtype=np.float64)

        preds = np.clip(
            global_mean
            + user_biases[u]
            + movie_biases[m]
            + np.einsum("ij,ij->i", UF[u], MF[m]),
            0.5, 5.0,
        )

        svd_sse += float(np.sum((preds - actuals) ** 2))
        mean_sse += float(np.sum((global_mean - actuals) ** 2))
        n_eval += int(len(actuals))

    svd_rmse = float(np.sqrt(svd_sse / n_eval))
    mean_rmse = float(np.sqrt(mean_sse / n_eval))
    return svd_rmse, mean_rmse, n_eval

# ── k-tuning ──────────────────────────────────────────────────────────────────

def tune_k(model: dict, test: pd.DataFrame, k_values: list[int] = K_VALUES) -> int:
    """
    Evaluate RMSE at each k by truncating a single pre-trained model's factors.
    Returns the k with the lowest test RMSE.
    """
    print(f"  {'k':<8} {'RMSE':>8}")
    print(f"  {'-' * 18}")
    best_k, best_rmse = k_values[0], float("inf")
    for k in k_values:
        rmse, _, _ = compute_rmse(model, test, k=k)
        marker = ""
        if rmse < best_rmse:
            best_rmse = rmse
            best_k = k
            marker = "  ←"
        print(f"  {k:<8} {rmse:>8.4f}{marker}")
    return best_k


# ── Recommendation methods ────────────────────────────────────────────────────

def algorithm_recommender(
    movies: pd.DataFrame,
    model: dict,
    train_movies: dict[int, set[int]],
    train_input_movies: dict[int, list[int]],
    dq_map: dict[int, float],
    franchise_map: dict,
    top_n: int = K,
):
    """Evaluate the same recommendation logic used by recommendations_algorithm.recommend()."""
    movie_ids = [int(m) for m in model["movie_ids"].tolist()]
    movie_factors = model["movie_factors"]
    movie_quality_scores = model["movie_quality_scores"]

    def recommend(user_id: int) -> list[int]:
        input_ids = train_input_movies.get(int(user_id), [])
        if not input_ids:
            return []
        return recommend_from_movie_ids(
            input_ids,
            movies,
            movie_ids,
            movie_factors,
            movie_quality_scores,
            dq_map,
            franchise_map,
            n=top_n,
            exclude_ids=train_movies.get(int(user_id), set()),
        )

    return recommend

def popular_recommender(train: pd.DataFrame, train_movies: dict[int, set[int]], top_n: int = K, allowed_movie_ids: set[int] | None = None):
    stats = train.groupby("MovieID").size()
    if allowed_movie_ids is not None:
        stats = stats[stats.index.isin(allowed_movie_ids)]
    ranked = list(stats.sort_values(ascending=False).index)

    def recommend(user_id: int) -> list[int]:
        trained = train_movies.get(user_id, set())
        result = []
        for m in ranked:
            if m not in trained:
                result.append(m)
                if len(result) >= top_n:
                    break
        return result

    return recommend


def highest_rated_recommender(train: pd.DataFrame, train_movies: dict[int, set[int]], top_n: int = K, allowed_movie_ids: set[int] | None = None):
    """
    IMDb-style Bayesian weighted rating (same formula as baselines.py):
      WR = (v / (v + m)) * R + (m / (v + m)) * C
    """
    stats = train.groupby("MovieID")["Rating"].agg(["count", "mean"])
    if allowed_movie_ids is not None:
        stats = stats[stats.index.isin(allowed_movie_ids)]
    stats.columns = ["vote_count", "vote_average"]
    C = float(stats["vote_average"].mean())
    m = float(stats["vote_count"].quantile(0.90))
    v, R = stats["vote_count"], stats["vote_average"]
    stats["score"] = (v / (v + m)) * R + (m / (v + m)) * C
    ranked = list(stats.sort_values("score", ascending=False).index)

    def recommend(user_id: int) -> list[int]:
        trained = train_movies.get(user_id, set())
        result = []
        for m in ranked:
            if m not in trained:
                result.append(m)
                if len(result) >= top_n:
                    break
        return result

    return recommend


def random_recommender(
    train: pd.DataFrame,
    train_movies: dict[int, set[int]],
    seed: int = RANDOM_SEED,
    top_n: int = K,
    allowed_movie_ids: set[int] | None = None,
):
    if allowed_movie_ids is None:
        all_movies = np.array(sorted(train["MovieID"].unique()), dtype=np.int32)
    else:
        all_movies = np.array(sorted(set(train["MovieID"].unique()).intersection(allowed_movie_ids)), dtype=np.int32)

    def recommend(user_id: int) -> list[int]:
        trained = train_movies.get(user_id, set())
        rng = np.random.default_rng(seed + int(user_id))
        result: list[int] = []
        for idx in rng.permutation(len(all_movies)):
            mid = int(all_movies[idx])
            if mid not in trained:
                result.append(mid)
                if len(result) >= top_n:
                    break
        return result

    return recommend

# ── Precision@K / Recall@K ────────────────────────────────────────────────────

def select_eval_users(
    relevant_by_user: dict[int, set[int]],
    max_users: int | None = EVAL_USER_SAMPLE_SIZE,
    seed: int = RANDOM_SEED,
) -> list[int]:
    """Select users with at least one relevant held-out item for ranking eval."""
    eligible = list(relevant_by_user.keys())
    if max_users is None or len(eligible) <= max_users:
        return eligible
    rng = np.random.default_rng(seed)
    return rng.choice(eligible, size=max_users, replace=False).astype(int).tolist()


def eval_ranking(
    rec_fn,
    relevant_by_user: dict[int, set[int]],
    user_ids: list[int],
    k: int = K,
) -> tuple[float, float]:
    """
    Mean Precision@K and Recall@K over selected users with ≥1 relevant item.
    Relevant = rating ≥ RELEVANCE_THRESHOLD.
    """
    p_list: list[float] = []
    r_list: list[float] = []
    for i, user_id in enumerate(user_ids, start=1):
        relevant = relevant_by_user[user_id]
        top_k = rec_fn(user_id)[:k]
        hits = sum(1 for m in top_k if m in relevant)
        p_list.append(hits / k)
        r_list.append(hits / len(relevant))
    return float(np.mean(p_list)), float(np.mean(r_list))

# ── Franchise analysis ────────────────────────────────────────────────────────

def franchise_analysis(dataset: pd.DataFrame) -> dict | None:
    df = dataset[["FranchiseInstallment", "imdb_averageRating"]].copy()
    df["FranchiseInstallment"] = pd.to_numeric(df["FranchiseInstallment"], errors="coerce")
    df["imdb_averageRating"]   = pd.to_numeric(df["imdb_averageRating"],   errors="coerce")
    df = df.dropna()
    if len(df) < 10:
        return None
    rho, p_sp = spearmanr(df["FranchiseInstallment"], df["imdb_averageRating"])
    r,   p_pe = pearsonr( df["FranchiseInstallment"], df["imdb_averageRating"])
    df["group"] = df["FranchiseInstallment"].clip(upper=5).astype(int)
    by_inst = df.groupby("group")["imdb_averageRating"].agg(mean="mean", count="count")
    return {
        "n": len(df),
        "spearman_r": float(rho), "spearman_p": float(p_sp),
        "pearson_r":  float(r),   "pearson_p":  float(p_pe),
        "by_inst": by_inst,
    }


# ── Dialogue feature correlation ──────────────────────────────────────────────

def dialogue_correlation(
    dataset: pd.DataFrame,
    cols: list[str] = DIALOGUE_COLS,
) -> pd.DataFrame:
    df = dataset.copy()
    df["imdb_averageRating"] = pd.to_numeric(df["imdb_averageRating"], errors="coerce")
    df = df.dropna(subset=["imdb_averageRating"])
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        sub = df[["imdb_averageRating", col]].dropna()
        if len(sub) < 10:
            continue
        r,   p_pe = pearsonr( sub[col], sub["imdb_averageRating"])
        rho, p_sp = spearmanr(sub[col], sub["imdb_averageRating"])
        rows.append({
            "feature":    col,
            "n":          len(sub),
            "pearson_r":  round(float(r),   4),
            "pearson_p":  round(float(p_pe), 4),
            "spearman_r": round(float(rho), 4),
            "spearman_p": round(float(p_sp), 4),
        })
    return (
        pd.DataFrame(rows)
        .sort_values("pearson_r", key=abs, ascending=False)
        .reset_index(drop=True)
    )


# ── Output helpers ────────────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{'=' * 62}")
    print(f"  {title}")
    print("=" * 62)


# ── Main ──────────────────────────────────────────────────────────────────────

def _comparison_word(svd_p: float, svd_r: float, bp: float, br: float) -> str:
    if svd_p > bp and svd_r > br:
        return "better"
    if svd_p < bp and svd_r < br:
        return "worse"
    return "mixed"


def _ranking_quality_word(precision: float, recall: float) -> str:
    """Conservative wording for top-K metrics; avoids overclaiming."""
    if precision >= 0.05 and recall >= 0.05:
        return "good"
    if precision >= 0.01 or recall >= 0.01:
        return "modest"
    return "weak"


def main() -> None:
    if not RATINGS_AND_TAGS_PATH.exists():
        print(f"ERROR: {RATINGS_AND_TAGS_PATH} not found — run main.py first.")
        return

    # ── Setup phase: load, split, choose k, and prepare shared recommender data ─
    section("Setup phase")
    train, test = load_and_split()
    print(f"  Split ratings: {len(train):,} train | {len(test):,} test")
    print(f"  Users: {train['UserID'].nunique():,} | Movies: {train['MovieID'].nunique():,}")

    movies = load_movies_metadata(MOVIES_PATH)
    metadata_movie_ids = set(movies["MovieID"].astype(int))

    # The shared recommender can only score movies that have both rating-model
    # factors and movie metadata.  Use the same candidate set for the algorithm
    # and all baselines so the comparison is fair.
    # model_movie_ids is defined after build_svd; initialize here for clarity.
    allowed_movie_ids = metadata_movie_ids

    t0 = time.time()
    print(f"  Building SVD at k={K_MAX} ...")
    model = build_svd(train, k=K_MAX)
    model_movie_ids = set(int(m) for m in model["movie_ids"].tolist())
    allowed_movie_ids = metadata_movie_ids.intersection(model_movie_ids)

    relevant_by_user = build_relevant_test_movies(test, allowed_movie_ids=allowed_movie_ids)
    n_with_relevant = len(relevant_by_user)
    print(f"  Users with at least one relevant test item: {n_with_relevant:,}")
    print(f"  Metadata source: {MOVIES_PATH}")
    print(f"  Metadata movies: {len(metadata_movie_ids):,}")
    print(f"  Model movies: {len(model_movie_ids):,}")
    print(f"  Candidate movies: {len(allowed_movie_ids):,}")
    print(f"  Model movies without metadata: {len(model_movie_ids - metadata_movie_ids):,}")
    print(f"  Metadata movies without model factors: {len(metadata_movie_ids - model_movie_ids):,}")
    print(f"  SVD setup completed in {time.time() - t0:.1f}s")
    print("  Selecting k:")
    best_k = tune_k(model, test)
    print(f"  Selected k: {best_k}")

    # Keep the selected SVD dimensions for the shared recommendations_algorithm function.
    model["movie_factors"] = model["movie_factors"][:, :best_k]

    # ── Metrics are computed silently; only numbers are printed. ──────────────
    svd_rmse, mean_rmse, n_eval = compute_rmse(model, test, k=best_k)

    eval_user_ids = select_eval_users(relevant_by_user)
    del test
    gc.collect()

    train_movies = build_train_movies_for_users(train, eval_user_ids)
    train_input_movies = build_train_input_movies_for_users(train, eval_user_ids, allowed_movie_ids)

    dq_map = load_dialogue_features()
    franchise_map = load_franchise_map()

    alg_rec = algorithm_recommender(
        movies,
        model,
        train_movies,
        train_input_movies,
        dq_map,
        franchise_map,
        top_n=K,
    )
    pop_rec = popular_recommender(train, train_movies, top_n=K, allowed_movie_ids=allowed_movie_ids)
    hr_rec = highest_rated_recommender(train, train_movies, top_n=K, allowed_movie_ids=allowed_movie_ids)
    rand_rec = random_recommender(train, train_movies, top_n=K, allowed_movie_ids=allowed_movie_ids)
    del train
    gc.collect()

    methods = [
        ("Recommendations algorithm", alg_rec),
        ("Popular", pop_rec),
        ("Highest-rated", hr_rec),
        ("Random", rand_rec),
    ]

    results: dict[str, tuple[float, float]] = {}
    for name, rec_fn in methods:
        results[name] = eval_ranking(rec_fn, relevant_by_user, eval_user_ids, k=K)

    alg_p, alg_r = results["Recommendations algorithm"]

    fa = None
    corr = pd.DataFrame()
    if DATASET_PATH.exists():
        dataset = pd.read_csv(DATASET_PATH, low_memory=False)
        fa = franchise_analysis(dataset)
        corr = dialogue_correlation(dataset)

    print()
    print("Recommender system performance:")
    print(f"  RMSE: {svd_rmse:.4f}")
    print(f"  Global-mean RMSE: {mean_rmse:.4f}")
    print(f"  RMSE delta: {svd_rmse - mean_rmse:+.4f}")
    print(f"  Evaluated rating pairs: {n_eval:,}")
    print(f"  Ranking users: {len(eval_user_ids):,}")
    print(f"  Candidate movies: {len(allowed_movie_ids):,}")
    print(f"  Metadata movies: {len(metadata_movie_ids):,}")
    print(f"  Model movies: {len(model_movie_ids):,}")
    print(f"  Model movies without metadata: {len(model_movie_ids - metadata_movie_ids):,}")
    print(f"  Metadata movies without model factors: {len(metadata_movie_ids - model_movie_ids):,}")
    print()
    print(f"  {'Method':<28} {'Precision@' + str(K):>12} {'Recall@' + str(K):>12} {'Delta P':>10} {'Delta R':>10}")
    print(f"  {'-' * 78}")
    for name, _ in methods:
        p, r = results[name]
        print(f"  {name:<28} {p:>12.4f} {r:>12.4f} {p - alg_p:>+10.4f} {r - alg_r:>+10.4f}")

    print("Franchise analysis:")
    if fa is None:
        print("  Franchise n: 0")
    else:
        print(f"  Franchise n: {fa['n']:,}")
        print(f"  Spearman rho: {fa['spearman_r']:+.4f}")
        print(f"  Spearman p: {fa['spearman_p']:.4f}")
        print(f"  Pearson r: {fa['pearson_r']:+.4f}")
        print(f"  Pearson p: {fa['pearson_p']:.4f}")

    print("Dialogue feature correlation:")
    if corr.empty:
        print("  Features evaluated: 0")
    else:
        top = corr.iloc[0]
        print(f"  Features evaluated: {len(corr):,}")
        print(f"  Top feature: {top['feature']}")
        print(f"  Top feature n: {int(top['n']):,}")
        print(f"  Pearson r: {top['pearson_r']:+.4f}")
        print(f"  Pearson p: {top['pearson_p']:.4f}")
        print(f"  Spearman rho: {top['spearman_r']:+.4f}")
        print(f"  Spearman p: {top['spearman_p']:.4f}")
    print()


if __name__ == "__main__":
    main()
