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

Run with --weight-sensitivity to instead sweep the top-level scoring weights
(W_CF, W_DIALOGUE, W_FRANCHISE, W_YEAR_PENALTY) and plot Precision@K/Recall@K
against each, on a smaller/faster sample than the full run above (see
--rows / --users / --weights).
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy.stats import pearsonr, spearmanr

import recommendations_algorithm as ra
from recommendations_algorithm import (
    MOVIES_PATH,
    load_movies_metadata,
    load_dialogue_features,
    load_franchise_map,
    load_installment_rating_trend,
    recommend_from_movie_ids,
)

# ── Config ────────────────────────────────────────────────────────────────────

RATINGS_AND_TAGS_PATH = Path("dataset_ratings_and_tags.csv")
DATASET_PATH          = Path("dataset.csv")

OUTPUT_DIR = Path("evaluation_outputs")   # saved log, CSVs, and plots from the latest run
PLOTS_DIR  = OUTPUT_DIR / "plots"

# Participant survey data (see save_survey_results()). Optional: skipped if absent.
SURVEY_RESPONSES_PATH = Path("survey-responses/responses_for_recommendations.csv")
SURVEY_SUMMARY_PATH   = Path("survey-responses/results_by_method.csv")

SURVEY_METHOD_ORDER = ["our_system", "popular", "highest_rated", "random"]
SURVEY_METHOD_LABELS = {
    "our_system": "Our system",
    "popular": "Popular",
    "highest_rated": "Highest-rated",
    "random": "Random",
}
SURVEY_METHOD_COLORS = {
    "our_system": "#4C72B0",
    "popular": "#DD8452",
    "highest_rated": "#55A868",
    "random": "#C44E52",
}

# Same 4 colors, keyed by the offline-evaluation method labels (methods list in
# _run()), so the two evaluations share one consistent color-per-method identity.
EVAL_METHOD_COLORS = {
    "Our system": "#4C72B0",
    "Popular": "#DD8452",
    "Highest-rated": "#55A868",
    "Random": "#C44E52",
}

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
EVAL_USER_SAMPLE_SIZE = 5000

# --- Weight sensitivity sweep (--weight-sensitivity) ---
# Smaller/faster sample than the full run above: the sweep reruns ranking
# evaluation once per grid point (~24 times total for the 4 weights below).
WEIGHT_SENSITIVITY_MAX_ROWS = 2_000_000   # pass --rows 0 for the full file
WEIGHT_SENSITIVITY_USER_SAMPLE = 300
WEIGHT_SENSITIVITY_SVD_K = 50              # fixed k; skips k-tuning

BASE_WEIGHTS = {
    "W_CF": ra.W_CF,
    "W_DIALOGUE": ra.W_DIALOGUE,
    "W_FRANCHISE": ra.W_FRANCHISE,
    "W_YEAR_PENALTY": ra.W_YEAR_PENALTY,
}
WEIGHT_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]
WEIGHT_LABELS = {
    "W_CF": "Collaborative filtering weight",
    "W_DIALOGUE": "Dialogue-quality weight",
    "W_FRANCHISE": "Franchise-quality weight",
    "W_YEAR_PENALTY": "Old-movie penalty weight",
}

# The two largest-magnitude features inside DIALOGUE_FEATURE_WEIGHTS (the
# composite that feeds W_DIALOGUE); swept the same way, but by rescaling
# feature-weight *magnitudes* since these weights carry signs.
DIALOGUE_SWEEP_FEATURES = ["negative_word_ratio", "type_token_ratio"]
BASE_DIALOGUE_WEIGHTS = dict(ra.DIALOGUE_FEATURE_WEIGHTS)
DIALOGUE_WEIGHT_LABELS = {
    "negative_word_ratio": "Negative-word-ratio weight",
    "type_token_ratio": "Type-token-ratio weight",
}

# negative_word_ratio's baseline sign is negative (it penalizes); sweep both
# signs for it to check whether flipping it to reward negative dialogue would
# actually help. type_token_ratio's baseline sign is already positive, so its
# sweep only needs to explore magnitude, not the opposite sign.
DIALOGUE_SIGNED_SWEEP_GRID = sorted(set([-v for v in WEIGHT_GRID] + WEIGHT_GRID))
DIALOGUE_SWEEP_GRIDS = {
    "negative_word_ratio": DIALOGUE_SIGNED_SWEEP_GRID,
    "type_token_ratio": WEIGHT_GRID,
}

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
#
# Why this script mixes two different kinds of baselines:
#
# RMSE and Precision@K/Recall@K test two different, complementary things, so
# they need two different kinds of baseline to be meaningful:
#   - RMSE asks "is the underlying rating-prediction model well-calibrated?"
#     (is the predicted score close to what the user actually gave?). That
#     question only makes sense for methods that output a predicted rating
#     value for a given (user, movie) pair, so its baselines here (below) are
#     all rating predictors of increasing complexity: global_mean -> item_mean
#     / user_mean -> bias_only -> the full svd model.
#   - Precision@K/Recall@K ask "is the final top-K list actually good?" (did
#     it contain movies the user liked?). That only needs a ranking, not a
#     calibrated score, so its baselines (popular_recommender,
#     highest_rated_recommender, random_recommender, further down) are simple
#     non-personalized ranking heuristics instead of rating predictors.
# The two families answer "does the rating model beat guessing the average?"
# and "does personalized ranking beat simple non-personalized heuristics?"
# respectively -- two separate validity checks, not one continuous ladder.

def compute_rmse(
    model: dict,
    test: pd.DataFrame,
    k: int | None = None,
    chunk_size: int = 500_000,
) -> tuple[dict[str, float], int]:
    """
    Return ({baseline_name: rmse}, n_evaluated) for every rating-prediction
    baseline, evaluated together in one pass over the test set:
      svd         = mu + b_u + b_i + U_u . V_i  (the full model)
      bias_only   = mu + b_u + b_i              (drop the latent-factor term)
      item_mean   = mu + b_i                    (= each movie's own mean rating)
      user_mean   = mu + b_u                    (= each user's own mean rating)
      global_mean = mu                          (the trivial baseline)

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

    sse = {name: 0.0 for name in ("svd", "bias_only", "item_mean", "user_mean", "global_mean")}
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

        ub = user_biases[u]
        mb = movie_biases[m]
        interaction = np.einsum("ij,ij->i", UF[u], MF[m])

        preds = {
            "svd":         global_mean + ub + mb + interaction,
            "bias_only":   global_mean + ub + mb,
            "item_mean":   global_mean + mb,
            "user_mean":   global_mean + ub,
            "global_mean": np.full_like(actuals, global_mean),
        }
        for name, pred in preds.items():
            clipped = np.clip(pred, 0.5, 5.0)
            sse[name] += float(np.sum((clipped - actuals) ** 2))
        n_eval += int(len(actuals))

    rmses = {name: float(np.sqrt(s / n_eval)) for name, s in sse.items()}
    return rmses, n_eval

# ── k-tuning ──────────────────────────────────────────────────────────────────

def tune_k(model: dict, test: pd.DataFrame, k_values: list[int] = K_VALUES) -> tuple[int, list[tuple[int, float]]]:
    """
    Evaluate RMSE at each k by truncating a single pre-trained model's factors.
    Returns (best_k, [(k, rmse), ...]) so the curve can be saved/plotted.
    """
    print(f"  {'k':<8} {'RMSE':>8}")
    print(f"  {'-' * 18}")
    best_k, best_rmse = k_values[0], float("inf")
    curve: list[tuple[int, float]] = []
    for k in k_values:
        rmses, _ = compute_rmse(model, test, k=k)
        rmse = rmses["svd"]
        curve.append((k, rmse))
        marker = ""
        if rmse < best_rmse:
            best_rmse = rmse
            best_k = k
            marker = "  ←"
        print(f"  {k:<8} {rmse:>8.4f}{marker}")
    return best_k, curve


# ── Recommendation methods ────────────────────────────────────────────────────

def algorithm_recommender(
    movies: pd.DataFrame,
    model: dict,
    train_movies: dict[int, set[int]],
    train_input_movies: dict[int, list[int]],
    dq_map: dict[int, float],
    franchise_map: dict,
    installment_trend: dict,
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
            installment_trend,
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
) -> tuple[float, float, list[float], list[float]]:
    """
    Mean Precision@K and Recall@K over selected users with ≥1 relevant item,
    plus the raw per-user lists (used to plot the score distribution, not just
    the mean). Relevant = rating ≥ RELEVANCE_THRESHOLD.
    """
    p_list: list[float] = []
    r_list: list[float] = []
    for i, user_id in enumerate(user_ids, start=1):
        relevant = relevant_by_user[user_id]
        top_k = rec_fn(user_id)[:k]
        hits = sum(1 for m in top_k if m in relevant)
        p_list.append(hits / k)
        r_list.append(hits / len(relevant))
    return float(np.mean(p_list)), float(np.mean(r_list)), p_list, r_list

# ── Weight sensitivity sweep ──────────────────────────────────────────────────
#
# Sweeps each top-level scoring weight in recommendations_algorithm.py one at a
# time, holding the other three fixed at their *relative* proportions (so all
# four keep summing to 1 at every grid point), and re-measures Precision@K /
# Recall@K at each setting. This shows empirically whether the chosen weights
# matter and whether they sit near the best point on each curve, instead of
# only asserting the choice in prose.

def weights_for(varied_name: str, varied_value: float) -> dict[str, float]:
    """
    Full weight dict where `varied_name` = varied_value and the other three
    weights are rescaled to keep their relative proportions while the four
    still sum to 1. This isolates the balance under study instead of also
    changing the overall magnitude of the score (which would not affect
    ranking on its own).
    """
    others = {k: v for k, v in BASE_WEIGHTS.items() if k != varied_name}
    others_sum = sum(others.values())
    remaining = max(0.0, 1.0 - varied_value)
    if others_sum <= 0:
        scaled = {k: remaining / len(others) for k in others}
    else:
        scaled = {k: remaining * (v / others_sum) for k, v in others.items()}
    scaled[varied_name] = varied_value
    return scaled


def apply_weights(weights: dict[str, float]) -> None:
    for name, value in weights.items():
        setattr(ra, name, value)


def run_weight_sweep(
    varied_name: str,
    grid,
    movies,
    model,
    train_movies,
    train_input_movies,
    dq_map,
    franchise_map,
    installment_trend,
    relevant_by_user,
    eval_user_ids,
) -> pd.DataFrame:
    rows = []
    for value in grid:
        weights = weights_for(varied_name, value)
        apply_weights(weights)
        t0 = time.time()
        rec_fn = algorithm_recommender(
            movies, model, train_movies, train_input_movies,
            dq_map, franchise_map, installment_trend, top_n=K,
        )
        p_mean, r_mean, _, _ = eval_ranking(rec_fn, relevant_by_user, eval_user_ids, k=K)
        elapsed = time.time() - t0
        print(f"  {varied_name}={value:.2f} (others rescaled): "
              f"Precision@{K}={p_mean:.4f}  Recall@{K}={r_mean:.4f}  [{elapsed:.1f}s]")
        rows.append({"varied_weight": varied_name, "value": value, **weights,
                      "precision": p_mean, "recall": r_mean})
    apply_weights(BASE_WEIGHTS)  # restore baseline before the next weight's sweep
    return pd.DataFrame(rows)


def dialogue_weights_for(varied_name: str, varied_value: float) -> dict[str, float]:
    """
    Full DIALOGUE_FEATURE_WEIGHTS-shaped dict where `varied_name` is set to
    `varied_value` (which may carry either sign, e.g. to test flipping a
    feature from a penalty to a reward) and the other features are rescaled
    to keep their relative magnitudes while all five magnitudes still sum to
    1. Mirrors weights_for(), generalized to signed feature weights.
    """
    others = {k: v for k, v in BASE_DIALOGUE_WEIGHTS.items() if k != varied_name}
    others_abs_sum = sum(abs(v) for v in others.values())
    remaining = max(0.0, 1.0 - abs(varied_value))
    if others_abs_sum <= 0:
        scaled = {k: remaining / len(others) for k in others}
    else:
        scaled = {k: remaining * (v / others_abs_sum) for k, v in others.items()}
    scaled[varied_name] = varied_value
    return scaled


def apply_dialogue_weights(weights: dict[str, float]) -> None:
    ra.DIALOGUE_FEATURE_WEIGHTS = dict(weights)


def run_dialogue_weight_sweep(
    varied_name: str,
    grid,
    movies,
    model,
    train_movies,
    train_input_movies,
    franchise_map,
    installment_trend,
    relevant_by_user,
    eval_user_ids,
) -> pd.DataFrame:
    """Like run_weight_sweep(), but for a DIALOGUE_FEATURE_WEIGHTS entry: each
    grid point needs dq_map recomputed from the new feature weights before
    scoring, since dialogue quality is precomputed per movie rather than
    read live per candidate."""
    rows = []
    for value in grid:
        weights = dialogue_weights_for(varied_name, value)
        apply_dialogue_weights(weights)
        dq_map = load_dialogue_features()
        t0 = time.time()
        rec_fn = algorithm_recommender(
            movies, model, train_movies, train_input_movies,
            dq_map, franchise_map, installment_trend, top_n=K,
        )
        p_mean, r_mean, _, _ = eval_ranking(rec_fn, relevant_by_user, eval_user_ids, k=K)
        elapsed = time.time() - t0
        print(f"  {varied_name}={value:+.2f} (others rescaled): "
              f"Precision@{K}={p_mean:.4f}  Recall@{K}={r_mean:.4f}  [{elapsed:.1f}s]")
        rows.append({"varied_weight": varied_name, "value": weights[varied_name], **weights,
                      "precision": p_mean, "recall": r_mean})
    apply_dialogue_weights(BASE_DIALOGUE_WEIGHTS)  # restore baseline before the next feature's sweep
    return pd.DataFrame(rows)


def save_sensitivity_plot(
    weight_results: pd.DataFrame,
    dialogue_results: pd.DataFrame,
    plots_dir: Path,
) -> None:
    names = list(BASE_WEIGHTS.keys())
    fig = plt.figure(figsize=(4.8 * len(names), 10))
    gs = fig.add_gridspec(2, len(names))

    top_axes = [fig.add_subplot(gs[0, i]) for i in range(len(names))]
    # Bottom row: same per-subplot size as the top row (one grid column each),
    # placed under the middle two columns so they read as centered.
    bottom_axes = [
        fig.add_subplot(gs[1, 1], sharey=top_axes[0]),
        fig.add_subplot(gs[1, 2], sharey=top_axes[0]),
    ]

    def draw(ax, sub, chosen_value, xlabel):
        ax.plot(sub["value"], sub["precision"], marker="o", linewidth=2.5, markersize=8,
                color="#4C72B0", label=f"Precision@{K}")
        ax.plot(sub["value"], sub["recall"], marker="s", linewidth=2.5, markersize=8,
                color="#DD8452", label=f"Recall@{K}")
        ax.axvline(chosen_value, color="#555555", linestyle="--", linewidth=1.5, label="Chosen value")
        ax.set_xlabel(xlabel, fontsize=19)
        ax.tick_params(axis="both", labelsize=16)

    for ax, name in zip(top_axes, names):
        sub = weight_results[weight_results["varied_weight"] == name].sort_values("value")
        draw(ax, sub, BASE_WEIGHTS[name], WEIGHT_LABELS[name])

    for ax, name in zip(bottom_axes, DIALOGUE_SWEEP_FEATURES):
        sub = dialogue_results[dialogue_results["varied_weight"] == name].sort_values("value")
        draw(ax, sub, BASE_DIALOGUE_WEIGHTS[name], DIALOGUE_WEIGHT_LABELS[name])

    top_axes[0].set_ylabel("Score", fontsize=19)
    bottom_axes[0].set_ylabel("Score", fontsize=19)
    top_axes[0].legend(fontsize=15, loc="best")
    fig.tight_layout()
    fig.subplots_adjust(top=0.88, hspace=0.45)
    fig.suptitle("Weight Sensitivity: Precision@K / Recall@K vs. Each Scoring Weight", fontsize=24, y=0.96)
    out_path = plots_dir / "weight_sensitivity.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot to {out_path}")


def run_weight_sensitivity(args: argparse.Namespace) -> None:
    max_rows = None if args.rows == 0 else args.rows

    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    print(f"Loading ratings (max_rows={max_rows}) ...")
    train, test = load_and_split(max_rating_rows=max_rows)
    print(f"  Train: {len(train):,}  Test: {len(test):,}")

    movies = load_movies_metadata(MOVIES_PATH)
    metadata_movie_ids = set(movies["MovieID"].astype(int))

    print(f"Building SVD at k={WEIGHT_SENSITIVITY_SVD_K} ...")
    model = build_svd(train, k=WEIGHT_SENSITIVITY_SVD_K)
    model["movie_factors"] = model["movie_factors"][:, :WEIGHT_SENSITIVITY_SVD_K]
    model_movie_ids = set(int(m) for m in model["movie_ids"].tolist())
    allowed_movie_ids = metadata_movie_ids.intersection(model_movie_ids)
    print(f"  Candidate movies: {len(allowed_movie_ids):,}")

    relevant_by_user = build_relevant_test_movies(test, allowed_movie_ids=allowed_movie_ids)
    eval_user_ids = select_eval_users(relevant_by_user, max_users=args.users)
    print(f"  Eval users: {len(eval_user_ids):,} (of {len(relevant_by_user):,} eligible)")

    train_movies = build_train_movies_for_users(train, eval_user_ids)
    train_input_movies = build_train_input_movies_for_users(train, eval_user_ids, allowed_movie_ids)

    dq_map = load_dialogue_features()
    franchise_map = load_franchise_map()
    installment_trend = load_installment_rating_trend()

    print("\nBaseline weights:", BASE_WEIGHTS)

    all_results = []
    for name in args.weights:
        print(f"\nSweeping {name} ...")
        df = run_weight_sweep(
            name, WEIGHT_GRID, movies, model, train_movies, train_input_movies,
            dq_map, franchise_map, installment_trend, relevant_by_user, eval_user_ids,
        )
        all_results.append(df)

    all_results = pd.concat(all_results, ignore_index=True)
    all_results.to_csv(OUTPUT_DIR / "weight_sensitivity.csv", index=False)
    print(f"\nSaved sweep results to {OUTPUT_DIR / 'weight_sensitivity.csv'}")

    dialogue_results = []
    for name in DIALOGUE_SWEEP_FEATURES:
        print(f"\nSweeping dialogue feature weight: {name} ...")
        df = run_dialogue_weight_sweep(
            name, DIALOGUE_SWEEP_GRIDS[name], movies, model, train_movies, train_input_movies,
            franchise_map, installment_trend, relevant_by_user, eval_user_ids,
        )
        dialogue_results.append(df)

    dialogue_results = pd.concat(dialogue_results, ignore_index=True)
    dialogue_results.to_csv(OUTPUT_DIR / "dialogue_weight_sensitivity.csv", index=False)
    print(f"\nSaved dialogue feature sweep results to {OUTPUT_DIR / 'dialogue_weight_sensitivity.csv'}")

    save_sensitivity_plot(all_results, dialogue_results, PLOTS_DIR)


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
    by_inst = df.groupby("group")["imdb_averageRating"].agg(mean="mean", count="count", std="std")
    by_inst["std"] = by_inst["std"].fillna(0.0)
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


class _Tee:
    """Write everything to both the original stream and a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def save_results_csv(
    output_dir: Path,
    rmses: dict[str, float],
    n_eval: int,
    best_k: int,
    eval_user_ids: list[int],
    allowed_movie_ids: set[int],
    metadata_movie_ids: set[int],
    model_movie_ids: set[int],
    methods: list[tuple[str, object]],
    results: dict[str, tuple[float, float]],
    k_tuning_curve: list[tuple[int, float]],
    fa: dict | None,
    corr: pd.DataFrame,
) -> None:
    """Save the run's headline metrics, method comparison, k-tuning curve, and
    franchise/dialogue diagnostics as CSVs so results can be tracked across runs
    without re-parsing the text log."""
    alg_p, alg_r = results["Our system"]

    summary = {
        "RMSE": rmses["svd"],
        "Bias_only_RMSE": rmses["bias_only"],
        "Item_mean_RMSE": rmses["item_mean"],
        "User_mean_RMSE": rmses["user_mean"],
        "Global_mean_RMSE": rmses["global_mean"],
        "RMSE_delta": rmses["svd"] - rmses["global_mean"],
        "Evaluated_rating_pairs": n_eval,
        "Selected_k": best_k,
        "Ranking_users": len(eval_user_ids),
        "Candidate_movies": len(allowed_movie_ids),
        "Metadata_movies": len(metadata_movie_ids),
        "Model_movies": len(model_movie_ids),
        f"Precision_at_{K}": alg_p,
        f"Recall_at_{K}": alg_r,
    }
    pd.DataFrame(summary.items(), columns=["metric", "value"]).to_csv(
        output_dir / "summary_metrics.csv", index=False
    )

    method_rows = [
        {
            "method": name,
            f"precision_at_{K}": p,
            f"recall_at_{K}": r,
            "delta_precision": p - alg_p,
            "delta_recall": r - alg_r,
        }
        for name, _ in methods
        for p, r in [results[name]]
    ]
    pd.DataFrame(method_rows).to_csv(output_dir / "method_comparison.csv", index=False)

    pd.DataFrame(k_tuning_curve, columns=["k", "rmse"]).to_csv(
        output_dir / "k_tuning.csv", index=False
    )

    if fa is not None:
        pd.DataFrame([{
            "n": fa["n"],
            "spearman_r": fa["spearman_r"],
            "spearman_p": fa["spearman_p"],
            "pearson_r": fa["pearson_r"],
            "pearson_p": fa["pearson_p"],
        }]).to_csv(output_dir / "franchise_summary.csv", index=False)
        fa["by_inst"].reset_index().to_csv(output_dir / "franchise_by_installment.csv", index=False)

    if not corr.empty:
        corr.to_csv(output_dir / "dialogue_correlation.csv", index=False)

    print(f"  Saved CSVs to {output_dir}/")


# Shared font sizes for evaluate.py's plots (kept larger than matplotlib
# defaults so labels are readable at presentation size, not just on screen).
PLOT_TITLE_FONT_SIZE = 26
PLOT_LABEL_FONT_SIZE = 20
PLOT_TICK_FONT_SIZE = 17
PLOT_LEGEND_FONT_SIZE = 17
PLOT_ANNOTATION_FONT_SIZE = 16


def save_plots(
    plots_dir: Path,
    methods: list[tuple[str, object]],
    results: dict[str, tuple[float, float]],
    raw_results: dict[str, tuple[list[float], list[float]]],
    k_tuning_curve: list[tuple[int, float]],
    best_k: int,
    rmses: dict[str, float],
) -> None:
    """Save the diagnostic plots used in the writeup for this evaluation run:
    the overall method comparison, the per-user outcome distribution, and the
    k-tuning curve. These complement (but do not replace) the feature-
    correlation plots in analyze_data.py."""
    plt.rcParams.update({"figure.max_open_warning": 0})
    names = [name for name, _ in methods]

    # All-factors comparison: RMSE across every rating-prediction baseline
    # (from the full SVD down to always guessing the global mean) alongside
    # Precision@K / Recall@K for every ranking method, so the headline
    # accuracy numbers sit next to the ranking metrics instead of only being
    # printed to the log. See the comment above compute_rmse() for why RMSE
    # and Precision/Recall use different baseline families.
    fig, (ax_rmse, ax_prec, ax_rec) = plt.subplots(
        1, 3, figsize=(19, 7.5), gridspec_kw={"width_ratios": [1.3, 1, 1]}
    )

    rmse_labels = ["Our system (SVD)", "Bias-only\n(no factors)", "Overall average\n(same for everyone)"]
    rmse_values = [rmses["svd"], rmses["bias_only"], rmses["global_mean"]]
    rmse_colors = [EVAL_METHOD_COLORS["Our system"], "#9E9E9E", "#D9D9D9"]
    bars = ax_rmse.bar(rmse_labels, rmse_values, color=rmse_colors)
    for bar, v in zip(bars, rmse_values):
        ax_rmse.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}",
                     ha="center", va="bottom", fontsize=PLOT_LABEL_FONT_SIZE)
    ax_rmse.set_xticks(range(len(rmse_labels)))
    ax_rmse.set_xticklabels(rmse_labels, rotation=35, ha="right", fontsize=PLOT_ANNOTATION_FONT_SIZE)
    ax_rmse.set_ylabel("RMSE (lower is better)", fontsize=PLOT_TITLE_FONT_SIZE)
    ax_rmse.set_title("Rating Prediction (RMSE)", fontsize=PLOT_TITLE_FONT_SIZE)
    ax_rmse.tick_params(axis="y", labelsize=PLOT_LABEL_FONT_SIZE)
    ax_rmse.margins(y=0.15)

    for ax, metric_idx, title in ((ax_prec, 0, f"Precision@{K}"), (ax_rec, 1, f"Recall@{K}")):
        values = [results[name][metric_idx] for name in names]
        colors = [EVAL_METHOD_COLORS.get(name, "#4C72B0") for name in names]
        bars = ax.bar(names, values, color=colors)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=PLOT_LABEL_FONT_SIZE)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=PLOT_LABEL_FONT_SIZE)
        ax.tick_params(axis="y", labelsize=PLOT_LABEL_FONT_SIZE)
        ax.set_ylabel("Score", fontsize=PLOT_TITLE_FONT_SIZE)
        ax.set_title(title, fontsize=PLOT_TITLE_FONT_SIZE)
        ax.margins(y=0.15)

    for ax in (ax_rmse, ax_prec, ax_rec):
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")

    fig.suptitle("Our System vs. Baselines: All Metrics", fontsize=PLOT_TITLE_FONT_SIZE + 6, y=0.98)
    fig.tight_layout(rect=(0, 0.05, 1, 0.965), w_pad=3.0)
    fig.savefig(plots_dir / "overall_metrics_comparison.png", dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)

    # Note: a separate "Precision@K/Recall@K by method (mean)" bar chart used
    # to be saved here (precision_recall_by_method.png), but it isn't used in
    # the writeup -- its numbers are already the center/right panels of
    # overall_metrics_comparison.png above -- so it's been dropped to keep
    # this function producing only plots that appear in the writeup.

    # Per-user Precision@K / Recall@K distribution by method. Precision@K
    # can only take K+1 discrete values (hits / K) and is heavily zero-inflated,
    # so a boxplot collapses to a flat line at 0 with everything else drawn as
    # "outlier" points, and an 11-column heatmap is too fine-grained to read at
    # a glance. Instead, bucket each user's outcome into 3 plain-language
    # categories per metric and stack them, the same traffic-light style as
    # survey_rank_distribution.png: red = found nothing, orange = found some,
    # green = found (nearly) everything.
    outcome_colors = ["#E53935", "#FFB74D", "#2E7D32"]  # worst -> best

    def draw_outcome_bars(ax, pct_matrix: np.ndarray, seg_labels: list[str], title: str) -> None:
        bottom = np.zeros(len(names))
        for seg_idx, (seg_label, color) in enumerate(zip(seg_labels, outcome_colors)):
            values = pct_matrix[:, seg_idx]
            bars = ax.bar(names, values, bottom=bottom, color=color, label=seg_label)
            for bar, v in zip(bars, values):
                if v >= 3:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + v / 2, f"{v:.0f}%",
                            ha="center", va="center", fontsize=PLOT_ANNOTATION_FONT_SIZE,
                            color="white", fontweight="bold")
            bottom += values
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=PLOT_TICK_FONT_SIZE)
        ax.tick_params(axis="y", labelsize=PLOT_TICK_FONT_SIZE)
        ax.set_ylabel("Users (%)", fontsize=PLOT_LABEL_FONT_SIZE)
        ax.set_title(title, fontsize=PLOT_LABEL_FONT_SIZE, pad=18)
        ax.legend(fontsize=PLOT_ANNOTATION_FONT_SIZE, loc="upper center",
                  bbox_to_anchor=(0.5, -0.26), ncol=3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

    # Precision@K: how many of the K recommended movies were relevant.
    precision_labels = ["No relevant picks", "1 relevant pick", "2+ relevant picks"]
    precision_matrix = np.zeros((len(names), 3))
    for i, name in enumerate(names):
        p_list = np.asarray(raw_results[name][0])
        hits = np.round(p_list * K).astype(int)
        precision_matrix[i] = [np.mean(hits == 0), np.mean(hits == 1), np.mean(hits >= 2)]
    precision_matrix *= 100
    draw_outcome_bars(ax1, precision_matrix, precision_labels,
                       f"Precision@{K}: Relevant Movies Found in Top-{K}")

    # Recall@K: what fraction of a user's held-out liked movies were surfaced.
    recall_labels = ["Found none", "Found some", "Found all"]
    recall_matrix = np.zeros((len(names), 3))
    for i, name in enumerate(names):
        r_list = np.asarray(raw_results[name][1])
        recall_matrix[i] = [np.mean(r_list <= 1e-9), np.mean((r_list > 1e-9) & (r_list < 1 - 1e-9)),
                             np.mean(r_list >= 1 - 1e-9)]
    recall_matrix *= 100
    draw_outcome_bars(ax2, recall_matrix, recall_labels, f"Recall@{K}: Coverage of a User's Liked Movies")

    fig.suptitle(f"Per-User Precision@{K} / Recall@{K} Outcomes by Method",
                 fontsize=PLOT_TITLE_FONT_SIZE, y=0.985)
    fig.text(0.5, 0.92, f"Each user's top-{K} recommendations are checked against their held-out liked movies "
                        f"(rating >= {RELEVANCE_THRESHOLD:g})",
             ha="center", fontsize=PLOT_ANNOTATION_FONT_SIZE, style="italic", color="dimgray")
    fig.tight_layout(rect=(0, 0.02, 1, 0.905), w_pad=2.0)
    fig.savefig(plots_dir / "precision_recall_distribution.png", dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    # k-tuning RMSE curve, with the RMSE value annotated at each point.
    ks = [k for k, _ in k_tuning_curve]
    curve_rmses = [rmse for _, rmse in k_tuning_curve]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(ks, curve_rmses, marker="o", markersize=11, linewidth=2.5, color="#4C72B0")
    for k, rmse in k_tuning_curve:
        ax.annotate(f"{rmse:.4f}", (k, rmse), textcoords="offset points", xytext=(0, 14),
                    ha="center", fontsize=PLOT_LABEL_FONT_SIZE)
    best_rmse = dict(k_tuning_curve)[best_k]
    ax.scatter([best_k], [best_rmse], color="#C44E52", s=150, zorder=5, label=f"Selected k={best_k}")
    ax.set_xlabel("k (latent factors)", fontsize=PLOT_TITLE_FONT_SIZE)
    ax.set_ylabel("RMSE", fontsize=PLOT_TITLE_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=PLOT_LABEL_FONT_SIZE)
    ax.set_title("SVD k-Tuning: RMSE vs. Number of Latent Factors", fontsize=PLOT_TITLE_FONT_SIZE)
    ax.margins(y=0.15)
    ax.legend(fontsize=PLOT_TITLE_FONT_SIZE)
    fig.tight_layout()
    fig.savefig(plots_dir / "k_tuning_rmse.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved plots to {plots_dir}/")


# ── Participant survey results ───────────────────────────────────────────────
#
# Separate from the offline metrics above: 15 participants each rated 3 movies
# from each of the 4 methods (blinded) and ranked the 4 method-groups best (1)
# to worst (4). This section is optional — it is skipped if the survey data in
# survey-responses/ is not present.

def load_survey_responses(path: Path = SURVEY_RESPONSES_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["method"] = pd.Categorical(df["method"], categories=SURVEY_METHOD_ORDER, ordered=True)
    return df


def summarize_survey_responses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute the per-method survey summary from the raw responses.

    Movie-level metrics (relevance, would-watch) are averaged over all rated
    movies (3 per submission per method). Group-level metrics (rank, novelty)
    are averaged once per submission/method, since they were only asked once
    per group, not once per movie.
    """
    df = df.copy()
    # Per-rating main score, so its std can be computed the same way as
    # relevance/would-watch's, rather than only existing as a scalar mean of
    # means with no spread information.
    df["_main_score_row"] = (df["relevance_1_5"] + df["would_watch_1_5"]) / 2

    movie_level = df.groupby("method", observed=True).agg(
        n_respondents=("submission_id", "nunique"),
        n_movie_ratings=("submission_id", "size"),
        avg_relevance=("relevance_1_5", "mean"),
        std_relevance=("relevance_1_5", "std"),
        avg_would_watch=("would_watch_1_5", "mean"),
        std_would_watch=("would_watch_1_5", "std"),
        avg_main_score=("_main_score_row", "mean"),
        std_main_score=("_main_score_row", "std"),
    )

    group_level = (
        df.drop_duplicates(subset=["submission_id", "method"])
        .groupby("method", observed=True)
        .agg(
            avg_group_rank=("group_rank_1_best", "mean"),
            std_group_rank=("group_rank_1_best", "std"),
            avg_novelty=("group_novelty_1_5", "mean"),
            std_novelty=("group_novelty_1_5", "std"),
            ranked_1st_count=("group_rank_1_best", lambda s: int((s == 1).sum())),
        )
    )

    summary = movie_level.join(group_level)
    summary["main_score"] = summary["avg_main_score"]
    return summary.loc[SURVEY_METHOD_ORDER]


def survey_rank_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Count how many times each method was ranked 1st..4th (1 = best)."""
    group_level = df.drop_duplicates(subset=["submission_id", "method"])
    counts = (
        group_level.groupby(["method", "group_rank_1_best"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    for rank in (1, 2, 3, 4):
        if rank not in counts.columns:
            counts[rank] = 0
    return counts.loc[SURVEY_METHOD_ORDER, [1, 2, 3, 4]]


def cross_check_survey_summary(summary: pd.DataFrame, summary_path: Path = SURVEY_SUMMARY_PATH) -> None:
    if not summary_path.exists():
        return
    existing = pd.read_csv(summary_path, encoding="utf-8-sig").set_index("method").loc[SURVEY_METHOD_ORDER]
    cols = ["avg_relevance", "avg_would_watch", "main_score", "ranked_1st_count", "avg_novelty"]
    diffs = (summary[cols] - existing[cols]).abs()
    if (diffs.max() > 1e-6).any():
        print("  WARNING: recomputed summary differs from results_by_method.csv:")
        print(diffs[diffs.max(axis=1) > 1e-6])
    else:
        print("  Recomputed summary matches results_by_method.csv.")


def save_survey_metrics_dashboard(summary: pd.DataFrame, plots_dir: Path) -> None:
    """
    One clustered bar chart: each of the four shared 1-5 scale metrics (relevance,
    would-watch, novelty, main score) gets a group of four bars, one per method.
    Avg. Rank and Times-Ranked-#1 are on different scales (1-4 and count-of-15)
    and are already shown clearly in survey_rank_distribution.png, so they are
    left out here rather than crammed into unrelated small subplots.
    """
    methods = list(summary.index)
    colors = [SURVEY_METHOD_COLORS[m] for m in methods]

    metrics = [
        ("avg_relevance", "std_relevance", "Relevance"),
        ("avg_would_watch", "std_would_watch", "Would-Watch"),
        ("avg_novelty", "std_novelty", "Novelty"),
        ("main_score", "std_main_score", "Main Score"),
    ]
    metric_labels = [label for _, _, label in metrics]

    n_methods = len(methods)
    bar_width = 0.8 / n_methods
    x = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(13, 8))
    for i, (method, color) in enumerate(zip(methods, colors)):
        values = [summary.loc[method, col] for col, _, _ in metrics]
        errs = [summary.loc[method, err_col] if err_col else 0.0 for _, err_col, _ in metrics]
        offset = (i - (n_methods - 1) / 2) * bar_width
        bars = ax.bar(x + offset, values, bar_width, yerr=errs, capsize=4,
                       error_kw={"linewidth": 1.5},
                       label=SURVEY_METHOD_LABELS[method], color=color)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, 0.1, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=PLOT_ANNOTATION_FONT_SIZE,
                    color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=PLOT_LABEL_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=PLOT_LABEL_FONT_SIZE)
    ax.set_ylabel("Score (1-5)", fontsize=PLOT_TITLE_FONT_SIZE)
    ax.set_title("Participant Survey Results by Method", fontsize=PLOT_TITLE_FONT_SIZE, pad=38)
    ax.text(0.5, 1.02, "Black lines = ± 1 standard deviation across respondents' ratings",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=PLOT_LABEL_FONT_SIZE, style="italic", color="dimgray")
    ax.legend(fontsize=PLOT_LEGEND_FONT_SIZE)
    fig.tight_layout()
    fig.savefig(plots_dir / "survey_metrics_by_method.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_survey_rank_distribution(counts: pd.DataFrame, plots_dir: Path) -> None:
    # Sort methods by their actual average rank (best to worst), rather than
    # relying on SURVEY_METHOD_ORDER already happening to be in that order.
    n_respondents = counts.sum(axis=1)
    avg_rank = (counts[1] * 1 + counts[2] * 2 + counts[3] * 3 + counts[4] * 4) / n_respondents
    counts = counts.loc[avg_rank.sort_values().index]

    names = [SURVEY_METHOD_LABELS[m] for m in counts.index]
    rank_colors = {1: "#2E7D32", 2: "#9CCC65", 3: "#FFB74D", 4: "#E53935"}  # best (1) -> worst (4)

    fig, ax = plt.subplots(figsize=(11, 8))
    bottom = np.zeros(len(counts))
    # Stack rank 4 first (bottom) up to rank 1 (top), so "Ranked 1" reads at
    # the top of each bar and "Ranked 4" at the bottom.
    for rank in (4, 3, 2, 1):
        color = rank_colors[rank]
        values = counts[rank].to_numpy(dtype=float)
        bars = ax.bar(names, values, bottom=bottom, color=color, label=f"Ranked {rank}")
        for bar, v in zip(bars, values):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + v / 2, int(v),
                        ha="center", va="center", fontsize=PLOT_ANNOTATION_FONT_SIZE, color="white",
                        fontweight="bold")
        bottom += values

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=PLOT_TITLE_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=PLOT_LABEL_FONT_SIZE)
    ax.set_ylabel("Number of respondents (of 15)", fontsize=PLOT_TITLE_FONT_SIZE)
    ax.set_title("How Often Each Method Was Ranked 1st-4th", fontsize=PLOT_TITLE_FONT_SIZE + 1, pad=20)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], fontsize=PLOT_LABEL_FONT_SIZE,
              bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(plots_dir / "survey_rank_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_survey_rating_distribution(df: pd.DataFrame, plots_dir: Path) -> None:
    """
    Shaded-line (line + filled area) chart of the rating distribution shape per
    method, replacing boxplots: ratings are integers 1-5, so a box-and-whisker
    summary hides the actual shape (e.g. how bimodal or concentrated it is).
    Plotting % of ratings at each value as a line with a shaded area under it
    shows that shape directly and makes the four methods easy to compare.
    """
    scale = [1, 2, 3, 4, 5]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    for ax, col, title in (
        (ax1, "relevance_1_5", "Relevance (1-5)"),
        (ax2, "would_watch_1_5", "Would-Watch (1-5)"),
    ):
        for method in SURVEY_METHOD_ORDER:
            values = df.loc[df["method"] == method, col].to_numpy()
            pct = np.array([np.mean(values == v) for v in scale]) * 100
            color = SURVEY_METHOD_COLORS[method]
            ax.plot(scale, pct, marker="o", markersize=9, linewidth=2.5,
                    color=color, label=SURVEY_METHOD_LABELS[method])
            ax.fill_between(scale, pct, color=color, alpha=0.15)
        ax.set_xticks(scale)
        ax.tick_params(axis="both", labelsize=PLOT_LABEL_FONT_SIZE)
        ax.set_xlabel(title, fontsize=PLOT_TITLE_FONT_SIZE)
        ax.set_ylabel("Ratings (%)", fontsize=PLOT_TITLE_FONT_SIZE)
        ax.margins(y=0.1)
        ax.legend(fontsize=PLOT_LABEL_FONT_SIZE)

    fig.suptitle("Per-Movie Rating Distribution by Method", fontsize=PLOT_TITLE_FONT_SIZE + 4)
    fig.tight_layout()
    fig.savefig(plots_dir / "survey_rating_distribution.png", dpi=150)
    plt.close(fig)


def save_survey_relevance_vs_watch_scatter(df: pd.DataFrame, plots_dir: Path) -> None:
    """
    Construct-validity check: relevance and would-watch are two separate survey
    questions asked about the same movie. If they measure related things, they
    should move together. Both are integers 1-5 with only 180 ratings total, so
    a scatter (even jittered) is mostly overlapping dots; a small-multiple
    heatmap per method (cell = number of ratings at that relevance/would-watch
    pair) shows the diagonal concentration without the overlap.
    """
    scale = [1, 2, 3, 4, 5]
    matrices = {}
    for method in SURVEY_METHOD_ORDER:
        sub = df[df["method"] == method]
        mat = np.zeros((5, 5), dtype=int)  # rows = would-watch, cols = relevance
        for rel, watch in zip(sub["relevance_1_5"].astype(int), sub["would_watch_1_5"].astype(int)):
            mat[watch - 1, rel - 1] += 1
        matrices[method] = mat
    max_count = max(mat.max() for mat in matrices.values())

    fig, axes = plt.subplots(2, 2, figsize=(13, 13))
    fig.subplots_adjust(hspace=0.4, wspace=0.3, top=0.83)
    im = None
    for ax, method in zip(axes.flat, SURVEY_METHOD_ORDER):
        mat = matrices[method]
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=max_count, origin="lower", aspect="auto")
        for r in range(5):
            for c in range(5):
                v = mat[r, c]
                if v > 0:
                    text_color = "white" if v > max_count * 0.55 else "#222222"
                    ax.text(c, r, str(v), ha="center", va="center",
                            fontsize=PLOT_LABEL_FONT_SIZE, color=text_color)
        ax.set_xticks(range(5))
        ax.set_xticklabels(scale, fontsize=PLOT_LABEL_FONT_SIZE)
        ax.set_yticks(range(5))
        ax.set_yticklabels(scale, fontsize=PLOT_LABEL_FONT_SIZE)
        ax.set_xlabel("Relevance", fontsize=PLOT_TITLE_FONT_SIZE)
        ax.set_ylabel("Would-watch", fontsize=PLOT_TITLE_FONT_SIZE)
        ax.set_title(SURVEY_METHOD_LABELS[method], fontsize=PLOT_TITLE_FONT_SIZE)

    cbar = fig.colorbar(im, ax=axes, fraction=0.04, pad=0.03)
    cbar.set_label("Number of ratings", fontsize=PLOT_TITLE_FONT_SIZE)
    cbar.ax.tick_params(labelsize=PLOT_LABEL_FONT_SIZE)

    r, _ = pearsonr(df["relevance_1_5"], df["would_watch_1_5"])
    fig.suptitle("Relevance vs. Would-Watch Ratings by Method", fontsize=PLOT_TITLE_FONT_SIZE + 3, y=0.965)
    fig.text(0.5, 0.91, f"Pearson r = {r:.2f}  (n = {len(df)} ratings)",
             ha="center", fontsize=PLOT_LABEL_FONT_SIZE, color="dimgray")
    fig.text(0.5, 0.885, "Diagonal concentration = the two ratings agree",
             ha="center", fontsize=PLOT_LABEL_FONT_SIZE, style="italic", color="dimgray")
    fig.savefig(plots_dir / "survey_relevance_vs_would_watch.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_survey_novelty_vs_relevance_scatter(df: pd.DataFrame, plots_dir: Path) -> None:
    """
    Accuracy-vs-diversity trade-off: for each method, plot its mean novelty
    rating against its mean relevance rating (averaged over respondents), with
    error bars for the spread across respondents. Plotting all 60 individual
    per-respondent points made the trend hard to see, so this shows one clearly
    positioned marker per method instead.
    """
    group_level = df.drop_duplicates(subset=["submission_id", "method"])[
        ["submission_id", "method", "group_novelty_1_5"]
    ]
    movie_avg = (
        df.groupby(["submission_id", "method"], observed=True)["relevance_1_5"]
        .mean()
        .reset_index(name="avg_relevance")
    )
    merged = group_level.merge(movie_avg, on=["submission_id", "method"])

    fig, ax = plt.subplots(figsize=(10, 8))

    for method in SURVEY_METHOD_ORDER:
        sub = merged[merged["method"] == method]
        color = SURVEY_METHOD_COLORS[method]
        ax.errorbar(
            sub["avg_relevance"].mean(), sub["group_novelty_1_5"].mean(),
            xerr=sub["avg_relevance"].std(), yerr=sub["group_novelty_1_5"].std(),
            fmt="o", markersize=20, color=color, ecolor=color, elinewidth=2, capsize=6,
            markeredgecolor="black", markeredgewidth=1.3, label=SURVEY_METHOD_LABELS[method], zorder=5,
        )

    ax.tick_params(axis="both", labelsize=PLOT_LABEL_FONT_SIZE)
    ax.set_xlabel("Mean relevance rating per respondent (1-5)", fontsize=PLOT_TITLE_FONT_SIZE)
    ax.set_ylabel("Mean novelty rating (1-5)", fontsize=PLOT_TITLE_FONT_SIZE)
    ax.set_title("Novelty vs. Relevance Trade-off by Method", fontsize=PLOT_TITLE_FONT_SIZE, pad=38)
    ax.text(0.5, 1.02, "marker = method mean; error bars = ± 1 std across respondents",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=PLOT_ANNOTATION_FONT_SIZE, style="italic", color="dimgray")
    ax.legend(fontsize=PLOT_LEGEND_FONT_SIZE)
    fig.tight_layout()
    fig.savefig(plots_dir / "survey_novelty_vs_relevance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_survey_results(plots_dir: Path, responses_path: Path = SURVEY_RESPONSES_PATH) -> None:
    """Load, print, and plot the participant survey results, if present."""
    if not responses_path.exists():
        print(f"  {responses_path} not found; skipping survey results.")
        return

    df = load_survey_responses(responses_path)
    summary = summarize_survey_responses(df)
    counts = survey_rank_distribution(df)

    print("  Summary by method:")
    print(summary.round(3).to_string())
    print()
    print("  Rank distribution (rows = method, columns = rank 1..4):")
    print(counts.to_string())
    cross_check_survey_summary(summary)

    save_survey_metrics_dashboard(summary, plots_dir)
    save_survey_rank_distribution(counts, plots_dir)
    save_survey_rating_distribution(df, plots_dir)
    save_survey_relevance_vs_watch_scatter(df, plots_dir)
    save_survey_novelty_vs_relevance_scatter(df, plots_dir)
    print(f"  Saved survey plots to {plots_dir}/")


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight-sensitivity", action="store_true",
                         help="Sweep the top-level scoring weights instead of running the full evaluation")
    parser.add_argument("--rows", type=int, default=WEIGHT_SENSITIVITY_MAX_ROWS,
                         help="[--weight-sensitivity] Max rating rows to load (0 = full file)")
    parser.add_argument("--users", type=int, default=WEIGHT_SENSITIVITY_USER_SAMPLE,
                         help="[--weight-sensitivity] Eval user sample size")
    parser.add_argument("--weights", nargs="*", default=list(BASE_WEIGHTS.keys()),
                         help="[--weight-sensitivity] Which weights to sweep")
    args = parser.parse_args()

    if args.weight_sensitivity:
        run_weight_sensitivity(args)
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
    log_path = OUTPUT_DIR / "evaluate_log.txt"
    log_file = open(log_path, "w")
    original_stdout = sys.stdout
    sys.stdout = _Tee(original_stdout, log_file)
    try:
        _run(log_path)
    finally:
        sys.stdout = original_stdout
        log_file.close()


def _run(log_path: Path) -> None:
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
    best_k, k_tuning_curve = tune_k(model, test)
    print(f"  Selected k: {best_k}")

    # Keep the selected SVD dimensions for the shared recommendations_algorithm function.
    model["movie_factors"] = model["movie_factors"][:, :best_k]

    # ── Metrics are computed silently; only numbers are printed. ──────────────
    rmses, n_eval = compute_rmse(model, test, k=best_k)

    eval_user_ids = select_eval_users(relevant_by_user)
    del test
    gc.collect()

    train_movies = build_train_movies_for_users(train, eval_user_ids)
    train_input_movies = build_train_input_movies_for_users(train, eval_user_ids, allowed_movie_ids)

    dq_map = load_dialogue_features()
    franchise_map = load_franchise_map()
    installment_trend = load_installment_rating_trend()

    alg_rec = algorithm_recommender(
        movies,
        model,
        train_movies,
        train_input_movies,
        dq_map,
        franchise_map,
        installment_trend,
        top_n=K,
    )
    pop_rec = popular_recommender(train, train_movies, top_n=K, allowed_movie_ids=allowed_movie_ids)
    hr_rec = highest_rated_recommender(train, train_movies, top_n=K, allowed_movie_ids=allowed_movie_ids)
    rand_rec = random_recommender(train, train_movies, top_n=K, allowed_movie_ids=allowed_movie_ids)
    del train
    gc.collect()

    methods = [
        ("Our system", alg_rec),
        ("Popular", pop_rec),
        ("Highest-rated", hr_rec),
        ("Random", rand_rec),
    ]

    print(f"  Scoring ranking metrics for {len(eval_user_ids):,} sampled users "
          f"(of {n_with_relevant:,} eligible) x {len(methods)} methods ...")
    t_rank = time.time()
    results: dict[str, tuple[float, float]] = {}
    raw_results: dict[str, tuple[list[float], list[float]]] = {}
    for name, rec_fn in methods:
        p_mean, r_mean, p_list, r_list = eval_ranking(rec_fn, relevant_by_user, eval_user_ids, k=K)
        results[name] = (p_mean, r_mean)
        raw_results[name] = (p_list, r_list)
    print(f"  Ranking evaluation completed in {time.time() - t_rank:.1f}s")

    alg_p, alg_r = results["Our system"]

    fa = None
    corr = pd.DataFrame()
    if DATASET_PATH.exists():
        dataset = pd.read_csv(DATASET_PATH, low_memory=False)
        fa = franchise_analysis(dataset)
        corr = dialogue_correlation(dataset)

    print()
    print("Recommender system performance:")
    print(f"  RMSE (svd): {rmses['svd']:.4f}")
    print(f"  RMSE (bias-only): {rmses['bias_only']:.4f}")
    print(f"  RMSE (item-mean): {rmses['item_mean']:.4f}")
    print(f"  RMSE (user-mean): {rmses['user_mean']:.4f}")
    print(f"  RMSE (global-mean): {rmses['global_mean']:.4f}")
    print(f"  RMSE delta (svd vs. global-mean): {rmses['svd'] - rmses['global_mean']:+.4f}")
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

    save_results_csv(
        OUTPUT_DIR,
        rmses,
        n_eval,
        best_k,
        eval_user_ids,
        allowed_movie_ids,
        metadata_movie_ids,
        model_movie_ids,
        methods,
        results,
        k_tuning_curve,
        fa,
        corr,
    )
    save_plots(PLOTS_DIR, methods, results, raw_results, k_tuning_curve, best_k, rmses)

    section("Participant survey results")
    save_survey_results(PLOTS_DIR)

    print(f"\n  Saved full log to {log_path}")


if __name__ == "__main__":
    main()
