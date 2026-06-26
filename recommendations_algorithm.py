import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

RATINGS_PATH = "dataset_ratings_and_tags.csv"
DATASET_PATH = "dataset.csv"
# Use the project dataset as the movie metadata source so evaluation and the
# interactive recommender consider the same movie universe.
MOVIES_PATH  = DATASET_PATH
SVD_K        = 50
K_VALUES     = [20, 50]  # match evaluate.py; k=50 was best and avoids extra memory

# Dialogue/language is a central project signal.
# Positive weights reward richer vocabulary; negative weights penalize more
# negative or repetitive dialogue. Each feature is normalized before combining.
DIALOGUE_FEATURE_WEIGHTS = {
    "negative_word_ratio": -0.35,
    "type_token_ratio": 0.25,
    "hapax_ratio": 0.15,
    "repeated_short_phrase_ratio": -0.15,
    "bigram_repetition_ratio": -0.10,
}

# Signal weights (must sum to 1.0). Collaborative filtering remains the main
# recommendation signal, while dialogue/language, franchise quality, and an
# old-movie penalty provide quality-aware novelty. Genre is used only as a
# post-filter on the final ranked list, not as a full-candidate scoring term.
W_CF        = 0.75
W_DIALOGUE  = 0.15
W_FRANCHISE = 0.05
W_YEAR_PENALTY = 0.05

# Inside the CF component, balance personalization with general item quality.
# This keeps recommendations relevant while still allowing universally liked
# movies to appear when they are good matches.
W_LATENT_SIM = 0.60
W_ITEM_BIAS  = 0.40

# How to build the general-quality prior that is passed as movie_biases_norm.
# Popularity remains useful, but rating quality receives more weight now that
# the main recommendation score is more personalized.
W_QUALITY_POPULARITY = 0.60
W_QUALITY_RATING     = 0.40



def load_movies_metadata(movies_path=MOVIES_PATH):
    """Load movie metadata from dataset.csv and normalize columns used by recommend()."""
    movies = pd.read_csv(movies_path, low_memory=False)
    if "MovieID" not in movies.columns:
        raise ValueError(f"{movies_path} must contain a MovieID column")

    movies["MovieID"] = pd.to_numeric(movies["MovieID"], errors="coerce")
    movies = movies.dropna(subset=["MovieID"]).copy()
    movies["MovieID"] = movies["MovieID"].astype(int)

    # Keep the interactive title matching working even if dataset.csv uses a
    # different title column name.
    if "Title" not in movies.columns:
        for candidate in ("title", "MovieTitle", "movie_title", "primaryTitle", "originalTitle"):
            if candidate in movies.columns:
                movies["Title"] = movies[candidate].astype(str)
                break
        else:
            movies["Title"] = movies["MovieID"].map(lambda mid: f"Movie {mid}")

    if "Year" not in movies.columns:
        for candidate in ("year", "startYear", "release_year", "ReleaseYear"):
            if candidate in movies.columns:
                movies["Year"] = pd.to_numeric(movies[candidate], errors="coerce")
                break
        else:
            movies["Year"] = np.nan

    # The old movies_clean.csv had Genre1..Genre8. If dataset.csv has a single
    # genre string, split it into those columns; otherwise create empty columns.
    genre_cols = [f"Genre{i}" for i in range(1, 9)]
    if not any(col in movies.columns for col in genre_cols):
        source_col = next((c for c in ("Genres", "genres", "Genre", "genre") if c in movies.columns), None)
        if source_col is not None:
            split_genres = movies[source_col].fillna("").astype(str).str.replace("|", ",", regex=False).str.split(",")
            for i, col in enumerate(genre_cols):
                movies[col] = split_genres.map(lambda xs, j=i: xs[j].strip() if j < len(xs) and xs[j].strip() else np.nan)
        else:
            for col in genre_cols:
                movies[col] = np.nan
    else:
        for col in genre_cols:
            if col not in movies.columns:
                movies[col] = np.nan

    return movies.drop_duplicates(subset=["MovieID"], keep="first").reset_index(drop=True)

def load_dialogue_features(dataset_path=DATASET_PATH):
    """
    Load per-movie dialogue quality scores from precomputed dataset.csv.

    Uses dialogue features that showed measurable correlation with IMDb ratings.
    The composite score is normalized to [0, 1].
    """
    print("Loading dialogue features from dataset...")
    if not os.path.exists(dataset_path):
        print(f"  {dataset_path} not found; dialogue features disabled.")
        return {}

    cols_needed = ["MovieID"] + list(DIALOGUE_FEATURE_WEIGHTS)
    available = pd.read_csv(dataset_path, nrows=0).columns.tolist()
    cols_to_read = [c for c in cols_needed if c in available]
    missing = set(cols_needed) - set(cols_to_read)
    if missing:
        print(f"  Missing dialogue columns in dataset.csv: {missing}")

    df = pd.read_csv(dataset_path, usecols=cols_to_read, low_memory=False)
    df["MovieID"] = pd.to_numeric(df["MovieID"], errors="coerce")
    df = df.dropna(subset=["MovieID"]).copy()
    df["MovieID"] = df["MovieID"].astype(int)

    for col in DIALOGUE_FEATURE_WEIGHTS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    df = df.dropna(subset=list(DIALOGUE_FEATURE_WEIGHTS))
    if df.empty:
        print("  No rows with complete dialogue features; dialogue scores disabled.")
        return {}

    normalized_features = []
    total_abs_weight = 0.0

    for col, weight in DIALOGUE_FEATURE_WEIGHTS.items():
        values = df[col].astype(float)
        mn, mx = float(values.min()), float(values.max())

        if mx > mn:
            normalized = (values - mn) / (mx - mn)
        else:
            normalized = pd.Series(0.5, index=df.index)

        normalized_features.append(weight * normalized)
        total_abs_weight += abs(float(weight))

    score = sum(normalized_features)

    # Convert the weighted sum back to [0, 1]. This keeps the dialogue score
    # comparable to the other recommendation components.
    if total_abs_weight > 0:
        score = (score + total_abs_weight) / (2 * total_abs_weight)
    else:
        score = pd.Series(0.5, index=df.index)

    score = score.clip(0.0, 1.0)
    return dict(zip(df["MovieID"], score.values))


def load_svd_model(ratings_path=RATINGS_PATH, k_values=K_VALUES, tune=True):
    """
    Fit biased SVD on user ratings.  Prediction: μ + b_u + b_i + U·V.

    When tune=True, holds out 10 % of ratings as a validation set and selects k
    from k_values by RMSE.  Returns (movie_ids, movie_factors,
    movie_biases_norm) where movie_biases_norm are item biases normalised to
    [0, 1] for use in the ranking score.
    """
    print("Building biased SVD model from ratings...")

    if not os.path.exists(ratings_path):
        fallback_paths = [
            "datasets/movies-1M/movies_ratings_clean.csv",
            "datasets/movies-1M/movies-ratings-clean.csv",
        ]
        for fallback_path in fallback_paths:
            if os.path.exists(fallback_path):
                print(f"  {ratings_path} not found; using fallback ratings file: {fallback_path}")
                ratings_path = fallback_path
                break
        else:
            raise FileNotFoundError(
                f"No ratings file found. Expected {ratings_path} or one of {fallback_paths}."
            )

    if str(ratings_path).endswith(".csv"):
        available_cols = pd.read_csv(ratings_path, nrows=0).columns.tolist()

        if "InteractionType" in available_cols:
            # Combined dataset_ratings_and_tags.csv. Keep only explicit ratings;
            # tags are not useful for SVD rating reconstruction.
            usecols = ["UserID", "MovieID", "Rating", "InteractionType"]
            if "SourceDataset" in available_cols:
                usecols.append("SourceDataset")

            df = pd.read_csv(ratings_path, usecols=usecols, low_memory=False)
            df = df[df["InteractionType"] == "rating"].copy()

            # If 1M and 32M are combined, their user IDs can overlap.
            # Offset 32M users so they are not treated as the same people.
            if "SourceDataset" in df.columns:
                mask_32m = df["SourceDataset"].astype(str).eq("MovieLens 32M")
                df.loc[mask_32m, "UserID"] = pd.to_numeric(
                    df.loc[mask_32m, "UserID"],
                    errors="coerce",
                ) + 300_000
                df = df.drop(columns=["SourceDataset"])

            df = df.drop(columns=["InteractionType"])
        else:
            # Cleaned MovieLens ratings file.
            df = pd.read_csv(ratings_path, usecols=["UserID", "MovieID", "Rating"], low_memory=False)
    else:
        ratings = []
        with open(ratings_path) as f:
            for line in f:
                uid, mid, rating, _ = line.strip().split("::")
                ratings.append((int(uid), int(mid), float(rating)))
        df = pd.DataFrame(ratings, columns=["UserID", "MovieID", "Rating"])

    df["UserID"] = pd.to_numeric(df["UserID"], errors="coerce")
    df["MovieID"] = pd.to_numeric(df["MovieID"], errors="coerce")
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df = df.dropna(subset=["UserID", "MovieID", "Rating"]).copy()

    df["UserID"] = df["UserID"].astype(int)
    df["MovieID"] = df["MovieID"].astype(int)
    df["Rating"] = df["Rating"].astype(float)

    if tune and len(k_values) > 1:
        rng   = np.random.default_rng(42)
        mask  = rng.random(len(df)) < 0.10
        val   = df[mask].copy()
        train = df[~mask].copy()
    else:
        train = df
        val   = None

    global_mean  = float(train['Rating'].mean())
    user_means   = train.groupby('UserID')['Rating'].mean()
    movie_means  = train.groupby('MovieID')['Rating'].mean()

    movie_ids = sorted(train['MovieID'].unique())
    user_ids  = sorted(train['UserID'].unique())
    user_idx  = {u: i for i, u in enumerate(user_ids)}
    movie_idx = {m: i for i, m in enumerate(movie_ids)}

    user_biases  = np.array(
        [float(user_means.get(u,  global_mean)) - global_mean for u in user_ids],
        dtype=np.float32,
    )
    movie_biases = np.array(
        [float(movie_means.get(m, global_mean)) - global_mean for m in movie_ids],
        dtype=np.float32,
    )

    b_u       = train['UserID'].map(user_means).fillna(global_mean) - global_mean
    b_i       = train['MovieID'].map(movie_means).fillna(global_mean) - global_mean
    residuals = (train['Rating'] - global_mean - b_u - b_i).values.astype(np.float32)

    rows = [user_idx[u] for u in train['UserID']]
    cols = [movie_idx[m] for m in train['MovieID']]
    R    = csr_matrix((residuals, (rows, cols)),
                      shape=(len(user_ids), len(movie_ids)), dtype=np.float32)

    k_max = min(max(k_values), min(R.shape) - 1)
    U, sigma, Vt = svds(R, k=k_max)
    order = np.argsort(-sigma)
    U, sigma, Vt = U[:, order], sigma[order], Vt[order]

    UF = U * sigma  # (n_users,  k_max)
    MF = Vt.T       # (n_movies, k_max)

    if val is not None and len(val) >= 10:
        print(f"  {'k':<8} {'RMSE':>8}")
        best_k, best_rmse = k_values[0], float('inf')
        for k in k_values:
            k = min(k, k_max)
            u_idx_series = val['UserID'].map(user_idx)
            m_idx_series = val['MovieID'].map(movie_idx)
            valid = u_idx_series.notna() & m_idx_series.notna()
            if valid.sum() < 10:
                continue
            ui  = u_idx_series[valid].astype(int).values
            mi  = m_idx_series[valid].astype(int).values
            act = val['Rating'][valid].values
            pred = np.clip(
                global_mean
                + user_biases[ui]
                + movie_biases[mi]
                + np.einsum('ij,ij->i', UF[:, :k][ui], MF[:, :k][mi]),
                0.5, 5.0,
            )
            rmse   = float(np.sqrt(np.mean((pred - act) ** 2)))
            marker = ''
            if rmse < best_rmse:
                best_rmse = rmse
                best_k    = k
                marker    = '  ←'
            print(f"  {k:<8} {rmse:>8.4f}{marker}")
        print(f"  Best k = {best_k}")
        UF = UF[:, :best_k]
        MF = MF[:, :best_k]
    else:
        k = min(SVD_K, k_max)
        UF = UF[:, :k]
        MF = MF[:, :k]

    # General-quality prior used for ranking.  It combines popularity and
    # rating quality so the algorithm can compete with the strong popularity
    # and highest-rated baselines while still adding personalization.
    movie_counts = train.groupby('MovieID').size()
    count_values = np.array([float(movie_counts.get(m, 0.0)) for m in movie_ids], dtype=np.float32)
    pop = np.log1p(count_values)
    pop_min, pop_max = float(pop.min()), float(pop.max())
    popularity_norm = (
        (pop - pop_min) / (pop_max - pop_min) if pop_max > pop_min
        else np.full_like(pop, 0.5, dtype=np.float32)
    )

    mb_min, mb_max = float(movie_biases.min()), float(movie_biases.max())
    rating_norm = (
        (movie_biases - mb_min) / (mb_max - mb_min) if mb_max > mb_min
        else np.full_like(movie_biases, 0.5, dtype=np.float32)
    )

    movie_biases_norm = (
        W_QUALITY_POPULARITY * popularity_norm
        + W_QUALITY_RATING * rating_norm
    ).astype(np.float32)

    return movie_ids, MF, movie_biases_norm


def load_franchise_map(dataset_path=DATASET_PATH):
    """
    Return franchise metadata by MovieID. The score later uses franchise name,
    installment number, and IMDb rating when available. Non-franchise movies
    are absent from the dict.
    """
    if not os.path.exists(dataset_path):
        print(f"  {dataset_path} not found; franchise awareness disabled.")
        return {}

    available = pd.read_csv(dataset_path, nrows=0).columns.tolist()
    cols = [c for c in [
        "MovieID", "FranchiseName", "FranchiseInstallment", "imdb_averageRating"
    ] if c in available]

    if "MovieID" not in cols or "FranchiseName" not in cols:
        print("  Franchise columns not found; franchise awareness disabled.")
        return {}

    df = pd.read_csv(dataset_path, usecols=cols, low_memory=False)
    df["MovieID"] = pd.to_numeric(df["MovieID"], errors="coerce")
    df = df.dropna(subset=["MovieID", "FranchiseName"]).copy()
    df["MovieID"] = df["MovieID"].astype(int)

    if "FranchiseInstallment" in df.columns:
        df["FranchiseInstallment"] = pd.to_numeric(df["FranchiseInstallment"], errors="coerce")
    else:
        df["FranchiseInstallment"] = np.nan

    if "imdb_averageRating" in df.columns:
        df["imdb_averageRating"] = pd.to_numeric(df["imdb_averageRating"], errors="coerce")
    else:
        df["imdb_averageRating"] = np.nan

    return {
        int(row.MovieID): {
            "name": row.FranchiseName,
            "installment": row.FranchiseInstallment,
            "imdb": row.imdb_averageRating,
        }
        for row in df.itertuples(index=False)
    }


def cosine_sim(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def build_old_movie_penalties(movies):
    """
    Return an old-movie penalty by MovieID.

    This corrects IMDb's possible old-movie bias by pushing older movies
    downward only. New and recent movies do not receive an extra bonus.
    Missing years are treated neutrally with no penalty.
    """
    if "Year" not in movies.columns:
        return {}

    df = movies[["MovieID", "Year"]].copy()
    df["MovieID"] = pd.to_numeric(df["MovieID"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["MovieID"]).copy()

    valid_years = df["Year"].dropna()
    if valid_years.empty:
        return {int(mid): 0.0 for mid in df["MovieID"]}

    old_cutoff = float(valid_years.quantile(0.10))
    recent_cutoff = float(valid_years.quantile(0.50))

    if recent_cutoff <= old_cutoff:
        return {int(mid): 0.0 for mid in df["MovieID"]}

    # penalty = 1 for very old movies, fades to 0 by the median year.
    # Movies newer than the median year get 0, not a bonus.
    df["old_movie_penalty"] = ((recent_cutoff - df["Year"]) / (recent_cutoff - old_cutoff)).clip(0.0, 1.0)
    df["old_movie_penalty"] = df["old_movie_penalty"].fillna(0.0)

    return dict(zip(df["MovieID"].astype(int), df["old_movie_penalty"].astype(float)))


def get_movie_genres(row):
    """Return normalized genre set for a metadata row."""
    genres = set()
    for i in range(1, 9):
        col = f"Genre{i}"
        if col in row.index and pd.notna(row.get(col)):
            value = str(row.get(col)).strip()
            if value:
                genres.add(value.lower())
    return genres


def is_nonstandard_movie_candidate(row):
    """
    Identify obvious non-standard movie candidates.

    These items can score well numerically because of recency or dialogue, but
    they are usually poor survey recommendations for narrative movie inputs.
    """
    title = str(row.get("Title", "")).lower()

    blocked_title_terms = [
        "stand-up",
        "stand up",
        "live at",
        "live in",
        "live from",
        "comedy special",
        "concert",
    ]
    if any(term in title for term in blocked_title_terms):
        return True

    # If IMDb titleType exists in dataset.csv, prefer standard theatrical/movie entries.
    title_type_col = next((c for c in ("titleType", "TitleType", "imdb_titleType") if c in row.index), None)
    if title_type_col is not None and pd.notna(row.get(title_type_col)):
        title_type = str(row.get(title_type_col)).strip().lower()
        if title_type != "movie":
            return True

    genres = get_movie_genres(row)
    blocked_genres = {
        "documentary",
        "short",
        "talk-show",
        "reality-tv",
        "news",
        "game-show",
    }
    return bool(genres & blocked_genres)


def passes_final_genre_filter(candidate_id, input_ids, movie_rows):
    """
    Lightweight post-filter used only after ranking.

    The model first ranks candidates normally. Then this function removes
    candidates that look unsuitable for the user's selected movies. This is much
    faster than calculating genre similarity for every candidate.
    """
    if candidate_id not in movie_rows.index:
        return False

    candidate_row = movie_rows.loc[candidate_id]
    if is_nonstandard_movie_candidate(candidate_row):
        return False

    input_genres = set()
    for input_id in input_ids:
        if input_id in movie_rows.index:
            input_genres.update(get_movie_genres(movie_rows.loc[input_id]))

    candidate_genres = get_movie_genres(candidate_row)

    # If genre metadata is missing, do not over-filter.
    if not input_genres or not candidate_genres:
        return True

    # Reject only clear genre mismatches. This avoids stand-up/comedy-only picks
    # for drama/adventure/sci-fi inputs, while keeping the main ranking unchanged.
    return bool(input_genres & candidate_genres)

def resolve_input_ids(input_titles, movies, mid_to_idx):
    """Resolve typed movie titles to MovieIDs with exact match first, contains second."""
    title_lower = movies["Title"].astype(str).str.lower()
    input_ids = []

    for title in input_titles:
        query = title.strip().lower()
        if not query:
            continue

        match = movies[title_lower == query]
        if match.empty:
            match = movies[title_lower.str.contains(query, regex=False, na=False)]
        if match.empty:
            print(f" Could not find movie: '{title}'")
            continue

        row = match.iloc[0]
        mid = int(row.MovieID)
        if mid not in mid_to_idx:
            print(f"  '{row.Title}' has no rating data, skipping.")
            continue

        input_ids.append(mid)
        year = int(row.Year) if pd.notna(row.Year) else "unknown year"
        print(f"   Found: {row.Title} ({year})")

    return input_ids


def franchise_signal(candidate_id, input_ids, franchise_map):
    """
    Conservative franchise signal:
      - unrelated movies are neutral (0.5),
      - same-franchise movies receive a bonus only if quality appears maintained,
      - later/weaker installments are penalized.
    """
    candidate = franchise_map.get(candidate_id)
    if not candidate:
        return 0.5

    same_franchise_inputs = [
        franchise_map[input_id]
        for input_id in input_ids
        if input_id in franchise_map
        and franchise_map[input_id].get("name") == candidate.get("name")
    ]
    if not same_franchise_inputs:
        return 0.5

    input_imdb = [x.get("imdb") for x in same_franchise_inputs if pd.notna(x.get("imdb"))]
    candidate_imdb = candidate.get("imdb")
    candidate_inst = candidate.get("installment")
    input_inst = [x.get("installment") for x in same_franchise_inputs if pd.notna(x.get("installment"))]

    score = 0.75  # same franchise, but not automatically a perfect match

    if input_imdb and pd.notna(candidate_imdb):
        imdb_delta = float(candidate_imdb) - float(np.mean(input_imdb))
        if imdb_delta >= -0.20:
            score = 1.0
        elif imdb_delta <= -0.75:
            score = 0.20
        else:
            score = 0.50

    # Later installments tend to score lower, so be cautious with them.
    if input_inst and pd.notna(candidate_inst) and float(candidate_inst) > max(map(float, input_inst)):
        score = max(0.0, score - 0.20)

    return float(np.clip(score, 0.0, 1.0))


def _recommend_scored_candidates(
    input_ids,
    movies,
    movie_ids,
    movie_factors,
    movie_biases_norm,
    dq_map,
    franchise_map,
    n=3,
    exclude_ids=None,
):
    """Shared implementation used by both the interactive recommender and evaluation."""
    mid_to_idx = {int(mid): i for i, mid in enumerate(movie_ids)}
    movie_rows = movies.set_index("MovieID", drop=False)

    clean_input_ids = [int(mid) for mid in input_ids if int(mid) in mid_to_idx]
    if len(clean_input_ids) < 1:
        return []

    excluded = set(int(mid) for mid in (exclude_ids or ()))
    excluded.update(clean_input_ids)

    user_profile = np.mean([movie_factors[mid_to_idx[mid]] for mid in clean_input_ids], axis=0)
    user_dq = float(np.mean([dq_map.get(mid, 0.5) for mid in clean_input_ids]))
    old_movie_penalties = build_old_movie_penalties(movies)

    # Vectorized latent similarity for all movies. This is faster than calling
    # cosine_sim for every candidate inside the loop.
    profile_norm = np.linalg.norm(user_profile)
    movie_norms = np.linalg.norm(movie_factors, axis=1)
    denom = movie_norms * profile_norm
    latent_scores = np.zeros(len(movie_ids), dtype=np.float32)
    valid_norm = denom > 0
    latent_scores[valid_norm] = (movie_factors[valid_norm] @ user_profile) / denom[valid_norm]
    latent_scores = np.clip(latent_scores, 0.0, 1.0)

    candidates = []
    for i, mid in enumerate(movie_ids):
        mid = int(mid)
        if mid in excluded or mid not in movie_rows.index:
            continue

        # Collaborative filtering: combine latent similarity with a stronger
        # movie-quality prior from item bias.
        cf_score = W_LATENT_SIM * float(latent_scores[i]) + W_ITEM_BIAS * float(movie_biases_norm[i])

        # Dialogue/language quality is a meaningful side signal.
        movie_dq = float(dq_map.get(mid, 0.5))
        dq_score = 1.0 - abs(movie_dq - user_dq)

        # Franchise awareness uses IMDb/installment metadata when available.
        franchise_score = franchise_signal(mid, clean_input_ids, franchise_map)

        # Old-movie correction: old movies are pushed downward, while newer movies
        # receive no extra bonus.
        old_movie_penalty = float(old_movie_penalties.get(mid, 0.0))

        score = (
            W_CF * cf_score
            + W_DIALOGUE * dq_score
            + W_FRANCHISE * franchise_score
            - W_YEAR_PENALTY * old_movie_penalty
        )
        candidates.append((mid, score, cf_score, dq_score, franchise_score, old_movie_penalty))

    candidates.sort(key=lambda x: x[1], reverse=True)

    # Post-filter only the ranked list. If a top candidate is unsuitable, replace
    # it with the next lower-ranked suitable candidate.
    filtered = []
    for candidate in candidates:
        mid = int(candidate[0])
        if passes_final_genre_filter(mid, clean_input_ids, movie_rows):
            filtered.append(candidate)
            if len(filtered) >= n:
                break

    return filtered


def recommend_from_movie_ids(
    input_ids,
    movies,
    movie_ids,
    movie_factors,
    movie_biases_norm,
    dq_map,
    franchise_map,
    n=3,
    exclude_ids=None,
):
    """Return recommended MovieIDs using the same scoring logic as recommend().

    This function is for evaluation, where users are represented by MovieIDs
    from their training ratings rather than by typed movie titles.
    """
    candidates = _recommend_scored_candidates(
        input_ids,
        movies,
        movie_ids,
        movie_factors,
        movie_biases_norm,
        dq_map,
        franchise_map,
        n=n,
        exclude_ids=exclude_ids,
    )
    return [mid for mid, *_ in candidates]


def recommend(
    input_titles,
    movies,
    movie_ids,
    movie_factors,
    movie_biases_norm,
    dq_map,
    franchise_map,
    n=3,
):
    mid_to_idx = {int(mid): i for i, mid in enumerate(movie_ids)}
    movie_rows = movies.set_index("MovieID", drop=False)

    input_ids = resolve_input_ids(input_titles, movies, mid_to_idx)
    if len(input_ids) < 1:
        print("No valid input movies found.")
        return []

    candidates = _recommend_scored_candidates(
        input_ids,
        movies,
        movie_ids,
        movie_factors,
        movie_biases_norm,
        dq_map,
        franchise_map,
        n=n,
    )

    results = []
    for mid, score, cf, dq, fs, yp in candidates:
        row = movie_rows.loc[mid]
        genres = [row[f"Genre{i}"] for i in range(1, 9) if pd.notna(row.get(f"Genre{i}"))]
        franchise_note = ""
        if fs >= 0.80:
            franchise_note = " [franchise quality maintained]"
        elif fs <= 0.30:
            franchise_note = " [franchise quality declined]"
        results.append({
            "title":     row["Title"],
            "year":      int(row["Year"]) if pd.notna(row["Year"]) else None,
            "genres":    ", ".join(genres),
            "score":     round(score, 3),
            "cf":        round(cf, 3),
            "dq":        round(dq, 3),
            "franchise": round(fs, 3),
            "old_movie_penalty": round(yp, 3),
            "note":      franchise_note,
        })
    return results

def main():
    movies                             = load_movies_metadata(MOVIES_PATH)
    dq_map                             = load_dialogue_features()
    movie_ids, movie_factors, mb_norm  = load_svd_model()
    franchise_map                      = load_franchise_map()

    while True:
        print("\nEnter 3 movies you like (or 'quit' to exit):")
        titles = []
        for i in range(1, 4):
            t = input(f"  Movie {i}: ").strip()
            if t.lower() == 'quit':
                return
            titles.append(t)

        results = recommend(
            titles, movies, movie_ids, movie_factors, mb_norm,
            dq_map, franchise_map,
        )

        if results:
            print("\n── Recommendations ──")
            for rank, r in enumerate(results, 1):
                print(f"\n  {rank}. {r['title']} ({r['year']}){r['note']}")
                print(f"     Genres:  {r['genres']}")
                print(f"     Score:   {r['score']}  "
                      f"(CF: {r['cf']} | Dialogue: {r['dq']} | "
                      f"Franchise: {r['franchise']} | "
                      f"Old penalty: {r['old_movie_penalty']})")

        again = input("\nTry again? (y/n): ").strip().lower()
        if again != 'y':
            break


if __name__ == '__main__':
    main()
