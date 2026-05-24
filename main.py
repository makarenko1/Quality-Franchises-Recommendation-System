from pathlib import Path
import re
import importlib.util
from collections import Counter

import pandas as pd


DATASETS_DIR = Path("datasets")

MOVIES_DIR = DATASETS_DIR / "movies-1M"
MOVIES_32M_DIR = DATASETS_DIR / "movies-32M"
OPENSUBTITLES_DIR = DATASETS_DIR / "opensubtitles"
IMDB_DIR = DATASETS_DIR / "imdb"

MOVIES_CSV = MOVIES_DIR / "movies_clean.csv"
MOVIES_32M_PREPROCESS_SCRIPT = MOVIES_32M_DIR / "movies_32m_preprocess.py"
MOVIES_32M_CSV = MOVIES_32M_DIR / "movies_clean.csv"
FRANCHISES_CSV = DATASETS_DIR / "franchises" / "franchises.csv"
SUBS_DIR = OPENSUBTITLES_DIR / "subs"

IMDB_RAW_DIR = IMDB_DIR / "raw"
IMDB_PREPROCESS_SCRIPT = IMDB_DIR / "imdb_preprocess.py"
IMDB_MOVIES_CSV = IMDB_DIR / "imdb_movies_clean.csv"
IMDB_RATINGS_CSV = IMDB_DIR / "imdb_ratings_clean.csv"
IMDB_LOG = IMDB_DIR / "imdb_cleaning_log.txt"

MAIN_DATASET = Path("dataset.csv")
ANALYSIS_SCRIPT = Path("analyze_data.py")

LANGUAGE_FEATURE_VERSION = 2
LANGUAGE_FEATURE_VERSION_COL = "language_feature_version"

SUBTITLE_FEATURE_COLUMNS = {
    # Basic dialogue size
    "num_lines",
    "num_tokens",
    "num_unique_tokens",
    "avg_line_length",
    "median_line_length",

    # Vocabulary richness
    "type_token_ratio",
    "hapax_ratio",
    "average_word_length",
    "long_word_ratio",
    "common_word_ratio",
    "rare_word_ratio",
    "simple_word_ratio",
    "complex_word_ratio",

    # Repetition / formulaic dialogue
    "top_word_frequency_ratio",
    "bigram_repetition_ratio",
    "trigram_repetition_ratio",
    "repeated_line_ratio",
    "duplicate_line_count",
    "most_common_line_frequency",
    "repeated_short_phrase_ratio",

    # Sentiment / emotion proxies
    "average_sentiment",
    "sentiment_variance",
    "positive_word_ratio",
    "negative_word_ratio",
    "anger_word_ratio",
    "fear_word_ratio",
    "joy_word_ratio",
    "sadness_word_ratio",

    # Conversational style
    "question_line_ratio",
    "exclamation_line_ratio",
    "first_person_pronoun_ratio",
    "second_person_pronoun_ratio",
    "contraction_ratio",

    # Readability
    "flesch_reading_ease",
    "average_sentence_length",
}

NORMALIZED_LANGUAGE_FEATURE_COLUMNS = {
    "subtitle_words_per_minute",
    "unique_subtitle_words_per_minute",
    "num_lines_per_minute",
}

LANGUAGE_FEATURE_COLUMNS = SUBTITLE_FEATURE_COLUMNS | NORMALIZED_LANGUAGE_FEATURE_COLUMNS | {LANGUAGE_FEATURE_VERSION_COL}


def ensure_preprocessed_datasets():
    """
    Create or update the main dataset.csv.

    Processing order:
        1. Ensure IMDb preprocessing exists.
        2. Ensure MovieLens 32M movie preprocessing exists.
        3. If dataset.csv does not exist, create it from all available
           MovieLens movies: 1M plus new 32M MovieIDs.
        4. Add OpenSubtitles dialogue features when available, without dropping
           movies that have no subtitle file.
        5. Add franchise metadata to all movies.
        6. Add IMDb movie metadata and ratings.
        7. Save everything back to dataset.csv.

    Franchise analysis is called separately after this function.
    """
    ensure_imdb_preprocessed()
    ensure_movies_32m_preprocessed()

    if not MAIN_DATASET.exists():
        print("Creating main dataset from MovieLens 1M and 32M movies...")

        dataset = create_all_movies_dataset(
            movies_1m_csv=MOVIES_CSV,
            movies_32m_csv=MOVIES_32M_CSV,
            output_path=MAIN_DATASET,
        )
    else:
        print(f"Loading existing main dataset from {MAIN_DATASET}")
        dataset = pd.read_csv(MAIN_DATASET, low_memory=False)

        dataset = remove_duplicate_movie_ids(dataset)

        print("Adding new MovieLens 32M movies to main dataset...")
        dataset = merge_movies_32m_into_dataset(dataset)

        dataset.to_csv(MAIN_DATASET, index=False)
        print(f"Saved MovieLens 32M additions to {MAIN_DATASET}")

    if False:
        print(
            "All current language feature columns already exist for movies "
            "with subtitles, skipping OpenSubtitles preprocessing."
        )
    else:
        missing_language_features = LANGUAGE_FEATURE_COLUMNS - set(dataset.columns)

        if missing_language_features:
            print(
                "Preprocessing OpenSubtitles because these language features are missing: "
                f"{sorted(missing_language_features)}"
            )
        else:
            print(
                "Preprocessing OpenSubtitles because existing subtitle feature rows "
                "were created with an old or missing language_feature_version."
            )

        subtitles = preprocess_opensubtitles(dataset_path=SUBS_DIR)

        print("Merging subtitle features into main dataset...")
        dataset = merge_movies_with_subtitle_features(
            movies=dataset,
            subtitles=subtitles,
        )

    print("Adding franchise metadata...")
    dataset = add_franchise_columns(dataset)

    print("Adding IMDb metadata and ratings...")
    dataset = add_imdb_columns(dataset)

    dataset.to_csv(MAIN_DATASET, index=False)
    print(f"Saved final dataset to {MAIN_DATASET}")

    return dataset


def remove_duplicate_movie_ids(
    dataset,
    output_path=MAIN_DATASET,
    keep="first",
):
    """
    Remove duplicate MovieID rows from the main dataset.

    This keeps one row per MovieID and deletes the duplicate copies. By default,
    the first occurrence is preserved because existing rows may already contain
    subtitle, franchise, or IMDb features.

    Parameters
    ----------
    dataset : pandas.DataFrame
        Main dataset.
    output_path : str or pathlib.Path
        Path where the deduplicated dataset should be saved.
    keep : {"first", "last"}
        Which duplicate row to preserve.

    Returns
    -------
    pandas.DataFrame
        Dataset with unique MovieID values.
    """
    if "MovieID" not in dataset.columns:
        raise ValueError("Column 'MovieID' not found in main dataset")

    dataset = dataset.copy()

    before = len(dataset)

    dataset["MovieID"] = pd.to_numeric(dataset["MovieID"], errors="coerce")
    dataset = dataset.dropna(subset=["MovieID"]).copy()
    dataset["MovieID"] = dataset["MovieID"].astype(int)

    invalid_movie_ids_removed = before - len(dataset)

    duplicate_mask = dataset.duplicated(subset="MovieID", keep=keep)
    duplicate_count = int(duplicate_mask.sum())

    if duplicate_count == 0 and invalid_movie_ids_removed == 0:
        print("No duplicate MovieID rows found.")
        return dataset

    dataset = dataset.drop_duplicates(subset="MovieID", keep=keep).copy()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)

    if invalid_movie_ids_removed:
        print(f"Removed {invalid_movie_ids_removed} rows with invalid MovieID")

    print(
        f"Removed {duplicate_count} duplicate MovieID rows; "
        f"kept {len(dataset):,} unique movies."
    )
    print(f"Saved deduplicated dataset to {output_path}")

    return dataset


def remove_duplicate_movie_ids_from_file(
    input_path=MAIN_DATASET,
    output_path=MAIN_DATASET,
    keep="first",
):
    """
    Remove duplicate MovieID rows directly from dataset.csv.

    This is useful if dataset.csv already exists and you want to clean it
    without rerunning the full preprocessing pipeline.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found")

    dataset = pd.read_csv(input_path)

    return remove_duplicate_movie_ids(
        dataset=dataset,
        output_path=output_path,
        keep=keep,
    )


def ensure_imdb_preprocessed():
    """
    Run IMDb preprocessing if the processed IMDb files do not already exist.

    Expected input files:
        datasets/imdb/raw/title.basics.tsv.gz
        datasets/imdb/raw/title.ratings.tsv.gz

    Also supports uncompressed:
        datasets/imdb/raw/title.basics.tsv
        datasets/imdb/raw/title.ratings.tsv

    Expected preprocessing script:
        datasets/imdb/imdb_preprocess.py

    Outputs:
        datasets/imdb/imdb_movies_clean.csv
        datasets/imdb/imdb_ratings_clean.csv
        datasets/imdb/imdb_cleaning_log.txt
    """
    if IMDB_MOVIES_CSV.exists() and IMDB_RATINGS_CSV.exists():
        print("Found existing IMDb processed files, skipping IMDb preprocessing.")
        return

    print("Preprocessing IMDb...")

    if not IMDB_PREPROCESS_SCRIPT.exists():
        raise FileNotFoundError(f"{IMDB_PREPROCESS_SCRIPT} not found")

    basics_path = IMDB_RAW_DIR / "title.basics.tsv.gz"
    ratings_path = IMDB_RAW_DIR / "title.ratings.tsv.gz"

    if not basics_path.exists():
        basics_path = IMDB_RAW_DIR / "title.basics.tsv"

    if not ratings_path.exists():
        ratings_path = IMDB_RAW_DIR / "title.ratings.tsv"

    if not basics_path.exists():
        raise FileNotFoundError(f"{basics_path} not found")

    if not ratings_path.exists():
        raise FileNotFoundError(f"{ratings_path} not found")

    spec = importlib.util.spec_from_file_location(
        "imdb_preprocess",
        IMDB_PREPROCESS_SCRIPT,
    )

    imdb_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imdb_module)

    movies, ratings, log_lines = imdb_module.preprocess_imdb(
        basics_path,
        ratings_path,
    )

    IMDB_DIR.mkdir(parents=True, exist_ok=True)

    movies.to_csv(IMDB_MOVIES_CSV, index=False)
    ratings.to_csv(IMDB_RATINGS_CSV, index=False)
    IMDB_LOG.write_text("\n".join(log_lines), encoding="utf-8")

    print(f"Saved IMDb movies-1M to {IMDB_MOVIES_CSV}")
    print(f"Saved IMDb ratings to {IMDB_RATINGS_CSV}")


def ensure_movies_32m_preprocessed():
    """
    Run MovieLens 32M movie preprocessing if movies_clean.csv does not exist.

    Expected input:
        datasets/movies-32M/raw/movies.csv

    Expected preprocessing script:
        datasets/movies-32M/movies_32m_preprocess.py

    Output:
        datasets/movies-32M/movies_clean.csv
    """
    if MOVIES_32M_CSV.exists():
        print("Found existing MovieLens 32M processed movies, skipping 32M preprocessing.")
        return

    print("Preprocessing MovieLens 32M movies...")

    if not MOVIES_32M_PREPROCESS_SCRIPT.exists():
        raise FileNotFoundError(f"{MOVIES_32M_PREPROCESS_SCRIPT} not found")

    raw_movies_path = MOVIES_32M_DIR / "raw" / "movies.csv"

    if not raw_movies_path.exists():
        raise FileNotFoundError(f"{raw_movies_path} not found")

    spec = importlib.util.spec_from_file_location(
        "movies_32m_preprocess",
        MOVIES_32M_PREPROCESS_SCRIPT,
    )

    movies_32m_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(movies_32m_module)

    if not hasattr(movies_32m_module, "preprocess_movies_32m"):
        raise AttributeError(
            f"{MOVIES_32M_PREPROCESS_SCRIPT} must define preprocess_movies_32m()"
        )

    movies_32m_module.preprocess_movies_32m(
        raw_dir=MOVIES_32M_DIR / "raw",
        out_dir=MOVIES_32M_DIR,
        require_ratings=False,
        require_tags=False,
        drop_movies_without_ratings=False,
    )

    if not MOVIES_32M_CSV.exists():
        raise FileNotFoundError(
            f"MovieLens 32M preprocessing finished but {MOVIES_32M_CSV} was not created"
        )

    print(f"Saved MovieLens 32M movies to {MOVIES_32M_CSV}")


def create_all_movies_dataset(
    movies_1m_csv=MOVIES_CSV,
    movies_32m_csv=MOVIES_32M_CSV,
    output_path=MAIN_DATASET,
):
    """
    Create dataset.csv from all available MovieLens movies.

    The base dataset starts with cleaned MovieLens 1M movies, then appends only
    MovieLens 32M rows whose MovieID is not already present. Existing 1M rows are
    preserved even if 32M has different metadata for the same MovieID.

    This makes franchise and IMDb analysis run on all available movies, not only
    movies that already have subtitles.
    """
    movies_1m_csv = Path(movies_1m_csv)
    movies_32m_csv = Path(movies_32m_csv)
    output_path = Path(output_path)

    if not movies_1m_csv.exists():
        raise FileNotFoundError(f"{movies_1m_csv} not found")

    movies_1m = pd.read_csv(movies_1m_csv)

    if "MovieID" not in movies_1m.columns:
        raise ValueError("Column 'MovieID' not found in MovieLens 1M dataset")

    movies_1m = movies_1m.copy()
    movies_1m["MovieID"] = pd.to_numeric(movies_1m["MovieID"], errors="coerce")
    movies_1m = movies_1m.dropna(subset=["MovieID"]).copy()
    movies_1m["MovieID"] = movies_1m["MovieID"].astype(int)
    movies_1m = movies_1m.drop_duplicates(subset="MovieID", keep="first").copy()

    movies_1m["MovieLens1MAvailable"] = True
    movies_1m["MovieLens32MAvailable"] = False

    dataset = merge_movies_32m_into_dataset(
        dataset=movies_1m,
        movies_32m_csv=movies_32m_csv,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)

    print(f"Saved all-movies base dataset to {output_path}")
    print(f"Base dataset rows: {len(dataset):,}")

    return dataset


def create_movies_with_subtitles_dataset(
    movies_csv=MOVIES_CSV,
    subs_dir=SUBS_DIR,
    output_path=MAIN_DATASET,
):
    """
    Create a filtered movies dataset containing only movies with subtitles.

    This helper is kept for optional subtitle-only experiments. The main project
    pipeline now uses create_all_movies_dataset() so franchise analysis can run
    on all available 1M and 32M movies.
    """
    movies_csv = Path(movies_csv)
    subs_dir = Path(subs_dir)
    output_path = Path(output_path)

    if not movies_csv.exists():
        raise FileNotFoundError(f"{movies_csv} not found")

    if not subs_dir.exists():
        raise FileNotFoundError(f"{subs_dir} not found")

    movies = pd.read_csv(movies_csv)

    if "MovieID" not in movies.columns:
        raise ValueError("Column 'MovieID' not found in movies-1M dataset")

    movie_ids_with_subs = set()

    for subtitle_path in subs_dir.glob("*.srt"):
        match = re.match(r"^(\d+)_", subtitle_path.name)

        if match:
            movie_ids_with_subs.add(int(match.group(1)))

    movies_with_subs = movies[movies["MovieID"].isin(movie_ids_with_subs)].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    movies_with_subs.to_csv(output_path, index=False)

    print(f"Found {len(movie_ids_with_subs)} subtitle files")
    print(f"Saved {len(movies_with_subs)} movies-1M with subtitles to {output_path}")

    return movies_with_subs


def merge_movies_32m_into_dataset(
    dataset,
    movies_32m_csv=MOVIES_32M_CSV,
):
    """
    Add MovieLens 32M movies to the main dataset without overwriting existing rows.

    Matching is done by MovieID.

    If a MovieID already exists in the dataset, the 32M row is skipped, even if
    the 32M metadata is different. New 32M-only MovieIDs are appended as new rows.

    Adds/refreshes:
        - MovieLens32MAvailable
    """
    movies_32m_csv = Path(movies_32m_csv)

    if not movies_32m_csv.exists():
        raise FileNotFoundError(f"{movies_32m_csv} not found")

    dataset = remove_duplicate_movie_ids(dataset)
    movies_32m = pd.read_csv(movies_32m_csv, low_memory=False)

    if "MovieID" not in dataset.columns:
        raise ValueError("Column 'MovieID' not found in main dataset")

    if "MovieID" not in movies_32m.columns:
        raise ValueError("Column 'MovieID' not found in MovieLens 32M dataset")

    dataset = dataset.copy()
    movies_32m = movies_32m.copy()

    dataset["MovieID"] = pd.to_numeric(dataset["MovieID"], errors="coerce")
    movies_32m["MovieID"] = pd.to_numeric(movies_32m["MovieID"], errors="coerce")

    dataset = dataset.dropna(subset=["MovieID"]).copy()
    movies_32m = movies_32m.dropna(subset=["MovieID"]).copy()

    dataset["MovieID"] = dataset["MovieID"].astype(int)
    movies_32m["MovieID"] = movies_32m["MovieID"].astype(int)

    movies_32m = movies_32m.drop_duplicates(subset="MovieID", keep="first").copy()

    if "MovieLens32MAvailable" in dataset.columns:
        dataset = dataset.drop(columns=["MovieLens32MAvailable"])

    all_32m_movie_ids = set(movies_32m["MovieID"])
    existing_movie_ids = set(dataset["MovieID"])

    dataset["MovieLens32MAvailable"] = dataset["MovieID"].isin(all_32m_movie_ids)

    new_movies_32m = movies_32m[
        ~movies_32m["MovieID"].isin(existing_movie_ids)
    ].copy()

    skipped_existing = len(movies_32m) - len(new_movies_32m)

    if new_movies_32m.empty:
        print(
            "MovieLens 32M merge: "
            f"skipped {skipped_existing:,} existing MovieIDs; "
            "added 0 new movies."
        )
        return dataset

    if "MovieLens1MAvailable" not in new_movies_32m.columns:
        new_movies_32m["MovieLens1MAvailable"] = False

    new_movies_32m["MovieLens32MAvailable"] = True

    all_columns = list(dataset.columns)

    for col in new_movies_32m.columns:
        if col not in all_columns:
            all_columns.append(col)

    for col in all_columns:
        if col not in dataset.columns:
            dataset[col] = pd.NA

        if col not in new_movies_32m.columns:
            new_movies_32m[col] = pd.NA

    dataset = pd.concat(
        [
            dataset[all_columns],
            new_movies_32m[all_columns],
        ],
        ignore_index=True,
    )

    print(
        "MovieLens 32M merge: "
        f"skipped {skipped_existing:,} existing MovieIDs; "
        f"added {len(new_movies_32m):,} new movies."
    )

    return dataset


def merge_movies_with_subtitle_features(movies, subtitles):
    """
    Merge main dataset rows with subtitle dialogue features.

    Existing subtitle feature columns are removed first, so rerunning the
    pipeline refreshes features instead of creating duplicate columns.
    """
    movies = movies.copy()
    subtitles = subtitles.copy()

    if "file_name" not in subtitles.columns:
        raise ValueError("Column 'file_name' not found in subtitles dataset")

    subtitles["MovieID"] = subtitles["file_name"].apply(extract_movie_id_from_filename)
    subtitles = subtitles.dropna(subset=["MovieID"])
    subtitles["MovieID"] = subtitles["MovieID"].astype(int)

    feature_cols = [
        col for col in subtitles.columns
        if col in SUBTITLE_FEATURE_COLUMNS
        or col == LANGUAGE_FEATURE_VERSION_COL
    ]

    if not feature_cols:
        raise ValueError("No subtitle feature columns were created")

    old_feature_cols = [
        col for col in movies.columns
        if col in SUBTITLE_FEATURE_COLUMNS
        or col in NORMALIZED_LANGUAGE_FEATURE_COLUMNS
        or col == LANGUAGE_FEATURE_VERSION_COL
    ]

    movies = movies.drop(columns=old_feature_cols, errors="ignore")
    subtitles = subtitles[["MovieID"] + feature_cols]

    dataset = movies.merge(
        subtitles,
        on="MovieID",
        how="left",
    )

    return dataset


def add_franchise_columns(
    movies=None,
    main_csv=MAIN_DATASET,
    franchises_csv=FRANCHISES_CSV,
):
    """
    Add franchise metadata to the main dataset.
    """
    if movies is None:
        main_csv = Path(main_csv)

        if not main_csv.exists():
            raise FileNotFoundError(f"{main_csv} not found")

        movies = pd.read_csv(main_csv)

    franchises_csv = Path(franchises_csv)
    movies = movies.copy()

    for col in [
        "FranchiseID",
        "FranchiseName",
        "FranchiseInstallment",
        "FranchiseLength",
    ]:
        if col in movies.columns:
            movies = movies.drop(columns=[col])

    if not franchises_csv.exists():
        print(f"No franchise file found at {franchises_csv}")
        return movies

    franchises = pd.read_csv(franchises_csv)

    required_cols = {
        "franchise",
        "part",
        "title",
        "year",
        "franchise_length",
    }

    missing_cols = required_cols - set(franchises.columns)

    if missing_cols:
        raise ValueError(
            f"{franchises_csv} is missing columns: {missing_cols}\n"
            f"Available columns: {list(franchises.columns)}"
        )

    franchises = franchises.copy()

    franchises["match_title"] = franchises["title"].apply(normalize_match_title)
    franchises["match_year"] = pd.to_numeric(franchises["year"], errors="coerce")

    movies["match_title"] = movies["Title"].apply(normalize_match_title)
    movies["match_year"] = pd.to_numeric(movies["Year"], errors="coerce")

    franchises["FranchiseID"] = franchises["franchise"].astype("category").cat.codes + 1

    franchise_lookup = franchises[
        [
            "match_title",
            "match_year",
            "FranchiseID",
            "franchise",
            "part",
            "franchise_length",
        ]
    ].rename(
        columns={
            "franchise": "FranchiseName",
            "part": "FranchiseInstallment",
            "franchise_length": "FranchiseLength",
        }
    )

    movies = movies.merge(
        franchise_lookup,
        on=["match_title", "match_year"],
        how="left",
    )

    movies = movies.drop(columns=["match_title", "match_year"])

    print("Added franchise metadata")
    print(f"Franchise movies: {movies['FranchiseID'].notna().sum()}")

    return movies


def add_imdb_columns(
    dataset,
    imdb_movies_csv=IMDB_MOVIES_CSV,
    imdb_ratings_csv=IMDB_RATINGS_CSV,
):
    """
    Add IMDb movie metadata and ratings to the main dataset.

    Matching is done by normalized title and release year.

    Adds:
        - imdb_tconst
        - imdb_Title
        - imdb_originalTitle
        - imdb_isAdult
        - imdb_runtimeMinutes
        - imdb_averageRating
        - imdb_numVotes
        - imdb_low_votes
    """
    imdb_movies_csv = Path(imdb_movies_csv)
    imdb_ratings_csv = Path(imdb_ratings_csv)

    if not imdb_movies_csv.exists():
        raise FileNotFoundError(f"{imdb_movies_csv} not found")

    if not imdb_ratings_csv.exists():
        raise FileNotFoundError(f"{imdb_ratings_csv} not found")

    dataset = dataset.copy()

    imdb_cols = [
        col for col in dataset.columns
        if col.startswith("imdb_")
    ]

    dataset = dataset.drop(columns=imdb_cols, errors="ignore")

    imdb_movies = pd.read_csv(imdb_movies_csv)
    imdb_ratings = pd.read_csv(imdb_ratings_csv)

    required_movie_cols = {
        "tconst",
        "Title",
        "Year",
    }

    required_rating_cols = {
        "tconst",
        "averageRating",
        "numVotes",
    }

    missing_movie_cols = required_movie_cols - set(imdb_movies.columns)
    missing_rating_cols = required_rating_cols - set(imdb_ratings.columns)

    if missing_movie_cols:
        raise ValueError(f"IMDb movies-1M missing columns: {missing_movie_cols}")

    if missing_rating_cols:
        raise ValueError(f"IMDb ratings missing columns: {missing_rating_cols}")

    imdb_rating_cols = ["tconst", "averageRating", "numVotes"]

    if "low_votes" in imdb_ratings.columns:
        imdb_rating_cols.append("low_votes")

    imdb = imdb_movies.merge(
        imdb_ratings[imdb_rating_cols],
        on="tconst",
        how="left",
    )

    imdb["match_title"] = imdb["Title"].apply(normalize_match_title)
    imdb["match_year"] = pd.to_numeric(imdb["Year"], errors="coerce")

    dataset["match_title"] = dataset["Title"].apply(normalize_match_title)
    dataset["match_year"] = pd.to_numeric(dataset["Year"], errors="coerce")

    imdb_lookup_cols = [
        "match_title",
        "match_year",
        "tconst",
        "Title",
        "averageRating",
        "numVotes",
    ]

    optional_imdb_cols = [
        "originalTitle_ascii",
        "isAdult",
        "runtimeMinutes",
        "low_votes",
    ]

    for col in optional_imdb_cols:
        if col in imdb.columns:
            imdb_lookup_cols.append(col)

    imdb_lookup = imdb[imdb_lookup_cols].rename(
        columns={
            "tconst": "imdb_tconst",
            "Title": "imdb_Title",
            "originalTitle_ascii": "imdb_originalTitle",
            "isAdult": "imdb_isAdult",
            "runtimeMinutes": "imdb_runtimeMinutes",
            "averageRating": "imdb_averageRating",
            "numVotes": "imdb_numVotes",
            "low_votes": "imdb_low_votes",
        }
    )

    imdb_lookup = imdb_lookup.drop_duplicates(
        subset=["match_title", "match_year"],
        keep="first",
    )

    dataset = dataset.merge(
        imdb_lookup,
        on=["match_title", "match_year"],
        how="left",
    )

    dataset = dataset.drop(columns=["match_title", "match_year"])

    dataset = add_normalized_language_features(dataset)

    print("Added IMDb metadata")
    print(f"IMDb matches: {dataset['imdb_tconst'].notna().sum()}")

    return dataset


def add_normalized_language_features(dataset):
    """
    Add subtitle features normalized by IMDb runtime.

    Raw subtitle counts are affected by movie length. These normalized
    features make dialogue quantity more comparable across films.
    """
    dataset = dataset.copy()

    required_cols = {
        "num_tokens",
        "num_unique_tokens",
        "num_lines",
        "imdb_runtimeMinutes",
    }

    if not required_cols.issubset(set(dataset.columns)):
        return dataset

    runtime = pd.to_numeric(
        dataset["imdb_runtimeMinutes"],
        errors="coerce",
    )

    valid_runtime = runtime > 0

    for col in NORMALIZED_LANGUAGE_FEATURE_COLUMNS:
        dataset[col] = pd.NA

    dataset.loc[valid_runtime, "subtitle_words_per_minute"] = (
        pd.to_numeric(dataset.loc[valid_runtime, "num_tokens"], errors="coerce")
        / runtime.loc[valid_runtime]
    )

    dataset.loc[valid_runtime, "unique_subtitle_words_per_minute"] = (
        pd.to_numeric(dataset.loc[valid_runtime, "num_unique_tokens"], errors="coerce")
        / runtime.loc[valid_runtime]
    )

    dataset.loc[valid_runtime, "num_lines_per_minute"] = (
        pd.to_numeric(dataset.loc[valid_runtime, "num_lines"], errors="coerce")
        / runtime.loc[valid_runtime]
    )

    return dataset


def run_analysis(analysis_script=ANALYSIS_SCRIPT):
    """
    Run analyze_data.py after dataset.csv has been created or updated.

    The analysis script is expected to read dataset.csv and save plots/results.
    """
    analysis_script = Path(analysis_script)

    if not analysis_script.exists():
        raise FileNotFoundError(f"{analysis_script} not found")

    spec = importlib.util.spec_from_file_location(
        "franchises_analysis",
        analysis_script,
    )

    analysis_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis_module)

    if not hasattr(analysis_module, "main"):
        raise AttributeError(
            f"{analysis_script} must define a main() function"
        )

    analysis_module.main()

    print("Finished franchise analysis.")


def has_language_features(dataset):
    """
    Check whether the main dataset already contains the current subtitle features.

    Rows without subtitles are allowed to have missing language features. This is
    expected because the main dataset now contains all 1M and 32M movies, while
    subtitle files may exist only for a subset.

    The version check is applied only to rows that actually have subtitle
    features.
    """
    if not LANGUAGE_FEATURE_COLUMNS.issubset(set(dataset.columns)):
        return False

    subtitle_rows = dataset["num_tokens"].notna()

    if not subtitle_rows.any():
        return False

    version_values = pd.to_numeric(
        dataset.loc[subtitle_rows, LANGUAGE_FEATURE_VERSION_COL],
        errors="coerce",
    )

    return version_values.eq(LANGUAGE_FEATURE_VERSION).all()


# -----------------------------
# Helper functions
# -----------------------------

def preprocess_opensubtitles(
    dataset_path: str | Path = SUBS_DIR,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Preprocess downloaded OpenSubtitles .srt files.

    Returns one row per subtitle/movie file with numeric dialogue features only.
    Does not save full subtitle text.
    """
    dataset_path = Path(dataset_path)
    files = list(dataset_path.rglob("*.srt"))

    if limit is not None:
        files = files[:limit]

    rows = []

    for file_path in files:
        try:
            raw_lines, dialogue_lines = _extract_opensubtitles_line_records(file_path)
            dialogue_text = " ".join(dialogue_lines)

            features = {
                **get_dialogue_length_features(dialogue_text, dialogue_lines),
                **get_dialogue_richness(dialogue_text),
                **get_dialogue_repetition(dialogue_text, dialogue_lines),
                **get_dialogue_style_features(raw_lines, dialogue_text),
                **get_dialogue_sentiment_features(dialogue_text),
                **get_dialogue_readability_features(raw_lines, dialogue_text),
            }

            rows.append({
                "file_name": file_path.name,
                LANGUAGE_FEATURE_VERSION_COL: LANGUAGE_FEATURE_VERSION,
                **features,
            })

        except Exception as e:
            print(f"Failed to process {file_path}: {e}")

    return pd.DataFrame(rows)


def _extract_opensubtitles_line_records(file_path: Path) -> tuple[list[str], list[str]]:
    """
    Extract raw and cleaned dialogue lines from one downloaded .srt subtitle file.

    Raw lines are used for punctuation-based features such as question and
    exclamation ratios. Cleaned lines are used for token-based features.
    """
    raw_lines = []
    cleaned_lines = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line:
                continue

            if line.isdigit():
                continue

            if "-->" in line:
                continue

            cleaned = clean_dialogue_text(line)

            if cleaned:
                raw_lines.append(line)
                cleaned_lines.append(cleaned)

    return raw_lines, cleaned_lines


def _extract_opensubtitles_lines(file_path: Path) -> list[str]:
    """
    Extract cleaned dialogue lines from one downloaded .srt subtitle file.
    """
    lines = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line:
                continue

            if line.isdigit():
                continue

            if "-->" in line:
                continue

            cleaned = clean_dialogue_text(line)

            if cleaned:
                lines.append(cleaned)

    return lines


def clean_dialogue_text(text: str) -> str:
    """
    Clean one subtitle dialogue line.
    """
    if not text:
        return ""

    text = text.lower()

    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\{.*?\}", " ", text)
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"\(.*?\)", " ", text)
    text = re.sub(r"^[a-zA-Z\s]{1,30}:\s*", " ", text)

    text = re.sub(r"[^a-z'\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def get_dialogue_richness(dialogue_text: str) -> dict:
    """
    Measure vocabulary richness.
    """
    tokens = _tokenize(dialogue_text)

    if not tokens:
        return {
            "num_tokens": 0,
            "num_unique_tokens": 0,
            "type_token_ratio": 0,
            "hapax_ratio": 0,
        }

    token_counts = Counter(tokens)
    hapax_count = sum(1 for count in token_counts.values() if count == 1)

    return {
        "num_tokens": len(tokens),
        "num_unique_tokens": len(token_counts),
        "type_token_ratio": len(token_counts) / len(tokens),
        "hapax_ratio": hapax_count / len(token_counts),
    }



def get_dialogue_length_features(
    dialogue_text: str,
    dialogue_lines: list[str] | None = None,
) -> dict:
    """
    Measure subtitle dialogue length.
    """
    tokens = _tokenize(dialogue_text)
    num_lines = len(dialogue_lines) if dialogue_lines else 0

    line_lengths = [
        len(_tokenize(line))
        for line in dialogue_lines
    ] if dialogue_lines else []

    return {
        "num_lines": num_lines,
        "avg_line_length": len(tokens) / num_lines if num_lines else 0,
        "median_line_length": _median(line_lengths),
    }


def get_dialogue_repetition(
    dialogue_text: str,
    dialogue_lines: list[str] | None = None,
) -> dict:
    """
    Measure repeated words, repeated phrases, and repeated lines.
    """
    tokens = _tokenize(dialogue_text)

    if not tokens:
        return {
            "top_word_frequency_ratio": 0,
            "bigram_repetition_ratio": 0,
            "trigram_repetition_ratio": 0,
            "repeated_line_ratio": 0,
            "duplicate_line_count": 0,
            "most_common_line_frequency": 0,
            "repeated_short_phrase_ratio": 0,
        }

    token_counts = Counter(tokens)

    bigrams = list(zip(tokens, tokens[1:]))
    trigrams = list(zip(tokens, tokens[1:], tokens[2:]))

    bigram_counts = Counter(bigrams)
    trigram_counts = Counter(trigrams)

    repeated_bigrams = sum(
        count for count in bigram_counts.values()
        if count > 1
    )
    repeated_trigrams = sum(
        count for count in trigram_counts.values()
        if count > 1
    )

    repeated_short_phrases = repeated_bigrams + repeated_trigrams
    total_short_phrases = len(bigrams) + len(trigrams)

    if dialogue_lines:
        line_counts = Counter(dialogue_lines)
        repeated_lines = sum(
            count for count in line_counts.values()
            if count > 1
        )
        duplicate_line_count = sum(
            count - 1 for count in line_counts.values()
            if count > 1
        )
        most_common_line_frequency = line_counts.most_common(1)[0][1]
        repeated_line_ratio = repeated_lines / len(dialogue_lines)
    else:
        duplicate_line_count = 0
        most_common_line_frequency = 0
        repeated_line_ratio = 0

    return {
        "top_word_frequency_ratio": token_counts.most_common(1)[0][1] / len(tokens),
        "bigram_repetition_ratio": repeated_bigrams / len(bigrams) if bigrams else 0,
        "trigram_repetition_ratio": repeated_trigrams / len(trigrams) if trigrams else 0,
        "repeated_line_ratio": repeated_line_ratio,
        "duplicate_line_count": duplicate_line_count,
        "most_common_line_frequency": most_common_line_frequency,
        "repeated_short_phrase_ratio": (
            repeated_short_phrases / total_short_phrases
            if total_short_phrases
            else 0
        ),
    }


COMMON_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their",
    "what", "so", "up", "out", "if", "about", "who", "get", "which", "go",
    "me", "when", "make", "can", "like", "time", "no", "just", "him",
    "know", "take", "people", "into", "year", "your", "good", "some",
    "could", "them", "see", "other", "than", "then", "now", "look",
    "only", "come", "its", "over", "think", "also", "back", "after",
    "use", "two", "how", "our", "work", "first", "well", "way", "even",
    "new", "want", "because", "any", "these", "give", "day", "most", "us",
}

POSITIVE_WORDS = {
    "good", "great", "best", "better", "love", "like", "happy", "hope",
    "yes", "beautiful", "wonderful", "nice", "fine", "win", "winner",
    "safe", "free", "friend", "thanks", "thank", "joy", "glad", "smile",
    "perfect", "amazing", "brave", "trust", "peace", "sweet",
}

NEGATIVE_WORDS = {
    "bad", "worse", "worst", "hate", "no", "not", "never", "death",
    "dead", "kill", "killed", "die", "died", "danger", "dangerous",
    "wrong", "sad", "cry", "pain", "hurt", "afraid", "fear", "angry",
    "mad", "sorry", "alone", "lost", "hell", "damn", "war", "fight",
}

ANGER_WORDS = {
    "angry", "mad", "hate", "kill", "fight", "damn", "hell", "rage",
    "revenge", "furious", "enemy", "attack", "destroy",
}

FEAR_WORDS = {
    "fear", "afraid", "scared", "terrified", "danger", "run", "hide",
    "death", "die", "dead", "murder", "threat", "monster",
}

JOY_WORDS = {
    "happy", "joy", "glad", "smile", "laugh", "love", "wonderful",
    "beautiful", "great", "party", "fun", "hope", "free",
}

SADNESS_WORDS = {
    "sad", "cry", "tears", "alone", "lost", "sorry", "pain", "hurt",
    "miss", "goodbye", "death", "dead", "grief",
}

FIRST_PERSON_PRONOUNS = {
    "i", "me", "my", "mine", "we", "us", "our", "ours",
}

SECOND_PERSON_PRONOUNS = {
    "you", "your", "yours", "yourself", "yourselves",
}


def get_dialogue_style_features(
    raw_lines: list[str],
    dialogue_text: str,
) -> dict:
    """
    Measure punctuation and conversational style features.
    """
    tokens = _tokenize(dialogue_text)
    num_tokens = len(tokens)
    num_lines = len(raw_lines)

    if not num_lines:
        return {
            "question_line_ratio": 0,
            "exclamation_line_ratio": 0,
            "first_person_pronoun_ratio": 0,
            "second_person_pronoun_ratio": 0,
            "contraction_ratio": 0,
        }

    question_lines = sum(1 for line in raw_lines if "?" in line)
    exclamation_lines = sum(1 for line in raw_lines if "!" in line)

    first_person_count = sum(1 for token in tokens if token in FIRST_PERSON_PRONOUNS)
    second_person_count = sum(1 for token in tokens if token in SECOND_PERSON_PRONOUNS)
    contraction_count = sum(1 for token in tokens if "'" in token)

    return {
        "question_line_ratio": question_lines / num_lines,
        "exclamation_line_ratio": exclamation_lines / num_lines,
        "first_person_pronoun_ratio": first_person_count / num_tokens if num_tokens else 0,
        "second_person_pronoun_ratio": second_person_count / num_tokens if num_tokens else 0,
        "contraction_ratio": contraction_count / num_tokens if num_tokens else 0,
    }


def get_dialogue_sentiment_features(dialogue_text: str) -> dict:
    """
    Estimate simple sentiment and emotion features using small built-in lexicons.

    This is intentionally lightweight and dependency-free. It is not a full
    sentiment model, but it can reveal whether simple emotional word ratios are
    useful for the project.
    """
    tokens = _tokenize(dialogue_text)

    if not tokens:
        return {
            "average_sentiment": 0,
            "sentiment_variance": 0,
            "positive_word_ratio": 0,
            "negative_word_ratio": 0,
            "anger_word_ratio": 0,
            "fear_word_ratio": 0,
            "joy_word_ratio": 0,
            "sadness_word_ratio": 0,
        }

    scores = []
    positive_count = 0
    negative_count = 0
    anger_count = 0
    fear_count = 0
    joy_count = 0
    sadness_count = 0

    for token in tokens:
        score = 0

        if token in POSITIVE_WORDS:
            positive_count += 1
            score += 1

        if token in NEGATIVE_WORDS:
            negative_count += 1
            score -= 1

        if token in ANGER_WORDS:
            anger_count += 1

        if token in FEAR_WORDS:
            fear_count += 1

        if token in JOY_WORDS:
            joy_count += 1

        if token in SADNESS_WORDS:
            sadness_count += 1

        scores.append(score)

    avg_sentiment = sum(scores) / len(scores)
    sentiment_variance = (
        sum((score - avg_sentiment) ** 2 for score in scores) / len(scores)
    )

    return {
        "average_sentiment": avg_sentiment,
        "sentiment_variance": sentiment_variance,
        "positive_word_ratio": positive_count / len(tokens),
        "negative_word_ratio": negative_count / len(tokens),
        "anger_word_ratio": anger_count / len(tokens),
        "fear_word_ratio": fear_count / len(tokens),
        "joy_word_ratio": joy_count / len(tokens),
        "sadness_word_ratio": sadness_count / len(tokens),
    }


def get_dialogue_readability_features(
    raw_lines: list[str],
    dialogue_text: str,
) -> dict:
    """
    Estimate readability and lexical complexity.

    Sentence-based readability must use raw subtitle lines, because the cleaned
    dialogue text removes punctuation. If no sentence punctuation is found, the
    number of subtitle lines is used as a conservative sentence proxy.
    """
    tokens = _tokenize(dialogue_text)

    if not tokens:
        return {
            "average_word_length": 0,
            "long_word_ratio": 0,
            "common_word_ratio": 0,
            "rare_word_ratio": 0,
            "simple_word_ratio": 0,
            "complex_word_ratio": 0,
            "flesch_reading_ease": 0,
            "average_sentence_length": 0,
        }

    word_lengths = [len(token.replace("'", "")) for token in tokens]
    syllable_counts = [_count_syllables(token) for token in tokens]

    long_words = sum(1 for length in word_lengths if length >= 7)
    simple_words = sum(1 for count in syllable_counts if count <= 1)
    complex_words = sum(1 for count in syllable_counts if count >= 3)
    common_words = sum(1 for token in tokens if token in COMMON_WORDS)
    rare_words = len(tokens) - common_words

    raw_text = " ".join(str(line) for line in raw_lines)
    sentence_count = len(re.findall(r"[.!?]+", raw_text))

    if sentence_count == 0:
        sentence_count = max(1, len(raw_lines))

    average_sentence_length = len(tokens) / sentence_count
    syllables_per_word = sum(syllable_counts) / len(tokens)

    flesch_reading_ease = (
        206.835
        - 1.015 * average_sentence_length
        - 84.6 * syllables_per_word
    )

    return {
        "average_word_length": sum(word_lengths) / len(word_lengths),
        "long_word_ratio": long_words / len(tokens),
        "common_word_ratio": common_words / len(tokens),
        "rare_word_ratio": rare_words / len(tokens),
        "simple_word_ratio": simple_words / len(tokens),
        "complex_word_ratio": complex_words / len(tokens),
        "flesch_reading_ease": flesch_reading_ease,
        "average_sentence_length": average_sentence_length,
    }


def _median(values: list[float]) -> float:
    """
    Compute a median without extra dependencies.
    """
    if not values:
        return 0

    sorted_values = sorted(values)
    n = len(sorted_values)
    middle = n // 2

    if n % 2 == 1:
        return sorted_values[middle]

    return (sorted_values[middle - 1] + sorted_values[middle]) / 2


def _count_syllables(word: str) -> int:
    """
    Approximate English syllable count for readability features.
    """
    word = re.sub(r"[^a-z]", "", str(word).lower())

    if not word:
        return 0

    vowels = "aeiouy"
    count = 0
    previous_was_vowel = False

    for char in word:
        is_vowel = char in vowels

        if is_vowel and not previous_was_vowel:
            count += 1

        previous_was_vowel = is_vowel

    if word.endswith("e") and count > 1:
        count -= 1

    return max(count, 1)


def extract_movie_id_from_filename(file_name):
    """
    Extract MovieID from subtitle filename.

    Example:
        1_toy_story_1995.srt -> 1
    """
    match = re.match(r"^(\d+)_", str(file_name))

    if not match:
        return None

    return int(match.group(1))


def normalize_match_title(title):
    """
    Normalize titles for matching MovieLens movies-1M, IMDb rows, and franchise rows.
    """
    title = str(title).lower().strip()
    title = re.sub(r"\(.*?\)", "", title)
    title = re.sub(r"[^a-z0-9\s]", " ", title)
    title = re.sub(r"\s+", " ", title)
    return title.strip()


def _tokenize(text: str) -> list[str]:
    """
    Tokenize cleaned dialogue text.
    """
    if not text:
        return []

    return re.findall(r"[a-z']+", text.lower())


if __name__ == "__main__":
    ensure_preprocessed_datasets()
    run_analysis()
