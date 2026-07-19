from pathlib import Path
import re
import importlib.util
import subprocess
import sys
from collections import Counter

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


REPO_ROOT = Path(__file__).resolve().parent

# Large files are gitignored (see .gitignore / README "Notes") and only exist
# locally after setup.sh has downloaded and unpacked them. Run it
# automatically here, but only if something required is actually missing, so
# a fresh clone doesn't need a separate manual `./setup.sh` step first.
REQUIRED_DATA_PATHS = [
    REPO_ROOT / "dataset_ratings_and_tags.csv",
    REPO_ROOT / "datasets" / "imdb" / "raw" / "title.basics.tsv.gz",
    REPO_ROOT / "datasets" / "imdb" / "raw" / "title.ratings.tsv.gz",
    REPO_ROOT / "datasets" / "movies-32M" / "raw" / "ratings.csv",
    REPO_ROOT / "datasets" / "movies-32M" / "movies_ratings_clean.csv",
    REPO_ROOT / "datasets" / "opensubtitles" / "subs",
]


def ensure_setup_data() -> None:
    def present(p: Path) -> bool:
        return any(p.iterdir()) if p.is_dir() else p.exists()

    missing = [p for p in REQUIRED_DATA_PATHS if not present(p)]
    if not missing:
        return

    print("==> Missing data detected, running setup.sh to fetch it:")
    for p in missing:
        print(f"      - {p.relative_to(REPO_ROOT)}")
    process = subprocess.Popen(["bash", str(REPO_ROOT / "setup.sh")], cwd=REPO_ROOT)
    while process.poll() is None:
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print("Downloading missing data...")
    if process.returncode != 0:
        sys.exit(f"setup.sh failed with exit code {process.returncode}")
    if any(not present(p) for p in REQUIRED_DATA_PATHS):
        sys.exit("setup.sh ran but required data is still missing.")


DATASETS_DIR = Path("datasets")

MOVIES_DIR = DATASETS_DIR / "movies-1M"
MOVIES_32M_DIR = DATASETS_DIR / "movies-32M"
OPENSUBTITLES_DIR = DATASETS_DIR / "opensubtitles"
IMDB_DIR = DATASETS_DIR / "imdb"

MOVIES_1M_PREPROCESS_SCRIPT = MOVIES_DIR / "movies_preprocess.py"
MOVIES_CSV = MOVIES_DIR / "movies_clean.csv"

MOVIES_32M_PREPROCESS_SCRIPT = MOVIES_32M_DIR / "movies_32m_preprocess.py"
MOVIES_32M_CSV = MOVIES_32M_DIR / "movies_clean.csv"

MOVIELENS_1M_RATINGS_CSV = MOVIES_DIR / "movies_ratings_clean.csv"
MOVIELENS_32M_RATINGS_CSV = MOVIES_32M_DIR / "movies_ratings_clean.csv"

MOVIELENS_1M_TAGS_CSV = MOVIES_DIR / "movies_tags_clean.csv"
MOVIELENS_32M_TAGS_CSV = MOVIES_32M_DIR / "movies_tags_clean.csv"

FRANCHISES_CSV = DATASETS_DIR / "franchises" / "franchises.csv"
SUBS_DIR = OPENSUBTITLES_DIR / "subs"

IMDB_RAW_DIR = IMDB_DIR / "raw"
IMDB_PREPROCESS_SCRIPT = IMDB_DIR / "imdb_preprocess.py"
IMDB_MOVIES_CSV = IMDB_DIR / "imdb_movies_clean.csv"
IMDB_RATINGS_CSV = IMDB_DIR / "imdb_ratings_clean.csv"
IMDB_LOG = IMDB_DIR / "imdb_cleaning_log.txt"

MAIN_DATASET = Path("dataset.csv")
RATINGS_AND_TAGS_DATASET = Path("dataset_ratings_and_tags.csv")
ANALYSIS_SCRIPT = Path("analyze_data.py")

LANGUAGE_FEATURE_VERSION = 4
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

    # Optional content-word vocabulary/repetition features.
    # These remove NLTK stopwords and stem tokens, but do not replace the
    # original unfiltered dialogue features.
    "content_stemmed_num_tokens",
    "content_stemmed_num_unique_tokens",
    "content_stemmed_type_token_ratio",
    "content_stemmed_hapax_ratio",
    "content_stemmed_top_word_frequency_ratio",
    "content_stemmed_bigram_repetition_ratio",
    "content_stemmed_trigram_repetition_ratio",
    "content_stemmed_repeated_short_phrase_ratio",

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

LOG_WIDTH = 72


def log_section(title):
    """
    Print a readable pipeline section header.
    """
    print("\n" + "=" * LOG_WIDTH)
    print(title)
    print("=" * LOG_WIDTH)


def log_step(message):
    """
    Print a readable pipeline step message.
    """
    print(f"\n--- {message}")


def log_info(message):
    """
    Print a readable pipeline detail message.
    """
    print(f"  {message}")


def ensure_preprocessed_datasets():
    """
    Create or update the main dataset.csv.

    Processing order:
        1. Ensure MovieLens 1M preprocessing exists.
        2. Ensure MovieLens 32M preprocessing exists.
        3. Create dataset.csv from MovieLens 1M only if it does not exist.
           Otherwise, load the existing dataset and append only new MovieLens rows.
        4. Append only new MovieLens 32M MovieIDs.
        5. Ensure IMDb preprocessing exists after the MovieLens movie dataset is ready.
        6. Append OpenSubtitles dialogue features only for movies that need them:
           newly appended rows, rows with missing language features, or rows with
           an outdated language feature version.
        7. Append franchise metadata when available, without dropping non-franchise movies.
        8. Append IMDb movie metadata and ratings when available, without dropping
           movies that do not match IMDb.
        9. Build dataset_ratings_and_tags.csv with both user ratings and user tags.
        10. Save everything back to dataset.csv.

    Franchise analysis is called separately after this function.
    """
    log_section("Preparing source datasets")
    ensure_movies_1m_preprocessed()
    ensure_movies_32m_preprocessed()

    log_section("Building movie-level dataset")

    if MAIN_DATASET.exists():
        log_step(f"Loading existing movie dataset from {MAIN_DATASET}")
        dataset = pd.read_csv(MAIN_DATASET, low_memory=False)
        original_rows = len(dataset)

        log_step("Appending missing MovieLens 1M movies")
        before_rows = len(dataset)
        dataset = merge_movies_1m_into_dataset(dataset)
        log_info(f"Rows after 1M append: {len(dataset):,} (before: {before_rows:,})")
    else:
        log_step("Creating movie dataset from all MovieLens 1M movies")
        dataset = create_movies_1m_dataset(
            movies_csv=MOVIES_CSV,
            output_path=MAIN_DATASET,
        )
        original_rows = 0
        log_info(f"Rows after 1M creation: {len(dataset):,}")

    log_step("Appending new MovieLens 32M movies")
    before_rows = len(dataset)
    dataset = merge_movies_32m_into_dataset(dataset)
    log_info(f"Rows after 32M append: {len(dataset):,} (before: {before_rows:,})")
    log_info(f"New movie rows added this run: {len(dataset) - original_rows:,}")

    dataset.to_csv(MAIN_DATASET, index=False)
    log_info(f"Saved MovieLens movie dataset to {MAIN_DATASET}")

    log_section("Preparing external metadata")
    ensure_imdb_preprocessed()

    log_section("Appending subtitle language features")
    movie_ids_needing_subtitles = get_movie_ids_needing_language_features(dataset)

    if not movie_ids_needing_subtitles:
        log_info(
            "All available subtitle files already have current language features; "
            "skipping OpenSubtitles preprocessing."
        )
    else:
        log_info(
            "Subtitle files selected for preprocessing: "
            f"{len(movie_ids_needing_subtitles):,}"
        )

        subtitles = preprocess_opensubtitles(
            dataset_path=SUBS_DIR,
            movie_ids=movie_ids_needing_subtitles,
        )

        if subtitles.empty:
            log_info("No selected subtitle files were preprocessed.")
        else:
            before_rows = len(dataset)
            dataset = merge_movies_with_subtitle_features(
                movies=dataset,
                subtitles=subtitles,
            )
            log_info(
                "Rows after subtitle feature append: "
                f"{len(dataset):,} (before: {before_rows:,})"
            )

    log_section("Appending franchise metadata")
    before_rows = len(dataset)
    dataset = add_franchise_columns(dataset)
    log_info(
        "Rows after franchise metadata append: "
        f"{len(dataset):,} (before: {before_rows:,})"
    )

    log_section("Appending IMDb metadata")
    before_rows = len(dataset)
    dataset = add_imdb_columns(dataset)
    log_info(
        "Rows after IMDb metadata append: "
        f"{len(dataset):,} (before: {before_rows:,})"
    )

    log_section("Building recommender interaction dataset")
    create_ratings_and_tags_dataset(
        movie_ids=set(dataset["MovieID"]),
        output_path=RATINGS_AND_TAGS_DATASET,
    )

    dataset.to_csv(MAIN_DATASET, index=False)

    log_section("Done")
    log_info(f"Saved final movie dataset to {MAIN_DATASET}")
    log_info(f"Saved ratings-and-tags dataset to {RATINGS_AND_TAGS_DATASET}")
    log_info(f"Final movie rows: {len(dataset):,}")

    return dataset


def ensure_movies_1m_preprocessed():
    """
    Run MovieLens 1M preprocessing if cleaned files do not already exist.

    Expected input:
        datasets/movies-1M/raw/movies.dat
        datasets/movies-1M/raw/ratings.dat
        datasets/movies-1M/raw/tags.dat

    Expected preprocessing script:
        datasets/movies-1M/movies_preprocess.py

    Outputs:
        datasets/movies-1M/movies_clean.csv
        datasets/movies-1M/movies_ratings_clean.csv
        datasets/movies-1M/movies_tags_clean.csv
    """
    expected_outputs = [
        MOVIES_CSV,
        MOVIELENS_1M_RATINGS_CSV,
        MOVIELENS_1M_TAGS_CSV,
    ]

    if all(path.exists() for path in expected_outputs):
        print("Found existing MovieLens 1M processed files, skipping 1M preprocessing.")
        return

    print("Preprocessing MovieLens 1M...")

    if not MOVIES_1M_PREPROCESS_SCRIPT.exists():
        raise FileNotFoundError(f"{MOVIES_1M_PREPROCESS_SCRIPT} not found")

    spec = importlib.util.spec_from_file_location(
        "movies_1m_preprocess",
        MOVIES_1M_PREPROCESS_SCRIPT,
    )

    movies_1m_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(movies_1m_module)

    if not hasattr(movies_1m_module, "preprocess_movies"):
        raise AttributeError(
            f"{MOVIES_1M_PREPROCESS_SCRIPT} must define preprocess_movies()"
        )

    movies_1m_module.preprocess_movies(
        raw_dir=MOVIES_DIR / "raw",
        out_dir=MOVIES_DIR,
    )

    missing_outputs = [
        path
        for path in expected_outputs
        if not path.exists()
    ]

    if missing_outputs:
        raise FileNotFoundError(
            "MovieLens 1M preprocessing finished but these outputs were not created: "
            f"{missing_outputs}"
        )

    print(f"Saved MovieLens 1M movies to {MOVIES_CSV}")
    print(f"Saved MovieLens 1M ratings to {MOVIELENS_1M_RATINGS_CSV}")
    print(f"Saved MovieLens 1M tags to {MOVIELENS_1M_TAGS_CSV}")


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
    Run MovieLens 32M preprocessing if cleaned files do not already exist.

    Expected input:
        datasets/movies-32M/raw/movies.csv
        datasets/movies-32M/raw/ratings.csv
        datasets/movies-32M/raw/tags.csv

    Expected preprocessing script:
        datasets/movies-32M/movies_32m_preprocess.py

    Outputs:
        datasets/movies-32M/movies_clean.csv
        datasets/movies-32M/movies_ratings_clean.csv
        datasets/movies-32M/movies_tags_clean.csv
    """
    expected_outputs = [
        MOVIES_32M_CSV,
        MOVIELENS_32M_RATINGS_CSV,
        MOVIELENS_32M_TAGS_CSV,
    ]

    if all(path.exists() for path in expected_outputs):
        print("Found existing MovieLens 32M processed files, skipping 32M preprocessing.")
        return

    print("Preprocessing MovieLens 32M...")

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

    missing_outputs = [
        path
        for path in expected_outputs
        if not path.exists()
    ]

    if missing_outputs:
        raise FileNotFoundError(
            "MovieLens 32M preprocessing finished but these outputs were not created: "
            f"{missing_outputs}"
        )

    print(f"Saved MovieLens 32M movies to {MOVIES_32M_CSV}")
    print(f"Saved MovieLens 32M ratings to {MOVIELENS_32M_RATINGS_CSV}")
    print(f"Saved MovieLens 32M tags to {MOVIELENS_32M_TAGS_CSV}")


def create_movies_1m_dataset(
    movies_csv=MOVIES_CSV,
    output_path=MAIN_DATASET,
):
    """
    Create dataset.csv from all cleaned MovieLens 1M movies.

    This is the main movie-level dataset for the recommender system. Movies are
    kept even when no subtitle file exists; subtitle features are merged later
    only for movies with downloaded subtitles.
    """
    movies_csv = Path(movies_csv)
    output_path = Path(output_path)

    if not movies_csv.exists():
        raise FileNotFoundError(f"{movies_csv} not found")

    movies = pd.read_csv(movies_csv, low_memory=False)

    if "MovieID" not in movies.columns:
        raise ValueError("Column 'MovieID' not found in movies-1M dataset")

    movies = movies.copy()
    movies["MovieID"] = pd.to_numeric(movies["MovieID"], errors="coerce")
    movies = movies.dropna(subset=["MovieID"]).copy()
    movies["MovieID"] = movies["MovieID"].astype(int)
    movies = movies.drop_duplicates(subset="MovieID", keep="first").copy()

    movies["MovieLens1MAvailable"] = True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    movies.to_csv(output_path, index=False)

    log_info(f"Saved {len(movies):,} MovieLens 1M movies to {output_path}")

    return movies


def merge_movies_1m_into_dataset(
    dataset,
    movies_1m_csv=MOVIES_CSV,
):
    """
    Add missing MovieLens 1M movies to the main dataset.

    Existing MovieIDs are kept unchanged. Only MovieLens 1M rows whose MovieID
    is not already present in dataset.csv are appended.
    """
    movies_1m_csv = Path(movies_1m_csv)

    if not movies_1m_csv.exists():
        raise FileNotFoundError(f"{movies_1m_csv} not found")

    dataset = dataset.copy()
    movies_1m = pd.read_csv(movies_1m_csv, low_memory=False)

    if "MovieID" not in dataset.columns:
        raise ValueError("Column 'MovieID' not found in main dataset")

    if "MovieID" not in movies_1m.columns:
        raise ValueError("Column 'MovieID' not found in MovieLens 1M dataset")

    dataset["MovieID"] = pd.to_numeric(dataset["MovieID"], errors="coerce")
    movies_1m["MovieID"] = pd.to_numeric(movies_1m["MovieID"], errors="coerce")

    dataset = dataset.dropna(subset=["MovieID"]).copy()
    movies_1m = movies_1m.dropna(subset=["MovieID"]).copy()

    dataset["MovieID"] = dataset["MovieID"].astype(int)
    movies_1m["MovieID"] = movies_1m["MovieID"].astype(int)

    before_dedup = len(dataset)
    dataset = dataset.drop_duplicates(subset="MovieID", keep="first").copy()
    removed_duplicates = before_dedup - len(dataset)

    if removed_duplicates:
        print(
            f"Removed {removed_duplicates:,} duplicate MovieID rows before 1M merge."
        )

    movies_1m = movies_1m.drop_duplicates(subset="MovieID", keep="first").copy()

    existing_movie_ids = set(dataset["MovieID"])
    movies_1m_ids = set(movies_1m["MovieID"])

    if "MovieLens1MAvailable" in dataset.columns:
        dataset = dataset.drop(columns=["MovieLens1MAvailable"])

    dataset["MovieLens1MAvailable"] = dataset["MovieID"].isin(movies_1m_ids)

    new_movies_1m = movies_1m[
        ~movies_1m["MovieID"].isin(existing_movie_ids)
    ].copy()

    skipped_existing = len(movies_1m) - len(new_movies_1m)

    if new_movies_1m.empty:
        print(
            "MovieLens 1M merge: "
            f"skipped {skipped_existing:,} existing MovieIDs; "
            "added 0 new movies."
        )
        return dataset

    new_movies_1m["MovieLens1MAvailable"] = True

    all_columns = list(dataset.columns)

    for col in new_movies_1m.columns:
        if col not in all_columns:
            all_columns.append(col)

    for col in all_columns:
        if col not in dataset.columns:
            dataset[col] = pd.NA

        if col not in new_movies_1m.columns:
            new_movies_1m[col] = pd.NA

    dataset = pd.concat(
        [
            dataset[all_columns],
            new_movies_1m[all_columns],
        ],
        ignore_index=True,
    )

    print(
        "MovieLens 1M merge: "
        f"skipped {skipped_existing:,} existing MovieIDs; "
        f"added {len(new_movies_1m):,} new movies."
    )

    return dataset


def create_movies_with_subtitles_dataset(
    movies_csv=MOVIES_CSV,
    subs_dir=SUBS_DIR,
    output_path=MAIN_DATASET,
):
    """
    Create dataset.csv containing only movies-1M that have downloaded subtitles.
    """
    movies_csv = Path(movies_csv)
    subs_dir = Path(subs_dir)
    output_path = Path(output_path)

    if not movies_csv.exists():
        raise FileNotFoundError(f"{movies_csv} not found")

    if not subs_dir.exists():
        raise FileNotFoundError(f"{subs_dir} not found")

    movies = pd.read_csv(movies_csv, low_memory=False)

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
    Merge MovieLens 32M movie metadata into the main dataset.

    Matching is done by MovieID. Existing MovieIDs are kept unchanged, even if
    MovieLens 32M has different metadata for the same MovieID. Only new 32M
    MovieIDs that are not already present in dataset.csv are appended.

    Adds:
        - MovieLens32MAvailable
    """
    movies_32m_csv = Path(movies_32m_csv)

    if not movies_32m_csv.exists():
        raise FileNotFoundError(f"{movies_32m_csv} not found")

    dataset = dataset.copy()
    movies_32m = pd.read_csv(movies_32m_csv, low_memory=False)

    if "MovieID" not in dataset.columns:
        raise ValueError("Column 'MovieID' not found in main dataset")

    if "MovieID" not in movies_32m.columns:
        raise ValueError("Column 'MovieID' not found in MovieLens 32M dataset")

    dataset["MovieID"] = pd.to_numeric(dataset["MovieID"], errors="coerce")
    movies_32m["MovieID"] = pd.to_numeric(movies_32m["MovieID"], errors="coerce")

    dataset = dataset.dropna(subset=["MovieID"]).copy()
    movies_32m = movies_32m.dropna(subset=["MovieID"]).copy()

    dataset["MovieID"] = dataset["MovieID"].astype(int)
    movies_32m["MovieID"] = movies_32m["MovieID"].astype(int)

    before_dedup = len(dataset)
    dataset = dataset.drop_duplicates(subset="MovieID", keep="first").copy()
    removed_duplicates = before_dedup - len(dataset)

    if removed_duplicates:
        print(
            f"Removed {removed_duplicates:,} duplicate MovieID rows before 32M merge."
        )

    movies_32m = movies_32m.drop_duplicates(subset="MovieID", keep="first").copy()

    old_32m_cols = [
        col for col in dataset.columns
        if col.endswith("_32M")
    ]
    dataset = dataset.drop(columns=old_32m_cols, errors="ignore")

    existing_movie_ids = set(dataset["MovieID"])
    movies_32m_ids = set(movies_32m["MovieID"])

    if "MovieLens32MAvailable" in dataset.columns:
        dataset = dataset.drop(columns=["MovieLens32MAvailable"])

    dataset["MovieLens32MAvailable"] = dataset["MovieID"].isin(movies_32m_ids)

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
    Merge subtitle dialogue features into the main dataset.

    Movies without subtitle files are kept. Existing subtitle feature columns are
    refreshed only for MovieIDs present in the subtitle dataframe.
    """
    movies = movies.copy()
    subtitles = subtitles.copy()

    if subtitles.empty:
        return movies

    if "file_name" not in subtitles.columns:
        raise ValueError("Column 'file_name' not found in subtitles dataset")

    if "MovieID" not in movies.columns:
        raise ValueError("Column 'MovieID' not found in main dataset")

    subtitles["MovieID"] = subtitles["file_name"].apply(extract_movie_id_from_filename)
    subtitles = subtitles.dropna(subset=["MovieID"]).copy()
    subtitles["MovieID"] = subtitles["MovieID"].astype(int)

    feature_cols = [
        col for col in subtitles.columns
        if col in SUBTITLE_FEATURE_COLUMNS
        or col == LANGUAGE_FEATURE_VERSION_COL
    ]

    if not feature_cols:
        raise ValueError("No subtitle feature columns were created")

    movies["MovieID"] = pd.to_numeric(movies["MovieID"], errors="coerce")
    movies = movies.dropna(subset=["MovieID"]).copy()
    movies["MovieID"] = movies["MovieID"].astype(int)

    subtitles = subtitles[["MovieID"] + feature_cols]
    subtitles = subtitles.drop_duplicates(subset="MovieID", keep="last")
    subtitles = subtitles.set_index("MovieID")

    movies = movies.set_index("MovieID", drop=False)

    for col in feature_cols:
        if col not in movies.columns:
            movies[col] = pd.NA

        update_values = subtitles[col].dropna()
        matching_index = update_values.index.intersection(movies.index)

        if len(matching_index) > 0:
            movies.loc[matching_index, col] = update_values.loc[matching_index]

    movies = movies.reset_index(drop=True)

    log_info(f"Updated subtitle features for {len(subtitles):,} subtitle files")
    log_info(f"Movies kept after subtitle merge: {len(movies):,}")

    return movies


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

    franchise_lookup = franchise_lookup.drop_duplicates(
        subset=["match_title", "match_year"],
        keep="first",
    )

    movies = movies.merge(
        franchise_lookup,
        on=["match_title", "match_year"],
        how="left",
    )

    movies = movies.drop(columns=["match_title", "match_year"])

    log_info("Added franchise metadata")
    log_info(f"Franchise movies matched: {movies['FranchiseID'].notna().sum():,}")

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

    log_info("Added IMDb metadata")
    log_info(f"IMDb matches: {dataset['imdb_tconst'].notna().sum():,}")

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


def create_ratings_and_tags_dataset(
    movie_ids,
    output_path=RATINGS_AND_TAGS_DATASET,
):
    """
    Create one final recommender interaction dataset.

    The output file, dataset_ratings_and_tags.csv, contains both:
        - explicit MovieLens user ratings;
        - optional MovieLens user tags.

    The movie-level metadata remains in dataset.csv. This file contains
    user-item interactions for the recommender system.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    movie_ids = {
        int(movie_id)
        for movie_id in movie_ids
        if pd.notna(movie_id)
    }

    if not movie_ids:
        raise ValueError("No movie IDs were provided for the ratings-and-tags dataset")

    if output_path.exists():
        output_path.unlink()

    wrote_any_rows = False
    total_rows = 0

    rating_sources = [
        ("MovieLens 1M", MOVIELENS_1M_RATINGS_CSV),
        ("MovieLens 32M", MOVIELENS_32M_RATINGS_CSV),
    ]

    for source_name, ratings_path in rating_sources:
        ratings_path = Path(ratings_path)

        if not ratings_path.exists():
            print(f"{ratings_path} not found for {source_name}; skipping.")
            continue

        rows_written = append_filtered_ratings_as_events(
            ratings_path=ratings_path,
            movie_ids=movie_ids,
            source_name=source_name,
            output_path=output_path,
            write_header=not wrote_any_rows,
        )

        if rows_written > 0:
            wrote_any_rows = True
            total_rows += rows_written

        print(f"{source_name} ratings kept: {rows_written:,}")

    tag_sources = [
        ("MovieLens 1M", MOVIELENS_1M_TAGS_CSV),
        ("MovieLens 32M", MOVIELENS_32M_TAGS_CSV),
    ]

    for source_name, tags_path in tag_sources:
        tags_path = Path(tags_path)

        if not tags_path.exists():
            print(f"{tags_path} not found for {source_name}; skipping.")
            continue

        rows_written = append_filtered_tags_as_events(
            tags_path=tags_path,
            movie_ids=movie_ids,
            source_name=source_name,
            output_path=output_path,
            write_header=not wrote_any_rows,
        )

        if rows_written > 0:
            wrote_any_rows = True
            total_rows += rows_written

        print(f"{source_name} tags kept: {rows_written:,}")

    if not wrote_any_rows:
        empty_columns = [
            "UserID",
            "MovieID",
            "InteractionType",
            "Rating",
            "Tag",
            "Timestamp",
            "SourceDataset",
        ]
        pd.DataFrame(columns=empty_columns).to_csv(output_path, index=False)
        print("No ratings or tags matched the movie-level dataset.")
    else:
        print(f"Total ratings and tags saved to {output_path}: {total_rows:,}")

    return output_path


def append_filtered_ratings_as_events(
    ratings_path,
    movie_ids,
    source_name,
    output_path,
    write_header,
    chunksize=500_000,
):
    """
    Append ratings for selected MovieIDs to dataset_ratings_and_tags.csv.

    Large MovieLens 32M rating files are processed in chunks to avoid loading
    the full file into memory.
    """
    ratings_path = Path(ratings_path)
    total_written = 0

    if ratings_path.stat().st_size == 0:
        print(f"{ratings_path} is empty; skipping ratings from this file.")
        return 0

    try:
        reader = pd.read_csv(
            ratings_path,
            low_memory=False,
            chunksize=chunksize,
        )
    except pd.errors.EmptyDataError:
        print(f"{ratings_path} has no readable columns; skipping ratings from this file.")
        return 0

    for chunk in reader:
        chunk = normalize_rating_columns(chunk)

        if "MovieID" not in chunk.columns:
            raise ValueError(f"Column 'MovieID' not found in {ratings_path}")

        chunk["MovieID"] = pd.to_numeric(chunk["MovieID"], errors="coerce")
        chunk = chunk.dropna(subset=["MovieID"]).copy()
        chunk["MovieID"] = chunk["MovieID"].astype(int)

        chunk = chunk[chunk["MovieID"].isin(movie_ids)].copy()

        if chunk.empty:
            continue

        chunk["InteractionType"] = "rating"
        chunk["SourceDataset"] = source_name
        chunk["Tag"] = pd.NA

        preferred_cols = [
            "UserID",
            "MovieID",
            "InteractionType",
            "Rating",
            "Tag",
            "Timestamp",
            "SourceDataset",
        ]

        output_cols = [col for col in preferred_cols if col in chunk.columns]
        chunk = chunk[output_cols]

        chunk.to_csv(
            output_path,
            mode="a",
            index=False,
            header=write_header,
        )

        write_header = False
        total_written += len(chunk)

    return total_written


def append_filtered_tags_as_events(
    tags_path,
    movie_ids,
    source_name,
    output_path,
    write_header,
    chunksize=500_000,
):
    """
    Append tags for selected MovieIDs to dataset_ratings_and_tags.csv.

    Large MovieLens 32M tag files are processed in chunks to avoid loading the
    full file into memory.
    """
    tags_path = Path(tags_path)
    total_written = 0

    if tags_path.stat().st_size == 0:
        print(f"{tags_path} is empty; skipping tags from this file.")
        return 0

    try:
        reader = pd.read_csv(
            tags_path,
            low_memory=False,
            chunksize=chunksize,
        )
    except pd.errors.EmptyDataError:
        print(f"{tags_path} has no readable columns; skipping tags from this file.")
        return 0

    for chunk in reader:
        chunk = normalize_tag_columns(chunk)

        if "MovieID" not in chunk.columns:
            raise ValueError(f"Column 'MovieID' not found in {tags_path}")

        chunk["MovieID"] = pd.to_numeric(chunk["MovieID"], errors="coerce")
        chunk = chunk.dropna(subset=["MovieID"]).copy()
        chunk["MovieID"] = chunk["MovieID"].astype(int)

        chunk = chunk[chunk["MovieID"].isin(movie_ids)].copy()

        if chunk.empty:
            continue

        chunk["InteractionType"] = "tag"
        chunk["SourceDataset"] = source_name
        chunk["Rating"] = pd.NA

        preferred_cols = [
            "UserID",
            "MovieID",
            "InteractionType",
            "Rating",
            "Tag",
            "Timestamp",
            "SourceDataset",
        ]

        output_cols = [col for col in preferred_cols if col in chunk.columns]
        chunk = chunk[output_cols]

        chunk.to_csv(
            output_path,
            mode="a",
            index=False,
            header=write_header,
        )

        write_header = False
        total_written += len(chunk)

    return total_written


def normalize_rating_columns(ratings):
    """
    Normalize common MovieLens rating column names.
    """
    ratings = ratings.copy()

    rename_map = {
        "userId": "UserID",
        "userid": "UserID",
        "user_id": "UserID",
        "movieId": "MovieID",
        "movieid": "MovieID",
        "movie_id": "MovieID",
        "rating": "Rating",
        "timestamp": "Timestamp",
    }

    ratings = ratings.rename(
        columns={
            col: rename_map.get(col, col)
            for col in ratings.columns
        }
    )

    return ratings


def normalize_tag_columns(tags):
    """
    Normalize common MovieLens tag column names.
    """
    tags = tags.copy()

    rename_map = {
        "userId": "UserID",
        "userid": "UserID",
        "user_id": "UserID",
        "movieId": "MovieID",
        "movieid": "MovieID",
        "movie_id": "MovieID",
        "tag": "Tag",
        "timestamp": "Timestamp",
    }

    tags = tags.rename(
        columns={
            col: rename_map.get(col, col)
            for col in tags.columns
        }
    )

    return tags


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
    Return True if no available subtitle file needs preprocessing.
    """
    return len(get_movie_ids_needing_language_features(dataset)) == 0


def get_movie_ids_needing_language_features(dataset):
    """
    Return MovieIDs whose subtitle files should be processed.

    A subtitle file is processed when:
        - its MovieID exists in dataset.csv, and
        - any language feature column is missing from dataset.csv, or
        - any language feature value for that row is missing, or
        - language_feature_version is not the current version.

    This means newly appended MovieLens rows with subtitle files are processed,
    while already-complete existing rows are skipped.
    """
    if "MovieID" not in dataset.columns:
        raise ValueError("Column 'MovieID' not found in main dataset")

    subtitle_movie_ids = get_movie_ids_with_subtitle_files(SUBS_DIR)

    if not subtitle_movie_ids:
        return set()

    dataset = dataset.copy()
    dataset["MovieID"] = pd.to_numeric(dataset["MovieID"], errors="coerce")
    dataset = dataset.dropna(subset=["MovieID"]).copy()
    dataset["MovieID"] = dataset["MovieID"].astype(int)

    dataset_movie_ids = set(dataset["MovieID"])
    candidate_movie_ids = subtitle_movie_ids & dataset_movie_ids

    if not candidate_movie_ids:
        return set()

    missing_columns = LANGUAGE_FEATURE_COLUMNS - set(dataset.columns)

    if missing_columns:
        return candidate_movie_ids

    subtitle_rows = dataset["MovieID"].isin(candidate_movie_ids)

    feature_cols = list(SUBTITLE_FEATURE_COLUMNS)
    feature_missing = dataset.loc[subtitle_rows, feature_cols].isna().any(axis=1)

    version_values = pd.to_numeric(
        dataset.loc[subtitle_rows, LANGUAGE_FEATURE_VERSION_COL],
        errors="coerce",
    )
    old_version = ~version_values.eq(LANGUAGE_FEATURE_VERSION)

    needs_processing_rows = dataset.loc[
        subtitle_rows,
        "MovieID",
    ][feature_missing | old_version]

    return set(needs_processing_rows.astype(int))


def get_movie_ids_with_subtitle_files(subs_dir=SUBS_DIR):
    """
    Return MovieIDs that have downloaded subtitle files.

    Subtitle filenames are expected to start with MovieID, for example:
        1_toy_story_1995.srt
    """
    subs_dir = Path(subs_dir)

    if not subs_dir.exists():
        return set()

    movie_ids = set()

    for subtitle_path in subs_dir.glob("*.srt"):
        movie_id = extract_movie_id_from_filename(subtitle_path.name)

        if movie_id is not None:
            movie_ids.add(movie_id)

    return movie_ids


# -----------------------------
# Helper functions
# -----------------------------

def preprocess_opensubtitles(
    dataset_path: str | Path = SUBS_DIR,
    limit: int | None = None,
    movie_ids: set[int] | None = None,
) -> pd.DataFrame:
    """
    Preprocess downloaded OpenSubtitles .srt files.

    If movie_ids is provided, only subtitle files whose filenames start with one
    of those MovieIDs are processed.

    Returns one row per subtitle/movie file with numeric dialogue features only.
    Does not save full subtitle text.
    """
    dataset_path = Path(dataset_path)
    files = list(dataset_path.rglob("*.srt"))

    if movie_ids is not None:
        movie_ids = {
            int(movie_id)
            for movie_id in movie_ids
            if pd.notna(movie_id)
        }
        files = [
            file_path
            for file_path in files
            if extract_movie_id_from_filename(file_path.name) in movie_ids
        ]

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
                **get_content_stemmed_vocabulary_features(dialogue_text),
                **get_content_stemmed_repetition_features(dialogue_text),
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


def get_content_stemmed_vocabulary_features(dialogue_text: str) -> dict:
    """
    Measure vocabulary richness after removing NLTK English stopwords and
    stemming tokens with PorterStemmer.

    These are optional additional features. They do not replace the original
    unfiltered and unstemmed vocabulary features, because stopwords and original
    word forms are still useful for dialogue style, readability, sentiment, and
    pronoun-based features.
    """
    tokens = _content_stemmed_tokens(dialogue_text)

    if not tokens:
        return {
            "content_stemmed_num_tokens": 0,
            "content_stemmed_num_unique_tokens": 0,
            "content_stemmed_type_token_ratio": 0,
            "content_stemmed_hapax_ratio": 0,
        }

    token_counts = Counter(tokens)
    hapax_count = sum(1 for count in token_counts.values() if count == 1)

    return {
        "content_stemmed_num_tokens": len(tokens),
        "content_stemmed_num_unique_tokens": len(token_counts),
        "content_stemmed_type_token_ratio": len(token_counts) / len(tokens),
        "content_stemmed_hapax_ratio": hapax_count / len(token_counts),
    }


def get_content_stemmed_repetition_features(dialogue_text: str) -> dict:
    """
    Measure repeated content words and phrases after removing NLTK English
    stopwords and stemming tokens with PorterStemmer.

    These features focus on repeated meaningful stems instead of repeated
    function words such as articles, pronouns, and prepositions.
    """
    tokens = _content_stemmed_tokens(dialogue_text)

    if not tokens:
        return {
            "content_stemmed_top_word_frequency_ratio": 0,
            "content_stemmed_bigram_repetition_ratio": 0,
            "content_stemmed_trigram_repetition_ratio": 0,
            "content_stemmed_repeated_short_phrase_ratio": 0,
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

    return {
        "content_stemmed_top_word_frequency_ratio": token_counts.most_common(1)[0][1] / len(tokens),
        "content_stemmed_bigram_repetition_ratio": repeated_bigrams / len(bigrams) if bigrams else 0,
        "content_stemmed_trigram_repetition_ratio": repeated_trigrams / len(trigrams) if trigrams else 0,
        "content_stemmed_repeated_short_phrase_ratio": (
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


try:
    nltk.download('stopwords')
    NLTK_STOPWORDS = set(stopwords.words("english"))
except LookupError as exc:
    raise LookupError(
        "NLTK stopwords corpus is missing. Run this once before executing the "
        "pipeline: import nltk; nltk.download('stopwords')"
    ) from exc

STEMMER = PorterStemmer()


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


def _content_stemmed_tokens(text: str) -> list[str]:
    """
    Tokenize cleaned dialogue text, remove NLTK English stopwords, and stem
    tokens with PorterStemmer.

    This is used only for optional content-word vocabulary and repetition
    features. The main dialogue features still use the original unstemmed tokens.
    """
    content_tokens = []

    for token in _tokenize(text):
        clean_token = token.replace("'", "")

        if len(clean_token) <= 1:
            continue

        if clean_token in NLTK_STOPWORDS:
            continue

        content_tokens.append(STEMMER.stem(clean_token))

    return content_tokens


if __name__ == "__main__":
    ensure_setup_data()
    ensure_preprocessed_datasets()
    run_analysis()
