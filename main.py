from pathlib import Path
import re
import importlib.util
from collections import Counter

import pandas as pd


DATASETS_DIR = Path("datasets")

MOVIES_DIR = DATASETS_DIR / "movies"
OPENSUBTITLES_DIR = DATASETS_DIR / "opensubtitles"
IMDB_DIR = DATASETS_DIR / "imdb"

MOVIES_CSV = MOVIES_DIR / "movies_clean.csv"
FRANCHISES_CSV = DATASETS_DIR / "franchises" / "franchises.csv"
SUBS_DIR = OPENSUBTITLES_DIR / "subs"

IMDB_RAW_DIR = IMDB_DIR / "raw"
IMDB_PREPROCESS_SCRIPT = IMDB_DIR / "imdb_preprocess.py"
IMDB_MOVIES_CSV = IMDB_DIR / "imdb_movies_clean.csv"
IMDB_RATINGS_CSV = IMDB_DIR / "imdb_ratings_clean.csv"
IMDB_LOG = IMDB_DIR / "imdb_cleaning_log.txt"

MAIN_DATASET = Path("dataset.csv")
FRANCHISE_ANALYSIS_SCRIPT = Path("franchises_analysis.py")

LANGUAGE_FEATURE_COLUMNS = {
    "num_tokens",
    "num_unique_tokens",
    "type_token_ratio",
    "hapax_ratio",
}


def ensure_preprocessed_datasets():
    """
    Create or update the main dataset.csv.

    Processing order:
        1. Ensure IMDb preprocessing exists.
        2. If dataset.csv does not exist, create it from movies that have subtitles.
        3. Add OpenSubtitles dialogue richness features only if missing.
        4. Add franchise metadata.
        5. Add IMDb movie metadata and ratings.
        6. Save everything back to dataset.csv.

    Franchise analysis is called separately after this function.
    """
    ensure_imdb_preprocessed()

    if not MAIN_DATASET.exists():
        print("Creating main dataset from movies with subtitles...")

        dataset = create_movies_with_subtitles_dataset(
            movies_csv=MOVIES_CSV,
            subs_dir=SUBS_DIR,
            output_path=MAIN_DATASET,
        )
    else:
        print(f"Loading existing main dataset from {MAIN_DATASET}")
        dataset = pd.read_csv(MAIN_DATASET)

    if has_language_features(dataset):
        print(
            "Language feature columns already exist, "
            "skipping OpenSubtitles preprocessing."
        )
    else:
        print("Preprocessing OpenSubtitles...")
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

    print(f"Saved IMDb movies to {IMDB_MOVIES_CSV}")
    print(f"Saved IMDb ratings to {IMDB_RATINGS_CSV}")


def create_movies_with_subtitles_dataset(
    movies_csv=MOVIES_CSV,
    subs_dir=SUBS_DIR,
    output_path=MAIN_DATASET,
):
    """
    Create dataset.csv containing only movies that have downloaded subtitles.
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
        raise ValueError("Column 'MovieID' not found in movies dataset")

    movie_ids_with_subs = set()

    for subtitle_path in subs_dir.glob("*.srt"):
        match = re.match(r"^(\d+)_", subtitle_path.name)

        if match:
            movie_ids_with_subs.add(int(match.group(1)))

    movies_with_subs = movies[movies["MovieID"].isin(movie_ids_with_subs)].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    movies_with_subs.to_csv(output_path, index=False)

    print(f"Found {len(movie_ids_with_subs)} subtitle files")
    print(f"Saved {len(movies_with_subs)} movies with subtitles to {output_path}")

    return movies_with_subs


def merge_movies_with_subtitle_features(movies, subtitles):
    """
    Merge main dataset rows with subtitle dialogue richness features.
    """
    movies = movies.copy()
    subtitles = subtitles.copy()

    if "file_name" not in subtitles.columns:
        raise ValueError("Column 'file_name' not found in subtitles dataset")

    subtitles["MovieID"] = subtitles["file_name"].apply(extract_movie_id_from_filename)
    subtitles = subtitles.dropna(subset=["MovieID"])
    subtitles["MovieID"] = subtitles["MovieID"].astype(int)

    excluded_cols = {
        "dataset",
        "file_name",
        "file_path",
    }

    subtitle_feature_cols = [
        col for col in subtitles.columns
        if (
            (col not in movies.columns or col == "MovieID")
            and col not in excluded_cols
        )
    ]

    subtitles = subtitles[subtitle_feature_cols]

    dataset = movies.merge(
        subtitles,
        on="MovieID",
        how="inner",
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
        raise ValueError(f"IMDb movies missing columns: {missing_movie_cols}")

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

    print("Added IMDb metadata")
    print(f"IMDb matches: {dataset['imdb_tconst'].notna().sum()}")

    return dataset


def run_franchise_analysis(analysis_script=FRANCHISE_ANALYSIS_SCRIPT):
    """
    Run franchises_analysis.py after dataset.csv has been created or updated.

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
    Check whether the main dataset already contains subtitle/dialogue features.
    """
    return LANGUAGE_FEATURE_COLUMNS.issubset(set(dataset.columns))


# -----------------------------
# Helper functions
# -----------------------------

def preprocess_opensubtitles(
    dataset_path: str | Path = SUBS_DIR,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Preprocess downloaded OpenSubtitles .srt files.

    Returns one row per subtitle/movie file with dialogue richness features only.
    Does not save full subtitle text.
    """
    dataset_path = Path(dataset_path)

    files = list(dataset_path.rglob("*.srt"))

    if limit is not None:
        files = files[:limit]

    rows = []

    for file_path in files:
        try:
            dialogue_lines = _extract_opensubtitles_lines(file_path)
            dialogue_text = " ".join(dialogue_lines)

            richness_features = get_dialogue_richness(dialogue_text)

            rows.append({
                "dataset": "opensubtitles",
                "file_name": file_path.name,
                "file_path": str(file_path),
                **richness_features,
            })

        except Exception as e:
            print(f"Failed to process {file_path}: {e}")

    return pd.DataFrame(rows)


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
    Normalize titles for matching MovieLens movies, IMDb rows, and franchise rows.
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
    run_franchise_analysis()
