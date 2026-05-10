from pathlib import Path
import re
from collections import Counter

import pandas as pd


DATASETS_DIR = Path("datasets")

MOVIES_DIR = DATASETS_DIR / "movies"
OPENSUBTITLES_DIR = DATASETS_DIR / "opensubtitles"

MOVIES_CSV = MOVIES_DIR / "movies_clean.csv"
FRANCHISES_CSV = DATASETS_DIR / "franchises" / "franchises.csv"
SUBS_DIR = OPENSUBTITLES_DIR / "subs"

MAIN_DATASET = Path("dataset.csv")


def ensure_preprocessed_datasets():
    """
    Create or update the main dataset.csv.

    Processing order:
        1. If dataset.csv does not exist, create it from movies that have subtitles.
        2. Preprocess OpenSubtitles.
        3. Merge subtitle dialogue features into dataset.csv.
        4. Add franchise metadata.
        5. Save everything back to dataset.csv.
    """
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

    print("Preprocessing OpenSubtitles...")
    subtitles = preprocess_opensubtitles(dataset_path=SUBS_DIR)

    print("Merging subtitle features into main dataset...")
    dataset = merge_movies_with_subtitle_features(
        movies=dataset,
        subtitles=subtitles,
    )

    dataset.to_csv(MAIN_DATASET, index=False)
    print(f"Saved subtitle features to {MAIN_DATASET}")

    print("Adding franchise metadata...")
    dataset = add_franchise_columns(dataset)

    dataset.to_csv(MAIN_DATASET, index=False)
    print(f"Saved final dataset to {MAIN_DATASET}")

    return dataset


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
    Normalize titles for matching MovieLens movies to franchise rows.
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