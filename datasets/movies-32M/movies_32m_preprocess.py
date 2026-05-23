import datetime as dt
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd

"""
Preprocesses the MovieLens 32M movies, ratings, and tags datasets.

Expected folder structure:
    datasets/
        movies-32M/
            movies_32m_preprocess.py
            raw/
                movies.csv
                ratings.csv        optional
                tags.csv           optional

Outputs:
    datasets/movies-32M/movies_clean.csv
    datasets/movies-32M/movies_ratings_clean.csv      if ratings.csv exists
    datasets/movies-32M/movies_tags_clean.csv         if tags.csv exists
    datasets/movies-32M/movies_cleaning_log.txt

Main cleaning steps:
    - Load raw MovieLens 32M CSV files
    - Rename MovieLens columns to the project format:
        movieId -> MovieID
        title -> Title
        genres -> Genres
        userId -> UserID
        rating -> Rating
        timestamp -> Timestamp
        tag -> Tag
    - Remove nulls, blanks, duplicates, and invalid rows
    - Convert titles, genres, and tags to ASCII
    - Extract movie year from title
    - Fix MovieLens-style trailing articles:
        "Matrix, The" -> "The Matrix"
        "Enfer, L'" -> "L'Enfer"
        "Savage Nights (Nuits fauves, Les)"
            -> "Savage Nights (Les Nuits fauves)"
    - Split pipe-separated genres into Genre1, Genre2, ...
    - If ratings/tags exist, validate them and remove orphan MovieIDs
    - Save cleaned CSV files and a cleaning log
"""

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
OUT_DIR = BASE_DIR

MOVIES_CSV = RAW_DIR / "movies.csv"
RATINGS_CSV = RAW_DIR / "ratings.csv"
TAGS_CSV = RAW_DIR / "tags.csv"

VALID_GENRES = {
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
}

ARTICLES = r"The|A|An|L'|Le|La|Les"
LIGATURES = {
    "æ": "ae",
    "Æ": "AE",
    "ø": "o",
    "Ø": "O",
    "ß": "ss",
}


def log(message, log_lines):
    """
    Print a cleaning message and save it to the log list.
    """
    print(message)
    log_lines.append(message)


def read_csv_utf8(path):
    """
    Read a CSV file using UTF-8, with a safe fallback for encoding issues.
    """
    path = Path(path)

    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def to_ascii(value):
    """
    Convert text to ASCII.

    Handles selected ligatures manually, then removes accents and
    other non-ASCII characters using Unicode normalization.
    """
    value = "".join(LIGATURES.get(c, c) for c in str(value))
    return (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii")
    )


def _move_trailing_article(text):
    """
    Move trailing articles from the end of a title to the beginning.

    Examples
    --------
    "Matrix, The" -> "The Matrix"
    "Valachi Papers,The" -> "The Valachi Papers"
    "Enfer, L'" -> "L'Enfer"
    "Nuits fauves, Les" -> "Les Nuits fauves"
    """
    text = str(text).strip()

    match = re.match(
        rf"^(.*?),\s*({ARTICLES})$",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return text

    main_title = match.group(1).strip()
    article = match.group(2).strip()

    if article.lower() == "l'":
        return f"L'{main_title}"

    return f"{article} {main_title}"


def fix_movie_title(title):
    """
    Fix MovieLens-style titles, including titles inside parentheses.

    Examples:
        "Matrix, The" -> "The Matrix"
        "Enfer, L'" -> "L'Enfer"
        "Savage Nights (Nuits fauves, Les)"
            -> "Savage Nights (Les Nuits fauves)"
    """
    title = str(title).strip()

    match = re.match(r"^(.*?)\s*(\((.*)\))?$", title)

    if not match:
        return _move_trailing_article(title)

    main_title = match.group(1).strip()
    parenthetical = match.group(3)

    fixed_main = _move_trailing_article(main_title)

    if parenthetical:
        fixed_parenthetical = _move_trailing_article(parenthetical)
        return f"{fixed_main} ({fixed_parenthetical})"

    return fixed_main


def preprocess_movies_32m(
    raw_dir=RAW_DIR,
    out_dir=OUT_DIR,
    require_ratings=False,
    require_tags=False,
    drop_movies_without_ratings=True,
):
    """
    Run the MovieLens 32M preprocessing pipeline.

    Parameters
    ----------
    raw_dir : str or pathlib.Path
        Directory containing raw MovieLens 32M files.
    out_dir : str or pathlib.Path
        Directory where cleaned CSV files and the cleaning log are saved.
    require_ratings : bool
        If True, raise an error when ratings.csv is missing.
        If False, process ratings only when ratings.csv exists.
    require_tags : bool
        If True, raise an error when tags.csv is missing.
        If False, process tags only when tags.csv exists.
    drop_movies_without_ratings : bool
        If ratings are available, remove movies that have no ratings.

    Returns
    -------
    tuple
        (movies, ratings, tags), where ratings/tags are None if their
        raw files were not available.
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_lines = []

    movies_path = raw_dir / "movies.csv"
    ratings_path = raw_dir / "ratings.csv"
    tags_path = raw_dir / "tags.csv"

    if not movies_path.exists():
        raise FileNotFoundError(f"{movies_path} not found")

    if require_ratings and not ratings_path.exists():
        raise FileNotFoundError(f"{ratings_path} not found")

    if require_tags and not tags_path.exists():
        raise FileNotFoundError(f"{tags_path} not found")

    movies = read_csv_utf8(movies_path)
    ratings = read_csv_utf8(ratings_path) if ratings_path.exists() else None
    tags = read_csv_utf8(tags_path) if tags_path.exists() else None

    movies = standardize_movies_columns(movies)

    if ratings is not None:
        ratings = standardize_ratings_columns(ratings)

    if tags is not None:
        tags = standardize_tags_columns(tags)

    log(
        "Loaded: "
        f"movies-32M={len(movies):,}, "
        f"ratings={len(ratings):,}" if ratings is not None else
        f"Loaded: movies-32M={len(movies):,}, ratings=not found",
        log_lines,
    )

    if tags is not None:
        log(f"Loaded tags={len(tags):,}", log_lines)
    else:
        log("Loaded tags=not found", log_lines)

    for name, df in [
        ("movies-32M", movies),
        ("ratings", ratings),
        ("tags", tags),
    ]:
        if df is not None:
            log(f"Nulls in {name}:\n{df.isna().sum().to_string()}\n", log_lines)

    movies = clean_movies(movies, log_lines)

    if ratings is not None:
        ratings = clean_ratings(ratings, log_lines)

    if tags is not None:
        tags = clean_tags(tags, log_lines)

    if ratings is not None:
        before = len(ratings)
        ratings = ratings[ratings["MovieID"].isin(movies["MovieID"])]
        log(
            f"ratings: dropped {before - len(ratings)} rows with orphan MovieID",
            log_lines,
        )

    if tags is not None:
        before = len(tags)
        tags = tags[tags["MovieID"].isin(movies["MovieID"])]
        log(
            f"tags: dropped {before - len(tags)} rows with orphan MovieID",
            log_lines,
        )

    if ratings is not None and drop_movies_without_ratings:
        before = len(movies)
        movies = movies[movies["MovieID"].isin(ratings["MovieID"])]
        log(
            f"movies-32M: dropped {before - len(movies)} movies with no ratings",
            log_lines,
        )

        if tags is not None:
            before = len(tags)
            tags = tags[tags["MovieID"].isin(movies["MovieID"])]
            log(
                f"tags: dropped {before - len(tags)} additional rows orphaned by movie pruning",
                log_lines,
            )

    validate_cleaned_data(movies, ratings, tags, log_lines)

    movies.to_csv(out_dir / "movies_clean.csv", index=False)

    if ratings is not None:
        ratings.to_csv(out_dir / "movies_ratings_clean.csv", index=False)

    if tags is not None:
        tags.to_csv(out_dir / "movies_tags_clean.csv", index=False)

    with open(out_dir / "movies_cleaning_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    print(f"Saved cleaned CSVs and log to {out_dir}")

    return movies, ratings, tags


def standardize_movies_columns(movies):
    """
    Convert MovieLens 32M movie column names to project column names.
    """
    rename_map = {
        "movieId": "MovieID",
        "title": "Title",
        "genres": "Genres",
    }

    movies = movies.rename(columns=rename_map)

    required = {"MovieID", "Title", "Genres"}
    missing = required - set(movies.columns)

    if missing:
        raise ValueError(f"movies.csv is missing columns: {missing}")

    return movies[["MovieID", "Title", "Genres"]].copy()


def standardize_ratings_columns(ratings):
    """
    Convert MovieLens 32M rating column names to project column names.
    """
    rename_map = {
        "userId": "UserID",
        "movieId": "MovieID",
        "rating": "Rating",
        "timestamp": "Timestamp",
    }

    ratings = ratings.rename(columns=rename_map)

    required = {"UserID", "MovieID", "Rating", "Timestamp"}
    missing = required - set(ratings.columns)

    if missing:
        raise ValueError(f"ratings.csv is missing columns: {missing}")

    return ratings[["UserID", "MovieID", "Rating", "Timestamp"]].copy()


def standardize_tags_columns(tags):
    """
    Convert MovieLens 32M tag column names to project column names.
    """
    rename_map = {
        "userId": "UserID",
        "movieId": "MovieID",
        "tag": "Tag",
        "timestamp": "Timestamp",
    }

    tags = tags.rename(columns=rename_map)

    required = {"UserID", "MovieID", "Tag", "Timestamp"}
    missing = required - set(tags.columns)

    if missing:
        raise ValueError(f"tags.csv is missing columns: {missing}")

    return tags[["UserID", "MovieID", "Tag", "Timestamp"]].copy()


def clean_movies(movies, log_lines):
    """
    Clean MovieLens 32M movie metadata.
    """
    for col in ["Title", "Genres"]:
        movies[col] = (
            movies[col]
            .astype(str)
            .str.strip()
            .replace({"": None, "nan": None})
        )

    before = len(movies)
    movies = movies.dropna()
    log(f"movies-32M: dropped {before - len(movies)} null/blank rows", log_lines)

    before = len(movies)
    movies = movies.drop_duplicates().drop_duplicates(subset="MovieID")
    log(f"movies-32M: dropped {before - len(movies)} duplicate rows", log_lines)

    title_changed = (~movies["Title"].map(str.isascii)).sum()
    genre_changed = (~movies["Genres"].map(str.isascii)).sum()

    movies["Title"] = movies["Title"].apply(to_ascii)
    movies["Genres"] = movies["Genres"].apply(to_ascii)

    log(
        f"movies-32M: transliterated {title_changed} titles and "
        f"{genre_changed} genres to ASCII",
        log_lines,
    )

    before = len(movies)
    movies["MovieID"] = pd.to_numeric(movies["MovieID"], errors="coerce")
    movies = movies.dropna(subset=["MovieID"])
    movies["MovieID"] = movies["MovieID"].astype(int)
    movies = movies[movies["MovieID"] >= 1]
    log(f"movies-32M: dropped {before - len(movies)} rows with invalid MovieID", log_lines)

    before = len(movies)
    year_extract = movies["Title"].str.extract(r"^(.*)\s*\((\d{4})\)\s*$")
    year_extract.columns = ["TitleOnly", "Year"]

    valid = year_extract["Year"].notna()
    movies = movies.loc[valid].copy()

    movies["Title"] = (
        year_extract
        .loc[valid, "TitleOnly"]
        .str.strip()
        .apply(fix_movie_title)
        .values
    )

    movies["Year"] = year_extract.loc[valid, "Year"].astype(int).values

    log(
        f"movies-32M: dropped {before - len(movies)} rows with malformed Title; "
        f"extracted Year column",
        log_lines,
    )

    before = len(movies)
    movies = movies[movies["Genres"] != "(no genres listed)"]
    log(f'movies-32M: dropped {before - len(movies)} rows with "(no genres listed)"', log_lines)

    before = len(movies)

    def genres_ok(genres):
        parts = str(genres).split("|")
        return len(parts) > 0 and all(part in VALID_GENRES for part in parts)

    movies = movies[movies["Genres"].apply(genres_ok)]
    log(f"movies-32M: dropped {before - len(movies)} rows with unknown genre tokens", log_lines)

    before = len(movies)
    movies = movies[movies["Title"].map(str.isascii) & movies["Genres"].map(str.isascii)]
    log(
        f"movies-32M: dropped {before - len(movies)} rows with non-ASCII characters",
        log_lines,
    )

    max_genres = movies["Genres"].str.split("|").map(len).max()
    genre_cols = [f"Genre{i + 1}" for i in range(max_genres)]

    split_df = movies["Genres"].str.split("|", expand=True).fillna("")
    split_df.columns = genre_cols

    movies = pd.concat([movies.drop(columns="Genres"), split_df], axis=1)
    movies = movies[["MovieID", "Title", "Year"] + genre_cols]

    log(f"movies-32M: split Genres into {max_genres} columns", log_lines)

    return movies


def clean_ratings(ratings, log_lines):
    """
    Clean MovieLens 32M ratings.
    """
    before = len(ratings)
    ratings = ratings.drop_duplicates()
    log(f"ratings: dropped {before - len(ratings)} exact-duplicate rows", log_lines)

    before = len(ratings)
    conflict_mask = ratings.duplicated(subset=["UserID", "MovieID"], keep=False)
    n_conflicts = conflict_mask.sum()
    ratings = ratings.drop_duplicates(subset=["UserID", "MovieID"])
    log(
        f"ratings: found {n_conflicts} rows in conflicting (UserID, MovieID) groups; "
        f"dropped {before - len(ratings)} to keep one rating per tuple",
        log_lines,
    )

    before = len(ratings)

    for col in ["UserID", "MovieID", "Rating", "Timestamp"]:
        ratings[col] = pd.to_numeric(ratings[col], errors="coerce")

    ratings = ratings.dropna()
    ratings = ratings[(ratings["Rating"] * 2) % 1 == 0]

    ratings = ratings.astype({
        "UserID": int,
        "MovieID": int,
        "Timestamp": int,
    })

    ratings["Rating"] = ratings["Rating"].astype(float)

    ratings = ratings[(ratings["Rating"] >= 0.5) & (ratings["Rating"] <= 5.0)]
    ratings = ratings[(ratings["UserID"] >= 1)]
    ratings = ratings[(ratings["MovieID"] >= 1)]

    ts_min = 631152000
    ts_max = int(time.time())

    ratings = ratings[
        (ratings["Timestamp"] >= ts_min)
        & (ratings["Timestamp"] <= ts_max)
    ]

    log(f"ratings: dropped {before - len(ratings)} rows with invalid values", log_lines)

    ratings["Timestamp"] = pd.to_datetime(ratings["Timestamp"], unit="s")
    ratings = ratings.rename(columns={"Timestamp": "Date"})

    return ratings


def clean_tags(tags, log_lines):
    """
    Clean MovieLens 32M tags.
    """
    tags["Tag"] = (
        tags["Tag"]
        .astype(str)
        .str.strip()
        .replace({"": None, "nan": None})
    )

    before = len(tags)
    tags = tags.dropna()
    log(f"tags: dropped {before - len(tags)} null/blank rows", log_lines)

    before = len(tags)
    tags = tags.drop_duplicates().drop_duplicates(subset=["UserID", "MovieID", "Tag"])
    log(f"tags: dropped {before - len(tags)} duplicate rows", log_lines)

    before = len(tags)

    for col in ["UserID", "MovieID", "Timestamp"]:
        tags[col] = pd.to_numeric(tags[col], errors="coerce")

    tags = tags.dropna(subset=["UserID", "MovieID", "Timestamp"])
    tags = tags.astype({
        "UserID": int,
        "MovieID": int,
        "Timestamp": int,
    })

    tags = tags[(tags["UserID"] >= 1)]
    tags = tags[(tags["MovieID"] >= 1)]

    ts_min = 631152000
    ts_max = int(time.time())

    tags = tags[(tags["Timestamp"] >= ts_min) & (tags["Timestamp"] <= ts_max)]

    tag_changed = (~tags["Tag"].map(str.isascii)).sum()
    tags["Tag"] = tags["Tag"].apply(to_ascii).str.strip()
    tags = tags[tags["Tag"] != ""]

    log(f"tags: transliterated {tag_changed} tags to ASCII", log_lines)
    log(f"tags: dropped {before - len(tags)} rows with invalid values or empty tag", log_lines)

    tags["Timestamp"] = pd.to_datetime(tags["Timestamp"], unit="s")
    tags = tags.rename(columns={"Timestamp": "Date"})

    return tags


def validate_cleaned_data(movies, ratings, tags, log_lines):
    """
    Validate the cleaned MovieLens 32M datasets.
    """
    assert movies["MovieID"].is_unique
    log("PASS: MovieID is unique in movies", log_lines)

    genre_cols = [c for c in movies.columns if c.startswith("Genre")]
    all_genre_tokens = set(movies[genre_cols].values.ravel()) - {""}
    unknown = all_genre_tokens - VALID_GENRES
    assert not unknown
    log("PASS: all movie genres are in the whitelist", log_lines)

    next_year = dt.date.today().year + 1
    bad_years = movies.loc[
        (movies["Year"] < 1800) | (movies["Year"] > next_year),
        "Year",
    ]
    assert bad_years.empty
    log("PASS: all movies have a valid Year", log_lines)

    bad_title_chars = movies.loc[~movies["Title"].map(str.isascii), "Title"]
    assert bad_title_chars.empty

    for col in genre_cols:
        bad = movies.loc[~movies[col].map(str.isascii), col]
        assert bad.empty

    log("PASS: all titles and genre columns use only ASCII characters", log_lines)

    if ratings is not None:
        bad_ratings = ratings.loc[~ratings["Rating"].between(0.5, 5.0), "Rating"]
        assert bad_ratings.empty

        non_half = ratings.loc[(ratings["Rating"] * 2) % 1 != 0, "Rating"]
        assert non_half.empty

        log(f"PASS: all {len(ratings):,} ratings are half-star in [0.5, 5.0]", log_lines)

        orphan_movies = set(ratings["MovieID"]) - set(movies["MovieID"])
        assert not orphan_movies
        log("PASS: all MovieIDs in ratings exist in movies", log_lines)

        dup_pairs = ratings[ratings.duplicated(subset=["UserID", "MovieID"], keep=False)]
        assert dup_pairs.empty
        log("PASS: every (UserID, MovieID) pair has exactly one rating", log_lines)

    if tags is not None:
        orphan_tag_movies = set(tags["MovieID"]) - set(movies["MovieID"])
        assert not orphan_tag_movies
        log("PASS: all MovieIDs in tags exist in movies", log_lines)

        bad_tag_chars = tags.loc[~tags["Tag"].map(str.isascii), "Tag"]
        assert bad_tag_chars.empty
        log("PASS: all tags use only ASCII characters", log_lines)


def fix_movie_titles_retroactively(
    input_path=OUT_DIR / "movies_clean.csv",
    output_path=OUT_DIR / "movies_clean_fixed.csv",
):
    """
    Fix MovieLens-style titles in an already processed CSV.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found")

    df = pd.read_csv(input_path)

    if "Title" not in df.columns:
        raise ValueError("Column 'Title' not found in dataset")

    original_titles = df["Title"].astype(str).copy()

    df["Title"] = df["Title"].astype(str).apply(fix_movie_title)

    changed_mask = original_titles != df["Title"]
    num_changed = int(changed_mask.sum())

    print(f"Fixed {num_changed} titles")

    if num_changed > 0:
        print("\nExamples:")
        changed_examples = df.loc[changed_mask, ["MovieID", "Title"]].head(10)

        for idx in changed_examples.index:
            old_title = original_titles.loc[idx]
            new_title = df.loc[idx, "Title"]
            movie_id = df.loc[idx, "MovieID"] if "MovieID" in df.columns else "unknown"

            print(f"- [{movie_id}] {old_title} -> {new_title}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nSaved fixed file to: {output_path}")

    return df


if __name__ == "__main__":
    preprocess_movies_32m()
