import datetime as dt
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd

"""
Preprocesses the MovieLens movies-1M, ratings, and tags datasets.

Expected folder structure:
    datasets/
        movies-1M/
            movies_preprocess.py
            raw/
                movies.dat
                ratings.dat
                tags.dat

Outputs:
    datasets/movies-1M/movies_clean.csv
    datasets/movies-1M/movies_ratings_clean.csv
    datasets/movies-1M/movies_tags_clean.csv
    datasets/movies-1M/movies_cleaning_log.txt

Main cleaning steps:
    - Load raw MovieLens .dat files
    - Remove nulls, blanks, duplicates, and invalid rows
    - Convert titles, genres, and tags to ASCII
    - Extract movie year from title
    - Fix MovieLens-style titles:
        "Matrix, The" -> "The Matrix"
    - Validate ratings, IDs, timestamps, genres, and references
    - Save cleaned CSV files
"""

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
OUT_DIR = BASE_DIR

MOVIES_DAT = RAW_DIR / "movies.dat"
RATINGS_DAT = RAW_DIR / "ratings.dat"
TAGS_DAT = RAW_DIR / "tags.dat"

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
RAW_ENCODING = "latin-1"


def log(message, log_lines):
    """
    Print a cleaning message and save it to the log list.

    Parameters
    ----------
    message : str
        Message to print and store.
    log_lines : list[str]
        Mutable list collecting log messages.
    """
    print(message)
    log_lines.append(message)


def to_ascii(value):
    """
    Convert text to ASCII.

    Handles selected ligatures manually, then removes accents and
    other non-ASCII characters using Unicode normalization.

    Parameters
    ----------
    value : Any
        Value to convert.

    Returns
    -------
    str
        ASCII-only string.
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

    This function fixes titles where the article is placed at the end,
    a common format in MovieLens-style datasets.

    Supported articles:
        - English: "The", "A", "An"
        - French: "L'", "Le", "La", "Les"

    Examples
    --------
    "Matrix, The" -> "The Matrix"
    "Valachi Papers,The" -> "The Valachi Papers"
    "Enfer, L'" -> "L'Enfer"
    "Nuits fauves, Les" -> "Les Nuits fauves"

    Parameters
    ----------
    text : str
        Input title string.

    Returns
    -------
    str
        Title with article moved to the front if pattern matches,
        otherwise the original string unchanged.
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

    # Fix main title before parentheses
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


def preprocess_movies(raw_dir=RAW_DIR, out_dir=OUT_DIR):
    """
    Run the full MovieLens preprocessing pipeline.

    Parameters
    ----------
    raw_dir : str or pathlib.Path
        Directory containing raw MovieLens files:
        movies.dat, ratings.dat, and tags.dat.
    out_dir : str or pathlib.Path
        Directory where cleaned CSV files and the cleaning log are saved.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        Cleaned movies, ratings, and tags DataFrames.
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_lines = []

    movies = pd.read_csv(
        raw_dir / "movies.dat",
        sep="::",
        engine="python",
        header=None,
        names=["MovieID", "Title", "Genres"],
        encoding=RAW_ENCODING
    )

    ratings = pd.read_csv(
        raw_dir / "ratings.dat",
        sep="::",
        engine="python",
        header=None,
        names=["UserID", "MovieID", "Rating", "Timestamp"],
        encoding=RAW_ENCODING
    )

    tags = pd.read_csv(
        raw_dir / "tags.dat",
        sep="::",
        engine="python",
        header=None,
        names=["UserID", "MovieID", "Tag", "Timestamp"],
        encoding=RAW_ENCODING
    )

    log(f"Loaded: movies={len(movies):,}, ratings={len(ratings):,}, tags={len(tags):,}", log_lines)

    for name, df in [("movies", movies), ("ratings", ratings), ("tags", tags)]:
        log(f"Nulls in {name}:\n{df.isna().sum().to_string()}\n", log_lines)

    for df, cols in [
        (movies, ["Title", "Genres"]),
        (ratings, []),
        (tags, ["Tag"]),
    ]:
        for col in cols:
            df[col] = df[col].astype(str).str.strip().replace({"": None, "nan": None})

    before = len(movies), len(ratings), len(tags)
    movies = movies.dropna()
    ratings = ratings.dropna()
    tags = tags.dropna()

    log(
        f"Dropped nulls/blanks: movies-1M={before[0] - len(movies)}, "
        f"ratings={before[1] - len(ratings)}, tags={before[2] - len(tags)}",
        log_lines,
    )

    before = len(movies)
    movies = movies.drop_duplicates().drop_duplicates(subset="MovieID")
    log(f"movies-1M: dropped {before - len(movies)} duplicate rows", log_lines)

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

    before = len(tags)
    tags = tags.drop_duplicates().drop_duplicates(subset=["UserID", "MovieID", "Tag"])
    log(f"tags: dropped {before - len(tags)} duplicate rows", log_lines)

    title_changed = (~movies["Title"].map(str.isascii)).sum()
    genre_changed = (~movies["Genres"].map(str.isascii)).sum()

    movies["Title"] = movies["Title"].apply(to_ascii)
    movies["Genres"] = movies["Genres"].apply(to_ascii)

    log(
        f"movies-1M: transliterated {title_changed} titles and {genre_changed} genres to ASCII",
        log_lines,
    )

    before = len(movies)
    movies["MovieID"] = pd.to_numeric(movies["MovieID"], errors="coerce")
    movies = movies.dropna(subset=["MovieID"])
    movies["MovieID"] = movies["MovieID"].astype(int)
    movies = movies[(movies["MovieID"] >= 1) & (movies["MovieID"] <= 65133)]
    log(f"movies-1M: dropped {before - len(movies)} rows with invalid MovieID", log_lines)

    before = len(movies)
    year_extract = movies["Title"].str.extract(r"^(.*)\s*\((\d{4})\)\s*$")
    year_extract.columns = ["TitleOnly", "Year"]

    valid = year_extract["Year"].notna()
    movies = movies.loc[valid].copy()

    movies["Title"] = year_extract.loc[valid, "TitleOnly"].str.strip().apply(fix_movie_title).values

    movies["Year"] = year_extract.loc[valid, "Year"].astype(int).values

    log(
        f"movies-1M: dropped {before - len(movies)} rows with malformed Title; "
        f"extracted Year column",
        log_lines,
    )

    before = len(movies)
    movies = movies[movies["Genres"] != "(no genres listed)"]
    log(f'movies-1M: dropped {before - len(movies)} rows with "(no genres listed)"', log_lines)

    before = len(movies)

    def genres_ok(genres):
        parts = str(genres).split("|")
        return len(parts) > 0 and all(part in VALID_GENRES for part in parts)

    movies = movies[movies["Genres"].apply(genres_ok)]
    log(f"movies-1M: dropped {before - len(movies)} rows with unknown genre tokens", log_lines)

    before = len(movies)
    movies = movies[movies["Title"].map(str.isascii) & movies["Genres"].map(str.isascii)]
    log(
        f"movies-1M: dropped {before - len(movies)} rows with non-ASCII characters",
        log_lines,
    )

    max_genres = movies["Genres"].str.split("|").map(len).max()
    genre_cols = [f"Genre{i + 1}" for i in range(max_genres)]

    split_df = movies["Genres"].str.split("|", expand=True).fillna("")
    split_df.columns = genre_cols

    movies = pd.concat([movies.drop(columns="Genres"), split_df], axis=1)
    movies = movies[["MovieID", "Title", "Year"] + genre_cols]

    log(f"movies-1M: split Genres into {max_genres} columns", log_lines)

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
    ratings = ratings[(ratings["UserID"] >= 1) & (ratings["UserID"] <= 71567)]
    ratings = ratings[(ratings["MovieID"] >= 1) & (ratings["MovieID"] <= 65133)]

    ts_min = 631152000
    ts_max = int(time.time())

    ratings = ratings[
        (ratings["Timestamp"] >= ts_min)
        & (ratings["Timestamp"] <= ts_max)
    ]

    log(f"ratings: dropped {before - len(ratings)} rows with invalid values", log_lines)

    ratings["Timestamp"] = pd.to_datetime(ratings["Timestamp"], unit="s")
    ratings = ratings.rename(columns={"Timestamp": "Date"})

    before = len(tags)

    for col in ["UserID", "MovieID", "Timestamp"]:
        tags[col] = pd.to_numeric(tags[col], errors="coerce")

    tags = tags.dropna(subset=["UserID", "MovieID", "Timestamp"])
    tags = tags.astype({
        "UserID": int,
        "MovieID": int,
        "Timestamp": int,
    })

    tags = tags[(tags["UserID"] >= 1) & (tags["UserID"] <= 71567)]
    tags = tags[(tags["MovieID"] >= 1) & (tags["MovieID"] <= 65133)]
    tags = tags[(tags["Timestamp"] >= ts_min) & (tags["Timestamp"] <= ts_max)]

    tag_changed = (~tags["Tag"].map(str.isascii)).sum()
    tags["Tag"] = tags["Tag"].apply(to_ascii).str.strip()
    tags = tags[tags["Tag"] != ""]

    log(f"tags: transliterated {tag_changed} tags to ASCII", log_lines)
    log(f"tags: dropped {before - len(tags)} rows with invalid values or empty tag", log_lines)

    tags["Timestamp"] = pd.to_datetime(tags["Timestamp"], unit="s")
    tags = tags.rename(columns={"Timestamp": "Date"})

    before = len(ratings)
    ratings = ratings[ratings["MovieID"].isin(movies["MovieID"])]
    log(f"ratings: dropped {before - len(ratings)} rows with orphan MovieID", log_lines)

    before = len(tags)
    tags = tags[tags["MovieID"].isin(movies["MovieID"])]
    log(f"tags: dropped {before - len(tags)} rows with orphan MovieID", log_lines)

    before = len(movies)
    movies = movies[movies["MovieID"].isin(ratings["MovieID"])]
    log(f"movies-1M: dropped {before - len(movies)} movies-1M with no ratings", log_lines)

    before = len(tags)
    tags = tags[tags["MovieID"].isin(movies["MovieID"])]
    log(f"tags: dropped {before - len(tags)} additional rows orphaned by movie pruning", log_lines)

    validate_cleaned_data(movies, ratings, tags, log_lines)

    movies.to_csv(out_dir / "movies_clean.csv", index=False)
    ratings.to_csv(out_dir / "movies_ratings_clean.csv", index=False)
    tags.to_csv(out_dir / "movies_tags_clean.csv", index=False)

    with open(out_dir / "movies_cleaning_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    print(f"Saved cleaned CSVs and log to {out_dir}")

    return movies, ratings, tags


def validate_cleaned_data(movies, ratings, tags, log_lines):
    """
    Validate the cleaned MovieLens datasets.

    Checks:
        - Ratings are half-star values between 0.5 and 5.0
        - Ratings and tags only reference existing MovieIDs
        - Genres belong to the allowed genre whitelist
        - Movie years are within a reasonable range
        - Titles, genres, and tags are ASCII-only
        - Each (UserID, MovieID) pair has exactly one rating

    Parameters
    ----------
    movies : pandas.DataFrame
        Cleaned movie metadata.
    ratings : pandas.DataFrame
        Cleaned user ratings.
    tags : pandas.DataFrame
        Cleaned user tags.
    log_lines : list[str]
        Mutable list collecting validation messages.
    """
    bad_ratings = ratings.loc[~ratings["Rating"].between(0.5, 5.0), "Rating"]
    assert bad_ratings.empty

    non_half = ratings.loc[(ratings["Rating"] * 2) % 1 != 0, "Rating"]
    assert non_half.empty

    log(f"PASS: all {len(ratings):,} ratings are half-star in [0.5, 5.0]", log_lines)

    orphan_movies = set(ratings["MovieID"]) - set(movies["MovieID"])
    assert not orphan_movies

    log("PASS: all MovieIDs in ratings exist in movies-1M", log_lines)

    orphan_tag_movies = set(tags["MovieID"]) - set(movies["MovieID"])
    assert not orphan_tag_movies

    log("PASS: all MovieIDs in tags exist in movies-1M", log_lines)

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

    log("PASS: all movies-1M have a valid Year", log_lines)

    bad_title_chars = movies.loc[~movies["Title"].map(str.isascii), "Title"]
    assert bad_title_chars.empty

    for col in genre_cols:
        bad = movies.loc[~movies[col].map(str.isascii), col]
        assert bad.empty

    bad_tag_chars = tags.loc[~tags["Tag"].map(str.isascii), "Tag"]
    assert bad_tag_chars.empty

    log("PASS: all titles, genre columns, and tags use only ASCII characters", log_lines)

    dup_pairs = ratings[ratings.duplicated(subset=["UserID", "MovieID"], keep=False)]
    assert dup_pairs.empty

    log("PASS: every (UserID, MovieID) pair has exactly one rating", log_lines)


def fix_movie_titles_retroactively(
    input_path=OUT_DIR / "movies_clean.csv",
    output_path=OUT_DIR / "movies_clean_fixed.csv",
):
    """
    Fix MovieLens-style titles in an already processed CSV.

    Handles both main-title and parenthetical-title article placement.

    Examples:
        "Matrix, The" -> "The Matrix"
        "Enfer, L'" -> "L'Enfer"
        "Savage Nights (Nuits fauves, Les)"
            -> "Savage Nights (Les Nuits fauves)"

    Reads movies_clean.csv, fixes titles, and saves a new file.
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
    preprocess_movies()
    # fix_movie_titles_retroactively()