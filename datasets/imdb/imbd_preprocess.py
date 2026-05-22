import unicodedata
import datetime
import os
from pathlib import Path

import pandas as pd
import numpy as np

# paths
DATA_DIR = Path("../movies")
OUTPUT_MOVIES  = DATA_DIR / "imdb_movies_clean.csv"
OUTPUT_RATINGS = DATA_DIR / "imdb_ratings_clean.csv"
OUTPUT_LOG     = DATA_DIR / "imdb_cleaning_log.txt"

MIN_VOTES    = 10
CURRENT_YEAR = datetime.date.today().year

# imdb uses \N for missing values, need to tell pandas about it
# genres that are actually valid according to imdb
VALID_GENRES = {
    "Action", "Adult", "Adventure", "Animation", "Biography", "Comedy",
    "Crime", "Documentary", "Drama", "Family", "Fantasy", "Film-Noir",
    "Game-Show", "History", "Horror", "Music", "Musical", "Mystery",
    "News", "Reality-TV", "Romance", "Sci-Fi", "Short", "Sport",
    "Talk-Show", "Thriller", "War", "Western",
}

# special characters that need manual handling before normalization
LIGATURES = {
    "æ": "ae", "Æ": "AE",
    "ø": "o",  "Ø": "O",
    "ß": "ss",
    "œ": "oe", "Œ": "OE",
    "đ": "d",  "Đ": "D",
}


def to_ascii(value):
    # convert accented characters to ascii, e.g. Amélie -> Amelie
    value = "".join(LIGATURES.get(c, c) for c in str(value))
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")


def log(message, log_lines):
    print(message)
    log_lines.append(message)


def read_imdb_file(path):
    return pd.read_csv(path, sep="\t", na_values=r"\N", low_memory=False)


def show_info(df, name, log_lines):
    null_pct = (df.isna().sum() / len(df) * 100).round(1)
    info = pd.DataFrame({
        "dtype":  df.dtypes,
        "null_%": null_pct,
        "sample": [df[c].dropna().iloc[0] if df[c].notna().any() else "N/A" for c in df.columns],
    })
    log(f"\n{name} - {df.shape[0]:,} rows x {df.shape[1]} cols\n{info.to_string()}", log_lines)


def validate(movies, ratings, log_lines):
    log("\n--- running checks ---", log_lines)

    assert movies["tconst"].duplicated().sum() == 0
    log(f"OK: no duplicate movies ({len(movies):,} rows)", log_lines)

    assert ratings["tconst"].duplicated().sum() == 0
    log(f"OK: no duplicate ratings ({len(ratings):,} rows)", log_lines)

    assert (~ratings["averageRating"].between(1.0, 10.0)).sum() == 0
    log("OK: all ratings between 1 and 10", log_lines)

    assert (ratings["numVotes"] < 0).sum() == 0
    log("OK: no negative vote counts", log_lines)

    # check that bad years got flagged
    valid_years = movies["Year"].dropna()
    bad = valid_years[(valid_years < 1880) | (valid_years > CURRENT_YEAR + 3)]
    assert len(bad[~movies.loc[bad.index, "year_flag"]]) == 0
    log("OK: bad years are flagged", log_lines)

    # check genre_flag makes sense
    genre_cols = [c for c in movies.columns if c.startswith("Genre")]
    for _, row in movies[movies["genre_flag"]].iterrows():
        tokens = [row[c] for c in genre_cols if pd.notna(row[c]) and row[c] != ""]
        assert any(t not in VALID_GENRES for t in tokens)
    log(f"OK: genre_flag looks right", log_lines)

    assert (~movies["Title"].apply(str.isascii)).sum() == 0
    log("OK: titles are ascii", log_lines)

    orphans = set(ratings["tconst"]) - set(movies["tconst"])
    assert not orphans
    log("OK: all ratings match a movie", log_lines)

    log("--- all checks passed ---", log_lines)


def preprocess_imdb(basics_path, ratings_path, min_votes=MIN_VOTES):
    log_lines = []

    # load
    log("loading files...", log_lines)
    basics  = read_imdb_file(basics_path)
    ratings = read_imdb_file(ratings_path)
    show_info(basics,  "title.basics (raw)", log_lines)
    show_info(ratings, "title.ratings (raw)", log_lines)
    log(f"loaded {len(basics):,} rows in basics, {len(ratings):,} in ratings", log_lines)

    # drop duplicates
    log("\nremoving duplicates...", log_lines)
    b0, r0 = len(basics), len(ratings)
    basics  = basics.drop_duplicates(subset="tconst", keep="first")
    ratings = ratings.drop_duplicates(subset="tconst", keep="first")
    log(f"  removed {b0 - len(basics):,} from basics, {r0 - len(ratings):,} from ratings", log_lines)

    # keep only movies
    log("\nkeeping only movies...", log_lines)
    b0 = len(basics)
    basics = basics[basics["titleType"] == "movie"].copy()
    log(f"  kept {len(basics):,} movies, dropped {b0 - len(basics):,} other rows", log_lines)

    # fix types
    log("\nfixing column types...", log_lines)
    basics["isAdult"]        = pd.to_numeric(basics["isAdult"], errors="coerce").fillna(0).astype(bool)
    basics["startYear"]      = pd.to_numeric(basics["startYear"], errors="coerce").astype("Int32")
    basics["runtimeMinutes"] = pd.to_numeric(basics["runtimeMinutes"], errors="coerce").astype("Int32")
    ratings["averageRating"] = pd.to_numeric(ratings["averageRating"], errors="coerce")
    ratings["numVotes"]      = pd.to_numeric(ratings["numVotes"], errors="coerce").astype("Int64")

    # flag weird years (before cinema existed or too far in future)
    log("\nflagging bad values...", log_lines)
    year_issues = basics["startYear"].notna() & (
        (basics["startYear"] < 1880) | (basics["startYear"] >= CURRENT_YEAR + 3)
    )
    basics["year_flag"] = year_issues
    log(f"  {year_issues.sum():,} rows with suspicious year", log_lines)

    # flag weird runtimes
    runtime_issues = basics["runtimeMinutes"].notna() & (
        (basics["runtimeMinutes"] < 1) | (basics["runtimeMinutes"] > 1440)
    )
    basics["runtime_flag"] = runtime_issues
    log(f"  {runtime_issues.sum():,} rows with suspicious runtime", log_lines)

    # fix bad ratings
    bad_ratings = ratings["averageRating"].notna() & ~ratings["averageRating"].between(1.0, 10.0)
    ratings.loc[bad_ratings, "averageRating"] = pd.NA
    log(f"  {bad_ratings.sum():,} ratings out of range, set to NA", log_lines)

    # convert titles to ascii
    log("\nconverting titles to ascii...", log_lines)
    n1 = (~basics["primaryTitle"].dropna().map(str.isascii)).sum()
    n2 = (~basics["originalTitle"].dropna().map(str.isascii)).sum()
    basics["Title"]               = basics["primaryTitle"].fillna("").apply(to_ascii)
    basics["originalTitle_ascii"] = basics["originalTitle"].fillna("").apply(to_ascii)
    log(f"  converted {n1:,} primary titles, {n2:,} original titles", log_lines)

    # split genres into separate columns + flag unknown ones
    log("\nprocessing genres...", log_lines)
    genres_split = basics["genres"].fillna("").apply(
        lambda s: [g for g in s.split(",") if g] if s else []
    )
    basics["genre_flag"] = genres_split.apply(
        lambda lst: bool(lst) and any(t not in VALID_GENRES for t in lst)
    )
    log(f"  {basics['genre_flag'].sum():,} rows with unknown genre", log_lines)

    max_genres = genres_split.map(len).max()
    genre_cols = [f"Genre{i+1}" for i in range(max_genres)]
    genre_df   = genres_split.apply(lambda lst: pd.Series(lst)).reindex(columns=range(max_genres))
    genre_df.columns = genre_cols
    genre_df = genre_df.replace("", np.nan)
    basics = pd.concat([basics, genre_df], axis=1)
    log(f"  split into {max_genres} genre columns", log_lines)

    # build final output tables
    log("\nbuilding output tables...", log_lines)
    movie_cols = ["tconst", "Title", "originalTitle_ascii", "isAdult", "startYear",
                  "runtimeMinutes"] + genre_cols + ["year_flag", "runtime_flag", "genre_flag"]
    movies_out  = basics[movie_cols].rename(columns={"startYear": "Year"})
    ratings_out = ratings[ratings["tconst"].isin(movies_out["tconst"])].copy()
    ratings_out["low_votes"] = ratings_out["numVotes"] < min_votes

    log(f"  movies: {len(movies_out):,} rows", log_lines)
    log(f"  ratings: {len(ratings_out):,} rows", log_lines)
    show_info(movies_out,  "imdb_movies_clean", log_lines)
    show_info(ratings_out, "imdb_ratings_clean", log_lines)

    validate(movies_out, ratings_out, log_lines)

    return movies_out, ratings_out, log_lines


def main():
    basics_path  = DATA_DIR / "title.basics.tsv.gz"
    ratings_path = DATA_DIR / "title.ratings.tsv.gz"

    if not basics_path.exists():
        basics_path = DATA_DIR / "title.basics.tsv"
    if not ratings_path.exists():
        ratings_path = DATA_DIR / "title.ratings.tsv"

    if not basics_path.exists() or os.path.getsize(basics_path) == 0:
        raise FileNotFoundError(
            "title.basics not found - download from https://datasets.imdbws.com/title.basics.tsv.gz"
        )
    if not ratings_path.exists() or os.path.getsize(ratings_path) == 0:
        raise FileNotFoundError(
            "title.ratings not found - download from https://datasets.imdbws.com/title.ratings.tsv.gz"
        )

    movies, ratings, log_lines = preprocess_imdb(basics_path, ratings_path)

    movies.to_csv(OUTPUT_MOVIES, index=False)
    print(f"saved {len(movies):,} rows to {OUTPUT_MOVIES.name}")

    ratings.to_csv(OUTPUT_RATINGS, index=False)
    print(f"saved {len(ratings):,} rows to {OUTPUT_RATINGS.name}")

    OUTPUT_LOG.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"log saved to {OUTPUT_LOG.name}")


if __name__ == "__main__":
    main()
