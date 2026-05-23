import gzip
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests


"""
Downloads English subtitle files from the OpenSubtitles API for movies listed
in cleaned MovieLens datasets.

Expected folder structure:
    datasets/
        movies-1M/
            movies_clean.csv
        movies-32M/
            movies_clean.csv
        opensubtitles/
            opensubtitles_scraper.py
            subs/
            missing_subtitles_1m.txt
            missing_subtitles_32m.txt

Main features:
    - Reads movie titles from cleaned MovieLens CSV files
    - Supports both movies-1M and movies-32M datasets
    - Fixes MovieLens-style titles before searching OpenSubtitles
    - Searches OpenSubtitles by title and year
    - Saves one .srt subtitle file per movie
    - Skips movies that already have subtitle files
    - Logs movies with no subtitle match or failed downloads
"""


BASE_DIR = Path(__file__).resolve().parent

MOVIES_1M_CSV = BASE_DIR.parent / "movies-1M" / "movies_clean.csv"
MOVIES_32M_CSV = BASE_DIR.parent / "movies-32M" / "movies_clean.csv"

# Backward-compatible default.
MOVIES_CSV = MOVIES_1M_CSV

SUBS_DIR = BASE_DIR / "subs"

MISSING_1M_LOG_PATH = BASE_DIR / "missing_subtitles_1m.txt"
MISSING_32M_LOG_PATH = BASE_DIR / "missing_subtitles_32m.txt"

# Backward-compatible default.
MISSING_LOG_PATH = MISSING_1M_LOG_PATH

DEFAULT_BASE_URL = "https://api.opensubtitles.com/api/v1"
USER_AGENT = "movie-recommender-project v1.0"

ARTICLES = r"The|A|An|L'|Le|La|Les"


def scrape_subtitles_for_all_movie_datasets(
    subs_dir=SUBS_DIR,
    language="en",
    delay_seconds=0.5,
    limit=None,
):
    """
    Download subtitles for both MovieLens 1M and MovieLens 32M cleaned datasets.

    Existing subtitle files are skipped, so rerunning this after the 1M scrape
    will mainly download subtitles for 32M movies that are not already present.

    Parameters
    ----------
    subs_dir : str or pathlib.Path
        Directory where subtitle .srt files will be saved.
    language : str
        Subtitle language code. Default is English: "en".
    delay_seconds : float
        Delay between API calls to avoid rate-limit errors.
    limit : int or None
        Optional number of rows to process from each dataset.
    """
    token, api_key, base_url = login()

    scrape_subtitles_for_movies(
        movies_csv=MOVIES_1M_CSV,
        subs_dir=subs_dir,
        language=language,
        delay_seconds=delay_seconds,
        limit=limit,
        missing_log_path=MISSING_1M_LOG_PATH,
        dataset_name="movies-1M",
        credentials=(token, api_key, base_url),
    )

    scrape_subtitles_for_movies(
        movies_csv=MOVIES_32M_CSV,
        subs_dir=subs_dir,
        language=language,
        delay_seconds=delay_seconds,
        limit=limit,
        missing_log_path=MISSING_32M_LOG_PATH,
        dataset_name="movies-32M",
        credentials=(token, api_key, base_url),
    )


def scrape_subtitles_for_1m_movies(
    subs_dir=SUBS_DIR,
    language="en",
    delay_seconds=0.5,
    limit=None,
):
    """
    Download subtitles for the cleaned MovieLens 1M movies dataset.
    """
    return scrape_subtitles_for_movies(
        movies_csv=MOVIES_1M_CSV,
        subs_dir=subs_dir,
        language=language,
        delay_seconds=delay_seconds,
        limit=limit,
        missing_log_path=MISSING_1M_LOG_PATH,
        dataset_name="movies-1M",
    )


def scrape_subtitles_for_32m_movies(
    subs_dir=SUBS_DIR,
    language="en",
    delay_seconds=0.5,
    limit=None,
):
    """
    Download subtitles for the cleaned MovieLens 32M movies dataset.
    """
    return scrape_subtitles_for_movies(
        movies_csv=MOVIES_32M_CSV,
        subs_dir=subs_dir,
        language=language,
        delay_seconds=delay_seconds,
        limit=limit,
        missing_log_path=MISSING_32M_LOG_PATH,
        dataset_name="movies-32M",
    )


def scrape_subtitles_for_movies(
    movies_csv=MOVIES_CSV,
    subs_dir=SUBS_DIR,
    language="en",
    delay_seconds=0.5,
    limit=None,
    missing_log_path=MISSING_LOG_PATH,
    dataset_name=None,
    credentials=None,
):
    """
    Download subtitles for movies listed in a cleaned movies CSV.

    Parameters
    ----------
    movies_csv : str or pathlib.Path
        Path to the cleaned movies CSV. Must contain MovieID, Title, and Year.
    subs_dir : str or pathlib.Path
        Directory where subtitle .srt files will be saved.
    language : str
        Subtitle language code. Default is English: "en".
    delay_seconds : float
        Delay between API calls to avoid rate-limit errors.
    limit : int or None
        Optional number of movies to process, useful for testing.
    missing_log_path : str or pathlib.Path
        Path to the missing/failed subtitle log.
    dataset_name : str or None
        Optional dataset label saved in the missing subtitle log.
    credentials : tuple[str, str, str] or None
        Optional precomputed (token, api_key, base_url). This avoids logging in
        twice when scraping both 1M and 32M datasets.

    Notes
    -----
    Movies are added to the missing subtitles log when:
        - OpenSubtitles returns no matching subtitle.
        - Search or download fails.
    """
    movies_csv = Path(movies_csv)
    subs_dir = Path(subs_dir)
    subs_dir.mkdir(parents=True, exist_ok=True)

    if not movies_csv.exists():
        raise FileNotFoundError(f"{movies_csv} not found")

    movies = pd.read_csv(movies_csv)

    required_cols = {"MovieID", "Title", "Year"}
    missing_cols = required_cols - set(movies.columns)

    if missing_cols:
        raise ValueError(
            f"{movies_csv} is missing columns: {missing_cols}. "
            f"Available columns: {list(movies.columns)}"
        )

    if limit is not None:
        movies = movies.head(limit)

    if credentials is None:
        token, api_key, base_url = login()
    else:
        token, api_key, base_url = credentials

    missing_titles = []

    print(f"Scraping subtitles from {movies_csv}")
    print(f"Movies to check: {len(movies):,}")

    for _, movie in movies.iterrows():
        movie_id = movie["MovieID"]
        title = fix_movie_title(movie["Title"])
        year = int(movie["Year"]) if not pd.isna(movie["Year"]) else None

        output_path = subs_dir / make_subtitle_filename(movie_id, title, year)

        if subtitle_file_exists_for_movie(subs_dir, movie_id):
            print(f"Skipping existing subtitle: {title} ({year})")
            continue

        try:
            subtitle = find_best_subtitle(
                title=title,
                year=year,
                language=language,
                token=token,
                api_key=api_key,
                base_url=base_url,
            )

            if subtitle is None:
                print(f"No subtitles found: {title} ({year})")
                missing_titles.append(
                    format_missing_entry(
                        movie_id=movie_id,
                        title=title,
                        year=year,
                        reason="no_subtitles_found",
                        dataset_name=dataset_name,
                    )
                )
                continue

            download_subtitle(
                subtitle=subtitle,
                output_path=output_path,
                token=token,
                api_key=api_key,
                base_url=base_url,
            )

            print(f"Saved: {output_path}")

        except Exception as e:
            print(f"Failed: {title} ({year}) | {e}")
            missing_titles.append(
                format_missing_entry(
                    movie_id=movie_id,
                    title=title,
                    year=year,
                    reason=f"download_failed: {e}",
                    dataset_name=dataset_name,
                )
            )

        if delay_seconds and delay_seconds > 0:
            time.sleep(delay_seconds)

    save_missing_titles(missing_titles, output_path=missing_log_path)

    return missing_titles


def login():
    """
    Authenticate with OpenSubtitles.

    Authentication priority:
        1. Environment variables:
           OPENSUBTITLES_TOKEN + OPENSUBTITLES_API_KEY

        2. Command line:
           python opensubtitles_scraper.py TOKEN API_KEY

        3. Environment variables:
           OPENSUBTITLES_USERNAME + OPENSUBTITLES_PASSWORD + OPENSUBTITLES_API_KEY

        4. Command line:
           python opensubtitles_scraper.py USERNAME PASSWORD API_KEY

    Returns
    -------
    tuple[str, str, str]
        token, api_key, base_url
    """
    token = os.getenv("OPENSUBTITLES_TOKEN")
    api_key = os.getenv("OPENSUBTITLES_API_KEY")

    if token and api_key:
        return token, api_key, DEFAULT_BASE_URL

    if len(sys.argv) == 3:
        token = sys.argv[1]
        api_key = sys.argv[2]
        return token, api_key, DEFAULT_BASE_URL

    username = os.getenv("OPENSUBTITLES_USERNAME")
    password = os.getenv("OPENSUBTITLES_PASSWORD")

    if username and password and api_key:
        return login_with_username_password(username, password, api_key)

    if len(sys.argv) == 4:
        username = sys.argv[1]
        password = sys.argv[2]
        api_key = sys.argv[3]
        return login_with_username_password(username, password, api_key)

    raise ValueError(
        "Missing OpenSubtitles credentials.\n\n"
        "Use one of these:\n"
        "1) export OPENSUBTITLES_TOKEN='...'\n"
        "   export OPENSUBTITLES_API_KEY='...'\n\n"
        "2) python opensubtitles_scraper.py TOKEN API_KEY\n\n"
        "3) export OPENSUBTITLES_USERNAME='...'\n"
        "   export OPENSUBTITLES_PASSWORD='...'\n"
        "   export OPENSUBTITLES_API_KEY='...'\n\n"
        "4) python opensubtitles_scraper.py USERNAME PASSWORD API_KEY"
    )


def login_with_username_password(username, password, api_key):
    """
    Log in to OpenSubtitles using username, password, and API key.
    """
    response = requests.post(
        f"{DEFAULT_BASE_URL}/login",
        headers={
            "Api-Key": api_key,
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
        json={
            "username": username,
            "password": password,
        },
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Login failed: {response.status_code}\n{response.text}"
        )

    data = response.json()
    token = data["token"]

    return token, api_key, DEFAULT_BASE_URL


def find_best_subtitle(title, year, language, token, api_key, base_url):
    """
    Search OpenSubtitles and return the best subtitle result.

    Results are ordered by download count, descending.

    Very short titles such as "M", "Pi", "Go", "It", or "Us" can make the
    OpenSubtitles API reject the request because the query is too short. In
    that case, this function retries with safer fallback queries that include
    the year and/or the word "movie".
    """
    last_error = None

    for query in build_subtitle_search_queries(title, year):
        try:
            results = search_subtitles_once(
                query=query,
                year=year,
                language=language,
                token=token,
                api_key=api_key,
                base_url=base_url,
            )

            if results:
                return results[0]

        except RuntimeError as e:
            last_error = e

            if is_query_too_short_error(e):
                print(f"Query too short, retrying with fallback: {query}")
                continue

            raise

    if last_error and is_query_too_short_error(last_error):
        return None

    return None


def search_subtitles_once(query, year, language, token, api_key, base_url):
    """
    Run one OpenSubtitles search request.
    """
    params = {
        "query": query,
        "languages": language,
        "type": "movie",
        "order_by": "download_count",
        "order_direction": "desc",
    }

    if year is not None:
        params["year"] = year

    response = requests.get(
        f"{base_url}/subtitles",
        headers=auth_headers(api_key, token),
        params=params,
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Subtitle search failed: {response.status_code}\n{response.text}"
        )

    return response.json().get("data", [])


def build_subtitle_search_queries(title, year=None):
    """
    Build OpenSubtitles search queries, including fallbacks for short titles.

    OpenSubtitles may reject very short queries. For those, appending the year
    usually makes the query long enough while still keeping it specific.

    Examples
    --------
    "M", 1931 -> ["M 1931", "M movie 1931", "M"]
    "Pi", 1998 -> ["Pi 1998", "Pi movie 1998", "Pi"]
    "Toy Story", 1995 -> ["Toy Story", "Toy Story 1995"]
    """
    title = str(title).strip()
    normalized_title = normalize_search_query(title)

    queries = []

    if is_short_search_query(normalized_title):
        if year is not None:
            queries.extend([
                f"{title} {year}",
                f"{title} movie {year}",
            ])

        queries.append(title)
    else:
        queries.append(title)

        title_without_parentheses = remove_parenthetical_text(title)

        if title_without_parentheses and title_without_parentheses != title:
            queries.append(title_without_parentheses)

        if year is not None:
            queries.append(f"{title} {year}")

    return unique_non_empty_strings(queries)


def normalize_search_query(text):
    """
    Normalize a search query for length checks.
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def is_short_search_query(query):
    """
    Return True when a query is likely too short for OpenSubtitles.
    """
    return len(normalize_search_query(query)) < 3


def is_query_too_short_error(error):
    """
    Detect OpenSubtitles errors caused by a too-short search query.
    """
    message = str(error).lower()

    return (
        "query is too short" in message
        or "query must be at least" in message
        or "query length" in message
    )


def remove_parenthetical_text(title):
    """
    Remove parenthetical alternate titles from a movie title.

    Example:
        "Seven (a.k.a. Se7en)" -> "Seven"
    """
    title = str(title).strip()
    title = re.sub(r"\s*\([^)]*\)\s*", " ", title)
    title = re.sub(r"\s+", " ", title)
    return title.strip()


def unique_non_empty_strings(values):
    """
    Return unique non-empty strings while preserving order.
    """
    seen = set()
    result = []

    for value in values:
        value = str(value).strip()

        if not value:
            continue

        key = value.lower()

        if key in seen:
            continue

        seen.add(key)
        result.append(value)

    return result


def download_subtitle(subtitle, output_path, token, api_key, base_url):
    """
    Download one subtitle file and save it to disk.
    """
    files = subtitle["attributes"].get("files", [])

    if not files:
        raise ValueError("Subtitle result has no downloadable files.")

    file_id = files[0]["file_id"]

    response = requests.post(
        f"{base_url}/download",
        headers={
            **auth_headers(api_key, token),
            "Content-Type": "application/json",
        },
        json={"file_id": file_id},
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Subtitle download request failed: {response.status_code}\n"
            f"{response.text}"
        )

    download_link = response.json()["link"]

    subtitle_response = requests.get(download_link, timeout=60)

    if subtitle_response.status_code != 200:
        raise RuntimeError(
            f"Subtitle file download failed: {subtitle_response.status_code}\n"
            f"{subtitle_response.text}"
        )

    content = subtitle_response.content

    if content[:2] == b"\x1f\x8b":
        content = gzip.decompress(content)

    output_path.write_bytes(content)


def auth_headers(api_key, token=None):
    """
    Build request headers for OpenSubtitles API calls.
    """
    headers = {
        "Api-Key": api_key,
        "User-Agent": USER_AGENT,
    }

    if token:
        headers["Authorization"] = f"Bearer {token}"

    return headers


def make_subtitle_filename(movie_id, title, year):
    """
    Create a safe subtitle filename.

    Returns
    -------
    str
        Filename in the format MovieID_title_year.srt.
    """
    clean_title = safe_filename(title)
    year_part = str(year) if year is not None else "unknown"
    return f"{movie_id}_{clean_title}_{year_part}.srt"


def subtitle_file_exists_for_movie(subs_dir, movie_id):
    """
    Check whether a subtitle file already exists for a movie.

    This uses MovieID as the stable identifier, so the movie is skipped even if
    the saved title text differs slightly from the current cleaned title.
    """
    subs_dir = Path(subs_dir)
    return any(subs_dir.glob(f"{movie_id}_*.srt"))


def safe_filename(text):
    """
    Convert a title to a filesystem-safe lowercase filename component.
    """
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")[:120]


def format_missing_entry(movie_id, title, year, reason, dataset_name=None):
    """
    Format one missing-subtitle log line.
    """
    year_part = str(year) if year is not None else "unknown"

    if dataset_name:
        return f"{dataset_name}\t{movie_id}\t{title}\t{year_part}\t{reason}"

    return f"{movie_id}\t{title}\t{year_part}\t{reason}"


def save_missing_titles(missing_titles, output_path=MISSING_LOG_PATH):
    """
    Save missing or failed subtitle downloads to a text file.
    """
    if not missing_titles:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    has_dataset_column = missing_titles[0].count("\t") == 4

    with open(output_path, "w", encoding="utf-8") as f:
        if has_dataset_column:
            f.write("Dataset\tMovieID\tTitle\tYear\tReason\n")
        else:
            f.write("MovieID\tTitle\tYear\tReason\n")

        for title in missing_titles:
            f.write(title + "\n")

    print(f"\nSaved missing titles to {output_path}")


def _move_trailing_article(text):
    """
    Move trailing articles from the end of a title to the beginning.
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
    Fix MovieLens-style title formatting.

    Handles article placement in both main titles and parenthetical alternate
    titles.
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


if __name__ == "__main__":
    # Scrape only the new 32M dataset by default.
    # Existing 1M subtitles in subs/ are skipped automatically if you call
    # scrape_subtitles_for_all_movie_datasets() instead.
    scrape_subtitles_for_32m_movies(
        language="en",
        delay_seconds=1.1,
        limit=None,
    )
