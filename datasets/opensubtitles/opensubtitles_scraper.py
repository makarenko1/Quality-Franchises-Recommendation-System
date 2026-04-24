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
in the cleaned MovieLens dataset.

Expected folder structure:
    datasets/
        movies/
            movies_clean.csv
        opensubtitles/
            opensubtitles_scraper.py
            subs/
            missing_subtitles.txt

Main features:
    - Reads movie titles from movies_clean.csv
    - Fixes MovieLens-style titles before searching OpenSubtitles
    - Searches OpenSubtitles by title and year
    - Saves one .srt subtitle file per movie
    - Skips movies that already have subtitle files
    - Logs movies with no subtitle match or failed downloads
"""


BASE_DIR = Path(__file__).resolve().parent

MOVIES_CSV = BASE_DIR.parent / "movies" / "movies_clean.csv"
MOVIES_CSV_FIXED = BASE_DIR.parent / "movies" / "movies_clean_fixed.csv"
SUBS_DIR = BASE_DIR / "subs"
MISSING_LOG_PATH = BASE_DIR / "missing_subtitles.txt"

DEFAULT_BASE_URL = "https://api.opensubtitles.com/api/v1"
USER_AGENT = "movie-recommender-project v1.0"

ARTICLES = r"The|A|An|L'|Le|La|Les"


def scrape_subtitles_for_movies(
    movies_csv=MOVIES_CSV,
    subs_dir=SUBS_DIR,
    language="en",
    delay_seconds=1.1,
    limit=None,
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

    Notes
    -----
    Movies are added to missing_subtitles.txt when:
        - OpenSubtitles returns no matching subtitle.
        - Search or download fails.
    """
    subs_dir = Path(subs_dir)
    subs_dir.mkdir(parents=True, exist_ok=True)

    movies = pd.read_csv(movies_csv)

    if limit is not None:
        movies = movies.head(limit)

    token, api_key, base_url = login()

    missing_titles = []

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
                    format_missing_entry(movie_id, title, year, "no_subtitles_found")
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
                format_missing_entry(movie_id, title, year, f"download_failed: {e}")
            )

        time.sleep(delay_seconds)

    save_missing_titles(missing_titles)


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

    Parameters
    ----------
    username : str
        OpenSubtitles username.
    password : str
        OpenSubtitles password.
    api_key : str
        OpenSubtitles API key.

    Returns
    -------
    tuple[str, str, str]
        token, api_key, base_url
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

    Parameters
    ----------
    title : str
        Movie title.
    year : int or None
        Movie release year.
    language : str
        Subtitle language code.
    token : str
        OpenSubtitles bearer token.
    api_key : str
        OpenSubtitles API key.
    base_url : str
        OpenSubtitles API base URL.

    Returns
    -------
    dict or None
        Best subtitle result, or None if no result was found.
    """
    params = {
        "query": title,
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

    results = response.json().get("data", [])

    if not results:
        return None

    return results[0]


def download_subtitle(subtitle, output_path, token, api_key, base_url):
    """
    Download one subtitle file and save it to disk.

    Parameters
    ----------
    subtitle : dict
        Subtitle result returned by find_best_subtitle.
    output_path : str or pathlib.Path
        Local .srt path where the subtitle should be saved.
    token : str
        OpenSubtitles bearer token.
    api_key : str
        OpenSubtitles API key.
    base_url : str
        OpenSubtitles API base URL.
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

    Parameters
    ----------
    api_key : str
        OpenSubtitles API key.
    token : str or None
        Optional bearer token.

    Returns
    -------
    dict
        HTTP headers.
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

    Parameters
    ----------
    movie_id : int or str
        MovieLens movie ID.
    title : str
        Movie title.
    year : int or None
        Movie release year.

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

    Parameters
    ----------
    subs_dir : str or pathlib.Path
        Subtitle folder.
    movie_id : int or str
        MovieLens movie ID.

    Returns
    -------
    bool
        True if a matching .srt file already exists, otherwise False.
    """
    subs_dir = Path(subs_dir)
    return any(subs_dir.glob(f"{movie_id}_*.srt"))


def safe_filename(text):
    """
    Convert a title to a filesystem-safe lowercase filename component.

    Parameters
    ----------
    text : str
        Raw text.

    Returns
    -------
    str
        Safe filename fragment.
    """
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")[:120]


def format_missing_entry(movie_id, title, year, reason):
    """
    Format one missing-subtitle log line.

    Parameters
    ----------
    movie_id : int or str
        MovieLens movie ID.
    title : str
        Movie title.
    year : int or None
        Movie release year.
    reason : str
        Reason the subtitle was not downloaded.

    Returns
    -------
    str
        Tab-separated log entry.
    """
    year_part = str(year) if year is not None else "unknown"
    return f"{movie_id}\t{title}\t{year_part}\t{reason}"


def save_missing_titles(missing_titles, output_path=MISSING_LOG_PATH):
    """
    Save missing or failed subtitle downloads to a text file.

    Parameters
    ----------
    missing_titles : list[str]
        Missing subtitle log entries.
    output_path : str or pathlib.Path
        Output text file path.
    """
    if not missing_titles:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("MovieID\tTitle\tYear\tReason\n")
        for title in missing_titles:
            f.write(title + "\n")

    print(f"\nSaved missing titles to {output_path}")


def _move_trailing_article(text):
    """
    Move trailing articles from the end of a title to the beginning.

    Supported articles:
        - English: The, A, An
        - French: L', Le, La, Les

    Examples
    --------
    "Matrix, The" -> "The Matrix"
    "Enfer, L'" -> "L'Enfer"
    "Nuits fauves, Les" -> "Les Nuits fauves"

    Parameters
    ----------
    text : str
        Input title string.

    Returns
    -------
    str
        Fixed title if a trailing article is found; otherwise unchanged.
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

    Examples
    --------
    "Matrix, The" -> "The Matrix"
    "Enfer, L'" -> "L'Enfer"
    "Savage Nights (Nuits fauves, Les)" -> "Savage Nights (Les Nuits fauves)"

    Parameters
    ----------
    title : str
        Movie title.

    Returns
    -------
    str
        Fixed title.
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
    scrape_subtitles_for_movies(
        movies_csv=MOVIES_CSV,
        subs_dir=SUBS_DIR,
        language="en",
        delay_seconds=1.1,
        limit=None,
    )