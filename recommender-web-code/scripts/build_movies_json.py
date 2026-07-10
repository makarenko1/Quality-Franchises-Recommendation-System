"""
Builds public/movies.json: a trimmed copy of dataset.csv (MovieID, Title,
Year, Genre1-8, hasFactors) for the frontend's client-side movie search.
Rerun whenever dataset.csv or model/movie_ids.npy change.

Usage (from this directory):
    python3 scripts/build_movies_json.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "dataset.csv"
MODEL_DIR = ROOT / "model"
OUT_PATH = ROOT / "public" / "movies.json"

GENRE_COLS = [f"Genre{i}" for i in range(1, 9)]


def main() -> None:
    df = pd.read_csv(DATASET_PATH, usecols=["MovieID", "Title", "Year"] + GENRE_COLS, low_memory=False)
    df = df.dropna(subset=["MovieID", "Title"]).copy()
    df["MovieID"] = df["MovieID"].astype(int)

    movie_ids_with_factors = set(int(mid) for mid in np.load(MODEL_DIR / "movie_ids.npy"))

    records = []
    for row in df.itertuples(index=False):
        genres = [g for g in (getattr(row, c) for c in GENRE_COLS) if isinstance(g, str) and g]
        records.append({
            "id": int(row.MovieID),
            "title": str(row.Title),
            "year": int(row.Year) if pd.notna(row.Year) else None,
            "genres": genres,
            "hasFactors": int(row.MovieID) in movie_ids_with_factors,
        })

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(records, f, separators=(",", ":"))

    print(f"Wrote {len(records):,} movies to {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
