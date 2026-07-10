"""
One-time (rerunnable) precompute step: fits the biased SVD on the full local
ratings file and saves the small resulting arrays as .npy files under model/.

This is the only place in this folder that needs the big
dataset_ratings_and_tags.csv (1.2GB, lives at the repo root, never copied into
this folder and never deployed to Vercel). Run this locally whenever the
ratings data changes; the deployed api/recommend.py only ever reads the
committed model/*.npy files plus the bundled dataset.csv.

Usage (from this directory):
    python3 scripts/precompute_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from recommendations_algorithm import load_svd_model  # noqa: E402

REPO_ROOT_RATINGS_PATH = Path(__file__).resolve().parent.parent.parent / "dataset_ratings_and_tags.csv"
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"


def main() -> None:
    ratings_path = REPO_ROOT_RATINGS_PATH
    if not ratings_path.exists():
        print(f"{ratings_path} not found; load_svd_model() will fall back to "
              f"the smaller MovieLens-1M ratings file instead.")

    movie_ids, movie_factors, movie_quality_scores = load_svd_model(ratings_path=ratings_path)

    MODEL_DIR.mkdir(exist_ok=True)
    np.save(MODEL_DIR / "movie_ids.npy", np.asarray(movie_ids, dtype=np.int64))
    np.save(MODEL_DIR / "movie_factors.npy", np.asarray(movie_factors, dtype=np.float32))
    np.save(MODEL_DIR / "movie_quality_scores.npy", np.asarray(movie_quality_scores, dtype=np.float32))

    print(f"Saved model arrays to {MODEL_DIR}/:")
    print(f"  movie_ids:            {np.asarray(movie_ids).shape}")
    print(f"  movie_factors:        {np.asarray(movie_factors).shape}")
    print(f"  movie_quality_scores: {np.asarray(movie_quality_scores).shape}")


if __name__ == "__main__":
    main()
