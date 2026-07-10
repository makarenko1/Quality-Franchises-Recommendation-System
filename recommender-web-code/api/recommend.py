"""
Vercel Python serverless function backing the recommender frontend.

Loads dataset.csv (movies/dialogue/franchise/installment data, bundled in this
folder) and the precomputed SVD arrays (model/*.npy, generated offline by
scripts/precompute_model.py) once at module import, so warm invocations reuse
them instead of reloading on every request. Exposes POST /api/recommend,
taking {"titles": [t1, t2, t3]} and returning the same JSON-shaped result list
that recommendations_algorithm.recommend() already produces, unmodified.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from recommendations_algorithm import (  # noqa: E402
    load_dialogue_features,
    load_franchise_map,
    load_installment_rating_trend,
    load_movies_metadata,
    recommend,
)

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "dataset.csv"
MODEL_DIR = ROOT / "model"

app = Flask(__name__)

# ── Loaded once per cold start, reused across warm invocations ──────────────
_movies = load_movies_metadata(DATASET_PATH)
_dq_map = load_dialogue_features(DATASET_PATH)
_franchise_map = load_franchise_map(DATASET_PATH)
_installment_trend = load_installment_rating_trend(DATASET_PATH)
_movie_ids = np.load(MODEL_DIR / "movie_ids.npy")
_movie_factors = np.load(MODEL_DIR / "movie_factors.npy")
_movie_quality_scores = np.load(MODEL_DIR / "movie_quality_scores.npy")


def _handle_recommend():
    body = request.get_json(silent=True) or {}
    titles = body.get("titles")
    if not isinstance(titles, list) or not (1 <= len(titles) <= 3):
        return jsonify({"error": "Expected JSON body {'titles': [1 to 3 movie titles]}"}), 400

    results = recommend(
        titles,
        _movies,
        _movie_ids,
        _movie_factors,
        _movie_quality_scores,
        _dq_map,
        _franchise_map,
        _installment_trend,
        n=3,
    )
    return jsonify({"results": results})


@app.route("/api/recommend", methods=["POST"])
def recommend_route():
    return _handle_recommend()


# Fallback: some Vercel Python runtime versions forward the request to "/"
# rather than the full /api/recommend path. Register both to be safe.
# GET is intentionally not handled here so a local dev server can serve
# public/index.html at "/" without a route conflict (see local_dev_server.py).
@app.route("/", methods=["POST"])
def root_route():
    return _handle_recommend()
