# Movie Recommender — Vercel web app

A static frontend + Python API version of the movie recommender, deployable
free on Vercel. Replaces the old Streamlit `app.py` (which can't run on
Vercel — its serverless functions are short-lived and stateless, while
Streamlit needs a persistent process holding a WebSocket per session).

## Why this folder is self-contained

Vercel only uploads the contents of the configured project root, so this
folder carries its own copy of `recommendations_algorithm.py` and
`dataset.csv` rather than reaching out to the repo root.

`load_svd_model()` normally needs `dataset_ratings_and_tags.csv` (1.2GB,
gitignored at the repo root — too large to bundle or fit live on every cold
start). Instead, the SVD factors are **precomputed once, locally**, and only
the small resulting arrays (`model/*.npy`, ~15MB total) are committed.
Everything else the recommender needs (movies, dialogue quality, franchise
map, installment trend) reads from the bundled `dataset.csv` (53MB, already
tracked in git) and runs unmodified at request time.

## Structure

```
recommender-web-code/
├── vercel.json                  # { outputDirectory: "public", framework: null }
├── requirements.txt             # flask, pandas, numpy
├── recommendations_algorithm.py # copy of the repo-root module
├── dataset.csv                  # copy of the repo-root dataset
├── model/                       # precomputed SVD arrays (movie_ids/movie_factors/movie_quality_scores)
├── scripts/
│   ├── precompute_model.py      # rerun locally if the ratings data changes
│   ├── build_movies_json.py     # rerun locally if dataset.csv or model/ changes
│   └── local_dev_server.py      # local-only: serves public/ + /api/recommend on one port
├── api/
│   └── recommend.py             # Flask app; POST /api/recommend {"titles": [...]} -> {"results": [...]}
└── public/
    ├── index.html / style.css / app.js   # ported from app.py's UI and design
    └── movies.json                       # trimmed movie list for client-side search
```

## Local development

```bash
cd recommender-web-code
pip install -r requirements.txt
python3 scripts/local_dev_server.py   # http://localhost:5055
```

## Regenerating the bundled data

Only needed if `dataset.csv` or the ratings data changes upstream:

```bash
cd recommender-web-code
python3 scripts/precompute_model.py    # needs ../dataset_ratings_and_tags.csv locally
python3 scripts/build_movies_json.py
```

## Deploying

1. Push this repo to GitHub (already done if you're reading this from a clone).
2. Create a free Vercel account (Hobby tier, no credit card required) at
   [vercel.com](https://vercel.com).
3. "New Project" → import this repo → set **Root Directory** to
   `recommender-web-code/` → Deploy.

That's it — Vercel installs from `requirements.txt`, serves `public/` as
static files, and runs `api/recommend.py` as a Python serverless function.
