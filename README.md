# Quality-Aware Franchise Movie Recommender

A movie recommendation system that combines collaborative filtering with **dialogue quality signals** extracted from subtitles and **franchise awareness** to surface films that are both a good personal fit and maintain high creative quality.

---

## How It Works

Recommendations are scored from three signals (weights sum to 1.0):

| Signal | Weight | Source |
|--------|--------|--------|
| Collaborative filtering (SVD) | 60% | MovieLens user ratings |
| Dialogue quality similarity | 25% | OpenSubtitles `.srt` files |
| Franchise quality continuity | 15% | Franchise metadata + dialogue |

**Dialogue quality** is measured by vocabulary richness, repetition patterns, sentiment, readability, and other features extracted from subtitle files. Within a franchise, a sequel is only recommended if its dialogue quality does not fall significantly below the films the user already likes.

---

## Project Structure

```
.
├── main.py                        # Data pipeline: builds dataset.csv
├── recommendations_algorithm.py   # SVD + dialogue + franchise scoring
├── baselines.py                   # Popular / highest-rated / random baselines
├── app.py                         # Streamlit web UI
├── analyze_data.py                # EDA and correlation analysis
├── dataset.csv                    # Movie-level feature dataset (built by main.py)
├── setup.sh                       # One-command repo setup
│
└── datasets/
    ├── movies-1M/                 # MovieLens 1M (movies, ratings, tags)
    ├── movies-32M/                # MovieLens 32M (movies, ratings, tags)
    ├── imdb/                      # IMDb title basics + ratings
    ├── opensubtitles/subs/        # Downloaded .srt subtitle files
    └── franchises/franchises.csv  # Franchise membership + installment order
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd Quality-Franchises-Recommendation-System
pip install -r requirements.txt   # pandas, numpy, scipy, streamlit, nltk, gdown
```

### 2. Download large data files and update the repo

```bash
./setup.sh
```

This script:
- Runs `git fetch` and `git pull`
- Downloads the large dataset archive from Google Drive via `gdown`
- Unpacks it into the correct directory structure

The archive covers all gitignored large files:
- `datasets/imdb/raw/` — IMDb title basics and ratings (`.tsv.gz`)
- `datasets/movies-32M/raw/` — MovieLens 32M raw ratings
- `datasets/movies-32M/movies_ratings_clean.csv` and `movies_tags_clean.csv`
- `datasets/opensubtitles/subs/` — subtitle `.srt` files
- `dataset_ratings_and_tags.csv` — combined user interaction dataset

### 3. Build the dataset

```bash
python main.py
```

This preprocesses all sources and produces `dataset.csv` (movie-level features) and `dataset_ratings_and_tags.csv` (user interactions). On subsequent runs it only processes new or updated rows.

---

## Running

### Web UI (Streamlit)

```bash
streamlit run app.py
```

Pick three movies you like; the app returns personalised recommendations using the full SVD + dialogue + franchise model.

### Command-line recommender

```bash
python recommendations_algorithm.py
```

Interactive loop — enter three titles, get scored recommendations with component breakdown.

### Baselines

```bash
python baselines.py
```

Runs the three non-personalised baselines: most popular, highest Bayesian-rated, and random.

### Analysis

```bash
python analyze_data.py
```

Produces correlation plots and summaries in `analysis_outputs/`.

---

## Data Sources

| Dataset | Description | Source |
|---------|-------------|--------|
| MovieLens 1M | 1M ratings from 6,000 users on 4,000 films | [grouplens.org](https://grouplens.org/datasets/movielens/1m/) |
| MovieLens 32M | 32M ratings, larger movie catalogue | [grouplens.org](https://grouplens.org/datasets/movielens/32m/) |
| IMDb | Title metadata and aggregated ratings | [imdb.com/interfaces](https://developer.imdb.com/non-commercial-datasets/) |
| OpenSubtitles | Community subtitle files (`.srt`) | [opensubtitles.org](https://www.opensubtitles.org/) |
| Franchises | Manually curated franchise/installment list | `datasets/franchises/franchises.csv` |
