# Quality-Aware Franchise Movie Recommender

A movie recommendation system that combines collaborative filtering with movie-quality signals from ratings, subtitle dialogue features, and franchise metadata. The system is designed to recommend movies that are personally relevant while avoiding low-quality franchise sequels.

---

## Project Overview

The project builds two main datasets:

- `dataset.csv` — movie-level metadata and features.
- `dataset_ratings_and_tags.csv` — user-item interaction data containing MovieLens ratings and tags.

The recommender uses:

1. **Collaborative filtering** from user ratings.
2. **General movie quality** based on popularity and weighted rating quality.
3. **Dialogue quality** from subtitle-derived language features.
4. **Franchise awareness** based on franchise membership, installment number, and IMDb rating.
5. **Old-movie penalty** to reduce bias toward older highly rated IMDb films without boosting newer movies.

The final interactive recommender and evaluation use `dataset.csv` as the movie metadata source so that all methods work with the same project movie universe.

---

## Recommendation Model

Recommendations are generated in `recommendations_algorithm.py`.

The final recommender uses one consistent scoring configuration for the command-line recommender, Streamlit app, evaluation, and participant recommendation survey. Collaborative filtering remains the main recommendation signal, but the ranking is adjusted using dialogue/language quality, franchise quality, and an old-movie penalty to reduce bias toward older highly rated IMDb films. After the main ranking is produced, a lightweight post-filter removes clearly unsuitable candidates such as stand-up specials, live concert/comedy items, shorts, documentaries, and obvious genre mismatches.

| Component | Weight | Description |
|----------|--------|-------------|
| Collaborative / quality score | 0.75 | Combines personalized latent similarity with a general item-quality prior |
| Dialogue / language quality | 0.15 | Uses subtitle-derived language features as a meaningful quality-aware signal |
| Franchise quality | 0.05 | Penalizes weaker later franchise installments and rewards maintained franchise quality |
| Old-movie penalty | -0.05 | Pushes older movies downward to reduce IMDb age bias; newer movies are not boosted |

Inside the collaborative / quality component:

| Subcomponent | Weight | Description |
|-------------|--------|-------------|
| Latent similarity | 0.60 | Similarity between the user's selected movies and candidate movies |
| Item-quality prior | 0.40 | General movie quality based on popularity and rating quality |

The item-quality prior combines:

| Item-quality signal | Weight |
|--------------------|--------|
| Popularity | 0.60 |
| Rating quality | 0.40 |

Dialogue quality is a composite score built from subtitle features. Each feature is normalized before being combined:

| Dialogue feature | Weight | Interpretation |
|-----------------|--------|----------------|
| `negative_word_ratio` | -0.35 | More negative wording lowers the score |
| `type_token_ratio` | 0.25 | Higher vocabulary diversity improves the score |
| `hapax_ratio` | 0.15 | More one-time words slightly improves the score |
| `repeated_short_phrase_ratio` | -0.15 | More repeated short phrases lowers the score |
| `bigram_repetition_ratio` | -0.10 | More repeated two-word phrases lowers the score |

For each user, the system compares candidate movies to the average dialogue-quality score of the user's selected input movies. This makes dialogue quality part of the personalized ranking rather than only a global filter.

The release-year correction is implemented as a penalty only: older movies can be pushed slightly downward, but newer movies do not receive an extra positive boost. This avoids over-recommending very recent items just because they are new.

After scoring and sorting candidates, the recommender applies a final lightweight suitability filter. If a top-ranked movie is clearly unsuitable, it is skipped and replaced with the next lower-ranked suitable candidate. This prevents subtitle-heavy items such as stand-up specials or live comedy/concert recordings from appearing as recommendations for narrative movie inputs, while keeping the main scoring process fast.

---

## Project Structure

```text
.
├── main.py                              # Main data pipeline
├── recommendations_algorithm.py         # Final recommender
├── evaluate.py                          # Offline evaluation
├── interactive_baselines.py             # Interactive popular / highest-rated / random baselines
├── build_recommendation_survey.py       # Builds participant recommendation table
├── app.py                               # Streamlit web app
├── analyze_data.py                      # Franchise and dialogue analysis
├── dataset.csv                          # Movie-level dataset
├── dataset_ratings_and_tags.csv         # Combined ratings/tags interaction dataset
├── initial_ratings_sheet.xlsx           # Cleaned participant input sheet
├── recommendations_wide_to_send.csv     # Recommendation table for follow-up survey
├── setup.sh                             # Downloads large files and sets up repo data
│
└── datasets/
    ├── movies-1M/                       # MovieLens 1M processed files
    ├── movies-32M/                      # MovieLens 32M processed files
    ├── imdb/                            # IMDb title and rating files
    ├── opensubtitles/subs/              # Subtitle .srt files
    └── franchises/franchises.csv        # Franchise metadata
```

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd Quality-Franchises-Recommendation-System
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

The project uses packages such as `pandas`, `numpy`, `scipy`, `streamlit`, `matplotlib`, `nltk`, and `gdown`.

### 3. Download large data files

```bash
./setup.sh
```

The setup script pulls the latest repository version, downloads the large data archive from Google Drive using `gdown`, unzips it into the project directory, and removes the temporary archive.

Large generated or raw data files are not stored directly in GitHub because of file-size limits.

---

## Building the Dataset

Run:

```bash
python main.py
```

This script:

1. Preprocesses MovieLens 1M.
2. Preprocesses MovieLens 32M.
3. Builds or updates `dataset.csv`.
4. Appends MovieLens 32M movies not already present.
5. Adds IMDb metadata and ratings.
6. Adds OpenSubtitles dialogue features.
7. Adds franchise metadata.
8. Builds `dataset_ratings_and_tags.csv`.

`dataset.csv` is the movie-level feature dataset.  
`dataset_ratings_and_tags.csv` is the interaction dataset used for SVD, evaluation, and baselines.

The interaction dataset contains explicit ratings and optional user tags. The recommendation model keeps only explicit rating rows when fitting SVD.

---

## Running the Recommender

### Streamlit app

```bash
streamlit run app.py
```

The app lets a user choose movies and returns recommendations from the final model.

### Command-line recommender

```bash
python recommendations_algorithm.py
```

The script loads:

- `dataset.csv` for movie metadata, dialogue features, and franchise metadata.
- `dataset_ratings_and_tags.csv` for fitting the SVD model.

If `dataset_ratings_and_tags.csv` is missing, the recommender can fall back to the cleaned MovieLens 1M ratings file.

---

## Interactive Baselines

Run:

```bash
python interactive_baselines.py
```

This script provides three baseline recommenders:

| Baseline | Description |
|---------|-------------|
| Popular | Recommends the most-rated movies |
| Highest-rated | Uses Bayesian weighted rating |
| Random | Randomly samples candidate movies |

The baselines read `dataset_ratings_and_tags.csv` and use the same `dataset.csv` movie universe as the main recommender.

---

## Evaluation

Run:

```bash
python evaluate.py
```

The evaluation script:

1. Loads ratings from `dataset_ratings_and_tags.csv`.
2. Keeps only explicit rating interactions.
3. Offsets MovieLens 32M user IDs to avoid collisions with MovieLens 1M user IDs.
4. Splits ratings into train and held-out test sets.
5. Fits a biased SVD model.
6. Tunes the number of latent factors.
7. Computes RMSE against a global-mean baseline.
8. Computes Precision@10 and Recall@10.
9. Compares the final recommender against:
   - popular baseline,
   - highest-rated baseline,
   - random baseline.
10. Reports franchise and dialogue-feature diagnostics.

The evaluation uses the same `recommend_from_movie_ids()` scoring path as the main recommender.

### Current evaluation summary

The final post-filtered model achieved:

| Metric | Result |
|--------|--------|
| RMSE | 0.8422 |
| Global-mean RMSE | 1.0603 |
| RMSE improvement | -0.2181 |
| Precision@10 | 0.0624 |
| Recall@10 | 0.0522 |

Baseline comparison:

| Method | Precision@10 | Recall@10 |
|--------|-------------:|----------:|
| Recommendations algorithm | 0.0624 | 0.0522 |
| Popular | 0.1162 | 0.0920 |
| Highest-rated | 0.0492 | 0.0440 |
| Random | 0.0000 | 0.0000 |

The final model outperforms the highest-rated and random baselines in top-10 ranking metrics. The popular baseline remains strongest, which is expected in MovieLens held-out evaluation because universally watched movies are more likely to appear in many users' test ratings. The proposed model is more novel than popularity because it combines collaborative filtering with dialogue quality, franchise quality, an old-movie penalty, and a post-ranking suitability filter.

Supporting analysis:

| Analysis | Result |
|----------|--------|
| Franchise installment vs IMDb rating | Spearman rho = -0.2985 |
| Top dialogue feature | `negative_word_ratio` |
| `negative_word_ratio` vs IMDb rating | Spearman rho = -0.1613 |

These results support the use of franchise and dialogue signals as quality-aware adjustments, even though collaborative filtering remains the strongest recommendation component.

---

## Analysis

Run:

```bash
python analyze_data.py
```

This script analyzes:

1. Whether later franchise installments tend to have lower IMDb ratings.
2. Whether subtitle dialogue features correlate with IMDb rating.
3. Whether dialogue features correlate with mean user rating.
4. Whether stopword-removed and stemmed content-word features add useful signal.
5. Genre, decade, and data-quality controls.

Outputs are saved in:

```text
analysis_outputs/
```

Important generated files include:

- franchise installment plots,
- dialogue-feature correlation plots,
- content-stemmed feature correlation plots,
- genre and decade heatmaps,
- `analysis_summary.txt`.

---

## Participant Recommendation Survey Workflow

The follow-up survey workflow uses manually collected participant movie preferences.

### Input file

Use the cleaned spreadsheet:

```text
initial_ratings_sheet.xlsx
```

Expected sheet name:

```text
initial_ratings_clean
```

Expected columns:

```text
submission_id
timestamp
respondent
movie_rank
MovieID
Title
Year
Rating_Original
RatingScale_Original
Rating_1_5
is_positive_input
```

The spreadsheet should already be normalized before running the script:

- one movie per row;
- Hebrew/English duplicate respondent names merged where needed;
- only the latest three movie rows kept for each unique respondent;
- all ratings converted to a 1–5 scale in `Rating_1_5`;
- old 1–10 ratings converted by dividing by 2;
- new 1–5 ratings left unchanged;
- missing ratings treated as selected liked movies;
- `is_positive_input = True` for missing ratings or `Rating_1_5 >= 3.5`.

The current cleaned survey sheet contains 15 unique respondents and 45 movie rows.

### Build the recommendation table

Run:

```bash
python build_recommendation_survey.py
```

The script outputs only:

```text
recommendations_wide_to_send.csv
```

This file has one row per participant/submission, with separate columns for each method:

```text
submission_id
respondent
timestamp
input_movies
our_system_1
our_system_2
our_system_3
popular_1
popular_2
popular_3
highest_rated_1
highest_rated_2
highest_rated_3
random_1
random_2
random_3
```

Each recommendation cell is formatted as:

```text
Movie Title (Year) [MovieID ID]
```

This wide file is intended for manually sending recommendations to participants for ranking.

---

## Data Sources

| Dataset | Description |
|---------|-------------|
| MovieLens 1M | Movie ratings, tags, and metadata |
| MovieLens 32M | Larger MovieLens ratings, tags, and movie metadata |
| IMDb | Title metadata and IMDb aggregate ratings |
| OpenSubtitles | Subtitle `.srt` files used for dialogue features |
| Franchise metadata | Curated franchise membership and installment order |

---

## Notes on Large Files

The following files are large and may be excluded from Git:

```text
dataset_ratings_and_tags.csv
datasets/movies-32M/
datasets/imdb/raw/
datasets/opensubtitles/subs/
analysis_outputs/
```

They can be recreated or restored using `setup.sh` and the project pipeline.

---

## Typical Workflow

```bash
./setup.sh
python main.py
python analyze_data.py
python evaluate.py
streamlit run app.py
```

For the participant follow-up recommendations:

```bash
python build_recommendation_survey.py
```
