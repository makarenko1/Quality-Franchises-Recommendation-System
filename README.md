# Quality-Aware Franchise Movie Recommender

A movie recommendation system that combines collaborative filtering with quality signals from ratings, subtitle dialogue, and franchise metadata — recommending movies that are personally relevant while avoiding low-quality franchise sequels.

---

## Overview

The project builds two datasets:

| File | Contents |
|---|---|
| `dataset.csv` | Movie-level metadata: genres, dialogue features, franchise info, IMDb ratings |
| `dataset_ratings_and_tags.csv` | MovieLens 1M + 32M user-item ratings and tags |

The recommender combines five signals:

1. **Collaborative filtering** — SVD on user ratings
2. **Item-quality prior** — popularity + Bayesian-weighted rating quality
3. **Dialogue quality** — subtitle-derived language features
4. **Franchise awareness** — franchise membership, installment number, IMDb rating
5. **Old-movie penalty** — corrects IMDb's bias toward older, highly-rated films

Every entry point (CLI, Streamlit app, evaluation, participant survey) scores through the same code path in `recommendations_algorithm.py`, so all of them rank the same `dataset.csv` movie universe the same way.

---

## How Scoring Works

| Component | Weight | Description |
|---|---:|---|
| Collaborative / quality score | 0.75 | Latent similarity (0.60) + item-quality prior (0.40) |
| Dialogue / language quality | 0.15 | Subtitle-derived language features |
| Franchise quality | 0.05 | Rewards maintained quality, penalizes weaker later installments |
| Old-movie penalty | -0.05 | Pushes older movies down; never boosts newer ones |

The item-quality prior is popularity (0.60) + rating quality (0.40). Dialogue quality is a weighted composite of 5 subtitle features (`negative_word_ratio` -0.35, `type_token_ratio` +0.25, `hapax_ratio` +0.15, `repeated_short_phrase_ratio` -0.15, `bigram_repetition_ratio` -0.10), compared against the average of the user's input movies — so it's part of the personalized ranking, not just a global filter.

Two refinements from milestone analysis, both on by default and toggleable via env var:

- **Dialogue normalization is grouped by decade × genre** (`USE_GROUPED_DIALOGUE_NORMALIZATION`, default on). Dialogue-feature correlations with IMDb rating vary a lot by genre (e.g. `negative_word_ratio` r ≈ -0.11 for Crime vs. -0.02 for Documentary). Each feature is now min-max normalized within decade × genre groups (≥30 movies) instead of the whole catalog, falling back to global normalization for sparse groups.
- **The later-installment penalty is graded, not flat** (`USE_INSTALLMENT_SHRINKAGE`, default on). Mean IMDb rating declines with installment number, but the estimate gets noisier at higher installments (fewer movies). The penalty (still capped at 0.20) is now scaled by a Bayesian-shrinkage-adjusted expected rating drop per installment bucket, instead of a flat -0.20 for any later installment.

A final lightweight post-filter removes clearly unsuitable candidates — stand-up specials, concerts, shorts, documentaries, genre mismatches — from the ranked list, replacing them with the next suitable candidate.

---

## Setup

```bash
git clone <repo-url> && cd Quality-Franchises-Recommendation-System
pip install -r requirements.txt   # pandas, numpy, scipy, streamlit, matplotlib, nltk, gdown
./setup.sh                        # downloads large data files (not stored in Git)
```

---

## Usage

| Task | Command | Notes |
|---|---|---|
| Build datasets | `python main.py` | Builds `dataset.csv` and `dataset_ratings_and_tags.csv` from MovieLens/IMDb/OpenSubtitles/franchise data |
| Run recommender (CLI) | `python recommendations_algorithm.py` | Prompts for 3 movies, prints ranked recommendations |
| Run recommender (web) | `streamlit run app.py` | Same scoring path, browser UI |
| Baselines | `python generate_recommendations.py --interactive` | Popular / highest-rated (Bayesian) / random, interactively |
| Evaluate | `python evaluate.py` | RMSE, Precision@10/Recall@10 vs. baselines, franchise/dialogue diagnostics |
| Analyze features | `python analyze_data.py` | Franchise/dialogue correlation analysis → `analysis_outputs/` |
| Build participant recs | `python generate_recommendations.py` | See [below](#participant-recommendation-survey) |

### Participant recommendation survey

Input: a cleaned spreadsheet `recommendations_initial.xlsx` (sheet `initial_ratings_clean`) with columns `submission_id, timestamp, respondent, movie_rank, MovieID, Title, Year, Rating_1_5, is_positive_input`, already normalized — one movie per row, ratings on a 1–5 scale, `is_positive_input = True` for missing ratings or `Rating_1_5 >= 3.5`.

`python generate_recommendations.py` outputs `recommendations_new.csv`: one row per participant, with 3 recommendations per method (`our_system`, `popular`, `highest_rated`, `random`), each formatted as `Movie Title (Year) [MovieID ID]`.

---

## Evaluation Results

`evaluate.py` splits `dataset_ratings_and_tags.csv` into train/held-out test sets, fits a biased SVD (tuning k), and reports RMSE against a ladder of rating-prediction baselines (bias-only, item-mean, user-mean, global-mean) plus Precision@10/Recall@10 against popular/highest-rated/random ranking baselines, plus franchise and dialogue diagnostics — using the same `recommend_from_movie_ids()` path as the main recommender. RMSE and Precision/Recall use different baseline families because they test different things: RMSE needs a predicted rating, so it only applies to methods that predict one; Precision/Recall only need a ranked list, so they apply to every method (see the comment above `compute_rmse()` in `evaluate.py`). It then also loads the participant survey data in `survey-responses/` (if present) and produces the survey plots described [below](#participant-survey-results), all in one run.

Current result (5,000 sampled ranking users, of ~204k eligible; `EVAL_USER_SAMPLE_SIZE` in `evaluate.py`), with both milestone refinements enabled (default):

| Metric | Result |
|---|---|
| RMSE | 0.8422 (bias-only 0.8799, user-mean 0.9488, item-mean 0.9606, global-mean 1.0603) |
| Precision@10 | 0.0561 |
| Recall@10 | 0.0512 |

| Method | Precision@10 | Recall@10 |
|---|---:|---:|
| Our system | 0.0561 | 0.0512 |
| Popular | 0.1201 | 0.1002 |
| Highest-rated | 0.0472 | 0.0416 |
| Random | 0.0002 | 0.0001 |

The popular baseline remains strongest, which is expected in MovieLens held-out evaluation since universally-watched movies appear in many users' test sets. The model is more novel than popularity because it combines collaborative filtering with dialogue quality, franchise quality, an old-movie penalty, and a post-filter.

**Ablation — grouped vs. global dialogue normalization:** run on the same split with `USE_GROUPED_DIALOGUE_NORMALIZATION=1` vs. `=0`. Global: 0.0624 / 0.0522. Grouped: 0.0632 / 0.0544 (+0.0008 / +0.0022) — a small, consistent gain, supporting the idea of comparing dialogue features to era/genre peers rather than the whole catalog. *(Measured on an earlier 500-user sample; not rechecked at the current 5,000-user sample size, though the relative comparison should still hold.)*

**Ablation — graded vs. flat installment penalty:** run with `USE_INSTALLMENT_SHRINKAGE=1` vs. `=0`. Both gave identical 0.0632 / 0.0544. Franchise quality is only a 0.05 weight, and the penalty only applies to same-franchise, later-installment candidates — too narrow a lever to move this sample. Kept as the default for being better-calibrated (it stops over-trusting sparse high-installment data), not for a measured lift. *(Also measured on the earlier 500-user sample.)*

Supporting analysis: franchise installment vs. IMDb rating, Spearman ρ = -0.2985; top dialogue feature `negative_word_ratio` vs. IMDb rating, Spearman ρ = -0.1613.

### Participant survey results

Separate from the offline metrics above: 15 respondents each rated 3 movies from each of the 4 methods (blinded) and ranked the 4 method-groups best (1) to worst (4). `evaluate.py` recomputes the summary from the raw responses and cross-checks it against the pre-aggregated `survey-responses/results_by_method.csv` before plotting; this step is skipped automatically if `survey-responses/` is absent.

| Method | Avg. relevance | Avg. would-watch | Avg. rank (1=best) | Ranked #1 |
|---|---:|---:|---:|---:|
| Our system | 4.07 | 4.04 | 1.60 | 9 / 15 |
| Popular | 3.84 | 3.89 | 1.87 | 4 / 15 |
| Highest-rated | 3.22 | 3.20 | 2.60 | 2 / 15 |
| Random | 2.09 | 2.13 | 3.93 | 0 / 15 |

Our system was ranked best by 9 of 15 participants and never ranked last, matching the offline evaluation's popular-baseline gap in the opposite direction: participants preferred it over the (offline-stronger) popular baseline on every subjective metric.

### Saved output

Every `python evaluate.py` run overwrites `evaluation_outputs/`:

```text
evaluation_outputs/
├── evaluate_log.txt                  # full console output of the run
├── summary_metrics.csv               # RMSE (svd/bias-only/item-mean/user-mean/global-mean),
│                                     # Precision@10/Recall@10, dataset sizes
├── method_comparison.csv             # Precision@10/Recall@10 per method
├── k_tuning.csv                      # RMSE at each candidate k
├── franchise_summary.csv             # franchise installment vs. rating correlation
├── franchise_by_installment.csv      # mean rating / count / std per installment bucket
├── dialogue_correlation.csv          # per-feature correlation with IMDb rating
└── plots/
    ├── overall_metrics_comparison.png       # RMSE (all baselines) + Precision@10 + Recall@10,
    │                                        # side by side
    ├── precision_recall_by_method.png       # Precision@10/Recall@10 mean, by method
    ├── precision_recall_distribution.png    # per-user outcome buckets (0/1/2+ relevant picks,
    │                                        # none/some/all liked movies found)
    ├── k_tuning_rmse.png                    # RMSE vs. k, with the selected k marked
    ├── survey_metrics_by_method.png         # clustered bars: relevance, would-watch,
    │                                        # novelty, main score, by method
    ├── survey_rank_distribution.png         # stacked bars: how often each method was
    │                                        # ranked 1st..4th
    ├── survey_rating_distribution.png       # per-movie relevance/would-watch distribution shape
    ├── survey_relevance_vs_would_watch.png  # small-multiple heatmaps: do the two ratings agree?
    └── survey_novelty_vs_relevance.png      # accuracy-vs-diversity trade-off, mean ± std by method
```

Note: `franchise_installment_mean_rating.png`, `imdb_rating_by_decade_genre_heatmap.png`, and `rating_dialogue_control_correlations.png` are produced separately by `analyze_data.py` into `analysis_outputs/`, not by `evaluate.py`.

---

## Project Structure

```text
.
├── main.py                        # Data pipeline: builds dataset.csv and dataset_ratings_and_tags.csv
├── recommendations_algorithm.py   # Final recommender, shared by every entry point below
├── evaluate.py                    # Offline evaluation (RMSE, Precision@K/Recall@K, diagnostics)
├── generate_recommendations.py    # Builds the participant recommendation table; --interactive queries baselines from the CLI
├── app.py                         # Streamlit web app
├── analyze_data.py                # Franchise and dialogue correlation analysis
├── setup.sh                       # Downloads large data files
├── dataset.csv                    # Movie-level dataset
├── dataset_ratings_and_tags.csv   # Combined ratings/tags interaction dataset
├── survey-responses/              # Collected participant survey responses
└── datasets/
    ├── movies-1M/                 # MovieLens 1M processed files
    ├── movies-32M/                # MovieLens 32M processed files
    ├── imdb/                      # IMDb title and rating files
    ├── opensubtitles/subs/        # Subtitle .srt files
    └── franchises/franchises.csv  # Franchise metadata
```

---

## Data Sources

| Dataset | Description |
|---|---|
| MovieLens 1M / 32M | Ratings, tags, movie metadata |
| IMDb | Title metadata and aggregate ratings |
| OpenSubtitles | Subtitle `.srt` files used for dialogue features |
| Franchise metadata | Curated franchise membership and installment order |

---

## Notes

Large/generated files excluded from Git — `dataset_ratings_and_tags.csv`, `datasets/movies-32M/`, `datasets/imdb/raw/`, `datasets/opensubtitles/subs/`, `analysis_outputs/` — are recreated via `./setup.sh` and `python main.py`.

Typical workflow:

```bash
./setup.sh
python main.py
python analyze_data.py
python evaluate.py
streamlit run app.py
python generate_recommendations.py   # participant follow-up recommendations
```
