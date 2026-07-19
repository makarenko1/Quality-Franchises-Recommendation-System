"""
analyze_data.py

Analyze two project questions using dataset.csv:

1. Is there a negative relationship between franchise installment number
   and IMDb rating?
2. Do dialogue features show a relationship with IMDb ratings?

Inputs:
    dataset.csv

Outputs:
    analysis_outputs/
        franchise_installment_vs_imdb_rating.png
        franchise_installment_mean_rating.png
        franchise_rating_distribution_by_installment.png
        dialogue_feature_correlations_installment_heatmap.png
        recommender_dialogue_features_by_installment.png
        dialogue_feature_correlations.png
        dialogue_feature_correlations_filtered.png
        dialogue_feature_correlations_imdb_rating_heatmap.png
        content_stemmed_feature_correlations.png
        dialogue_feature_correlations_mean_user_rating_heatmap.png
        rating_dialogue_control_correlations.png
        rating_by_genre.png
        imdb_rating_by_decade_genre_heatmap.png
        negative_word_ratio_by_decade_genre_heatmap.png
        vocabulary_diversity_imdb_correlation_by_decade_genre.png
        rating_by_year.png
        dialogue_feature_year_correlations.png
        dialogue_feature_redundancy_heatmap.png
        analysis_summary.txt

    Note: a handful of plots that used to be saved here were removed because
    they were pure duplicates of another plot in this list (same numbers,
    different chart type or a strict subset) or because the underlying
    result was consistently non-significant, per-feature noise (e.g. dozens
    of individual weak-correlation scatter/year plots). The statistics
    behind every removed plot are still computed and printed to
    analysis_summary.txt; only the redundant/uninformative image-saving
    calls were dropped.

Run:
    python analyze_data.py
"""

from pathlib import Path
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from recommendations_algorithm import DIALOGUE_FEATURE_WEIGHTS, ensure_setup_data


DATASET_PATH = Path("dataset.csv")
RATINGS_AND_TAGS_DATASET_PATH = Path("dataset_ratings_and_tags.csv")
OUTPUT_DIR = Path("analysis_outputs")

TITLE_FONT_SIZE = 40
AXIS_LABEL_FONT_SIZE = 36
TICK_FONT_SIZE = 32
LEGEND_FONT_SIZE = 32
ANNOTATION_FONT_SIZE = 30

PRESENTATION_TITLE_FONT_SIZE = TITLE_FONT_SIZE + 2
PRESENTATION_AXIS_LABEL_FONT_SIZE = AXIS_LABEL_FONT_SIZE + 2
PRESENTATION_TICK_FONT_SIZE = TICK_FONT_SIZE + 1
PRESENTATION_ANNOTATION_FONT_SIZE = ANNOTATION_FONT_SIZE + 1

SCATTER_TITLE_FONT_SIZE = TITLE_FONT_SIZE + 2
SCATTER_AXIS_LABEL_FONT_SIZE = AXIS_LABEL_FONT_SIZE + 2
SCATTER_TICK_FONT_SIZE = TICK_FONT_SIZE + 1
SCATTER_ANNOTATION_FONT_SIZE = ANNOTATION_FONT_SIZE + 1


def add_text_shadow(
    x,
    y,
    text,
    *,
    fontsize,
    ha="center",
    va="bottom",
    shadow_offset=(0.035, -0.035),
    shadow_alpha=0.35,
    **kwargs,
):
    """
    Add text without a duplicate/shadow copy.

    The function name is kept for compatibility with older plotting code, but
    the visual text-shadow effect is intentionally disabled.
    """
    plt.text(
        x,
        y,
        text,
        fontsize=fontsize,
        ha=ha,
        va=va,
        **kwargs,
    )


RATING_COL = "imdb_averageRating"
INSTALLMENT_COL = "FranchiseInstallment"
FRANCHISE_ID_COL = "FranchiseID"
FRANCHISE_NAME_COL = "FranchiseName"
RUNTIME_COL = "imdb_runtimeMinutes"
VOTES_COL = "imdb_numVotes"
IMDB_TITLE_COL = "imdb_Title"
GENRE_PREFIX = "Genre"

MIN_SUBTITLE_TOKENS = 500
MIN_SUBTITLE_LINES = 100
MAX_SUBTITLE_WORDS_PER_MINUTE = 180
MAX_REPEATED_LINE_RATIO = 0.4

DIALOGUE_FEATURES = [
    # Basic dialogue size
    "num_lines",
    "num_tokens",
    "num_unique_tokens",
    "subtitle_words_per_minute",
    "unique_subtitle_words_per_minute",
    "num_lines_per_minute",
    "avg_line_length",
    "median_line_length",

    # Vocabulary richness
    "type_token_ratio",
    "hapax_ratio",
    "average_word_length",
    "long_word_ratio",
    "common_word_ratio",
    "rare_word_ratio",
    "simple_word_ratio",
    "complex_word_ratio",

    # Repetition
    "top_word_frequency_ratio",
    "bigram_repetition_ratio",
    "trigram_repetition_ratio",
    "repeated_short_phrase_ratio",
    "repeated_line_ratio",
    "duplicate_line_count",
    "most_common_line_frequency",

    # Optional stopword-removed and stemmed content-word features
    "content_stemmed_num_tokens",
    "content_stemmed_num_unique_tokens",
    "content_stemmed_type_token_ratio",
    "content_stemmed_hapax_ratio",
    "content_stemmed_top_word_frequency_ratio",
    "content_stemmed_bigram_repetition_ratio",
    "content_stemmed_trigram_repetition_ratio",
    "content_stemmed_repeated_short_phrase_ratio",

    # Sentiment / emotion
    "average_sentiment",
    "sentiment_variance",
    "positive_word_ratio",
    "negative_word_ratio",
    "anger_word_ratio",
    "fear_word_ratio",
    "joy_word_ratio",
    "sadness_word_ratio",

    # Conversational style
    "question_line_ratio",
    "exclamation_line_ratio",
    "first_person_pronoun_ratio",
    "second_person_pronoun_ratio",
    "contraction_ratio",

    # Readability
    "flesch_reading_ease",
    "average_sentence_length",
]

DIALOGUE_FEATURE_LABELS = {
    "num_lines": "Number of subtitle lines",
    "num_tokens": "Total subtitle words",
    "num_unique_tokens": "Unique subtitle words",
    "subtitle_words_per_minute": "Subtitle words per minute",
    "unique_subtitle_words_per_minute": "Unique subtitle words per minute",
    "num_lines_per_minute": "Subtitle lines per minute",
    "avg_line_length": "Average subtitle line length",
    "median_line_length": "Median subtitle line length",

    "type_token_ratio": "Vocabulary diversity ratio",
    "hapax_ratio": "One-time word ratio",
    "average_word_length": "Average word length",
    "long_word_ratio": "Long-word ratio",
    "common_word_ratio": "Common-word ratio",
    "rare_word_ratio": "Non-common word ratio",
    "simple_word_ratio": "Simple-word ratio",
    "complex_word_ratio": "Complex-word ratio",

    "top_word_frequency_ratio": "Most frequent word share",
    "bigram_repetition_ratio": "Repeated two-word phrase ratio",
    "trigram_repetition_ratio": "Repeated three-word phrase ratio",
    "repeated_short_phrase_ratio": "Repeated short phrase ratio",
    "repeated_line_ratio": "Repeated subtitle line ratio",
    "duplicate_line_count": "Duplicate subtitle line count",
    "most_common_line_frequency": "Most common line frequency",

    "content_stemmed_num_tokens": "Content stemmed words",
    "content_stemmed_num_unique_tokens": "Unique content stemmed words",
    "content_stemmed_type_token_ratio": "Content stemmed vocabulary diversity",
    "content_stemmed_hapax_ratio": "Content stemmed one-time word ratio",
    "content_stemmed_top_word_frequency_ratio": "Most frequent content stem share",
    "content_stemmed_bigram_repetition_ratio": "Repeated content two-stem phrase ratio",
    "content_stemmed_trigram_repetition_ratio": "Repeated content three-stem phrase ratio",
    "content_stemmed_repeated_short_phrase_ratio": "Repeated content short phrase ratio",

    "average_sentiment": "Average sentiment score",
    "sentiment_variance": "Sentiment variance",
    "positive_word_ratio": "Positive word ratio",
    "negative_word_ratio": "Negative word ratio",
    "anger_word_ratio": "Anger word ratio",
    "fear_word_ratio": "Fear word ratio",
    "joy_word_ratio": "Joy word ratio",
    "sadness_word_ratio": "Sadness word ratio",

    "question_line_ratio": "Question line ratio",
    "exclamation_line_ratio": "Exclamation line ratio",
    "first_person_pronoun_ratio": "First-person pronoun ratio",
    "second_person_pronoun_ratio": "Second-person pronoun ratio",
    "contraction_ratio": "Contraction ratio",

    "flesch_reading_ease": "Flesch reading ease",
    "average_sentence_length": "Average sentence length",
}


YEAR_TREND_DIALOGUE_FEATURES = [
    "repeated_line_ratio",
    "avg_line_length",
    "median_line_length",
    "long_word_ratio",
    "complex_word_ratio",
    "subtitle_words_per_minute",
    "num_lines_per_minute",
    "average_sentence_length",
    "flesch_reading_ease",
]


def main():
    ensure_setup_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(DATASET_PATH)

    summary_lines = []
    summary_lines.append("Franchise and Dialogue Analysis")
    summary_lines.append("=" * 40)
    summary_lines.append(f"Dataset rows: {len(dataset):,}")
    summary_lines.append("")

    franchise_summary = analyze_franchise_rating_relationship(dataset)
    summary_lines.extend(franchise_summary)

    summary_lines.append("")
    franchise_dialogue_summary = analyze_franchise_dialogue_relationship(dataset)
    summary_lines.extend(franchise_dialogue_summary)

    summary_lines.append("")
    franchise_dialogue_corr_summary = analyze_franchise_dialogue_feature_correlations(dataset)
    summary_lines.extend(franchise_dialogue_corr_summary)

    summary_lines.append("")
    dialogue_summary = analyze_dialogue_rating_relationship(dataset)
    summary_lines.extend(dialogue_summary)

    summary_lines.append("")
    content_stemmed_summary = analyze_content_stemmed_feature_relationship(dataset)
    summary_lines.extend(content_stemmed_summary)

    summary_lines.append("")
    user_rating_dialogue_summary = analyze_dialogue_mean_user_rating_relationship(dataset)
    summary_lines.extend(user_rating_dialogue_summary)

    summary_lines.append("")
    diagnostics_summary = analyze_data_quality_and_controls(dataset)
    summary_lines.extend(diagnostics_summary)

    summary_path = OUTPUT_DIR / "analysis_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n".join(summary_lines))
    print(f"\nSaved plots and summary to: {OUTPUT_DIR}")


def load_dataset(path):
    """
    Load dataset.csv and validate required columns.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    dataset = pd.read_csv(path, low_memory=False)

    required_cols = {
        "Title",
        "Year",
        RATING_COL,
    }

    missing_cols = required_cols - set(dataset.columns)

    if missing_cols:
        raise ValueError(
            f"dataset.csv is missing required columns: {missing_cols}"
        )

    return dataset


def analyze_franchise_rating_relationship(dataset):
    """
    Analyze whether later franchise installments have lower IMDb ratings.

    Creates:
        - scatter plot with trend line
        - mean rating by installment plot

    Returns:
        list[str]
            Summary lines for analysis_summary.txt.
    """
    lines = []
    lines.append("1. Franchise installment number vs IMDb rating")
    lines.append("-" * 40)

    required_cols = {
        FRANCHISE_ID_COL,
        INSTALLMENT_COL,
        RATING_COL,
    }

    missing_cols = required_cols - set(dataset.columns)

    if missing_cols:
        lines.append(
            f"Skipped franchise analysis because columns are missing: {missing_cols}"
        )
        return lines

    franchise_df = dataset[
        [
            "Title",
            "Year",
            FRANCHISE_ID_COL,
            INSTALLMENT_COL,
            RATING_COL,
        ]
    ].copy()

    franchise_df[INSTALLMENT_COL] = pd.to_numeric(
        franchise_df[INSTALLMENT_COL],
        errors="coerce",
    )

    franchise_df[RATING_COL] = pd.to_numeric(
        franchise_df[RATING_COL],
        errors="coerce",
    )

    franchise_df = franchise_df.dropna(
        subset=[FRANCHISE_ID_COL, INSTALLMENT_COL, RATING_COL]
    )

    if franchise_df.empty:
        lines.append("No franchise rows with IMDb ratings were found.")
        return lines

    pearson_corr = franchise_df[INSTALLMENT_COL].corr(
        franchise_df[RATING_COL],
        method="pearson",
    )

    spearman_corr = franchise_df[INSTALLMENT_COL].corr(
        franchise_df[RATING_COL],
        method="spearman",
    )

    slope, intercept = np.polyfit(
        franchise_df[INSTALLMENT_COL],
        franchise_df[RATING_COL],
        deg=1,
    )

    lines.append(f"Franchise rows with ratings: {len(franchise_df):,}")
    lines.append(
        f"Unique franchises: {franchise_df[FRANCHISE_ID_COL].nunique():,}"
    )
    lines.append(f"Pearson correlation: {pearson_corr:.4f}")
    lines.append(f"Spearman correlation: {spearman_corr:.4f}")
    lines.append(f"Linear trend slope: {slope:.4f} IMDb rating points per installment")

    if slope < 0:
        lines.append(
            "Interpretation: the estimated trend is negative, which supports "
            "the hypothesis that later installments tend to score lower."
        )
    else:
        lines.append(
            "Interpretation: the estimated trend is not negative, so this dataset "
            "does not support the hypothesis in this simple analysis."
        )

    plot_franchise_scatter(franchise_df, slope, intercept)
    plot_mean_rating_by_installment(franchise_df)
    plot_rating_distribution_by_installment(franchise_df)

    rating_change_summary = analyze_rating_change_from_first(franchise_df)
    lines.extend(rating_change_summary)

    return lines


def analyze_rating_change_from_first(franchise_df):
    """
    Check whether later installments decline relative to the first movie
    within the same franchise.
    """
    lines = []
    df = franchise_df.copy()

    first_ratings = (
        df[df[INSTALLMENT_COL] == 1]
        .groupby(FRANCHISE_ID_COL)[RATING_COL]
        .first()
        .rename("first_installment_rating")
    )

    df = df.merge(
        first_ratings,
        left_on=FRANCHISE_ID_COL,
        right_index=True,
        how="left",
    )

    df = df.dropna(subset=["first_installment_rating"])
    df["rating_change_from_first"] = (
        df[RATING_COL] - df["first_installment_rating"]
    )

    later_df = df[df[INSTALLMENT_COL] > 1].copy()

    if later_df.empty:
        lines.append("No later installments could be compared to a first installment.")
        return lines

    mean_change = later_df["rating_change_from_first"].mean()
    median_change = later_df["rating_change_from_first"].median()
    share_lower = (later_df["rating_change_from_first"] < 0).mean()

    lines.append("")
    lines.append("Within-franchise comparison:")
    lines.append(
        f"Mean rating change from first installment: {mean_change:.4f}"
    )
    lines.append(
        f"Median rating change from first installment: {median_change:.4f}"
    )
    lines.append(
        f"Share of later installments rated below the first: {share_lower:.2%}"
    )

    return lines


def plot_franchise_scatter(franchise_df, slope, intercept):
    """
    Plot IMDb rating against franchise installment number.
    """
    x = franchise_df[INSTALLMENT_COL]
    y = franchise_df[RATING_COL]

    # Small deterministic jitter so overlapping points are more visible.
    rng = np.random.default_rng(seed=42)
    x_jitter = x + rng.normal(0, 0.04, size=len(x))

    plt.figure(figsize=(16, 11))
    plt.scatter(x_jitter, y, alpha=0.35, s=65)

    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept

    plt.plot(x_line, y_line, linewidth=3)

    plt.title(
        "IMDb Rating vs Franchise Installment Number",
        fontsize=SCATTER_TITLE_FONT_SIZE,
        pad=22,
    )
    plt.xlabel(
        "Franchise installment number",
        fontsize=SCATTER_AXIS_LABEL_FONT_SIZE,
    )
    plt.ylabel(
        "IMDb rating",
        fontsize=SCATTER_AXIS_LABEL_FONT_SIZE,
    )
    plt.tick_params(axis="both", labelsize=SCATTER_TICK_FONT_SIZE)

    slope_text = f"Trend: {slope:.2f} rating points per installment"
    plt.text(
        x.min(),
        y.max(),
        slope_text,
        fontsize=SCATTER_ANNOTATION_FONT_SIZE,
        ha="left",
        va="top",
    )

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "franchise_installment_vs_imdb_rating.png"
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_mean_rating_by_installment(franchise_df):
    """
    Plot average IMDb rating by installment number.

    The shaded band shows the minimum-to-maximum IMDb rating range for each
    installment number. The n= labels show how many movies are included in each
    installment-number average.
    """
    grouped = (
        franchise_df
        .groupby(INSTALLMENT_COL)[RATING_COL]
        .agg(["mean", "min", "max", "count"])
        .reset_index()
        .sort_values(INSTALLMENT_COL)
    )

    plt.figure(figsize=(15, 10))

    # Min-max range shadow/band.
    plt.fill_between(
        grouped[INSTALLMENT_COL],
        grouped["min"],
        grouped["max"],
        alpha=0.18,
        label="Min-max rating range",
    )

    plt.plot(
        grouped[INSTALLMENT_COL],
        grouped["mean"],
        marker="o",
        linewidth=3,
        markersize=9,
        label="Mean IMDb rating",
    )

    for _, row in grouped.iterrows():
        plt.text(
            row[INSTALLMENT_COL],
            row["mean"],
            f"n={int(row['count'])}",
            fontsize=max(1, PRESENTATION_ANNOTATION_FONT_SIZE - 2),
            ha="center",
            va="bottom",
        )

    plt.title(
        "Mean IMDb Rating by Franchise Installment",
        fontsize=PRESENTATION_TITLE_FONT_SIZE + 4,
        pad=20,
    )
    plt.xlabel(
        "Franchise installment number",
        fontsize=PRESENTATION_AXIS_LABEL_FONT_SIZE + 4,
    )
    plt.ylabel(
        "Mean IMDb rating",
        fontsize=PRESENTATION_AXIS_LABEL_FONT_SIZE + 4,
    )
    plt.tick_params(axis="both", labelsize=PRESENTATION_TICK_FONT_SIZE + 4)
    plt.legend(fontsize=LEGEND_FONT_SIZE + 4)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "franchise_installment_mean_rating.png"
    plt.savefig(output_path, dpi=200)
    plt.close()


def analyze_franchise_dialogue_relationship(dataset):
    """
    Analyze whether later franchise installments have worse dialogue quality —
    the same installment-decline question as
    analyze_franchise_rating_relationship(), but for negative_word_ratio (the
    most heavily weighted feature in the recommender's DIALOGUE_FEATURE_WEIGHTS)
    instead of IMDb rating.
    """
    lines = []
    lines.append("2. Franchise installment number vs negative_word_ratio")
    lines.append("-" * 40)

    feature_col = "negative_word_ratio"
    required_cols = {FRANCHISE_ID_COL, INSTALLMENT_COL, feature_col}
    missing_cols = required_cols - set(dataset.columns)

    if missing_cols:
        lines.append(
            f"Skipped franchise dialogue analysis because columns are missing: {missing_cols}"
        )
        return lines

    franchise_df = dataset[
        ["Title", "Year", FRANCHISE_ID_COL, INSTALLMENT_COL, feature_col]
    ].copy()

    franchise_df[INSTALLMENT_COL] = pd.to_numeric(
        franchise_df[INSTALLMENT_COL], errors="coerce"
    )
    franchise_df[feature_col] = pd.to_numeric(
        franchise_df[feature_col], errors="coerce"
    )
    franchise_df = franchise_df.dropna(
        subset=[FRANCHISE_ID_COL, INSTALLMENT_COL, feature_col]
    )

    if franchise_df.empty:
        lines.append("No franchise rows with negative_word_ratio were found.")
        return lines

    pearson_corr = franchise_df[INSTALLMENT_COL].corr(
        franchise_df[feature_col], method="pearson"
    )
    spearman_corr = franchise_df[INSTALLMENT_COL].corr(
        franchise_df[feature_col], method="spearman"
    )
    slope, intercept = np.polyfit(
        franchise_df[INSTALLMENT_COL], franchise_df[feature_col], deg=1
    )

    lines.append(f"Franchise rows with negative_word_ratio: {len(franchise_df):,}")
    lines.append(f"Unique franchises: {franchise_df[FRANCHISE_ID_COL].nunique():,}")
    lines.append(f"Pearson correlation: {pearson_corr:.4f}")
    lines.append(f"Spearman correlation: {spearman_corr:.4f}")
    lines.append(
        f"Linear trend slope: {slope:.6f} negative_word_ratio points per installment"
    )

    if slope > 0:
        lines.append(
            "Interpretation: the estimated trend is positive, which supports "
            "the hypothesis that later installments have more negative dialogue."
        )
    else:
        lines.append(
            "Interpretation: the estimated trend is not positive, so this dataset "
            "does not support the hypothesis in this simple analysis."
        )

    return lines


def analyze_franchise_dialogue_feature_correlations(dataset):
    """
    Examine every dialogue feature (not just negative_word_ratio) for a
    franchise-installment relationship: does any dialogue feature trend with
    installment number the way IMDb rating does? Reuses
    compute_feature_correlations_with_target() and mirrors
    plot_dialogue_mean_user_rating_correlation_heatmap()'s one-column heatmap,
    against franchise installment number instead of mean user rating.
    """
    lines = []
    lines.append("3. All dialogue features vs franchise installment number")
    lines.append("-" * 40)

    required_cols = {FRANCHISE_ID_COL, INSTALLMENT_COL}
    missing_cols = required_cols - set(dataset.columns)

    if missing_cols:
        lines.append(f"Skipped: missing columns {missing_cols}.")
        return lines

    available_features = [
        feature for feature in DIALOGUE_FEATURES
        if feature in dataset.columns
    ]

    if not available_features:
        lines.append("Skipped: no dialogue feature columns were found.")
        return lines

    franchise_df = dataset[
        [FRANCHISE_ID_COL, INSTALLMENT_COL] + available_features
    ].copy()
    franchise_df[INSTALLMENT_COL] = pd.to_numeric(
        franchise_df[INSTALLMENT_COL], errors="coerce"
    )
    franchise_df = franchise_df.dropna(subset=[FRANCHISE_ID_COL, INSTALLMENT_COL])

    if franchise_df.empty:
        lines.append("Skipped: no franchise rows with an installment number were found.")
        return lines

    corr_df = compute_feature_correlations_with_target(
        dataset=franchise_df,
        features=available_features,
        target_col=INSTALLMENT_COL,
    )

    if corr_df.empty:
        lines.append(
            "Skipped: no dialogue features had enough valid rows for "
            "installment correlation analysis."
        )
        return lines

    corr_df = corr_df.sort_values("abs_pearson", ascending=False).reset_index(drop=True)

    lines.append(f"Franchise rows used: {len(franchise_df):,}")
    lines.append(f"Unique franchises: {franchise_df[FRANCHISE_ID_COL].nunique():,}")
    lines.append(f"Dialogue features tested: {len(corr_df)}")
    lines.append("Top 5 by |Pearson correlation| with installment number:")
    for _, row in corr_df.head(5).iterrows():
        lines.append(
            f"  {row['feature']}: pearson={row['pearson']:.4f}, "
            f"spearman={row['spearman']:.4f}, n={int(row['n'])}"
        )

    plot_franchise_dialogue_correlation_heatmap(corr_df)

    lines.append(
        "Saved all-feature franchise-installment correlation heatmap to "
        "dialogue_feature_correlations_installment_heatmap.png."
    )

    plot_recommender_dialogue_features_by_installment(franchise_df, corr_df)

    lines.append(
        "Saved recommender-dialogue-feature-by-installment small multiples to "
        "recommender_dialogue_features_by_installment.png."
    )

    return lines


def plot_franchise_dialogue_correlation_heatmap(corr_df):
    """
    Plot all dialogue-feature correlations with franchise installment number
    as a compact one-column heatmap, mirroring
    plot_dialogue_mean_user_rating_correlation_heatmap().
    """
    corr_df = corr_df.copy().sort_values("pearson")

    labels = [
        get_feature_label(feature)
        for feature in corr_df["feature"]
    ]

    values = corr_df["pearson"].to_numpy().reshape(-1, 1)

    fig, ax = plt.subplots(figsize=(18, max(14, 0.72 * len(corr_df))))
    image = ax.imshow(values, aspect="auto", vmin=-0.25, vmax=0.25)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(
        "Pearson correlation with installment number",
        fontsize=PRESENTATION_AXIS_LABEL_FONT_SIZE,
    )
    cbar.ax.tick_params(labelsize=PRESENTATION_TICK_FONT_SIZE)

    y = np.arange(len(labels))
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=max(9, PRESENTATION_TICK_FONT_SIZE - 5))
    ax.set_xticks([0])
    ax.set_xticklabels(["Installment\nnumber"], fontsize=PRESENTATION_TICK_FONT_SIZE)

    for i, value in enumerate(corr_df["pearson"]):
        text_color = "white" if abs(value) >= 0.12 else "black"
        ax.text(
            0,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=max(1, PRESENTATION_ANNOTATION_FONT_SIZE - 4),
            color=text_color,
        )

    ax.set_title(
        "Dialogue Feature Correlations\nwith Installment Number",
        fontsize=PRESENTATION_TITLE_FONT_SIZE,
        pad=34,
    )
    ax.set_ylabel("Dialogue feature", fontsize=PRESENTATION_AXIS_LABEL_FONT_SIZE)

    fig.subplots_adjust(left=0.66, right=0.86, top=0.88, bottom=0.08)

    plt.savefig(
        OUTPUT_DIR / "dialogue_feature_correlations_installment_heatmap.png",
        dpi=260,
    )
    plt.close()


def plot_dialogue_features_by_installment_grid(
    franchise_df, corr_df, features, title, output_filename,
):
    """
    Small multiples: mean-by-installment (with min-max band) for the given
    dialogue features. Used by plot_recommender_dialogue_features_by_installment()
    for the fixed 5 features in DIALOGUE_FEATURE_WEIGHTS.
    """
    # 3 columns so a 5-feature grid fits in 2 compact rows (3 + 2, centered).
    n_cols = 3
    n_rows = (len(features) + n_cols - 1) // n_cols

    # Use a fine-grained GridSpec (2 sub-columns per real column) so a partial
    # last row is centered instead of left-aligned with blank cells on the
    # right. Each subplot spans 2 sub-columns out of n_cols*2; a row with
    # items_in_row < n_cols items is shifted right by (n_cols - items_in_row)
    # sub-columns, splitting the leftover space evenly on both sides.
    fig = plt.figure(figsize=(9.5 * n_cols, 7.5 * n_rows))
    gs = fig.add_gridspec(n_rows, n_cols * 2, hspace=0.85, wspace=0.9)

    axes = []
    for i in range(len(features)):
        row, col_in_row = divmod(i, n_cols)
        items_in_row = min(n_cols, len(features) - row * n_cols)
        row_offset = n_cols - items_in_row
        start = row_offset + col_in_row * 2
        axes.append(fig.add_subplot(gs[row, start:start + 2]))

    for ax, feature in zip(axes, features):
        grouped = (
            franchise_df
            .groupby(INSTALLMENT_COL)[feature]
            .agg(["mean", "min", "max", "count"])
            .reset_index()
            .sort_values(INSTALLMENT_COL)
        )

        ax.fill_between(
            grouped[INSTALLMENT_COL], grouped["min"], grouped["max"],
            alpha=0.18, color="#4C72B0",
        )
        ax.plot(
            grouped[INSTALLMENT_COL], grouped["mean"],
            marker="o", linewidth=2.5, markersize=7, color="#4C72B0",
        )

        for i, (_, row) in enumerate(grouped.iterrows()):
            ax.annotate(
                f"n={int(row['count'])}",
                (row[INSTALLMENT_COL], row["mean"]),
                textcoords="offset points",
                xytext=(0, 17 if i % 2 == 0 else -26),
                ha="center",
                fontsize=27,
            )

        feature_corr = corr_df.loc[corr_df["feature"] == feature].iloc[0]
        # ρ on its own line keeps the title's horizontal footprint short and
        # consistent regardless of feature-name length, so long labels (e.g.
        # "Repeated two-word phrase ratio") don't collide with a neighbor.
        ax.set_title(
            f"{get_feature_label(feature)}\n(ρ={feature_corr['spearman']:.2f})",
            fontsize=44,
        )
        ax.set_xlabel("Franchise installment number", fontsize=37)
        # Only features literally named "*_ratio" are proportions; a handful of
        # top-correlated features (e.g. median_subtitle_line_length) are counts
        # or lengths, so "Ratio" would be an incorrect label for those.
        y_label = "Ratio" if feature.endswith("_ratio") else get_feature_label(feature)
        ax.set_ylabel(y_label, fontsize=37)
        ax.tick_params(axis="both", labelsize=30)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=52)
    # Tighten the outer margins (matplotlib's default left/right/bottom
    # fractions leave large blank borders at this figure size) and drop
    # "top" further so the two-line subplot titles don't run into the
    # suptitle above them.
    fig.subplots_adjust(left=0.035, right=0.99, bottom=0.06, top=0.78)

    plt.savefig(
        OUTPUT_DIR / output_filename,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()


def plot_recommender_dialogue_features_by_installment(franchise_df, corr_df):
    """
    Small multiples for exactly the 5 dialogue features used in the
    recommender's DIALOGUE_FEATURE_WEIGHTS (recommendations_algorithm.py),
    regardless of their correlation ranking, so the features actually driving
    W_DIALOGUE can be inspected directly for a franchise-installment trend.
    """
    features = [f for f in DIALOGUE_FEATURE_WEIGHTS if f in corr_df["feature"].values]

    plot_dialogue_features_by_installment_grid(
        franchise_df,
        corr_df,
        features,
        "Spearman Correlations of Dialogue Features with Franchise Installment Number",
        "recommender_dialogue_features_by_installment.png",
    )


def plot_rating_distribution_by_installment(franchise_df):
    """
    Plot the distribution of IMDb ratings by franchise installment number.

    This complements the mean/trend plots by showing whether later installments
    are consistently lower or whether the trend is driven by outliers.
    """
    df = franchise_df[[INSTALLMENT_COL, RATING_COL]].copy()
    df[INSTALLMENT_COL] = pd.to_numeric(df[INSTALLMENT_COL], errors="coerce")
    df[RATING_COL] = pd.to_numeric(df[RATING_COL], errors="coerce")
    df = df.dropna()

    if df.empty:
        return

    grouped = (
        df
        .groupby(INSTALLMENT_COL)[RATING_COL]
        .apply(list)
        .reset_index()
        .sort_values(INSTALLMENT_COL)
    )

    # Limit the plot to installment positions with enough observations so that
    # the distribution remains readable.
    counts = df.groupby(INSTALLMENT_COL)[RATING_COL].count()
    valid_installments = counts[counts >= 3].index
    grouped = grouped[grouped[INSTALLMENT_COL].isin(valid_installments)].copy()

    if grouped.empty:
        return

    values = grouped[RATING_COL].tolist()
    labels = [str(int(x)) for x in grouped[INSTALLMENT_COL]]

    plt.figure(figsize=(16, 11))
    box = plt.boxplot(
        values,
        tick_labels=labels,
        patch_artist=True,
        showmeans=True,
    )

    for median in box["medians"]:
        median.set_linewidth(2)

    for i, installment in enumerate(grouped[INSTALLMENT_COL], start=1):
        count = int(counts.loc[installment])
        median_value = np.median(grouped.loc[grouped[INSTALLMENT_COL] == installment, RATING_COL].iloc[0])
        plt.text(
            i,
            median_value,
            f"n={count}",
            fontsize=ANNOTATION_FONT_SIZE,
            ha="center",
            va="bottom",
        )

    plt.title("IMDb Rating Distribution by Franchise Installment", fontsize=TITLE_FONT_SIZE)
    plt.xlabel("Franchise installment number", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel("IMDb rating", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "franchise_rating_distribution_by_installment.png", dpi=200)
    plt.close()


def analyze_dialogue_rating_relationship(dataset):
    """
    Analyze relationships between dialogue features and IMDb ratings.

    Creates:
        - bar plot of correlations
        - scatter plot with trend line for each available dialogue feature

    Returns:
        list[str]
            Summary lines for analysis_summary.txt.
    """
    lines = []
    lines.append("2. Dialogue features vs IMDb rating")
    lines.append("-" * 40)

    available_features = [
        feature for feature in DIALOGUE_FEATURES
        if feature in dataset.columns
    ]

    if not available_features:
        lines.append("No dialogue feature columns were found.")
        return lines

    dialogue_df = dataset[[RATING_COL] + available_features].copy()

    dialogue_df[RATING_COL] = pd.to_numeric(
        dialogue_df[RATING_COL],
        errors="coerce",
    )

    for feature in available_features:
        dialogue_df[feature] = pd.to_numeric(
            dialogue_df[feature],
            errors="coerce",
        )

    dialogue_df = dialogue_df.dropna(subset=[RATING_COL])

    correlations = []

    for feature in available_features:
        feature_df = dialogue_df[[feature, RATING_COL]].dropna()

        if len(feature_df) < 2:
            continue

        pearson_corr = feature_df[feature].corr(
            feature_df[RATING_COL],
            method="pearson",
        )

        spearman_corr = feature_df[feature].corr(
            feature_df[RATING_COL],
            method="spearman",
        )

        correlations.append(
            {
                "feature": feature,
                "pearson": pearson_corr,
                "spearman": spearman_corr,
                "n": len(feature_df),
            }
        )

        # A per-feature scatter plot used to be saved here for every one of
        # the ~46 dialogue features. Nearly all of them show |pearson| well
        # under 0.2 (see the interpretation below), so the scatters were
        # visually just noise clouds with no discernible trend -- dropped as
        # non-significant. The correlation values themselves are still
        # computed above and summarized in the bar chart/heatmap below.

    if not correlations:
        lines.append("No dialogue features had enough valid rows for analysis.")
        return lines

    corr_df = pd.DataFrame(correlations)
    corr_df["abs_pearson"] = corr_df["pearson"].abs()
    corr_df = corr_df.sort_values("abs_pearson", ascending=False)

    lines.append(f"Rows with IMDb ratings: {dialogue_df[RATING_COL].notna().sum():,}")
    lines.append("Available dialogue features:")
    for feature in available_features:
        lines.append(f"  - {get_feature_label(feature)}")

    lines.append("")
    lines.append("Correlations with IMDb rating:")
    for _, row in corr_df.iterrows():
        lines.append(
            f"  {get_feature_label(row['feature'])}: "
            f"Pearson={row['pearson']:.4f}, "
            f"Spearman={row['spearman']:.4f}, "
            f"n={int(row['n'])}"
        )

    strongest = corr_df.iloc[0]

    lines.append("")
    lines.append(
        f"Strongest absolute Pearson relationship: "
        f"{get_feature_label(strongest['feature'])} "
        f"({strongest['pearson']:.4f})."
    )

    if abs(strongest["pearson"]) >= 0.2:
        lines.append(
            "Interpretation: at least one dialogue feature shows a noticeable "
            "linear relationship with IMDb rating."
        )
    elif abs(strongest["pearson"]) >= 0.1:
        lines.append(
            "Interpretation: dialogue features show weak but non-zero relationships "
            "with IMDb rating."
        )
    else:
        lines.append(
            "Interpretation: dialogue features show only very weak relationships "
            "with IMDb rating in this simple correlation analysis."
        )

    plot_dialogue_correlation_bar(corr_df)

    plot_dialogue_feature_correlations_imdb_rating_heatmap(corr_df)
    lines.append(
        "Saved all-feature IMDb-rating correlation heatmap to "
        "dialogue_feature_correlations_imdb_rating_heatmap.png."
    )

    return lines


CONTENT_STEMMED_FEATURES = [
    "content_stemmed_num_tokens",
    "content_stemmed_num_unique_tokens",
    "content_stemmed_type_token_ratio",
    "content_stemmed_hapax_ratio",
    "content_stemmed_top_word_frequency_ratio",
    "content_stemmed_bigram_repetition_ratio",
    "content_stemmed_trigram_repetition_ratio",
    "content_stemmed_repeated_short_phrase_ratio",
]


def analyze_content_stemmed_feature_relationship(dataset):
    """
    Analyze only the optional stopword-removed and stemmed content-word features.

    This creates separate plots so these robustness features can be compared
    with the original dialogue features without replacing them.
    """
    lines = []
    lines.append("Optional content-stemmed features vs IMDb rating")
    lines.append("-" * 40)

    available_features = [
        feature for feature in CONTENT_STEMMED_FEATURES
        if feature in dataset.columns
    ]

    if not available_features:
        lines.append(
            "Skipped: no content-stemmed feature columns were found. "
            "Run main.py after adding the NLTK-based features first."
        )
        return lines

    if RATING_COL not in dataset.columns:
        lines.append(f"Skipped: {RATING_COL} is missing.")
        return lines

    corr_df = compute_feature_correlations_with_target(
        dataset=dataset,
        features=available_features,
        target_col=RATING_COL,
    )

    if corr_df.empty:
        lines.append("Skipped: no content-stemmed feature had enough valid rows.")
        return lines

    lines.append("Correlations with IMDb rating:")
    for _, row in corr_df.sort_values("abs_pearson", ascending=False).iterrows():
        lines.append(
            f"  {get_feature_label(row['feature'])}: "
            f"Pearson={row['pearson']:.4f}, "
            f"Spearman={row['spearman']:.4f}, "
            f"n={int(row['n'])}"
        )

    # Only one plot here: CONTENT_STEMMED_FEATURES has 8 entries, so a
    # "top 10" and "all" version would be identical -- the separate
    # content_stemmed_feature_correlations_all.png this used to also save
    # was dropped as a pure duplicate.
    top_corr_df = corr_df.sort_values("abs_pearson", ascending=False).head(10)
    plot_correlation_bar(
        top_corr_df,
        title="Content-Stemmed Feature Correlations with IMDb Rating",
        output_path=OUTPUT_DIR / "content_stemmed_feature_correlations.png",
    )

    lines.append(
        "Saved content-stemmed feature correlation plot to "
        "content_stemmed_feature_correlations.png."
    )

    return lines


def load_mean_user_ratings(
    ratings_path=RATINGS_AND_TAGS_DATASET_PATH,
):
    """
    Load dataset_ratings_and_tags.csv and compute mean user rating per movie.

    The ratings/tags dataset contains both rating and tag events. Rating rows are
    identified by a non-empty numeric Rating value.
    """
    ratings_path = Path(ratings_path)

    if not ratings_path.exists():
        return None, f"Skipped: {ratings_path} not found."

    required_cols = {"MovieID", "Rating"}
    chunks = []

    try:
        reader = pd.read_csv(
            ratings_path,
            usecols=["MovieID", "Rating"],
            low_memory=False,
            chunksize=500_000,
        )

        for chunk in reader:
            chunk["MovieID"] = pd.to_numeric(chunk["MovieID"], errors="coerce")
            chunk["Rating"] = pd.to_numeric(chunk["Rating"], errors="coerce")
            chunk = chunk.dropna(subset=["MovieID", "Rating"])

            if chunk.empty:
                continue

            chunk["MovieID"] = chunk["MovieID"].astype(int)
            chunks.append(chunk)

    except ValueError:
        return None, (
            f"Skipped: {ratings_path} must contain columns {required_cols}."
        )
    except pd.errors.EmptyDataError:
        return None, f"Skipped: {ratings_path} is empty."

    if not chunks:
        return None, f"Skipped: no rating rows found in {ratings_path}."

    ratings = pd.concat(chunks, ignore_index=True)

    mean_ratings = (
        ratings
        .groupby("MovieID")["Rating"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_user_rating",
                "count": "user_rating_count",
            }
        )
    )

    return mean_ratings, None


def analyze_dialogue_mean_user_rating_relationship(dataset):
    """
    Analyze relationships between dialogue features and mean user ratings.

    Mean user rating is computed from dataset_ratings_and_tags.csv using the
    Rating column grouped by MovieID. The computed values are used only inside
    this function for the correlation plots; dataset.csv is not modified.
    """
    lines = []
    lines.append("Dialogue features vs mean user rating")
    lines.append("-" * 40)

    if "MovieID" not in dataset.columns:
        lines.append("Skipped: MovieID column is missing from dataset.csv.")
        return lines

    mean_ratings, error_message = load_mean_user_ratings()

    if error_message is not None:
        lines.append(error_message)
        return lines

    available_features = [
        feature for feature in DIALOGUE_FEATURES
        if feature in dataset.columns
    ]

    if not available_features:
        lines.append("Skipped: no dialogue feature columns were found.")
        return lines

    mean_rating_map = mean_ratings.set_index("MovieID")["mean_user_rating"]

    df = dataset[["MovieID"] + available_features].copy()
    df["MovieID"] = pd.to_numeric(df["MovieID"], errors="coerce")
    df = df.dropna(subset=["MovieID"])
    df["MovieID"] = df["MovieID"].astype(int)
    df["mean_user_rating"] = df["MovieID"].map(mean_rating_map)
    df = df.dropna(subset=["mean_user_rating"])

    if df.empty:
        lines.append(
            "Skipped: no movies in dataset.csv matched rating rows in "
            f"{RATINGS_AND_TAGS_DATASET_PATH}."
        )
        return lines

    corr_df = compute_feature_correlations_with_target(
        dataset=df,
        features=available_features,
        target_col="mean_user_rating",
    )

    if corr_df.empty:
        lines.append(
            "Skipped: no dialogue features had enough valid rows for "
            "mean-user-rating correlation analysis."
        )
        return lines

    lines.append(
        "Mean user rating computed from "
        f"{RATINGS_AND_TAGS_DATASET_PATH} using Rating grouped by MovieID."
    )
    lines.append(f"Movies with mean user ratings used in this computation: {len(df):,}")
    lines.append("Correlations with mean user rating:")

    for _, row in corr_df.sort_values("abs_pearson", ascending=False).iterrows():
        lines.append(
            f"  {get_feature_label(row['feature'])}: "
            f"Pearson={row['pearson']:.4f}, "
            f"Spearman={row['spearman']:.4f}, "
            f"n={int(row['n'])}"
        )

    strongest = corr_df.sort_values("abs_pearson", ascending=False).iloc[0]

    lines.append("")
    lines.append(
        f"Strongest absolute Pearson relationship with mean user rating: "
        f"{get_feature_label(strongest['feature'])} "
        f"({strongest['pearson']:.4f})."
    )

    # Only the heatmap is saved here: separate top-10 and all-feature bar
    # charts used to also be saved, but they show the exact same numbers as
    # this heatmap in a different chart type, so they were dropped as
    # duplicates.
    plot_dialogue_mean_user_rating_correlation_heatmap(corr_df)

    lines.append(
        "Saved dialogue-feature heatmap with mean user rating to "
        "dialogue_feature_correlations_mean_user_rating_heatmap.png."
    )

    return lines


def compute_feature_correlations_with_target(dataset, features, target_col):
    """
    Compute Pearson and Spearman correlations for features against a target.
    """
    df = dataset[[target_col] + features].copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    correlations = []

    for feature in features:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")
        feature_df = df[[feature, target_col]].dropna()

        if len(feature_df) < 2:
            continue

        correlations.append({
            "feature": feature,
            "pearson": feature_df[feature].corr(feature_df[target_col], method="pearson"),
            "spearman": feature_df[feature].corr(feature_df[target_col], method="spearman"),
            "n": len(feature_df),
        })

    corr_df = pd.DataFrame(correlations)

    if not corr_df.empty:
        corr_df["abs_pearson"] = corr_df["pearson"].abs()

    return corr_df


def plot_dialogue_mean_user_rating_correlation_heatmap(corr_df):
    """
    Plot dialogue-feature correlations with mean user rating as a compact heatmap.

    This is a one-column heatmap, so it focuses only on the target variable
    instead of showing a full pairwise correlation matrix.
    """
    corr_df = corr_df.copy().sort_values("pearson")

    labels = [
        get_feature_label(feature)
        for feature in corr_df["feature"]
    ]

    values = corr_df["pearson"].to_numpy().reshape(-1, 1)

    fig, ax = plt.subplots(figsize=(13, max(14, 0.72 * len(corr_df))))
    image = ax.imshow(values, aspect="auto", vmin=-0.25, vmax=0.25)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(
        "Pearson correlation with mean user rating",
        fontsize=PRESENTATION_AXIS_LABEL_FONT_SIZE,
    )
    cbar.ax.tick_params(labelsize=PRESENTATION_TICK_FONT_SIZE)

    y = np.arange(len(labels))
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=max(9, PRESENTATION_TICK_FONT_SIZE - 5))
    ax.set_xticks([0])
    ax.set_xticklabels(["Mean user rating"], fontsize=PRESENTATION_TICK_FONT_SIZE)

    for i, value in enumerate(corr_df["pearson"]):
        text_color = "white" if abs(value) >= 0.12 else "black"
        ax.text(
            0,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=max(1, PRESENTATION_ANNOTATION_FONT_SIZE - 4),
            color=text_color,
        )

    ax.set_title(
        "Dialogue Feature Correlations\nwith Mean User Rating",
        fontsize=PRESENTATION_TITLE_FONT_SIZE,
        pad=34,
    )
    ax.set_ylabel("Dialogue feature", fontsize=PRESENTATION_AXIS_LABEL_FONT_SIZE)

    fig.subplots_adjust(left=0.56, right=0.86, top=0.88, bottom=0.08)

    plt.savefig(
        OUTPUT_DIR / "dialogue_feature_correlations_mean_user_rating_heatmap.png",
        dpi=260,
    )
    plt.close()


def plot_dialogue_feature_correlations_imdb_rating_heatmap(corr_df):
    """
    Plot all 46 dialogue-feature correlations with IMDb rating as a compact
    one-column heatmap, mirroring
    plot_dialogue_mean_user_rating_correlation_heatmap().
    """
    corr_df = corr_df.copy().sort_values("pearson")

    labels = [
        get_feature_label(feature)
        for feature in corr_df["feature"]
    ]

    values = corr_df["pearson"].to_numpy().reshape(-1, 1)

    fig, ax = plt.subplots(figsize=(18, max(14, 0.72 * len(corr_df))))
    image = ax.imshow(values, aspect="auto", vmin=-0.25, vmax=0.25)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(
        "Pearson correlation with IMDb rating",
        fontsize=PRESENTATION_AXIS_LABEL_FONT_SIZE,
    )
    cbar.ax.tick_params(labelsize=PRESENTATION_TICK_FONT_SIZE)

    y = np.arange(len(labels))
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=max(9, PRESENTATION_TICK_FONT_SIZE - 5))
    ax.set_xticks([0])
    ax.set_xticklabels(["IMDb rating"], fontsize=PRESENTATION_TICK_FONT_SIZE)

    for i, value in enumerate(corr_df["pearson"]):
        text_color = "white" if abs(value) >= 0.12 else "black"
        ax.text(
            0,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=max(1, PRESENTATION_ANNOTATION_FONT_SIZE - 4),
            color=text_color,
        )

    ax.set_title(
        "Dialogue Feature Correlations\nwith IMDb Rating",
        fontsize=PRESENTATION_TITLE_FONT_SIZE,
        pad=34,
    )
    ax.set_ylabel("Dialogue feature", fontsize=PRESENTATION_AXIS_LABEL_FONT_SIZE)

    fig.subplots_adjust(left=0.66, right=0.86, top=0.88, bottom=0.08)

    plt.savefig(
        OUTPUT_DIR / "dialogue_feature_correlations_imdb_rating_heatmap.png",
        dpi=260,
    )
    plt.close()


def get_feature_label(feature):
    """
    Return a readable plot label for a dialogue feature column.
    """
    return DIALOGUE_FEATURE_LABELS.get(
        feature,
        str(feature).replace("_", " ").title(),
    )


def plot_dialogue_correlation_bar(corr_df):
    """
    Plot Pearson and Spearman correlations for the top-10 dialogue features
    by |Pearson| with IMDb rating.

    A second "all features" version of this bar chart used to also be saved,
    but it showed the same ~46 correlation values as
    plot_dialogue_feature_correlations_imdb_rating_heatmap() (just as bars
    instead of a heatmap), so it was dropped as a duplicate; the heatmap
    stays as the all-feature view and this stays as the top-10 highlight.
    """
    corr_df = corr_df.copy()

    top_corr_df = corr_df.sort_values("abs_pearson", ascending=False).head(10)
    plot_correlation_bar(
        top_corr_df,
        title="Top Dialogue Feature Correlations with IMDb Rating",
        output_path=OUTPUT_DIR / "dialogue_feature_correlations.png",
    )


def plot_correlation_bar(
    corr_df,
    title,
    output_path,
    height=6,
    x_label="Correlation with IMDb rating",
):
    """
    Plot horizontal Pearson and Spearman correlation bars.
    """
    corr_df = corr_df.copy().sort_values("pearson")

    y = np.arange(len(corr_df))
    height_bar = 0.35

    feature_labels = [
        get_feature_label(feature)
        for feature in corr_df["feature"]
    ]

    plt.figure(figsize=(18, max(height, 12)))
    plt.barh(y - height_bar / 2, corr_df["pearson"], height_bar, label="Pearson")
    plt.barh(y + height_bar / 2, corr_df["spearman"], height_bar, label="Spearman")

    plt.axvline(0, linewidth=1)
    plt.yticks(y, feature_labels, fontsize=max(9, TICK_FONT_SIZE - 3))
    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.title(title, fontsize=TITLE_FONT_SIZE)
    plt.xlabel(x_label, fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel("Dialogue feature", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=200)
    plt.close()


# -----------------------------
# Diagnostic and robustness checks
# -----------------------------

def analyze_data_quality_and_controls(dataset):
    """
    Run extra checks that help decide whether the exploratory results are reliable.

    Checks:
        - Subtitle-quality outliers and filtered dialogue correlations
        - Genre differences
        - Runtime/year confounding
        - First-vs-last franchise decline
        - Simple regression with controls
        - Dialogue feature redundancy
        - IMDb matching quality
        - Franchise vs non-franchise differences
    """
    lines = []
    lines.append("3. Robustness checks and diagnostics")
    lines.append("-" * 40)

    lines.extend(analyze_imdb_match_quality(dataset))
    lines.append("")

    lines.extend(analyze_subtitle_quality(dataset))
    lines.append("")

    lines.extend(analyze_dialogue_correlations_filtered(dataset))
    lines.append("")

    lines.extend(analyze_runtime_and_year_controls(dataset))
    lines.append("")

    lines.extend(analyze_rating_and_dialogue_by_year(dataset))
    lines.append("")

    lines.extend(analyze_genre_patterns(dataset))
    lines.append("")

    lines.extend(analyze_rating_by_decade_and_genre(dataset))
    lines.append("")

    lines.extend(analyze_negative_word_ratio_by_decade_and_genre(dataset))
    lines.append("")

    lines.extend(analyze_vocabulary_diversity_correlation_by_decade_and_genre(dataset))
    lines.append("")

    lines.extend(analyze_first_vs_last_franchise(dataset))
    lines.append("")

    lines.extend(analyze_franchise_vs_nonfranchise(dataset))
    lines.append("")

    lines.extend(analyze_feature_redundancy(dataset))
    lines.append("")

    lines.extend(analyze_rating_dialogue_control_heatmap(dataset))
    lines.append("")

    lines.extend(run_simple_regression_checks(dataset))

    return lines


def get_available_dialogue_features(dataset):
    """
    Return dialogue features that exist in the dataset.
    """
    return [
        feature for feature in DIALOGUE_FEATURES
        if feature in dataset.columns
    ]


def get_numeric_dataset(dataset, columns):
    """
    Return a copy containing numeric versions of selected columns.
    """
    df = dataset.copy()

    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def subtitle_feature_mask(dataset):
    """
    Mark rows that have extracted subtitle/dialogue features.

    In the expanded 1M + 32M dataset, many movies are expected to have no
    subtitle file. Those rows should not be counted as failed subtitle-quality
    checks.
    """
    if "num_tokens" not in dataset.columns or "num_lines" not in dataset.columns:
        return pd.Series(False, index=dataset.index)

    num_tokens = pd.to_numeric(dataset["num_tokens"], errors="coerce")
    num_lines = pd.to_numeric(dataset["num_lines"], errors="coerce")

    return num_tokens.notna() | num_lines.notna()


def subtitle_quality_mask(dataset):
    """
    Mark rows that pass basic subtitle-quality checks.

    Rows without subtitle features are marked False, but they are not counted as
    failed quality rows in analyze_subtitle_quality(). They simply have no
    subtitle file/features.
    """
    df = get_numeric_dataset(
        dataset,
        [
            "num_tokens",
            "num_lines",
            "subtitle_words_per_minute",
            "repeated_line_ratio",
            "type_token_ratio",
            "hapax_ratio",
        ],
    )

    mask = subtitle_feature_mask(df)

    if "num_tokens" in df.columns:
        mask &= df["num_tokens"] >= MIN_SUBTITLE_TOKENS

    if "num_lines" in df.columns:
        mask &= df["num_lines"] >= MIN_SUBTITLE_LINES

    if "subtitle_words_per_minute" in df.columns:
        mask &= (
            df["subtitle_words_per_minute"].isna()
            | (df["subtitle_words_per_minute"] <= MAX_SUBTITLE_WORDS_PER_MINUTE)
        )

    if "repeated_line_ratio" in df.columns:
        mask &= (
            df["repeated_line_ratio"].isna()
            | (df["repeated_line_ratio"] <= MAX_REPEATED_LINE_RATIO)
        )

    if "type_token_ratio" in df.columns:
        mask &= (
            df["type_token_ratio"].isna()
            | (df["type_token_ratio"] < 1)
        )

    if "hapax_ratio" in df.columns:
        mask &= (
            df["hapax_ratio"].isna()
            | (df["hapax_ratio"] > 0)
        )

    return mask.fillna(False)


def analyze_subtitle_quality(dataset):
    """
    Summarize subtitle availability and subtitle-quality filters.

    The expanded dataset contains many movies without subtitle files. Those rows
    are reported separately and are not counted as subtitle-quality failures.
    """
    lines = []
    lines.append("Subtitle availability and quality checks:")

    required_cols = {"num_tokens", "num_lines"}
    if not required_cols.issubset(set(dataset.columns)):
        lines.append("Skipped: subtitle count columns are missing.")
        return lines

    df = get_numeric_dataset(
        dataset,
        [
            "num_tokens",
            "num_lines",
            "subtitle_words_per_minute",
            "repeated_line_ratio",
            "type_token_ratio",
            "hapax_ratio",
        ],
    )

    has_subtitles = subtitle_feature_mask(df)
    subtitle_df = df.loc[has_subtitles].copy()

    if subtitle_df.empty:
        lines.append(f"Rows in dataset: {len(df):,}")
        lines.append("Rows with subtitle features: 0")
        lines.append(f"Rows without subtitle features: {len(df):,}")
        lines.append("Skipped subtitle-quality filters because no subtitle features were found.")
        return lines

    quality_mask = subtitle_quality_mask(subtitle_df)

    lines.append(f"Rows in dataset: {len(df):,}")
    lines.append(f"Rows with subtitle features: {int(has_subtitles.sum()):,}")
    lines.append(f"Rows without subtitle features: {int((~has_subtitles).sum()):,}")
    lines.append(f"Rows passing subtitle-quality checks: {int(quality_mask.sum()):,}")
    lines.append(
        "Rows removed by subtitle-quality filters among subtitle rows: "
        f"{int((~quality_mask).sum()):,}"
    )

    checks = {
        f"num_tokens < {MIN_SUBTITLE_TOKENS}": (
            subtitle_df["num_tokens"] < MIN_SUBTITLE_TOKENS
            if "num_tokens" in subtitle_df.columns
            else pd.Series(False, index=subtitle_df.index)
        ),
        f"num_lines < {MIN_SUBTITLE_LINES}": (
            subtitle_df["num_lines"] < MIN_SUBTITLE_LINES
            if "num_lines" in subtitle_df.columns
            else pd.Series(False, index=subtitle_df.index)
        ),
        f"subtitle_words_per_minute > {MAX_SUBTITLE_WORDS_PER_MINUTE}": (
            subtitle_df["subtitle_words_per_minute"] > MAX_SUBTITLE_WORDS_PER_MINUTE
            if "subtitle_words_per_minute" in subtitle_df.columns
            else pd.Series(False, index=subtitle_df.index)
        ),
        f"repeated_line_ratio > {MAX_REPEATED_LINE_RATIO}": (
            subtitle_df["repeated_line_ratio"] > MAX_REPEATED_LINE_RATIO
            if "repeated_line_ratio" in subtitle_df.columns
            else pd.Series(False, index=subtitle_df.index)
        ),
        "type_token_ratio == 1": (
            subtitle_df["type_token_ratio"] == 1
            if "type_token_ratio" in subtitle_df.columns
            else pd.Series(False, index=subtitle_df.index)
        ),
        "hapax_ratio == 0": (
            subtitle_df["hapax_ratio"] == 0
            if "hapax_ratio" in subtitle_df.columns
            else pd.Series(False, index=subtitle_df.index)
        ),
    }

    for label, check_mask in checks.items():
        lines.append(f"  {label}: {int(check_mask.fillna(False).sum()):,}")

    return lines


def analyze_dialogue_correlations_filtered(dataset):
    """
    Recompute dialogue-feature correlations after subtitle-quality filtering.
    """
    lines = []
    lines.append("Dialogue correlations after subtitle-quality filtering:")

    available_features = get_available_dialogue_features(dataset)

    if not available_features:
        lines.append("Skipped: no dialogue feature columns found.")
        return lines

    if RATING_COL not in dataset.columns:
        lines.append(f"Skipped: {RATING_COL} is missing.")
        return lines

    filtered = dataset.loc[subtitle_quality_mask(dataset)].copy()

    if filtered.empty:
        lines.append("Skipped: no subtitle rows passed subtitle-quality filtering.")
        return lines

    corr_df = compute_feature_correlations(filtered, available_features)

    if corr_df.empty:
        lines.append("Skipped: no feature had enough valid filtered rows.")
        return lines

    lines.append(f"Filtered rows with IMDb ratings: {filtered[RATING_COL].notna().sum():,}")

    top = corr_df.sort_values("abs_pearson", ascending=False).head(10)
    lines.append("Top filtered correlations:")
    for _, row in top.iterrows():
        lines.append(
            f"  {get_feature_label(row['feature'])}: "
            f"Pearson={row['pearson']:.4f}, "
            f"Spearman={row['spearman']:.4f}, "
            f"n={int(row['n'])}"
        )

    plot_correlation_bar(
        top,
        title="Top Dialogue Correlations after Subtitle-Quality Filtering",
        output_path=OUTPUT_DIR / "dialogue_feature_correlations_filtered.png",
    )

    return lines


def compute_feature_correlations(dataset, features):
    """
    Compute Pearson and Spearman correlations for a list of numeric features.
    """
    df = dataset[[RATING_COL] + features].copy()

    df[RATING_COL] = pd.to_numeric(df[RATING_COL], errors="coerce")

    correlations = []

    for feature in features:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")
        feature_df = df[[feature, RATING_COL]].dropna()

        if len(feature_df) < 2:
            continue

        correlations.append({
            "feature": feature,
            "pearson": feature_df[feature].corr(feature_df[RATING_COL], method="pearson"),
            "spearman": feature_df[feature].corr(feature_df[RATING_COL], method="spearman"),
            "n": len(feature_df),
        })

    corr_df = pd.DataFrame(correlations)

    if not corr_df.empty:
        corr_df["abs_pearson"] = corr_df["pearson"].abs()

    return corr_df


def analyze_runtime_and_year_controls(dataset):
    """
    Check whether runtime or release year may confound dialogue/rating patterns.
    """
    lines = []
    lines.append("Runtime and release-year checks:")

    cols = [RATING_COL, "Year", RUNTIME_COL, "num_tokens", "num_unique_tokens"]
    available_cols = [col for col in cols if col in dataset.columns]

    if RATING_COL not in available_cols:
        lines.append("Skipped: IMDb rating column is missing.")
        return lines

    df = get_numeric_dataset(dataset[available_cols].copy(), available_cols)

    pairs = [
        ("Year", RATING_COL),
        (RUNTIME_COL, RATING_COL),
        (RUNTIME_COL, "num_tokens"),
        (RUNTIME_COL, "num_unique_tokens"),
        ("Year", "num_tokens"),
    ]

    for x_col, y_col in pairs:
        if x_col not in df.columns or y_col not in df.columns:
            continue

        pair_df = df[[x_col, y_col]].dropna()

        if len(pair_df) < 2:
            continue

        pearson = pair_df[x_col].corr(pair_df[y_col], method="pearson")
        spearman = pair_df[x_col].corr(pair_df[y_col], method="spearman")
        lines.append(
            f"  {x_col} vs {y_col}: "
            f"Pearson={pearson:.4f}, Spearman={spearman:.4f}, n={len(pair_df):,}"
        )

    return lines


def get_genre_columns(dataset):
    """
    Return movie genre columns.
    """
    return [
        col for col in dataset.columns
        if col.startswith(GENRE_PREFIX)
    ]


def get_primary_genre_series(dataset):
    """
    Use Genre1 as a simple primary genre label.
    """
    genre_cols = get_genre_columns(dataset)

    if not genre_cols:
        return pd.Series(pd.NA, index=dataset.index)

    return dataset[genre_cols[0]].replace("", pd.NA)


def analyze_rating_and_dialogue_by_year(dataset):
    """
    Analyze whether movie year may explain rating and dialogue-feature patterns.

    This is useful because older movies, especially silent or early films, may
    receive different IMDb ratings and have unusual subtitle/dialogue patterns.
    """
    lines = []
    lines.append("Rating and dialogue features by release year:")

    required_cols = {"Year", RATING_COL}
    missing_cols = required_cols - set(dataset.columns)

    if missing_cols:
        lines.append(f"Skipped: missing columns {missing_cols}.")
        return lines

    available_features = [
        feature for feature in YEAR_TREND_DIALOGUE_FEATURES
        if feature in dataset.columns
    ]

    if not available_features:
        lines.append("Skipped: no selected dialogue features found.")
        return lines

    cols = ["Year", RATING_COL] + available_features
    df = dataset[cols].copy()

    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Year"])
    df["Year"] = df["Year"].astype(int)

    # Use only years with enough movies for a meaningful yearly average.
    yearly = (
        df
        .groupby("Year")
        .agg(
            movie_count=(RATING_COL, "count"),
            mean_rating=(RATING_COL, "mean"),
            **{
                f"mean_{feature}": (feature, "mean")
                for feature in available_features
            }
        )
        .reset_index()
    )

    yearly = yearly[yearly["movie_count"] >= 10].copy()

    if yearly.empty:
        lines.append("Skipped: no year had at least 10 rated movies.")
        return lines

    year_rating_corr = yearly["Year"].corr(yearly["mean_rating"], method="pearson")
    lines.append(
        f"Year vs mean IMDb rating by year: Pearson={year_rating_corr:.4f}, "
        f"years={len(yearly):,}"
    )

    lines.append("Year vs mean dialogue features by year:")
    year_feature_rows = []

    for feature in available_features:
        mean_feature_col = f"mean_{feature}"
        feature_df = yearly[["Year", mean_feature_col]].dropna()

        if len(feature_df) < 2:
            continue

        pearson = feature_df["Year"].corr(feature_df[mean_feature_col], method="pearson")
        spearman = feature_df["Year"].corr(feature_df[mean_feature_col], method="spearman")

        year_feature_rows.append({
            "feature": feature,
            "pearson": pearson,
            "spearman": spearman,
            "n": len(feature_df),
        })

        lines.append(
            f"  {get_feature_label(feature)}: "
            f"Pearson={pearson:.4f}, Spearman={spearman:.4f}, "
            f"years={len(feature_df):,}"
        )

    plot_rating_by_year(yearly)

    corr_df = pd.DataFrame(year_feature_rows)

    if not corr_df.empty:
        corr_df["abs_pearson"] = corr_df["pearson"].abs()
        plot_year_feature_correlation_bar(corr_df)

    # A separate mean-by-year line plot per dialogue feature used to also be
    # saved here (9 files, one per feature in YEAR_TREND_DIALOGUE_FEATURES).
    # plot_year_feature_correlation_bar() above already summarizes all 9
    # features' correlation with year in one chart, so the per-feature detail
    # plots were dropped as duplicates of that summary.

    return lines


def plot_rating_by_year(yearly):
    """
    Plot mean IMDb rating by release year.
    """
    plt.figure(figsize=(17, 11))
    plt.plot(yearly["Year"], yearly["mean_rating"], marker="o", linewidth=2.5, markersize=8)

    if yearly["Year"].nunique() > 1:
        slope, intercept = np.polyfit(yearly["Year"], yearly["mean_rating"], deg=1)
        x_line = np.linspace(yearly["Year"].min(), yearly["Year"].max(), 100)
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line, linewidth=2)

    plt.title("Mean IMDb Rating by Release Year", fontsize=TITLE_FONT_SIZE)
    plt.xlabel("Release year", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel("Mean IMDb rating", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "rating_by_year.png", dpi=200)
    plt.close()


def plot_year_feature_correlation_bar(corr_df):
    """
    Plot correlations between release year and yearly dialogue-feature averages.
    """
    corr_df = corr_df.sort_values("pearson").copy()

    y = np.arange(len(corr_df))
    height_bar = 0.35

    feature_labels = [
        get_feature_label(feature)
        for feature in corr_df["feature"]
    ]

    plt.figure(figsize=(18, max(12, 0.95 * len(corr_df))))
    plt.barh(y - height_bar / 2, corr_df["pearson"], height_bar, label="Pearson")
    plt.barh(y + height_bar / 2, corr_df["spearman"], height_bar, label="Spearman")

    plt.axvline(0, linewidth=1)
    plt.yticks(y, feature_labels, fontsize=max(9, TICK_FONT_SIZE - 3))
    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.title("Dialogue Feature Trends by Release Year", fontsize=TITLE_FONT_SIZE)
    plt.xlabel("Correlation with release year", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel("Dialogue feature", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "dialogue_feature_year_correlations.png", dpi=200)
    plt.close()


def analyze_genre_patterns(dataset):
    """
    Check whether ratings and dialogue-feature correlations differ by genre.
    """
    lines = []
    lines.append("Genre checks:")

    if RATING_COL not in dataset.columns:
        lines.append("Skipped: IMDb rating column is missing.")
        return lines

    genre = get_primary_genre_series(dataset)

    if genre.isna().all():
        lines.append("Skipped: no genre columns found.")
        return lines

    df = dataset.copy()
    df["PrimaryGenre"] = genre
    df[RATING_COL] = pd.to_numeric(df[RATING_COL], errors="coerce")

    grouped = (
        df.dropna(subset=["PrimaryGenre", RATING_COL])
        .groupby("PrimaryGenre")[RATING_COL]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    if grouped.empty:
        lines.append("Skipped: no valid genre/rating rows.")
        return lines

    lines.append("Mean IMDb rating by primary genre:")
    for _, row in grouped.iterrows():
        lines.append(
            f"  {row['PrimaryGenre']}: mean={row['mean']:.3f}, n={int(row['count'])}"
        )

    plot_rating_by_genre(grouped)

    interesting_features = [
        "repeated_line_ratio",
        "content_stemmed_type_token_ratio",
        "content_stemmed_repeated_short_phrase_ratio",
        "content_stemmed_top_word_frequency_ratio",
        "avg_line_length",
        "median_line_length",
        "long_word_ratio",
        "complex_word_ratio",
    ]
    available_features = [
        feature for feature in interesting_features
        if feature in df.columns
    ]

    if available_features:
        lines.append("Dialogue-feature correlations by genre, selected features:")
        for genre_name, genre_df in df.groupby("PrimaryGenre"):
            if len(genre_df) < 50:
                continue

            lines.append(f"  {genre_name}:")
            for feature in available_features:
                temp = genre_df[[feature, RATING_COL]].copy()
                temp[feature] = pd.to_numeric(temp[feature], errors="coerce")
                temp = temp.dropna()

                if len(temp) < 30:
                    continue

                pearson = temp[feature].corr(temp[RATING_COL], method="pearson")
                lines.append(
                    f"    {get_feature_label(feature)}: Pearson={pearson:.4f}, n={len(temp):,}"
                )

    return lines


def plot_rating_by_genre(grouped):
    """
    Plot mean IMDb rating by primary genre.
    """
    plot_df = grouped.sort_values("mean", ascending=True)

    plt.figure(figsize=(18, max(12, 0.8 * len(plot_df))))
    y = np.arange(len(plot_df))
    plt.barh(y, plot_df["mean"])

    labels = [
        f"{row['PrimaryGenre']} (n={int(row['count'])})"
        for _, row in plot_df.iterrows()
    ]

    plt.yticks(y, labels, fontsize=max(9, TICK_FONT_SIZE - 3))
    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.xlabel("Mean IMDb rating", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel("Primary genre", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.title("Mean IMDb Rating by Primary Genre", fontsize=TITLE_FONT_SIZE)
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "rating_by_genre.png", dpi=200)
    plt.close()


def analyze_rating_by_decade_and_genre(dataset):
    """
    Analyze IMDb ratings by release decade and primary genre.

    This helps check whether rating patterns differ across both time period and
    genre, instead of looking only at one variable at a time.
    """
    lines = []
    lines.append("IMDb rating by decade and genre:")

    required_cols = {"Year", RATING_COL}
    missing_cols = required_cols - set(dataset.columns)

    if missing_cols:
        lines.append(f"Skipped: missing columns {missing_cols}.")
        return lines

    genre = get_primary_genre_series(dataset)

    if genre.isna().all():
        lines.append("Skipped: no genre columns found.")
        return lines

    df = dataset[["Year", RATING_COL]].copy()
    df["PrimaryGenre"] = genre
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df[RATING_COL] = pd.to_numeric(df[RATING_COL], errors="coerce")
    df = df.dropna(subset=["Year", RATING_COL, "PrimaryGenre"])

    if df.empty:
        lines.append("Skipped: no valid rows with year, genre, and IMDb rating.")
        return lines

    df["Decade"] = (df["Year"].astype(int) // 10) * 10
    df["DecadeLabel"] = df["Decade"].astype(int).astype(str) + "s"

    # Keep only reasonably populated decade/genre cells, so the heatmap is not
    # dominated by very small groups.
    grouped = (
        df
        .groupby(["PrimaryGenre", "DecadeLabel"], observed=True)[RATING_COL]
        .agg(["mean", "count"])
        .reset_index()
    )

    min_cell_count = 10
    grouped = grouped[grouped["count"] >= min_cell_count].copy()

    if grouped.empty:
        lines.append(
            "Skipped: no decade/genre group had enough movies "
            f"(minimum {min_cell_count})."
        )
        return lines

    plot_rating_by_decade_genre_heatmap(grouped)

    lines.append(
        "Saved IMDb rating by decade and genre heatmap to "
        "imdb_rating_by_decade_genre_heatmap.png."
    )

    return lines


def plot_rating_by_decade_genre_heatmap(grouped):
    """
    Plot mean IMDb rating by primary genre and release decade as a heatmap.
    """
    heatmap_df = grouped.pivot(
        index="PrimaryGenre",
        columns="DecadeLabel",
        values="mean",
    )

    # Sort genres by their overall mean rating and decades chronologically.
    genre_order = (
        grouped
        .groupby("PrimaryGenre")["mean"]
        .mean()
        .sort_values(ascending=True)
        .index
    )

    decade_order = sorted(
        heatmap_df.columns,
        key=lambda value: int(str(value).replace("s", "")),
    )

    heatmap_df = heatmap_df.reindex(index=genre_order, columns=decade_order)

    # Keep the canvas wide, but crop the right side and give more room to the
    # left side for long genre labels.
    fig_width = max(17, 1.35 * len(heatmap_df.columns))
    fig_height = max(10, 0.65 * len(heatmap_df.index))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(heatmap_df.values, aspect="auto", vmin=4.5, vmax=7.5)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Mean IMDb rating", fontsize=AXIS_LABEL_FONT_SIZE)
    cbar.ax.tick_params(labelsize=TICK_FONT_SIZE)

    x = np.arange(len(heatmap_df.columns))
    y = np.arange(len(heatmap_df.index))

    ax.set_xticks(x)
    ax.set_yticks(y)
    ax.set_xticklabels(heatmap_df.columns, rotation=45, ha="right", fontsize=TICK_FONT_SIZE)
    ax.set_yticklabels(heatmap_df.index, fontsize=TICK_FONT_SIZE)

    for i in range(len(heatmap_df.index)):
        for j in range(len(heatmap_df.columns)):
            value = heatmap_df.iloc[i, j]

            if pd.isna(value):
                continue

            text_color = "white" if value < 5.8 else "black"
            ax.text(
                j,
                i,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=max(1, ANNOTATION_FONT_SIZE - 6),
                color=text_color,
            )

    ax.set_title(
        "Mean IMDb Rating by Decade and Genre",
        fontsize=TITLE_FONT_SIZE,
        pad=24,
    )
    ax.set_xlabel("Release decade", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel("Primary genre", fontsize=AXIS_LABEL_FONT_SIZE)

    # More left margin for the y-axis label and genre names; less right margin
    # so the figure is cropped more tightly after the colorbar.
    fig.subplots_adjust(left=0.26, right=0.89, top=0.90, bottom=0.18)

    plt.savefig(OUTPUT_DIR / "imdb_rating_by_decade_genre_heatmap.png", dpi=220)
    plt.close()


def analyze_negative_word_ratio_by_decade_and_genre(dataset):
    """
    Analyze negative_word_ratio (the most heavily weighted dialogue feature in
    the recommender's DIALOGUE_FEATURE_WEIGHTS) by release decade and primary
    genre, mirroring analyze_rating_by_decade_and_genre() so the two heatmaps
    are directly comparable.
    """
    lines = []
    lines.append("negative_word_ratio by decade and genre:")

    feature_col = "negative_word_ratio"
    required_cols = {"Year", feature_col}
    missing_cols = required_cols - set(dataset.columns)

    if missing_cols:
        lines.append(f"Skipped: missing columns {missing_cols}.")
        return lines

    genre = get_primary_genre_series(dataset)

    if genre.isna().all():
        lines.append("Skipped: no genre columns found.")
        return lines

    df = dataset[["Year", feature_col]].copy()
    df["PrimaryGenre"] = genre
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df[feature_col] = pd.to_numeric(df[feature_col], errors="coerce")
    df = df.dropna(subset=["Year", feature_col, "PrimaryGenre"])

    if df.empty:
        lines.append("Skipped: no valid rows with year, genre, and negative_word_ratio.")
        return lines

    df["Decade"] = (df["Year"].astype(int) // 10) * 10
    df["DecadeLabel"] = df["Decade"].astype(int).astype(str) + "s"

    # Keep only reasonably populated decade/genre cells, so the heatmap is not
    # dominated by very small groups.
    grouped = (
        df
        .groupby(["PrimaryGenre", "DecadeLabel"], observed=True)[feature_col]
        .agg(["mean", "count"])
        .reset_index()
    )

    min_cell_count = 10
    grouped = grouped[grouped["count"] >= min_cell_count].copy()

    if grouped.empty:
        lines.append(
            "Skipped: no decade/genre group had enough movies "
            f"(minimum {min_cell_count})."
        )
        return lines

    plot_negative_word_ratio_by_decade_genre_heatmap(grouped)

    lines.append(
        "Saved negative_word_ratio by decade and genre heatmap to "
        "negative_word_ratio_by_decade_genre_heatmap.png."
    )

    return lines


def plot_negative_word_ratio_by_decade_genre_heatmap(grouped):
    """
    Plot mean negative_word_ratio by primary genre and release decade as a
    heatmap, in the same layout as plot_rating_by_decade_genre_heatmap().
    """
    heatmap_df = grouped.pivot(
        index="PrimaryGenre",
        columns="DecadeLabel",
        values="mean",
    )

    # Sort genres by their overall mean ratio (ascending = least negative
    # dialogue first) and decades chronologically.
    genre_order = (
        grouped
        .groupby("PrimaryGenre")["mean"]
        .mean()
        .sort_values(ascending=True)
        .index
    )

    decade_order = sorted(
        heatmap_df.columns,
        key=lambda value: int(str(value).replace("s", "")),
    )

    heatmap_df = heatmap_df.reindex(index=genre_order, columns=decade_order)

    # Data-driven color range (5th-95th percentile of cell means) instead of a
    # fixed range, since negative_word_ratio has no natural bounds like a
    # 1-10 rating scale and is long-tailed.
    cell_values = heatmap_df.values
    valid_values = cell_values[~pd.isna(cell_values)]
    vmin = float(np.percentile(valid_values, 5))
    vmax = float(np.percentile(valid_values, 95))
    midpoint = (vmin + vmax) / 2

    # Keep the canvas wide, but crop the right side and give more room to the
    # left side for long genre labels.
    fig_width = max(17, 1.35 * len(heatmap_df.columns))
    fig_height = max(10, 0.65 * len(heatmap_df.index))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(heatmap_df.values, aspect="auto", cmap="Reds", vmin=vmin, vmax=vmax)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Mean Negative-Word Ratio (× 1000)", fontsize=AXIS_LABEL_FONT_SIZE)
    cbar.ax.tick_params(labelsize=TICK_FONT_SIZE)
    cbar.ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda t, _pos: f"{t * 1000:.0f}")
    )

    x = np.arange(len(heatmap_df.columns))
    y = np.arange(len(heatmap_df.index))

    ax.set_xticks(x)
    ax.set_yticks(y)
    ax.set_xticklabels(heatmap_df.columns, rotation=45, ha="right", fontsize=TICK_FONT_SIZE)
    ax.set_yticklabels(heatmap_df.index, fontsize=TICK_FONT_SIZE)

    for i in range(len(heatmap_df.index)):
        for j in range(len(heatmap_df.columns)):
            value = heatmap_df.iloc[i, j]

            if pd.isna(value):
                continue

            text_color = "white" if value > midpoint else "black"
            ax.text(
                j,
                i,
                f"{value * 1000:.0f}",
                ha="center",
                va="center",
                fontsize=max(1, ANNOTATION_FONT_SIZE - 6),
                color=text_color,
            )

    ax.set_title(
        "Mean Negative-Word Ratio by Decade and Genre",
        fontsize=TITLE_FONT_SIZE,
        pad=24,
    )
    ax.set_xlabel("Release decade", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel("Primary genre", fontsize=AXIS_LABEL_FONT_SIZE)

    # More left margin for the y-axis label and genre names; less right margin
    # so the figure is cropped more tightly after the colorbar.
    fig.subplots_adjust(left=0.26, right=0.89, top=0.90, bottom=0.18)

    plt.savefig(OUTPUT_DIR / "negative_word_ratio_by_decade_genre_heatmap.png", dpi=220)
    plt.close()


def analyze_vocabulary_diversity_correlation_by_decade_and_genre(dataset):
    """
    Analyze the correlation between vocabulary diversity ratio and IMDb rating
    separately by release decade and primary genre.
    """
    lines = []
    lines.append("Vocabulary diversity vs IMDb rating by decade and genre:")

    feature_col = "type_token_ratio"
    required_cols = {"Year", RATING_COL, feature_col}
    missing_cols = required_cols - set(dataset.columns)

    if missing_cols:
        lines.append(f"Skipped: missing columns {missing_cols}.")
        return lines

    genre = get_primary_genre_series(dataset)

    if genre.isna().all():
        lines.append("Skipped: no genre columns found.")
        return lines

    df = dataset[["Year", RATING_COL, feature_col]].copy()
    df["PrimaryGenre"] = genre
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df[RATING_COL] = pd.to_numeric(df[RATING_COL], errors="coerce")
    df[feature_col] = pd.to_numeric(df[feature_col], errors="coerce")
    df = df.dropna(subset=["Year", RATING_COL, feature_col, "PrimaryGenre"])

    if df.empty:
        lines.append(
            "Skipped: no valid rows with year, genre, IMDb rating, and "
            "vocabulary diversity ratio."
        )
        return lines

    df["Decade"] = (df["Year"].astype(int) // 10) * 10
    df["DecadeLabel"] = df["Decade"].astype(int).astype(str) + "s"

    min_cell_count = 30
    rows = []

    for (genre_name, decade_label), group in df.groupby(
        ["PrimaryGenre", "DecadeLabel"],
        observed=True,
    ):
        if len(group) < min_cell_count:
            continue

        if group[feature_col].nunique() <= 1 or group[RATING_COL].nunique() <= 1:
            continue

        correlation = group[feature_col].corr(group[RATING_COL], method="pearson")

        if pd.isna(correlation):
            continue

        rows.append({
            "PrimaryGenre": genre_name,
            "DecadeLabel": decade_label,
            "correlation": correlation,
            "count": len(group),
        })

    corr_df = pd.DataFrame(rows)

    if corr_df.empty:
        lines.append(
            "Skipped: no decade/genre group had enough valid movies "
            f"(minimum {min_cell_count})."
        )
        return lines

    plot_vocabulary_diversity_correlation_by_decade_genre_heatmap(corr_df)

    lines.append(
        "Saved vocabulary diversity vs IMDb rating correlation heatmap to "
        "vocabulary_diversity_imdb_correlation_by_decade_genre.png."
    )

    return lines


def plot_vocabulary_diversity_correlation_by_decade_genre_heatmap(corr_df):
    """
    Plot Pearson correlation between vocabulary diversity ratio and IMDb rating
    by primary genre and release decade.
    """
    heatmap_df = corr_df.pivot(
        index="PrimaryGenre",
        columns="DecadeLabel",
        values="correlation",
    )

    # Sort genres by their overall average correlation and decades
    # chronologically, so the visual order is meaningful and stable.
    genre_order = (
        corr_df
        .groupby("PrimaryGenre")["correlation"]
        .mean()
        .sort_values(ascending=True)
        .index
    )

    decade_order = sorted(
        heatmap_df.columns,
        key=lambda value: int(str(value).replace("s", "")),
    )

    heatmap_df = heatmap_df.reindex(index=genre_order, columns=decade_order)

    # Slightly larger in both directions than before.
    fig_width = max(19, 1.55 * len(heatmap_df.columns))
    fig_height = max(12, 0.82 * len(heatmap_df.index))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(heatmap_df.values, aspect="auto", vmin=-0.5, vmax=0.5)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(
        "Pearson correlation with IMDb rating",
        fontsize=AXIS_LABEL_FONT_SIZE,
    )
    cbar.ax.tick_params(labelsize=TICK_FONT_SIZE)

    x = np.arange(len(heatmap_df.columns))
    y = np.arange(len(heatmap_df.index))

    ax.set_xticks(x)
    ax.set_yticks(y)
    ax.set_xticklabels(
        heatmap_df.columns,
        rotation=45,
        ha="right",
        fontsize=TICK_FONT_SIZE,
    )
    ax.set_yticklabels(heatmap_df.index, fontsize=TICK_FONT_SIZE)

    for i in range(len(heatmap_df.index)):
        for j in range(len(heatmap_df.columns)):
            value = heatmap_df.iloc[i, j]

            if pd.isna(value):
                continue

            text_color = "white" if abs(value) >= 0.25 else "black"
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=max(1, ANNOTATION_FONT_SIZE - 8),
                color=text_color,
            )

    ax.set_title(
        "Vocabulary Diversity Ratio and IMDb Rating Correlation\nby Decade and Genre",
        fontsize=TITLE_FONT_SIZE - 4,
        pad=26,
    )
    ax.set_xlabel("Release decade", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel("Primary genre", fontsize=AXIS_LABEL_FONT_SIZE)

    # Larger canvas with balanced room on all sides.
    fig.subplots_adjust(left=0.24, right=0.90, top=0.88, bottom=0.20)

    plt.savefig(
        OUTPUT_DIR / "vocabulary_diversity_imdb_correlation_by_decade_genre.png",
        dpi=220,
    )
    plt.close()


def analyze_first_vs_last_franchise(dataset):
    """
    Compare first and last installments within franchises.
    """
    lines = []
    lines.append("First-vs-last franchise check:")

    required_cols = {FRANCHISE_ID_COL, INSTALLMENT_COL, RATING_COL}

    if not required_cols.issubset(set(dataset.columns)):
        lines.append(f"Skipped: missing columns {required_cols - set(dataset.columns)}")
        return lines

    df = dataset[[FRANCHISE_ID_COL, INSTALLMENT_COL, RATING_COL]].copy()
    df[INSTALLMENT_COL] = pd.to_numeric(df[INSTALLMENT_COL], errors="coerce")
    df[RATING_COL] = pd.to_numeric(df[RATING_COL], errors="coerce")
    df = df.dropna()

    comparisons = []

    for franchise_id, group in df.groupby(FRANCHISE_ID_COL):
        group = group.sort_values(INSTALLMENT_COL)

        if group[INSTALLMENT_COL].nunique() < 2:
            continue

        first = group.iloc[0]
        last = group.iloc[-1]

        comparisons.append({
            "FranchiseID": franchise_id,
            "first_rating": first[RATING_COL],
            "last_rating": last[RATING_COL],
            "difference": last[RATING_COL] - first[RATING_COL],
            "first_installment": first[INSTALLMENT_COL],
            "last_installment": last[INSTALLMENT_COL],
        })

    comp_df = pd.DataFrame(comparisons)

    if comp_df.empty:
        lines.append("Skipped: no franchise had both first and later installments.")
        return lines

    mean_first = comp_df["first_rating"].mean()
    mean_last = comp_df["last_rating"].mean()
    mean_diff = comp_df["difference"].mean()
    median_diff = comp_df["difference"].median()
    share_lower = (comp_df["difference"] < 0).mean()

    lines.append(f"Comparable franchises: {len(comp_df):,}")
    lines.append(f"Mean first-installment rating: {mean_first:.4f}")
    lines.append(f"Mean last-installment rating: {mean_last:.4f}")
    lines.append(f"Mean last-minus-first rating difference: {mean_diff:.4f}")
    lines.append(f"Median last-minus-first rating difference: {median_diff:.4f}")
    lines.append(f"Share of franchises where last movie < first movie: {share_lower:.2%}")

    return lines


def analyze_franchise_vs_nonfranchise(dataset):
    """
    Compare franchise and non-franchise movies-1M.
    """
    lines = []
    lines.append("Franchise vs non-franchise comparison:")

    if FRANCHISE_ID_COL not in dataset.columns or RATING_COL not in dataset.columns:
        lines.append("Skipped: franchise or rating column is missing.")
        return lines

    df = dataset.copy()
    df[RATING_COL] = pd.to_numeric(df[RATING_COL], errors="coerce")
    df["is_franchise"] = df[FRANCHISE_ID_COL].notna()

    compare_cols = [
        RATING_COL,
        RUNTIME_COL,
        "avg_line_length",
        "repeated_line_ratio",
        "long_word_ratio",
        "subtitle_words_per_minute",
    ]

    compare_cols = [col for col in compare_cols if col in df.columns]

    if not compare_cols:
        lines.append("Skipped: no comparable numeric columns found.")
        return lines

    for col in compare_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    grouped = df.groupby("is_franchise")[compare_cols].mean(numeric_only=True)

    for is_franchise, row in grouped.iterrows():
        label = "franchise movies-1M" if is_franchise else "non-franchise movies-1M"
        count = int((df["is_franchise"] == is_franchise).sum())
        lines.append(f"  {label}: n={count:,}")
        for col in compare_cols:
            value = row[col]
            if pd.notna(value):
                label_col = get_feature_label(col) if col in DIALOGUE_FEATURE_LABELS else col
                lines.append(f"    mean {label_col}: {value:.4f}")

    return lines


def analyze_feature_redundancy(dataset):
    """
    Identify highly correlated dialogue features.
    """
    lines = []
    lines.append("Dialogue feature redundancy:")

    features = get_available_dialogue_features(dataset)

    if len(features) < 2:
        lines.append("Skipped: fewer than two dialogue features found.")
        return lines

    df = dataset[features].copy()

    for feature in features:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")

    corr = df.corr(method="pearson")

    pairs = []
    for i, feature_a in enumerate(features):
        for feature_b in features[i + 1:]:
            value = corr.loc[feature_a, feature_b]

            if pd.isna(value):
                continue

            if abs(value) >= 0.85:
                pairs.append((feature_a, feature_b, value))

    if not pairs:
        lines.append("No dialogue feature pairs had |correlation| >= 0.85.")
    else:
        pairs = sorted(pairs, key=lambda item: abs(item[2]), reverse=True)
        lines.append("Highly redundant feature pairs, |r| >= 0.85:")
        for feature_a, feature_b, value in pairs[:20]:
            lines.append(
                f"  {get_feature_label(feature_a)} vs {get_feature_label(feature_b)}: "
                f"r={value:.4f}"
            )

    plot_feature_redundancy_heatmap(corr)

    return lines


def plot_feature_redundancy_heatmap(corr):
    """
    Plot a compact heatmap of dialogue-feature correlations.
    """
    if corr.empty:
        return

    # Keep the first 25 features to avoid an unreadable plot.
    if len(corr.columns) > 25:
        corr = corr.iloc[:25, :25]

    labels = [
        get_feature_label(feature)
        for feature in corr.columns
    ]

    plt.figure(figsize=(16, 13))
    plt.imshow(corr.values, aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(label="Pearson correlation")

    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=90, fontsize=8)
    plt.yticks(ticks, labels, fontsize=8)
    plt.title("Dialogue Feature Redundancy Heatmap", fontsize=TITLE_FONT_SIZE)
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "dialogue_feature_redundancy_heatmap.png", dpi=200)
    plt.close()


def analyze_imdb_match_quality(dataset):
    """
    Summarize IMDb matching coverage and possible mismatches.
    """
    lines = []
    lines.append("IMDb match-quality checks:")

    if RATING_COL not in dataset.columns:
        lines.append(f"Skipped: {RATING_COL} is missing.")
        return lines

    total = len(dataset)
    matched = dataset[RATING_COL].notna().sum()

    lines.append(f"Rows: {total:,}")
    lines.append(f"Rows with IMDb rating: {matched:,} ({matched / total:.2%})")

    if "imdb_tconst" in dataset.columns:
        tconst_matches = dataset["imdb_tconst"].notna().sum()
        lines.append(f"Rows with IMDb title ID: {tconst_matches:,} ({tconst_matches / total:.2%})")

    if IMDB_TITLE_COL in dataset.columns:
        sample_cols = ["Title", "Year", IMDB_TITLE_COL, RATING_COL]
        sample_cols = [col for col in sample_cols if col in dataset.columns]
        title_norm = dataset["Title"].apply(normalize_title_for_comparison)
        imdb_title_norm = dataset[IMDB_TITLE_COL].apply(normalize_title_for_comparison)

        mismatch = dataset[
            dataset[IMDB_TITLE_COL].notna()
            & (title_norm != imdb_title_norm)
        ]

        lines.append(
            f"Rows where MovieLens title text differs from IMDb title text: {len(mismatch):,}"
        )

        if not mismatch.empty:
            lines.append("Sample title differences:")
            for _, row in mismatch[sample_cols].head(10).iterrows():
                lines.append(
                    f"  MovieLens='{row.get('Title')}', "
                    f"IMDb='{row.get(IMDB_TITLE_COL)}', "
                    f"Year={row.get('Year')}, "
                    f"Rating={row.get(RATING_COL)}"
                )

    return lines


def normalize_title_for_comparison(title):
    """
    Normalize titles for IMDb match-quality diagnostics.

    Removes alternate-title parentheticals so cases like
    "Seven (a.k.a. Se7en)" and "Seven" are not counted as suspicious.
    """
    title = str(title).lower().strip()
    title = re.sub(r"\(.*?\)", "", title)
    title = re.sub(r"[^a-z0-9\s]", " ", title)
    title = re.sub(r"\s+", " ", title)
    return title.strip()


def analyze_rating_dialogue_control_heatmap(dataset):
    """
    Create a simple correlation plot for IMDb rating against selected metadata
    and dialogue features.

    This replaces the busy full correlation heatmap with a clearer visualization:
    one horizontal bar per variable, showing only its correlation with IMDb rating.
    """
    lines = []
    lines.append("Rating, metadata, and dialogue-feature correlation plot:")

    selected_cols = [
        RATING_COL,
        "Year",
        RUNTIME_COL,
        "repeated_line_ratio",
        "avg_line_length",
        "median_line_length",
        "long_word_ratio",
        "complex_word_ratio",
        "negative_word_ratio",
        "anger_word_ratio",
        "subtitle_words_per_minute",
        "num_lines_per_minute",
        "type_token_ratio",
        "flesch_reading_ease",
        "average_sentence_length",
    ]

    selected_cols = [
        col for col in selected_cols
        if col in dataset.columns
    ]

    if RATING_COL not in selected_cols or len(selected_cols) < 3:
        lines.append("Skipped: IMDb rating or selected numeric columns are unavailable.")
        return lines

    df = dataset[selected_cols].copy()

    for col in selected_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    rows = []

    for col in selected_cols:
        if col == RATING_COL:
            continue

        pair_df = df[[RATING_COL, col]].dropna()

        if len(pair_df) < 30 or pair_df[col].nunique() <= 1:
            continue

        rows.append(
            {
                "feature": col,
                "correlation": pair_df[col].corr(pair_df[RATING_COL], method="pearson"),
                "spearman": pair_df[col].corr(pair_df[RATING_COL], method="spearman"),
                "n": len(pair_df),
            }
        )

    corr_df = pd.DataFrame(rows)

    if corr_df.empty:
        lines.append("Skipped: no selected column had enough valid values.")
        return lines

    corr_df["abs_correlation"] = corr_df["correlation"].abs()
    corr_df = corr_df.sort_values("correlation")

    # A one-column heatmap of the same values used to also be saved here
    # (rating_dialogue_control_heatmap.png). It showed the identical numbers
    # as the bar chart below in a different chart type, so it was dropped as
    # a duplicate.
    plot_rating_dialogue_control_correlation_bars(corr_df)

    strongest = corr_df.sort_values("abs_correlation", ascending=False).iloc[0]
    strongest_label = (
        get_feature_label(strongest["feature"])
        if strongest["feature"] in DIALOGUE_FEATURE_LABELS
        else {
            "Year": "Release year",
            RUNTIME_COL: "Runtime in minutes",
        }.get(strongest["feature"], str(strongest["feature"]).replace("_", " ").title())
    )

    lines.append(
        "Strongest absolute correlation with IMDb rating in this plot: "
        f"{strongest_label} ({strongest['correlation']:.4f}, n={int(strongest['n']):,})."
    )
    lines.append(
        "Saved rating/metadata/dialogue correlation plot to "
        "rating_dialogue_control_correlations.png."
    )

    return lines


def plot_rating_dialogue_control_correlation_bars(corr_df):
    """
    Plot selected metadata and dialogue features by correlation with IMDb rating.

    This is a simpler alternative to a full heatmap because it focuses only on
    the relationship with the outcome variable.
    """
    corr_df = corr_df.copy().sort_values("spearman")

    labels = [
        get_feature_label(feature) if feature in DIALOGUE_FEATURE_LABELS else {
            "Year": "Release year",
            RUNTIME_COL: "Runtime in minutes",
        }.get(feature, str(feature).replace("_", " ").title())
        for feature in corr_df["feature"]
    ]

    y = np.arange(len(corr_df))

    # Slim canvas, but still tall enough for readable labels.
    fig, ax = plt.subplots(figsize=(19, max(13, 1.05 * len(corr_df))))
    bars = ax.barh(y, corr_df["spearman"], height=0.78)
    ax.axvline(0, linewidth=2.4)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=PRESENTATION_TICK_FONT_SIZE)
    ax.tick_params(axis="x", labelsize=PRESENTATION_TICK_FONT_SIZE)

    max_abs = max(0.01, corr_df["spearman"].abs().max())

    # Extra space is needed because all value labels are now outside the bars.
    ax.set_xlim(-max_abs * 1.55, max_abs * 1.55)

    value_label_font_size = max(1, PRESENTATION_ANNOTATION_FONT_SIZE - 1)
    label_offset = max_abs * 0.055

    for bar, value in zip(bars, corr_df["spearman"]):
        y_pos = bar.get_y() + bar.get_height() / 2

        # Place every label outside its bar: right of positive bars,
        # left of negative bars.
        if value >= 0:
            x_pos = value + label_offset
            ha = "left"
        else:
            x_pos = value - label_offset
            ha = "right"

        ax.text(
            x_pos,
            y_pos,
            f"{value:.2f}",
            va="center",
            ha=ha,
            fontsize=value_label_font_size,
        )

    ax.set_title(
        "Movie Feature Spearman Correlations\nwith IMDb Rating",
        fontsize=PRESENTATION_TITLE_FONT_SIZE,
        pad=18,
    )
    ax.set_xlabel(
        "Spearman correlation with IMDb rating",
        fontsize=PRESENTATION_AXIS_LABEL_FONT_SIZE,
    )
    ax.set_ylabel(
        "Feature",
        fontsize=PRESENTATION_AXIS_LABEL_FONT_SIZE,
    )
    ax.grid(True, axis="x", alpha=0.3)

    # Slimmer canvas means long y labels need enough left margin.
    fig.subplots_adjust(left=0.42, right=0.96, top=0.90, bottom=0.14)

    plt.savefig(OUTPUT_DIR / "rating_dialogue_control_correlations.png", dpi=260)
    plt.close()


def run_simple_regression_checks(dataset):
    """
    Run lightweight linear regressions with controls.

    These are not final predictive models. They check whether:
        1. franchise installment remains negative inside franchise movies-1M;
        2. dialogue features remain useful across all IMDb-matched movies-1M.
    """
    lines = []
    lines.append("Simple regression checks:")

    if RATING_COL not in dataset.columns:
        lines.append("Skipped: IMDb rating column is missing.")
        return lines

    franchise_features = [
        INSTALLMENT_COL,
        "avg_line_length",
        "repeated_line_ratio",
        "long_word_ratio",
        "subtitle_words_per_minute",
        "Year",
        RUNTIME_COL,
    ]

    all_movie_dialogue_features = [
        "avg_line_length",
        "repeated_line_ratio",
        "long_word_ratio",
        "subtitle_words_per_minute",
        "Year",
        RUNTIME_COL,
    ]

    lines.extend(
        run_one_regression(
            dataset=dataset,
            features=franchise_features,
            title="Franchise-only regression with controls",
            require_columns=[INSTALLMENT_COL],
        )
    )

    lines.append("")

    lines.extend(
        run_one_regression(
            dataset=dataset,
            features=all_movie_dialogue_features,
            title="All-movies-1M dialogue-feature regression with controls",
            require_columns=[],
        )
    )

    return lines


def run_one_regression(
    dataset,
    features,
    title,
    require_columns,
):
    """
    Run one standardized OLS regression and return printable summary lines.
    """
    lines = []
    lines.append(title + ":")

    selected_features = [
        col for col in features
        if col in dataset.columns
    ]

    required = set(require_columns)
    missing_required = required - set(dataset.columns)
    if missing_required:
        lines.append(f"Skipped: missing required columns {missing_required}.")
        return lines

    if len(selected_features) < 2:
        lines.append("Skipped: not enough numeric predictors.")
        return lines

    df = dataset[[RATING_COL] + selected_features].copy()

    for col in [RATING_COL] + selected_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    genre = get_primary_genre_series(dataset)

    if not genre.isna().all():
        genre_dummies = pd.get_dummies(genre, prefix="genre", drop_first=True)
        counts = genre.value_counts()
        keep_genres = counts[counts >= 100].index
        genre_dummies = genre_dummies[
            [
                col for col in genre_dummies.columns
                if col.replace("genre_", "") in keep_genres
            ]
        ]
        df = pd.concat([df, genre_dummies], axis=1)

    df = df.dropna()

    if len(df) < 50:
        lines.append("Skipped: fewer than 50 complete rows.")
        return lines

    y = df[RATING_COL].astype(float).to_numpy()
    X_df = df.drop(columns=[RATING_COL]).astype(float)

    coefficients, r_squared = fit_linear_regression(X_df, y)

    lines.append(f"Rows used in regression: {len(df):,}")
    lines.append(f"R-squared: {r_squared:.4f}")
    lines.append("Selected coefficients:")

    for feature in selected_features:
        if feature in coefficients:
            label = get_feature_label(feature) if feature in DIALOGUE_FEATURE_LABELS else feature
            lines.append(f"  {label}: {coefficients[feature]:.4f}")

    return lines


def fit_linear_regression(X_df, y):
    """
    Fit ordinary least squares using numpy.

    Continuous columns are standardized so coefficients are comparable.
    Dummy columns stay as 0/1 controls.
    """
    X = X_df.copy()

    standardized_cols = []
    for col in X.columns:
        values = X[col].astype(float)
        unique_values = set(values.dropna().unique())

        if unique_values.issubset({0.0, 1.0}):
            continue

        std = values.std()

        if std == 0 or pd.isna(std):
            X[col] = 0
        else:
            X[col] = (values - values.mean()) / std
            standardized_cols.append(col)

    X_matrix = np.column_stack([
        np.ones(len(X)),
        X.to_numpy(dtype=float),
    ])

    beta, *_ = np.linalg.lstsq(X_matrix, y, rcond=None)

    y_pred = X_matrix @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot else 0

    coefficients = {
        col: beta[i + 1]
        for i, col in enumerate(X.columns)
    }

    return coefficients, r_squared


if __name__ == "__main__":
    main()
