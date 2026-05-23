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
        franchise_rating_change_from_first.png
        dialogue_feature_correlations.png
        dialogue_feature_correlations_all.png
        dialogue_feature_vs_imdb_rating_<feature>.png
        dialogue_feature_correlations_filtered.png
        dialogue_feature_redundancy_heatmap.png
        rating_by_genre.png
        analysis_summary.txt

Run:
    python analyze_data.py
"""

from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASET_PATH = Path("dataset.csv")
OUTPUT_DIR = Path("analysis_outputs")

TITLE_FONT_SIZE = 22
AXIS_LABEL_FONT_SIZE = 18
TICK_FONT_SIZE = 14
LEGEND_FONT_SIZE = 14
ANNOTATION_FONT_SIZE = 12

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


def main():
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
    dialogue_summary = analyze_dialogue_rating_relationship(dataset)
    summary_lines.extend(dialogue_summary)

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

    dataset = pd.read_csv(path)

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

    plot_rating_change_from_first(df)

    return lines


def plot_rating_change_from_first(franchise_df):
    """
    Plot average rating change from the first installment by installment number.
    """
    df = franchise_df.dropna(subset=["rating_change_from_first"]).copy()

    grouped = (
        df
        .groupby(INSTALLMENT_COL)["rating_change_from_first"]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values(INSTALLMENT_COL)
    )

    plt.figure(figsize=(9, 6))
    plt.plot(grouped[INSTALLMENT_COL], grouped["mean"], marker="o")
    plt.axhline(0, linewidth=1)

    for _, row in grouped.iterrows():
        plt.text(
            row[INSTALLMENT_COL],
            row["mean"],
            f"n={int(row['count'])}",
            fontsize=ANNOTATION_FONT_SIZE,
            ha="center",
            va="bottom",
        )

    plt.title("Rating Change from First Franchise Movie", fontsize=TITLE_FONT_SIZE)
    plt.xlabel("Franchise installment number", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel("IMDb rating change from first movie", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "franchise_rating_change_from_first.png"
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_franchise_scatter(franchise_df, slope, intercept):
    """
    Plot IMDb rating against franchise installment number.
    """
    x = franchise_df[INSTALLMENT_COL]
    y = franchise_df[RATING_COL]

    # Small deterministic jitter so overlapping points are more visible.
    rng = np.random.default_rng(seed=42)
    x_jitter = x + rng.normal(0, 0.04, size=len(x))

    plt.figure(figsize=(9, 6))
    plt.scatter(x_jitter, y, alpha=0.35)

    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept

    plt.plot(x_line, y_line, linewidth=2)

    plt.title("IMDb Rating vs Franchise Installment Number", fontsize=TITLE_FONT_SIZE)
    plt.xlabel("Franchise installment number", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel("IMDb rating", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "franchise_installment_vs_imdb_rating.png"
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_mean_rating_by_installment(franchise_df):
    """
    Plot average IMDb rating by installment number.

    The n= labels show how many movies-1M are included in each
    installment-number average.
    """
    grouped = (
        franchise_df
        .groupby(INSTALLMENT_COL)[RATING_COL]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values(INSTALLMENT_COL)
    )

    plt.figure(figsize=(9, 6))
    plt.plot(grouped[INSTALLMENT_COL], grouped["mean"], marker="o")

    for _, row in grouped.iterrows():
        plt.text(
            row[INSTALLMENT_COL],
            row["mean"],
            f"n={int(row['count'])}",
            fontsize=ANNOTATION_FONT_SIZE,
            ha="center",
            va="bottom",
        )

    plt.title("Mean IMDb Rating by Franchise Installment", fontsize=TITLE_FONT_SIZE)
    plt.xlabel("Franchise installment number", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel("Mean IMDb rating", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "franchise_installment_mean_rating.png"
    plt.savefig(output_path, dpi=200)
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

        plot_dialogue_feature_scatter(feature_df, feature)

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

    return lines


def get_feature_label(feature):
    """
    Return a readable plot label for a dialogue feature column.
    """
    return DIALOGUE_FEATURE_LABELS.get(
        feature,
        str(feature).replace("_", " ").title(),
    )


def plot_dialogue_feature_scatter(feature_df, feature):
    """
    Plot one dialogue feature against IMDb rating.
    """
    x = feature_df[feature]
    y = feature_df[RATING_COL]
    feature_label = get_feature_label(feature)

    plt.figure(figsize=(9, 6))
    plt.scatter(x, y, alpha=0.35)

    if x.nunique() > 1:
        slope, intercept = np.polyfit(x, y, deg=1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line, linewidth=2)

    plt.title(f"IMDb Rating vs {feature_label}", fontsize=TITLE_FONT_SIZE)
    plt.xlabel(feature_label, fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel("IMDb rating", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = OUTPUT_DIR / f"dialogue_feature_vs_imdb_rating_{feature}.png"
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_dialogue_correlation_bar(corr_df):
    """
    Plot Pearson and Spearman correlations for dialogue features.

    Saves:
        - a readable top-10 plot
        - a larger plot with all available dialogue features
    """
    corr_df = corr_df.copy()

    top_corr_df = corr_df.sort_values("abs_pearson", ascending=False).head(10)
    plot_correlation_bar(
        top_corr_df,
        title="Top Dialogue Feature Correlations with IMDb Rating",
        output_path=OUTPUT_DIR / "dialogue_feature_correlations.png",
    )

    all_corr_df = corr_df.sort_values("pearson")
    plot_correlation_bar(
        all_corr_df,
        title="All Dialogue Feature Correlations with IMDb Rating",
        output_path=OUTPUT_DIR / "dialogue_feature_correlations_all.png",
        height=max(8, 0.45 * len(all_corr_df)),
    )


def plot_correlation_bar(
    corr_df,
    title,
    output_path,
    height=6,
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

    plt.figure(figsize=(12, height))
    plt.barh(y - height_bar / 2, corr_df["pearson"], height_bar, label="Pearson")
    plt.barh(y + height_bar / 2, corr_df["spearman"], height_bar, label="Spearman")

    plt.axvline(0, linewidth=1)
    plt.yticks(y, feature_labels, fontsize=max(9, TICK_FONT_SIZE - 3))
    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.title(title, fontsize=TITLE_FONT_SIZE)
    plt.xlabel("Correlation with IMDb rating", fontsize=AXIS_LABEL_FONT_SIZE)
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

    lines.extend(analyze_genre_patterns(dataset))
    lines.append("")

    lines.extend(analyze_first_vs_last_franchise(dataset))
    lines.append("")

    lines.extend(analyze_franchise_vs_nonfranchise(dataset))
    lines.append("")

    lines.extend(analyze_feature_redundancy(dataset))
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


def subtitle_quality_mask(dataset):
    """
    Mark rows that pass basic subtitle-quality checks.

    These filters remove very short, very dense, or highly repetitive subtitle
    files that may be parsing/matching artifacts rather than real dialogue.
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

    mask = pd.Series(True, index=df.index)

    if "num_tokens" in df.columns:
        mask &= df["num_tokens"].fillna(0) >= MIN_SUBTITLE_TOKENS

    if "num_lines" in df.columns:
        mask &= df["num_lines"].fillna(0) >= MIN_SUBTITLE_LINES

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

    return mask


def analyze_subtitle_quality(dataset):
    """
    Summarize subtitle-quality filters.
    """
    lines = []
    lines.append("Subtitle-quality checks:")

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

    mask = subtitle_quality_mask(df)

    lines.append(f"Rows before subtitle-quality filtering: {len(df):,}")
    lines.append(f"Rows passing subtitle-quality checks: {int(mask.sum()):,}")
    lines.append(f"Rows removed by subtitle-quality checks: {int((~mask).sum()):,}")

    checks = {
        f"num_tokens < {MIN_SUBTITLE_TOKENS}": (
            df["num_tokens"] < MIN_SUBTITLE_TOKENS
            if "num_tokens" in df.columns
            else pd.Series(False, index=df.index)
        ),
        f"num_lines < {MIN_SUBTITLE_LINES}": (
            df["num_lines"] < MIN_SUBTITLE_LINES
            if "num_lines" in df.columns
            else pd.Series(False, index=df.index)
        ),
        f"subtitle_words_per_minute > {MAX_SUBTITLE_WORDS_PER_MINUTE}": (
            df["subtitle_words_per_minute"] > MAX_SUBTITLE_WORDS_PER_MINUTE
            if "subtitle_words_per_minute" in df.columns
            else pd.Series(False, index=df.index)
        ),
        f"repeated_line_ratio > {MAX_REPEATED_LINE_RATIO}": (
            df["repeated_line_ratio"] > MAX_REPEATED_LINE_RATIO
            if "repeated_line_ratio" in df.columns
            else pd.Series(False, index=df.index)
        ),
        "type_token_ratio == 1": (
            df["type_token_ratio"] == 1
            if "type_token_ratio" in df.columns
            else pd.Series(False, index=df.index)
        ),
        "hapax_ratio == 0": (
            df["hapax_ratio"] == 0
            if "hapax_ratio" in df.columns
            else pd.Series(False, index=df.index)
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
        lines.append("Skipped: no rows passed subtitle-quality filtering.")
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

    plt.figure(figsize=(10, max(6, 0.35 * len(plot_df))))
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

    plt.figure(figsize=(12, 10))
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
