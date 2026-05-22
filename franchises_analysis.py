"""
franchises_analysis.py

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
        dialogue_feature_correlations.png
        dialogue_feature_vs_imdb_rating_<feature>.png
        analysis_summary.txt

Run:
    python franchises_analysis.py
"""

from pathlib import Path
import math

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

DIALOGUE_FEATURES = [
    "num_tokens",
    "num_unique_tokens",
    "type_token_ratio",
    "hapax_ratio",
    "top_word_frequency_ratio",
    "bigram_repetition_ratio",
    "repeated_line_ratio",
]

DIALOGUE_FEATURE_LABELS = {
    "num_tokens": "Total subtitle words",
    "num_unique_tokens": "Unique subtitle words",
    "type_token_ratio": "Vocabulary diversity ratio",
    "hapax_ratio": "Rare-word ratio",
    "top_word_frequency_ratio": "Most frequent word share",
    "bigram_repetition_ratio": "Repeated two-word phrase ratio",
    "repeated_line_ratio": "Repeated subtitle line ratio",
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

    The n= labels show how many movies are included in each
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
        lines.append(f"  - {feature}")

    lines.append("")
    lines.append("Correlations with IMDb rating:")
    for _, row in corr_df.iterrows():
        lines.append(
            f"  {row['feature']}: "
            f"Pearson={row['pearson']:.4f}, "
            f"Spearman={row['spearman']:.4f}, "
            f"n={int(row['n'])}"
        )

    strongest = corr_df.iloc[0]

    lines.append("")
    lines.append(
        f"Strongest absolute Pearson relationship: {strongest['feature']} "
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
    return DIALOGUE_FEATURE_LABELS.get(feature, feature.replace("_", " ").title())

def plot_dialogue_feature_scatter(feature_df, feature):
    """
    Plot one dialogue feature against IMDb rating.
    """
    x = feature_df[feature]
    y = feature_df[RATING_COL]

    plt.figure(figsize=(9, 6))
    plt.scatter(x, y, alpha=0.35)

    if x.nunique() > 1:
        slope, intercept = np.polyfit(x, y, deg=1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line, linewidth=2)

    feature_label = get_feature_label(feature)

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
    """
    corr_df = corr_df.sort_values("pearson")

    x = np.arange(len(corr_df))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, corr_df["pearson"], width, label="Pearson")
    plt.bar(x + width / 2, corr_df["spearman"], width, label="Spearman")

    plt.axhline(0, linewidth=1)
    feature_labels = [get_feature_label(feature) for feature in corr_df["feature"]]
    plt.xticks(x, feature_labels, rotation=35, ha="right", fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)
    plt.title("Dialogue Feature Correlations with IMDb Rating", fontsize=TITLE_FONT_SIZE)
    plt.xlabel("Dialogue feature", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel("Correlation with IMDb rating", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "dialogue_feature_correlations.png"
    plt.savefig(output_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
