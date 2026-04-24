import gzip
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import pandas as pd


DATASETS_DIR = Path("datasets")


# -----------------------------
# Generic text cleaning
# -----------------------------

def clean_dialogue_text(text: str) -> str:
    if not text:
        return ""

    text = text.lower()

    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\{.*?\}", " ", text)
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"\(.*?\)", " ", text)
    text = re.sub(r"^[a-zA-Z\s]{1,30}:\s*", " ", text)

    text = re.sub(r"[^a-z'\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# -----------------------------
# OpenSubtitles preprocessing
# -----------------------------

def preprocess_opensubtitles(dataset_path: str | Path = DATASETS_DIR / "opensubtitles",
                             limit: int | None = None) -> pd.DataFrame:
    """
    Preprocess OpenSubtitles XML.GZ files.

    Returns one row per subtitle/movie file with cleaned dialogue lines.
    Feature extraction is intentionally kept separate.
    """
    dataset_path = Path(dataset_path)
    files = list(dataset_path.rglob("*.xml.gz"))

    if limit is not None:
        files = files[:limit]

    rows = []

    for file_path in files:
        try:
            dialogue_lines = _extract_opensubtitles_lines(file_path)

            rows.append({
                "dataset": "opensubtitles",
                "file_name": file_path.name,
                "file_path": str(file_path),
                "dialogue_lines": dialogue_lines,
                "dialogue_text": " ".join(dialogue_lines),
            })

        except Exception as e:
            print(f"Failed to process {file_path}: {e}")

    return pd.DataFrame(rows)


def _extract_opensubtitles_lines(file_path: Path) -> list[str]:
    """
    Internal helper for extracting cleaned dialogue lines from one OpenSubtitles file.
    """
    lines = []

    with gzip.open(file_path, "rt", encoding="utf-8", errors="ignore") as f:
        tree = ET.parse(f)
        root = tree.getroot()

        for sentence in root.iter("s"):
            words = [w.text for w in sentence.iter("w") if w.text]
            sentence_text = " ".join(words)

            cleaned = clean_dialogue_text(sentence_text)
            if cleaned:
                lines.append(cleaned)

    return lines


# -----------------------------
# Dialogue feature extraction
# -----------------------------

def get_dialogue_richness(dialogue_text: str) -> dict:
    """
    Measures vocabulary richness.
    """
    tokens = _tokenize(dialogue_text)

    if not tokens:
        return {
            "num_tokens": 0,
            "num_unique_tokens": 0,
            "type_token_ratio": 0,
            "hapax_ratio": 0,
        }

    token_counts = Counter(tokens)
    hapax_count = sum(1 for count in token_counts.values() if count == 1)

    return {
        "num_tokens": len(tokens),
        "num_unique_tokens": len(token_counts),
        "type_token_ratio": len(token_counts) / len(tokens),
        "hapax_ratio": hapax_count / len(token_counts),
    }


def get_dialogue_repetition(dialogue_text: str,
                            dialogue_lines: list[str] | None = None) -> dict:
    """
    Measures repeated words, repeated lines, and repeated bigrams.
    """
    tokens = _tokenize(dialogue_text)

    if not tokens:
        return {
            "top_word_frequency_ratio": 0,
            "bigram_repetition_ratio": 0,
            "repeated_line_ratio": 0,
        }

    token_counts = Counter(tokens)

    bigrams = list(zip(tokens, tokens[1:]))
    bigram_counts = Counter(bigrams)

    repeated_bigrams = sum(count for count in bigram_counts.values() if count > 1)

    if dialogue_lines:
        line_counts = Counter(dialogue_lines)
        repeated_lines = sum(count for count in line_counts.values() if count > 1)
        repeated_line_ratio = repeated_lines / len(dialogue_lines)
    else:
        repeated_line_ratio = None

    return {
        "top_word_frequency_ratio": token_counts.most_common(1)[0][1] / len(tokens),
        "bigram_repetition_ratio": repeated_bigrams / len(bigrams) if bigrams else 0,
        "repeated_line_ratio": repeated_line_ratio,
    }


def get_dialogue_length_features(dialogue_text: str,
                                 dialogue_lines: list[str] | None = None) -> dict:
    """
    Measures dialogue size and average line length.
    """
    tokens = _tokenize(dialogue_text)

    num_lines = len(dialogue_lines) if dialogue_lines else 0

    return {
        "num_lines": num_lines,
        "num_tokens": len(tokens),
        "avg_line_length": len(tokens) / num_lines if num_lines else 0,
    }


def extract_dialogue_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds dialogue feature columns to a preprocessed dialogue dataframe.

    Expected columns:
    - dialogue_text
    - dialogue_lines
    """
    rows = []

    for _, row in df.iterrows():
        dialogue_text = row.get("dialogue_text", "")
        dialogue_lines = row.get("dialogue_lines", [])

        features = {
            **get_dialogue_length_features(dialogue_text, dialogue_lines),
            **get_dialogue_richness(dialogue_text),
            **get_dialogue_repetition(dialogue_text, dialogue_lines),
        }

        rows.append(features)

    feature_df = pd.DataFrame(rows)

    return pd.concat(
        [df.reset_index(drop=True), feature_df.reset_index(drop=True)],
        axis=1
    )


# -----------------------------
# Internal helpers
# -----------------------------

def _tokenize(text: str) -> list[str]:
    if not text:
        return []

    return re.findall(r"[a-z']+", text.lower())