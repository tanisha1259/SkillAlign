from pathlib import Path
import re

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

NETSOL_FILE = DATA_DIR / "netsol_clean.csv"


# ============================================================
# Loading
# ============================================================

def load_netsol():
    df = pd.read_csv(NETSOL_FILE)

    required_columns = {
        "candidate_id",
        "resume",
        "requirement",
        "label",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    return df


# ============================================================
# Text normalization
# ============================================================

def normalize_text(text):
    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9+#.\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# Build classification text
# ============================================================

def build_pair_text(row):
    """
    Combine requirement and resume into a single
    classification input.
    """

    requirement = normalize_text(
        row["requirement"]
    )

    resume = normalize_text(
        row["resume"]
    )

    return (
        "requirement: "
        + requirement
        + " resume: "
        + resume
    )


# ============================================================
# Candidate-grouped split
# ============================================================

def create_splits(df):
    """
    Split by candidate so no candidate appears
    in more than one split.
    """

    missing_candidates = df["candidate_id"].isna().sum()

    print(
        f"\nRows with missing candidate_id: "
        f"{missing_candidates:,}"
    )

    if missing_candidates > 0:
        print("Dropping rows with missing candidate_id.")

        df = df.dropna(
            subset=["candidate_id"]
        ).copy()

    splitter_1 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.15,
        random_state=42,
    )

    train_val_idx, test_idx = next(
        splitter_1.split(
            df,
            groups=df["candidate_id"],
        )
    )

    train_val = df.iloc[train_val_idx].copy()
    test = df.iloc[test_idx].copy()

    splitter_2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.1765,
        random_state=42,
    )

    train_idx, val_idx = next(
        splitter_2.split(
            train_val,
            groups=train_val["candidate_id"],
        )
    )

    train = train_val.iloc[train_idx].copy()
    val = train_val.iloc[val_idx].copy()

    return train, val, test

# ============================================================
# TF-IDF cosine similarity
# ============================================================

def build_tfidf_similarity(train, val, test):
    """
    Calculate TF-IDF cosine similarity between each
    requirement and its corresponding resume.

    The TF-IDF vocabulary is fitted only on the training set.
    """

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    # Fit vocabulary only on training text.
    train_text = pd.concat(
        [
            train["requirement"],
            train["resume"],
        ]
    )

    vectorizer.fit(
        train_text.map(normalize_text)
    )

    def calculate_similarity(split):

        requirement_vectors = vectorizer.transform(
            split["requirement"].map(normalize_text)
        )

        resume_vectors = vectorizer.transform(
            split["resume"].map(normalize_text)
        )

        similarities = cosine_similarity(
            requirement_vectors,
            resume_vectors,
            dense_output=False,
        )

        # Only take the diagonal:
        # each requirement compared with its own resume.
        return similarities.diagonal()

    train_scores = calculate_similarity(train)
    val_scores = calculate_similarity(val)
    test_scores = calculate_similarity(test)

    return (
        train_scores,
        val_scores,
        test_scores,
    )

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    df = load_netsol()

    print(f"NETSOL rows: {len(df):,}")
    print(
        f"Unique candidates: "
        f"{df['candidate_id'].nunique(dropna=True):,}"
    )

    # --------------------------------------------------------
    # Leakage-safe split
    # --------------------------------------------------------

    train, val, test = create_splits(df)

    print("\nSplit sizes:")
    print("-" * 60)

    print(f"Train: {len(train):,}")
    print(f"Validation: {len(val):,}")
    print(f"Test: {len(test):,}")

    train_candidates = set(
        train["candidate_id"]
    )

    val_candidates = set(
        val["candidate_id"]
    )

    test_candidates = set(
        test["candidate_id"]
    )

    print("\nCandidate overlap:")
    print("-" * 60)

    print(
        "Train ∩ Validation:",
        len(train_candidates & val_candidates),
    )

    print(
        "Train ∩ Test:",
        len(train_candidates & test_candidates),
    )

    print(
        "Validation ∩ Test:",
        len(val_candidates & test_candidates),
    )

    # --------------------------------------------------------
    # TF-IDF cosine similarity baseline
    # --------------------------------------------------------

    train_similarity, val_similarity, test_similarity = (
        build_tfidf_similarity(
            train,
            val,
            test,
        )
    )

    print("\nTF-IDF cosine similarity:")
    print("-" * 60)

    print(
        f"Train mean: "
        f"{train_similarity.mean():.4f}"
    )

    print(
        f"Validation mean: "
        f"{val_similarity.mean():.4f}"
    )

    print(
        f"Test mean: "
        f"{test_similarity.mean():.4f}"
    )

    print("\nSimilarity by label — validation:")
    print("-" * 60)

    val_analysis = val.copy()

    val_analysis["similarity"] = val_similarity

    print(
        val_analysis
        .groupby("label")["similarity"]
        .describe()
        .round(4)
    )


    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    X_train = vectorizer.fit_transform(
        train.apply(build_pair_text, axis=1)
    )

    X_val = vectorizer.transform(
        val.apply(build_pair_text, axis=1)
    )

    X_test = vectorizer.transform(
        test.apply(build_pair_text, axis=1)
    )

    y_train = train["label"].astype(int)
    y_val = val["label"].astype(int)
    y_test = test["label"].astype(int)

    print("\nTF-IDF shapes:")
    print("-" * 60)

    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    predictions = model.predict(X_test)

    print("\nTest results:")
    print("=" * 60)

    print(
        f"Accuracy : "
        f"{accuracy_score(y_test, predictions):.4f}"
    )

    print(
        f"Precision: "
        f"{precision_score(y_test, predictions):.4f}"
    )

    print(
        f"Recall   : "
        f"{recall_score(y_test, predictions):.4f}"
    )

    print(
        f"F1       : "
        f"{f1_score(y_test, predictions):.4f}"
    )

    print("\nConfusion matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
        )
    )