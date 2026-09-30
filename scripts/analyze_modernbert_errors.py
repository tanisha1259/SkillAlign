from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import GroupShuffleSplit
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

NETSOL_FILE = DATA_DIR / "netsol_clean.csv"

MODEL_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "modernbert_netsol"
)


# ============================================================
# Loading
# ============================================================

def load_netsol():

    df = pd.read_csv(NETSOL_FILE)

    df = df.dropna(
        subset=[
            "candidate_id",
            "resume",
            "requirement",
        ]
    ).copy()

    df = df[
        df["resume"]
        .astype(str)
        .str.strip()
        .ne("")
    ].copy()

    df["label"] = df["label"].astype(int)

    return df


# ============================================================
# Same split as training
# ============================================================

def create_splits(df):

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
# Main
# ============================================================

if __name__ == "__main__":

    print("Loading data...")

    df = load_netsol()

    _, _, test = create_splits(df)

    print(
        f"Test rows: {len(test):,}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR
    )

    print("Loading ModernBERT...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)

    model.eval()

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = []

    probabilities = []

    batch_size = 4

    for start in range(
        0,
        len(test),
        batch_size,
    ):

        batch = test.iloc[
            start:start + batch_size
        ]

        requirements = (
            batch["requirement"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        resumes = (
            batch["resume"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        encoded = tokenizer(
            text=requirements,
            text_pair=resumes,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.no_grad():

            outputs = model(
                **encoded
            )

            probs = torch.softmax(
                outputs.logits,
                dim=-1,
            )

        batch_predictions = (
            torch.argmax(
                probs,
                dim=-1,
            )
            .cpu()
            .numpy()
        )

        predictions.extend(
            batch_predictions.tolist()
        )

        probabilities.extend(
            probs[:, 1]
            .cpu()
            .numpy()
            .tolist()
        )

    # --------------------------------------------------------
    # Attach predictions
    # --------------------------------------------------------

    test = test.reset_index(
        drop=True
    )

    test["prediction"] = predictions

    test["probability_met"] = probabilities

    test["correct"] = (
        test["label"]
        == test["prediction"]
    )

    # --------------------------------------------------------
    # Error counts
    # --------------------------------------------------------

    false_positives = test[
        (test["label"] == 0)
        & (test["prediction"] == 1)
    ]

    false_negatives = test[
        (test["label"] == 1)
        & (test["prediction"] == 0)
    ]

    print("\nError counts:")
    print("=" * 60)

    print(
        f"False positives: "
        f"{len(false_positives)}"
    )

    print(
        f"False negatives: "
        f"{len(false_negatives)}"
    )

    # --------------------------------------------------------
    # Most confident errors
    # --------------------------------------------------------

    print("\nMost confident false positives:")
    print("=" * 60)

    fp_display = (
        false_positives
        .sort_values(
            "probability_met",
            ascending=False,
        )
        .head(10)
    )

    for _, row in fp_display.iterrows():

        print(
            f"\nProbability met: "
            f"{row['probability_met']:.4f}"
        )

        print(
            f"Requirement: "
            f"{row['requirement']}"
        )

        print(
            f"Resume: "
            f"{str(row['resume'])[:500]}"
        )

    print("\nMost confident false negatives:")
    print("=" * 60)

    fn_display = (
        false_negatives
        .sort_values(
            "probability_met",
            ascending=True,
        )
        .head(10)
    )

    for _, row in fn_display.iterrows():

        print(
            f"\nProbability met: "
            f"{row['probability_met']:.4f}"
        )

        print(
            f"Requirement: "
            f"{row['requirement']}"
        )

        print(
            f"Resume: "
            f"{str(row['resume'])[:500]}"
        )