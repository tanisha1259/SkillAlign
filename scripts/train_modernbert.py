from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from sklearn.model_selection import GroupShuffleSplit

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from torch.utils.data import Dataset


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

NETSOL_FILE = DATA_DIR / "netsol_clean.csv"

MODEL_NAME = "answerdotai/ModernBERT-base"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "modernbert_netsol"


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

    # Candidate identity is required for leakage-safe splitting.
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
# Candidate-grouped split
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
# Dataset
# ============================================================

class RequirementResumeDataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer,
        max_length=512,
    ):

        self.requirements = (
            dataframe["requirement"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        self.resumes = (
            dataframe["resume"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        self.labels = (
            dataframe["label"]
            .astype(int)
            .tolist()
        )

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        encoded = self.tokenizer(
            text=self.requirements[idx],
            text_pair=self.resumes[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            key: value.squeeze(0)
            for key, value in encoded.items()
        }

        item["labels"] = torch.tensor(
            self.labels[idx],
            dtype=torch.long,
        )

        return item


# ============================================================
# Metrics
# ============================================================

def compute_metrics(eval_prediction):

    logits, labels = eval_prediction

    predictions = np.argmax(
        logits,
        axis=-1,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            zero_division=0,
        )
    )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print(
        f"PyTorch: {torch.__version__}"
    )

    print(
        f"CUDA available: "
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_netsol()

    print(
        f"\nUsable NETSOL rows: "
        f"{len(df):,}"
    )

    print(
        f"Unique candidates: "
        f"{df['candidate_id'].nunique():,}"
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train, val, test = create_splits(df)

    print("\nSplit sizes:")
    print("-" * 60)

    print(f"Train: {len(train):,}")
    print(f"Validation: {len(val):,}")
    print(f"Test: {len(test):,}")

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = RequirementResumeDataset(
        train,
        tokenizer,
    )

    val_dataset = RequirementResumeDataset(
        val,
        tokenizer,
    )

    test_dataset = RequirementResumeDataset(
        test,
        tokenizer,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("Loading ModernBERT...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    # --------------------------------------------------------
    # Training configuration
    # --------------------------------------------------------

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),

        num_train_epochs=3,

        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,

        gradient_accumulation_steps=4,

        learning_rate=2e-5,
        weight_decay=0.01,

        eval_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        logging_steps=25,

        fp16=torch.cuda.is_available(),

        report_to="none",

        save_total_limit=1,
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    trainer = Trainer(
        model=model,
        args=training_args,

        train_dataset=train_dataset,
        eval_dataset=val_dataset,

        compute_metrics=compute_metrics,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nStarting training...")

    trainer.train()

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("\nValidation results:")

    val_results = trainer.evaluate(
        val_dataset
    )

    for key, value in val_results.items():

        if key.startswith("eval_"):

            print(
                f"{key}: {value:.4f}"
                if isinstance(value, float)
                else f"{key}: {value}"
            )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    print("\nFinal test results:")

    test_results = trainer.evaluate(
        test_dataset
    )

    for key, value in test_results.items():

        if key.startswith("eval_"):

            print(
                f"{key}: {value:.4f}"
                if isinstance(value, float)
                else f"{key}: {value}"
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    trainer.save_model(
        str(OUTPUT_DIR)
    )

    tokenizer.save_pretrained(
        str(OUTPUT_DIR)
    )

    print(
        f"\nModel saved to:\n{OUTPUT_DIR}"
    )