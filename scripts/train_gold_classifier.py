import random
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

GOLD_PATH = "data/gold_annotations.csv"
MODEL_NAME = "answerdotai/ModernBERT-base"
OUTPUT_DIR = "outputs/modernbert_skillalign"

LABELS = {
    "Unsupported": 0,
    "Partial": 1,
    "Strong": 2,
}

ID2LABEL = {
    0: "Unsupported",
    1: "Partial",
    2: "Strong",
}

# ---------------------------------------------------------
# Load Gold
# ---------------------------------------------------------

df = pd.read_csv(
    GOLD_PATH,
    dtype={"annotation_id": str}
)

df["resume_evidence"] = (
    df["resume_evidence"]
    .fillna("")
    .astype(str)
)

df["requirement"] = (
    df["requirement"]
    .fillna("")
    .astype(str)
)

df["candidate_id"] = (
    df["candidate_id"]
    .fillna("")
    .astype(str)
)

df["label_id"] = df["label"].map(LABELS)

df = df[
    df["label_id"].notna()
    & df["requirement"].str.strip().ne("")
].copy()

print("=" * 70)
print("SKILLALIGN — GOLD 3-CLASS CLASSIFIER")
print("=" * 70)

print(f"Total usable rows: {len(df)}")
print("\nLabels:")
print(df["label"].value_counts())

print(f"\nUnique candidates: {df['candidate_id'].nunique()}")

# ---------------------------------------------------------
# Candidate-grouped split
# 70% train / 15% validation / 15% test
# ---------------------------------------------------------

gss1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=SEED,
)

train_idx, temp_idx = next(
    gss1.split(
        df,
        groups=df["candidate_id"]
    )
)

train_df = df.iloc[train_idx].copy()
temp_df = df.iloc[temp_idx].copy()

gss2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=SEED,
)

val_idx, test_idx = next(
    gss2.split(
        temp_df,
        groups=temp_df["candidate_id"]
    )
)

val_df = temp_df.iloc[val_idx].copy()
test_df = temp_df.iloc[test_idx].copy()

print("\nSplit:")
print(f"Train: {len(train_df)}")
print(f"Val:   {len(val_df)}")
print(f"Test:  {len(test_df)}")

print("\nCandidate overlap:")
print(
    "Train/Val:",
    len(
        set(train_df.candidate_id)
        & set(val_df.candidate_id)
    )
)
print(
    "Train/Test:",
    len(
        set(train_df.candidate_id)
        & set(test_df.candidate_id)
    )
)
print(
    "Val/Test:",
    len(
        set(val_df.candidate_id)
        & set(test_df.candidate_id)
    )
)

# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

class GoldDataset(torch.utils.data.Dataset):

    def __init__(self, frame, tokenizer):

        self.labels = frame["label_id"].astype(int).tolist()

        requirements = frame["requirement"].tolist()
        evidence = frame["resume_evidence"].tolist()

        self.encodings = tokenizer(
            requirements,
            evidence,
            truncation=True,
            padding="max_length",
            max_length=512,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        item = {
            key: torch.tensor(value[idx])
            for key, value in self.encodings.items()
        }

        item["labels"] = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return item


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

print("\nLoading tokenizer/model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=ID2LABEL,
    label2id=LABELS,
)

train_dataset = GoldDataset(
    train_df,
    tokenizer
)

val_dataset = GoldDataset(
    val_df,
    tokenizer
)

test_dataset = GoldDataset(
    test_df,
    tokenizer
)

# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    return {
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=8,

    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,

    gradient_accumulation_steps=2,

    learning_rate=2e-5,
    weight_decay=0.01,

    eval_strategy="epoch",
    save_strategy="epoch",

    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,

    logging_steps=10,

    fp16=torch.cuda.is_available(),

    report_to="none",

    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=train_dataset,
    eval_dataset=val_dataset,

    compute_metrics=compute_metrics,
)

print("\nStarting training...\n")

trainer.train()

# ---------------------------------------------------------
# Final test evaluation
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

predictions = trainer.predict(
    test_dataset
)

predicted_labels = np.argmax(
    predictions.predictions,
    axis=-1
)

true_labels = np.array(
    test_df["label_id"].tolist()
)

print(
    classification_report(
        true_labels,
        predicted_labels,
        labels=[0, 1, 2],
        target_names=[
            "Unsupported",
            "Partial",
            "Strong",
        ],
        digits=4,
        zero_division=0,
    )
)

print("Confusion matrix:")
print(
    confusion_matrix(
        true_labels,
        predicted_labels,
        labels=[0, 1, 2],
    )
)

metrics = compute_metrics(
    (predictions.predictions, true_labels)
)

print("\nSummary:")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")

# ---------------------------------------------------------
# Save model
# ---------------------------------------------------------

trainer.save_model(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)

print("\nModel saved to:")
print(OUTPUT_DIR)
