import re
import sys
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


sys.path.insert(0, "scripts")

from build_evidence_dataset import build_evidence_records


GOLD_PATH = "data/gold_annotations.csv"
NETSOL_PATH = "data/netsol_clean.csv"

MODEL_NAME = "BAAI/bge-m3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
TOP_K = 10


def normalize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_f1(gold, retrieved):
    g = set(normalize(gold).split())
    r = set(normalize(retrieved).split())

    if not g or not r:
        return 0.0

    overlap = len(g & r)
    precision = overlap / len(r)
    recall = overlap / len(g)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def is_hit(gold, retrieved):
    g = normalize(gold)
    r = normalize(retrieved)

    if not g or not r:
        return False

    if g in r or r in g:
        return True

    return token_f1(gold, retrieved) >= 0.50


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(
        last_hidden_state.size()
    ).float()

    summed = torch.sum(
        last_hidden_state * mask,
        dim=1
    )

    counts = torch.clamp(
        mask.sum(dim=1),
        min=1e-9
    )

    return summed / counts


@torch.no_grad()
def encode(texts, tokenizer, model):

    embeddings = []

    for start in range(0, len(texts), BATCH_SIZE):

        batch = texts[start:start + BATCH_SIZE]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        encoded = {
            k: v.to(DEVICE)
            for k, v in encoded.items()
        }

        output = model(**encoded)

        pooled = mean_pool(
            output.last_hidden_state,
            encoded["attention_mask"]
        )

        pooled = torch.nn.functional.normalize(
            pooled,
            p=2,
            dim=1
        )

        embeddings.append(
            pooled.cpu().numpy()
        )

    return np.vstack(embeddings)


def main():

    print("=" * 70)
    print("SKILLALIGN — BGE-M3 SEMANTIC EVIDENCE RETRIEVAL")
    print("=" * 70)

    gold = pd.read_csv(
        GOLD_PATH,
        dtype={"annotation_id": str}
    )

    gold = gold[
        gold["label"].isin(["Strong", "Partial"])
    ].copy()

    gold["resume_evidence"] = (
        gold["resume_evidence"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    gold = gold[
        gold["resume_evidence"].ne("")
    ].copy()

    df = pd.read_csv(NETSOL_PATH)
    evidence_df = build_evidence_records(df)

    print(f"Gold queries: {len(gold)}")
    print(f"Evidence chunks: {len(evidence_df)}")
    print(f"Device: {DEVICE}")

    print("\nLoading BGE-M3...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)

    model.eval()

    # --------------------------------------------------
    # Encode all evidence chunks once
    # --------------------------------------------------

    chunks = (
        evidence_df["resume_chunk"]
        .astype(str)
        .tolist()
    )

    print("\nEncoding evidence chunks...")

    chunk_embeddings = encode(
        chunks,
        tokenizer,
        model
    )

    # --------------------------------------------------
    # Evaluate each Gold requirement
    # --------------------------------------------------

    results = []

    for i, (_, row) in enumerate(gold.iterrows(), start=1):

        candidate_id = str(row["candidate_id"])
        requirement = str(row["requirement"])
        gold_evidence = str(row["resume_evidence"])

        candidate_mask = (
            evidence_df["candidate_id"]
            .astype(str)
            .values
            == candidate_id
        )

        candidate_indices = np.where(
            candidate_mask
        )[0]

        if len(candidate_indices) == 0:
            continue

        query_embedding = encode(
            [requirement],
            tokenizer,
            model
        )[0]

        candidate_vectors = chunk_embeddings[
            candidate_indices
        ]

        scores = candidate_vectors @ query_embedding

        order = np.argsort(-scores)[:TOP_K]

        hit_ranks = []
        best_f1 = 0.0
        top1 = ""

        for rank, local_idx in enumerate(order, start=1):

            global_idx = candidate_indices[local_idx]

            text = evidence_df.iloc[
                global_idx
            ]["resume_chunk"]

            if rank == 1:
                top1 = text

            f1 = token_f1(
                gold_evidence,
                text
            )

            best_f1 = max(
                best_f1,
                f1
            )

            if is_hit(
                gold_evidence,
                text
            ):
                hit_ranks.append(rank)

        first_hit = (
            min(hit_ranks)
            if hit_ranks
            else None
        )

        results.append({
            "annotation_id": row["annotation_id"],
            "candidate_id": candidate_id,
            "label": row["label"],
            "requirement": requirement,
            "gold_evidence": gold_evidence,
            "hit@1": first_hit is not None and first_hit <= 1,
            "hit@3": first_hit is not None and first_hit <= 3,
            "hit@5": first_hit is not None and first_hit <= 5,
            "hit@10": first_hit is not None and first_hit <= 10,
            "rr": 1.0 / first_hit if first_hit else 0.0,
            "best_f1": best_f1,
            "top1": top1,
        })

        if i % 25 == 0:
            print(f"Evaluated {i}/{len(gold)}")

    results_df = pd.DataFrame(results)

    def report(name, subset):

        print(f"\n{name}")
        print("-" * 50)
        print(f"Queries: {len(subset)}")
        print(f"Hit@1:  {subset['hit@1'].mean():.4f}")
        print(f"Hit@3:  {subset['hit@3'].mean():.4f}")
        print(f"Hit@5:  {subset['hit@5'].mean():.4f}")
        print(f"Hit@10: {subset['hit@10'].mean():.4f}")
        print(f"MRR:    {subset['rr'].mean():.4f}")
        print(
            f"Mean best token-F1: "
            f"{subset['best_f1'].mean():.4f}"
        )

    report("ALL", results_df)

    report(
        "STRONG",
        results_df[
            results_df["label"] == "Strong"
        ]
    )

    report(
        "PARTIAL",
        results_df[
            results_df["label"] == "Partial"
        ]
    )

    output = "data/semantic_evidence_results.csv"

    results_df.to_csv(
        output,
        index=False
    )

    print("\n" + "=" * 70)
    print(f"Saved → {output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
