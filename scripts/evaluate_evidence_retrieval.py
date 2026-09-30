import re
import sys

import pandas as pd

# Allow imports from scripts/
sys.path.insert(0, "scripts")

from retrieve_evidence import build_bm25, retrieve_evidence
from build_evidence_dataset import build_evidence_records


GOLD_PATH = "data/gold_annotations.csv"
NETSOL_PATH = "data/netsol_clean.csv"


def normalize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_f1(gold, retrieved):
    gold_tokens = set(normalize(gold).split())
    retrieved_tokens = set(normalize(retrieved).split())

    if not gold_tokens or not retrieved_tokens:
        return 0.0

    overlap = len(gold_tokens & retrieved_tokens)

    precision = overlap / len(retrieved_tokens)
    recall = overlap / len(gold_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def is_hit(gold_evidence, retrieved_text):
    """
    A retrieved chunk counts as a hit when it has meaningful
    lexical overlap with the manually annotated Gold evidence.
    """

    gold_norm = normalize(gold_evidence)
    retrieved_norm = normalize(retrieved_text)

    if not gold_norm or not retrieved_norm:
        return False

    # Exact containment
    if gold_norm in retrieved_norm or retrieved_norm in gold_norm:
        return True

    # Token overlap
    return token_f1(gold_evidence, retrieved_text) >= 0.50


def main():

    print("=" * 70)
    print("SKILLALIGN — GOLD EVIDENCE RETRIEVAL EVALUATION")
    print("=" * 70)

    # --------------------------------------------------
    # Load Gold annotations
    # --------------------------------------------------

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

    print(f"\nGold evidence queries: {len(gold)}")

    # --------------------------------------------------
    # Load NETSOL resumes
    # --------------------------------------------------

    df = pd.read_csv(NETSOL_PATH)

    evidence_df = build_evidence_records(df)

    print(f"Evidence chunks: {len(evidence_df)}")

    # Confirm expected schema
    required_columns = {
        "candidate_id",
        "resume_chunk_id",
        "resume_chunk",
    }

    missing = required_columns - set(evidence_df.columns)

    if missing:
        raise ValueError(
            f"Evidence dataset missing columns: {missing}"
        )

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------

    results = []

    for _, row in gold.iterrows():

        candidate_id = str(row["candidate_id"])
        requirement = str(row["requirement"])
        gold_evidence = str(row["resume_evidence"])

        # IMPORTANT:
        # Only retrieve from THIS candidate's resume.
        candidate_chunks = evidence_df[
            evidence_df["candidate_id"].astype(str)
            == candidate_id
        ].copy()

        if candidate_chunks.empty:

            results.append({
                "annotation_id": row["annotation_id"],
                "candidate_id": candidate_id,
                "label": row["label"],
                "requirement": requirement,
                "gold_evidence": gold_evidence,
                "hit@1": False,
                "hit@3": False,
                "hit@5": False,
                "hit@10": False,
                "rr": 0.0,
                "best_f1": 0.0,
                "top1": "",
            })

            continue

        # Build BM25 over this candidate's evidence chunks.
        candidate_chunk_texts = candidate_chunks["resume_chunk"].tolist()

        bm25 = build_bm25(
            candidate_chunk_texts
        )

        retrieved = retrieve_evidence(
            requirement,
            candidate_chunk_texts,
            bm25,
            top_k=10
        )

        hit_ranks = []
        best_f1 = 0.0

        for rank, item in enumerate(retrieved, start=1):

            text = item["chunk"]

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
            "hit@1": (
                first_hit is not None
                and first_hit <= 1
            ),
            "hit@3": (
                first_hit is not None
                and first_hit <= 3
            ),
            "hit@5": (
                first_hit is not None
                and first_hit <= 5
            ),
            "hit@10": (
                first_hit is not None
                and first_hit <= 10
            ),
            "rr": (
                1.0 / first_hit
                if first_hit
                else 0.0
            ),
            "best_f1": best_f1,
            "top1": (
                retrieved[0]["chunk"]
                if retrieved
                else ""
            ),
        })

    results_df = pd.DataFrame(results)

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    def report(name, subset):

        if len(subset) == 0:
            return

        print(f"\n{name}")
        print("-" * 50)

        print(f"Queries: {len(subset)}")
        print(
            f"Hit@1:  {subset['hit@1'].mean():.4f}"
        )
        print(
            f"Hit@3:  {subset['hit@3'].mean():.4f}"
        )
        print(
            f"Hit@5:  {subset['hit@5'].mean():.4f}"
        )
        print(
            f"Hit@10: {subset['hit@10'].mean():.4f}"
        )
        print(
            f"MRR:    {subset['rr'].mean():.4f}"
        )
        print(
            f"Mean best token-F1: "
            f"{subset['best_f1'].mean():.4f}"
        )

    report(
        "ALL",
        results_df
    )

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

    # --------------------------------------------------
    # Save detailed results
    # --------------------------------------------------

    output_path = (
        "data/evidence_retrieval_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 70)
    print(
        f"Saved detailed results → "
        f"{output_path}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()