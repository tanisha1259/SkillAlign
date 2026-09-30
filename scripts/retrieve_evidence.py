from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi


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
        "job_description",
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
# Evidence chunking
# ============================================================

def clean_text(text):
    text = str(text)

    text = text.lower()

    text = " ".join(text.split())

    return text


# ============================================================
# BM25 evidence retrieval
# ============================================================

def build_bm25(evidence_chunks):
    """Build a BM25 index over resume evidence chunks."""

    tokenized_chunks = [
        clean_text(chunk).split()
        for chunk in evidence_chunks
    ]

    return BM25Okapi(tokenized_chunks)


def retrieve_evidence(
    requirement,
    evidence_chunks,
    bm25,
    top_k=5,
):
    """
    Retrieve the most relevant resume evidence chunks
    for a requirement using a pre-built BM25 index.
    """

    if not evidence_chunks:
        return []

    query_tokens = clean_text(requirement).split()

    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    results = []

    for idx in ranked_indices:
        results.append(
            {
                "chunk": evidence_chunks[idx],
                "score": float(scores[idx]),
            }
        )

    return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    df = load_netsol()

    print(f"NETSOL rows: {len(df):,}")

    # --------------------------------------------------------
    # Select one candidate + requirement
    # --------------------------------------------------------

    example = df.iloc[0]

    candidate_id = example["candidate_id"]
    requirement = example["requirement"]

    print("\nExample requirement:")
    print(requirement)

    print("\nCandidate:")
    print(candidate_id)

    # --------------------------------------------------------
    # Get this candidate's resume
    # --------------------------------------------------------

    candidate_resume = (
        df[df["candidate_id"] == candidate_id]
        ["resume"]
        .iloc[0]
    )

    # --------------------------------------------------------
    # Temporary chunking
    # --------------------------------------------------------

    from build_evidence_dataset import split_into_chunks

    evidence_chunks = split_into_chunks(
        candidate_resume
    )

    print(
        f"\nEvidence chunks for candidate: "
        f"{len(evidence_chunks):,}"
    )

    bm25 = build_bm25(evidence_chunks)

    results = retrieve_evidence(
        requirement=requirement,
        evidence_chunks=evidence_chunks,
        bm25=bm25,
        top_k=5,
    )

    print("\nTop evidence:")
    print("=" * 70)

    for rank, result in enumerate(results, start=1):

        print(f"\n[{rank}] BM25 score: {result['score']:.4f}")

        print(result["chunk"])