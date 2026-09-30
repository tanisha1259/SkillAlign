from pathlib import Path
import ast
import re

import numpy as np
import pandas as pd

from rank_bm25 import BM25Okapi
# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


# ============================================================
# Dataset loading
# ============================================================

DATASET_DIR = DATA_DIR / "candidate-matching-synthetic"

RESUME_FILE = DATASET_DIR / "data" / "resumes-00000-of-00001.parquet"
JOB_FILE = DATASET_DIR / "jobs" / "train-00000-of-00001.parquet"
MATCH_FILE = DATASET_DIR / "matches" / "train-00000-of-00001.parquet"

def load_data():
    resumes = pd.read_parquet(RESUME_FILE)
    jobs = pd.read_parquet(JOB_FILE)
    matches = pd.read_parquet(MATCH_FILE)

    print(f"Resumes: {len(resumes):,}")
    print(f"Jobs: {len(jobs):,}")
    print(f"Matches: {len(matches):,}")

    return resumes, jobs, matches


# ============================================================
# Helpers
# ============================================================

def ensure_list(value):
    """Convert dataset list-like fields into Python lists."""
    if value is None:
        return []

    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)

    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
        except (ValueError, SyntaxError):
            pass

        return [value]

    return [value]


def normalize_skill(skill):
    """Normalize a skill for exact matching."""
    skill = str(skill).strip().lower()

    skill = re.sub(r"[\(\)\[\]\{\},.:;]", " ", skill)
    skill = re.sub(r"\s+", " ", skill)

    return skill.strip()


def build_resume_skill_sets(resumes):
    """Create normalized skill sets indexed by resume_id."""
    return {
        row["resume_id"]: {
            normalize_skill(skill)
            for skill in ensure_list(row["skills"])
            if str(skill).strip()
        }
        for _, row in resumes.iterrows()
    }

# ============================================================
# Exact skill-overlap retrieval
# ============================================================

def get_relevant_ids(matches_df, job_id):
    """Return ground-truth relevant resume IDs for a job."""
    row = matches_df.loc[matches_df["job_id"] == job_id].iloc[0]
    return set(ensure_list(row["relevant_resume_ids"]))


def rank_by_skill_overlap(job, resume_skill_sets):
    """Rank resumes by exact must-have skill overlap."""
    must_have = {
        normalize_skill(skill)
        for skill in ensure_list(job["must_have_skills"])
        if str(skill).strip()
    }

    scored = []

    for resume_id, resume_skills in resume_skill_sets.items():
        overlap = len(must_have & resume_skills)
        ratio = overlap / len(must_have) if must_have else 0.0

        scored.append((resume_id, overlap, ratio))

    scored.sort(
        key=lambda x: (x[1], x[2]),
        reverse=True
    )

    return [resume_id for resume_id, _, _ in scored]


def recall_at_k(ranked_ids, relevant_ids, k):
    """Recall@K using the dataset's 30 relevant resumes per job."""
    retrieved = set(ranked_ids[:k])
    return len(retrieved & relevant_ids) / len(relevant_ids)


def evaluate_skill_overlap(jobs_df, matches_df, resume_skill_sets):
    """Evaluate exact skill-overlap retrieval."""
    ks = [1, 3, 5, 10, 25, 50, 100]

    recalls = {k: [] for k in ks}

    for _, job in jobs_df.iterrows():
        relevant_ids = get_relevant_ids(
            matches_df,
            job["job_id"]
        )

        ranked_ids = rank_by_skill_overlap(
            job,
            resume_skill_sets
        )

        for k in ks:
            recalls[k].append(
                recall_at_k(
                    ranked_ids,
                    relevant_ids,
                    k
                )
            )

    return {
        f"R@{k}": float(np.mean(recalls[k]))
        for k in ks
    }

# ============================================================
# Skill overlap -> BM25 tie-break
# ============================================================

def build_bm25(resumes_df):
    """Build BM25 index over normalized resume skills."""
    corpus = [
        [
            normalize_skill(skill)
            for skill in ensure_list(row["skills"])
            if str(skill).strip()
        ]
        for _, row in resumes_df.iterrows()
    ]

    return BM25Okapi(corpus)


def rank_by_skill_overlap_bm25(
    job,
    resume_skill_sets,
    bm25,
):
    """
    Rank primarily by exact must-have skill overlap.
    Use BM25 as the secondary tie-breaker.
    """

    must_have = {
        normalize_skill(skill)
        for skill in ensure_list(job["must_have_skills"])
        if str(skill).strip()
    }

    query = list(must_have)
    bm25_scores = bm25.get_scores(query)

    scored = []

    for idx, (resume_id, resume_skills) in enumerate(
        resume_skill_sets.items()
    ):
        overlap = len(must_have & resume_skills)

        scored.append(
            (
                resume_id,
                overlap,
                float(bm25_scores[idx]),
            )
        )

    # Primary: exact skill overlap
    # Secondary: BM25
    scored.sort(
        key=lambda x: (x[1], x[2]),
        reverse=True
    )

    return [resume_id for resume_id, _, _ in scored]


def evaluate_hybrid(
    jobs_df,
    matches_df,
    resume_skill_sets,
    bm25,
):
    """Evaluate skill-overlap -> BM25 retrieval."""

    ks = [1, 3, 5, 10, 25, 50, 100]
    recalls = {k: [] for k in ks}

    for job_idx, (_, job) in enumerate(jobs_df.iterrows()):

        relevant_ids = get_relevant_ids(
            matches_df,
            job["job_id"]
        )

        ranked_ids = rank_by_skill_overlap_bm25(
            job,
            resume_skill_sets,
            bm25,
        )

        for k in ks:
            recalls[k].append(
                recall_at_k(
                    ranked_ids,
                    relevant_ids,
                    k
                )
            )

        if (job_idx + 1) % 500 == 0:
            print(
                f"Processed {job_idx + 1:,}/"
                f"{len(jobs_df):,} jobs"
            )

    return {
        f"R@{k}": float(np.mean(recalls[k]))
        for k in ks
    }

## ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    resumes_df, jobs_df, matches_df = load_data()

    resume_skill_sets = build_resume_skill_sets(resumes_df)
    print("\nBuilding BM25 index...")
    bm25 = build_bm25(resumes_df)
    print("BM25 ready.")

    print(f"Resume skill sets: {len(resume_skill_sets):,}")

    match_sizes = matches_df["relevant_resume_ids"].apply(
        lambda x: len(ensure_list(x))
    )

    print(
        f"Relevant resumes per job: "
        f"min={match_sizes.min()}, "
        f"max={match_sizes.max()}, "
        f"mean={match_sizes.mean():.1f}"
    )

    print("\nEvaluating exact skill overlap...")

    results = evaluate_skill_overlap(
        jobs_df,
        matches_df,
        resume_skill_sets
    )

    print("\nExact Skill Overlap Results")
    print("-" * 35)

    for metric, value in results.items():
        print(f"{metric:<8} {value:.6f}")

    print("\nEvaluating skill overlap -> BM25 hybrid...")

    hybrid_results = evaluate_hybrid(
        jobs_df,
        matches_df,
        resume_skill_sets,
        bm25,
    )

    print("\nHybrid Results")
    print("-" * 35)

    for metric, value in hybrid_results.items():
        print(f"{metric:<8} {value:.6f}")