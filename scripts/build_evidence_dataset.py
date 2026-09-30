from pathlib import Path
import re

import pandas as pd


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
# Resume chunking
# ============================================================

def clean_text(text):
    """Normalize whitespace while preserving sentence boundaries."""
    text = str(text)
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def split_into_chunks(text):
    """
    Create evidence-sized chunks from resume text while
    filtering contact metadata and standalone section headings.
    """

    text = clean_text(text)

    # Normalize bullet characters.
    text = re.sub(r"[•▪●◦]", "\n", text)

    lines = [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]

    section_pattern = re.compile(
        r"^(education\s*&\s*qualifications|"
        r"professional experience|"
        r"work experience|"
        r"employment history|"
        r"education|"
        r"experience|"
        r"skills\s*&\s*interests|"
        r"technical skills|"
        r"skills|"
        r"projects|"
        r"certifications?|"
        r"achievements?|"
        r"honors?\s*&\s*awards?|"
        r"awards?|"
        r"additional experience|"
        r"additional information|"
        r"summary|"
        r"profile|"
        r"objective|"
        r"references?)"
        r"\s*:?\s*$",
        re.IGNORECASE,
    )

    chunks = []

    for line in lines:

        # Page numbers.
        if re.fullmatch(r"\d{1,3}", line):
            continue

        # Contact / personal metadata.
        if re.search(
            r"(address\s*:|cell\s*:|phone\s*:|"
            r"email\s*:|linkedin|github|"
            r"date of birth|dob\s*:)",
            line,
            re.IGNORECASE,
        ):
            continue

        # Standalone URLs.
        if re.fullmatch(r"https?://\S+", line):
            continue

        # Section headings.
        if section_pattern.match(line):
            continue

        # Ignore obvious PDF extraction artifacts.
        if re.search(r"\b\w+\s+\w+\b", line):
            words = line.split()

            # Too many isolated short fragments often indicate
            # broken PDF text extraction.
            if len(words) >= 3 and sum(len(w) <= 2 for w in words) >= 2:
                continue

        # Ignore very short fragments.
        if len(line) < 25:
            continue

        # Long lines → sentence-level chunks.
        if len(line) > 350:

            sentences = re.split(
                r"(?<=[.!?])\s+",
                line
            )

            for sentence in sentences:
                sentence = sentence.strip()

                if len(sentence) >= 25:
                    chunks.append(sentence[:500])

        else:
            chunks.append(line)

    return chunks

# ============================================================
# Build evidence records
# ============================================================

def build_evidence_records(df):
    """
    Chunk each candidate resume once into reusable
    evidence passages.
    """

    # Each candidate should have one resume.
    candidate_resumes = (
        df[["candidate_id", "resume"]]
        .drop_duplicates("candidate_id")
    )

    records = []

    # Chunk each resume exactly once.
    for _, row in candidate_resumes.iterrows():

        chunks = split_into_chunks(row["resume"])

        for chunk_idx, chunk in enumerate(chunks):

            records.append(
                {
                    "candidate_id": row["candidate_id"],
                    "resume_chunk_id": chunk_idx,
                    "resume_chunk": chunk,
                }
            )

    evidence_df = pd.DataFrame(records)

    return evidence_df


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    df = load_netsol()

    print(f"NETSOL rows: {len(df):,}")
    print(
        f"Unique candidates: "
        f"{df['candidate_id'].nunique():,}"
    )
    print(
        f"Unique requirements: "
        f"{df['requirement'].nunique():,}"
    )

    evidence_df = build_evidence_records(df)

    print(
        f"\nEvidence chunks generated: "
        f"{len(evidence_df):,}"
    )

    print(
        f"Unique candidates chunked: "
        f"{evidence_df['candidate_id'].nunique():,}"
    )

    print("\nSample evidence chunks:")
    print("=" * 70)

    for i in range(min(10, len(evidence_df))):
        row = evidence_df.iloc[i]

        print(f"\n[{i}] Candidate: {row['candidate_id']}")
        print(row["resume_chunk"])