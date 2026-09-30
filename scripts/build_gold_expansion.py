import pandas as pd
from pathlib import Path

NETSOL_PATH = Path("data/netsol_clean.csv")
GOLD_PATH = Path("data/gold_annotations.csv")
OUTPUT_PATH = Path("data/gold_expansion_candidates.csv")

df = pd.read_csv(NETSOL_PATH)

# Same cleaning used for the existing Gold pilot
df = df.dropna(
    subset=["candidate_id", "resume", "requirement"]
).copy()

df = df[
    df["resume"].astype(str).str.strip().ne("")
].copy()

# Remove duplicate requirement/resume annotations
df = df.drop_duplicates(
    subset=["candidate_id", "requirement"]
).copy()

# Existing Gold requirements
gold = pd.read_csv(GOLD_PATH, dtype=str)

existing = set(
    zip(
        gold["candidate_id"].fillna(""),
        gold["requirement"].fillna("")
    )
)

# Keep only annotations not already present in Gold
df["key"] = list(
    zip(
        df["candidate_id"].astype(str),
        df["requirement"].astype(str)
    )
)

remaining = df[~df["key"].isin(existing)].copy()

# Select a diverse batch instead of simply taking the next 150 rows.
remaining["requirement_lower"] = (
    remaining["requirement"]
    .astype(str)
    .str.lower()
)

def classify_candidate_requirement(text):
    text = text.lower()

    if any(x in text for x in [
        "bachelor",
        "master",
        "degree",
        "mba",
        "b.com",
        "bba",
        "computer science"
    ]):
        return "education"

    if any(x in text for x in [
        "years of experience",
        "years relevant",
        "experience in",
        "experience with"
    ]):
        return "experience"

    if any(x in text for x in [
        "communication",
        "leadership",
        "teamwork",
        "mentoring",
        "interpersonal",
        "multitask",
        "manage multiple"
    ]):
        return "soft_skill"

    if any(x in text for x in [
        "salesforce",
        "hubspot",
        "crm",
        "microsoft office",
        "python",
        "sql",
        "java",
        "power bi",
        "excel",
        "tableau"
    ]):
        return "tool"

    return "other"


remaining["candidate_type"] = remaining[
    "requirement_lower"
].apply(classify_candidate_requirement)

# Take a balanced sample across requirement categories.
parts = []

target_per_type = 30

for req_type in [
    "education",
    "experience",
    "tool",
    "soft_skill",
    "other",
]:
    subset = remaining[
        remaining["candidate_type"] == req_type
    ].head(target_per_type)

    parts.append(subset)

selected = pd.concat(parts, ignore_index=True)

# If fewer than 150 were obtained, fill from remaining rows.
if len(selected) < 150:
    selected_keys = set(selected["key"])

    extra = remaining[
        ~remaining["key"].isin(selected_keys)
    ].head(150 - len(selected))

    selected = pd.concat(
        [selected, extra],
        ignore_index=True
    )

selected = selected.head(150).copy()

# Create annotation IDs continuing from 51.
selected.insert(
    0,
    "annotation_id",
    range(51, 51 + len(selected))
)

selected = selected[
    [
        "annotation_id",
        "candidate_id",
        "requirement",
        "resume",
        "candidate_type",
    ]
]

selected.to_csv(
    OUTPUT_PATH,
    index=False
)

print(f"Created: {OUTPUT_PATH}")
print(f"Rows: {len(selected)}")
print()
print("Requirement category distribution:")
print(selected["candidate_type"].value_counts())