import pandas as pd
from collections import Counter

GOLD_PATH = "data/gold_annotations.csv"

df = pd.read_csv(GOLD_PATH, dtype={"annotation_id": str})

errors = []

# -----------------------------
# Basic checks
# -----------------------------
if len(df) != 200:
    errors.append(f"Expected 200 rows, found {len(df)}")

if df["annotation_id"].duplicated().any():
    dupes = df.loc[df["annotation_id"].duplicated(), "annotation_id"].tolist()
    errors.append(f"Duplicate annotation IDs: {dupes}")

expected_ids = {str(i) for i in range(1, 201)}
actual_ids = set(df["annotation_id"].astype(str))

missing_ids = expected_ids - actual_ids
extra_ids = actual_ids - expected_ids

if missing_ids:
    errors.append(f"Missing annotation IDs: {sorted(missing_ids)}")

if extra_ids:
    errors.append(f"Unexpected annotation IDs: {sorted(extra_ids)}")

# -----------------------------
# Required fields
# -----------------------------
required_cols = [
    "annotation_id",
    "requirement_id",
    "candidate_id",
    "requirement",
    "requirement_type",
    "importance",
    "resume_evidence",
    "label",
    "evidence_start",
    "evidence_end",
    "notes",
]

for col in required_cols:
    if col not in df.columns:
        errors.append(f"Missing column: {col}")

# -----------------------------
# Label checks
# -----------------------------
valid_labels = {"Strong", "Partial", "Unsupported"}

bad_labels = df.loc[
    ~df["label"].fillna("").isin(valid_labels),
    ["annotation_id", "label"]
]

if len(bad_labels):
    errors.append(
        f"Invalid/missing labels: {bad_labels.to_dict('records')}"
    )

# -----------------------------
# Requirement type checks
# -----------------------------
valid_types = {
    "education",
    "experience",
    "tool",
    "soft_skill",
    "capability",
    "responsibility",
    "technical_skill",
}

bad_types = df.loc[
    ~df["requirement_type"].fillna("").isin(valid_types),
    ["annotation_id", "requirement_type"]
]

if len(bad_types):
    errors.append(
        f"Invalid/missing requirement types: {bad_types.to_dict('records')}"
    )

# -----------------------------
# Evidence consistency
# -----------------------------
evidence = df["resume_evidence"].fillna("").astype(str).str.strip()

supported = df["label"].isin(["Strong", "Partial"])
unsupported = df["label"].eq("Unsupported")

missing_evidence = df.loc[
    supported & evidence.eq(""),
    ["annotation_id", "label", "requirement"]
]

if len(missing_evidence):
    errors.append(
        f"Strong/Partial rows missing evidence: "
        f"{missing_evidence.to_dict('records')}"
    )

unsupported_with_evidence = df.loc[
    unsupported & evidence.ne(""),
    ["annotation_id", "requirement", "resume_evidence"]
]

if len(unsupported_with_evidence):
    errors.append(
        f"Unsupported rows contain evidence: "
        f"{unsupported_with_evidence.to_dict('records')}"
    )

# -----------------------------
# Candidate / requirement checks
# -----------------------------
for col in ["candidate_id", "requirement", "requirement_id"]:
    missing = df[col].fillna("").astype(str).str.strip().eq("")
    if missing.any():
        errors.append(
            f"{col} has {missing.sum()} missing/empty values"
        )

# -----------------------------
# Duplicate/conflicting annotation checks
# -----------------------------
group_cols = ["candidate_id", "requirement"]

group_sizes = df.groupby(group_cols, dropna=False).size()
duplicate_groups = group_sizes[group_sizes > 1]

if len(duplicate_groups):
    conflicts = []

    for (candidate, requirement), group in df.groupby(
        group_cols, dropna=False
    ):
        labels = set(group["label"].dropna())

        if len(labels) > 1:
            conflicts.append({
                "candidate_id": candidate,
                "requirement": requirement,
                "labels": sorted(labels),
            })

    if conflicts:
        errors.append(
            f"Conflicting duplicate annotations: {conflicts}"
        )

# -----------------------------
# Report
# -----------------------------
print("=" * 60)
print("GOLD DATASET VALIDATION")
print("=" * 60)

print(f"Rows: {len(df)}")
print(
    f"Annotation IDs: "
    f"{df.annotation_id.astype(int).min()}-"
    f"{df.annotation_id.astype(int).max()}"
)

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nRequirement-type distribution:")
print(df["requirement_type"].value_counts())

print("\nEvidence coverage:")
print(
    f"Strong/Partial with evidence: "
    f"{(supported & evidence.ne('')).sum()}/{supported.sum()}"
)
print(
    f"Unsupported without evidence: "
    f"{(unsupported & evidence.eq('')).sum()}/{unsupported.sum()}"
)

print(f"\nDuplicate candidate+requirement groups: {len(duplicate_groups)}")

if errors:
    print("\n❌ VALIDATION FAILED")
    print("-" * 60)

    for error in errors:
        print(f"- {error}")

    raise SystemExit(1)

print("\n" + "=" * 60)
print("✅ ALL GOLD VALIDATION CHECKS PASSED")
print("=" * 60)
