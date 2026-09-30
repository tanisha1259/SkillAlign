import pandas as pd

import re

GOLD_PATH = "data/gold_annotations.csv"
WORKSHEET_PATH = "data/gold_pilot_worksheet.csv"

VALID_LABELS = {"Strong", "Partial", "Unsupported"}

VALID_TYPES = {
    "technical_skill",
    "tool",
    "programming_language",
    "framework",
    "domain_knowledge",
    "experience",
    "education",
    "certification",
    "soft_skill",
    "responsibility",
    "capability",
    "other",
}

gold = pd.read_csv(GOLD_PATH, dtype=str).fillna("")
worksheet = pd.read_csv(WORKSHEET_PATH, dtype=str).fillna("")

print(f"Gold rows: {len(gold)}")
print(f"Worksheet rows: {len(worksheet)}")

errors = []

# --------------------------------------------------
# 1. Required columns
# --------------------------------------------------

required_columns = [
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

for col in required_columns:
    if col not in gold.columns:
        errors.append(f"Missing column: {col}")

# --------------------------------------------------
# 2. Annotation IDs
# --------------------------------------------------

if gold["annotation_id"].duplicated().any():
    errors.append("Duplicate annotation IDs found.")

expected_ids = {str(i) for i in range(1, 51)}
actual_ids = set(gold["annotation_id"])

missing_ids = expected_ids - actual_ids
extra_ids = actual_ids - expected_ids

if missing_ids:
    errors.append(f"Missing annotation IDs: {sorted(missing_ids)}")

if extra_ids:
    errors.append(f"Unexpected annotation IDs: {sorted(extra_ids)}")

# --------------------------------------------------
# 3. Missing labels / types
# --------------------------------------------------

missing_labels = gold["label"].eq("")
if missing_labels.any():
    ids = gold.loc[missing_labels, "annotation_id"].tolist()
    errors.append(f"Missing labels: {ids}")

missing_types = gold["requirement_type"].eq("")
if missing_types.any():
    ids = gold.loc[missing_types, "annotation_id"].tolist()
    errors.append(f"Missing requirement types: {ids}")

# --------------------------------------------------
# 4. Valid label values
# --------------------------------------------------

invalid_labels = gold.loc[
    ~gold["label"].isin(VALID_LABELS),
    ["annotation_id", "label"],
]

if len(invalid_labels) > 0:
    errors.append(
        f"Invalid labels: {invalid_labels.to_dict('records')}"
    )

# --------------------------------------------------
# 5. Valid requirement types
# --------------------------------------------------

invalid_types = gold.loc[
    ~gold["requirement_type"].isin(VALID_TYPES),
    ["annotation_id", "requirement_type"],
]

if len(invalid_types) > 0:
    errors.append(
        f"Invalid requirement types: {invalid_types.to_dict('records')}"
    )

# --------------------------------------------------
# 6. Evidence rules
# --------------------------------------------------

unsupported_with_evidence = gold[
    (gold["label"] == "Unsupported")
    & gold["resume_evidence"].str.strip().ne("")
]

if len(unsupported_with_evidence) > 0:
    errors.append(
        "Unsupported rows contain evidence: "
        + str(unsupported_with_evidence["annotation_id"].tolist())
    )

supported_without_evidence = gold[
    (gold["label"].isin(["Strong", "Partial"]))
    & gold["resume_evidence"].str.strip().eq("")
]

if len(supported_without_evidence) > 0:
    errors.append(
        "Strong/Partial rows have no evidence: "
        + str(supported_without_evidence["annotation_id"].tolist())
    )

# --------------------------------------------------
# 7. Evidence presence check
# --------------------------------------------------

# Gold evidence may be a normalized representation of
# PDF-extracted text, so exact substring matching is not
# used as a hard validation failure here.
#
# We only verify that:
# - Strong/Partial rows have evidence
# - Unsupported rows have no evidence
#
# Exact evidence-span alignment will be validated later
# on the expanded Gold dataset.

supported_without_evidence = gold[
    (gold["label"].isin(["Strong", "Partial"]))
    & gold["resume_evidence"].str.strip().eq("")
]

if len(supported_without_evidence) > 0:
    errors.append(
        "Strong/Partial rows have no evidence: "
        + str(supported_without_evidence["annotation_id"].tolist())
    )

# --------------------------------------------------
# 8. Duplicate annotation content
# --------------------------------------------------

duplicate_content = gold.duplicated(
    subset=[
        "candidate_id",
        "requirement",
        "resume_evidence",
        "label",
    ],
    keep=False,
)

if duplicate_content.any():
    duplicate_ids = gold.loc[
        duplicate_content, "annotation_id"
    ].tolist()

    print(
        "\nNote: duplicate annotation content detected "
        "(not necessarily an error):"
    )
    print(duplicate_ids)

# --------------------------------------------------
# Final report
# --------------------------------------------------

print("\n" + "=" * 50)
print("GOLD PILOT VALIDATION")
print("=" * 50)

if errors:
    print(f"\n❌ Validation failed: {len(errors)} issue(s)\n")

    for i, error in enumerate(errors, 1):
        print(f"{i}. {error}")

    raise SystemExit(1)

print("\n✅ All validation checks passed.")

print("\nLabel distribution:")
print(gold["label"].value_counts())

print("\nRequirement-type distribution:")
print(gold["requirement_type"].value_counts())

print("\nEvidence coverage:")
supported = gold["label"].isin(["Strong", "Partial"])
has_evidence = gold["resume_evidence"].str.strip().ne("")

print(
    f"Strong/Partial with evidence: "
    f"{(supported & has_evidence).sum()}/{supported.sum()}"
)

print(
    f"Unsupported with empty evidence: "
    f"{((gold['label'] == 'Unsupported') & ~has_evidence).sum()}/"
    f"{(gold['label'] == 'Unsupported').sum()}"
)