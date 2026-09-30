from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

NETSOL_FILE = PROJECT_ROOT / "data" / "netsol_clean.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "gold_annotations.csv"


def main():
    df = pd.read_csv(NETSOL_FILE)

    # Keep only usable rows.
    df = df.dropna(
        subset=["candidate_id", "resume", "requirement"]
    ).copy()

    df = df[
        df["resume"].astype(str).str.strip().ne("")
    ].copy()

    # We are NOT treating the NETSOL label as our Gold label.
    # It is only being used to select source examples.
    pilot = (
        df[
            [
                "candidate_id",
                "requirement",
                "resume",
                "label",
            ]
        ]
        .drop_duplicates()
        .head(50)
        .copy()
    )

    pilot.insert(
        0,
        "annotation_id",
        range(1, len(pilot) + 1),
    )

    pilot.insert(
        1,
        "requirement_id",
        [
            f"req_{i:04d}"
            for i in range(1, len(pilot) + 1)
        ],
    )

    pilot["requirement_type"] = ""
    pilot["importance"] = ""
    pilot["resume_evidence"] = ""
    pilot["label"] = ""
    pilot["evidence_start"] = ""
    pilot["evidence_end"] = ""
    pilot["notes"] = ""

    pilot = pilot[
        [
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
    ]

    pilot.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"Created pilot: {OUTPUT_FILE}")
    print(f"Rows: {len(pilot)}")


if __name__ == "__main__":
    main()