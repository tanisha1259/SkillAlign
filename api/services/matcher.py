import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from analyze_match import SkillAlignClassifier


_classifier = None


def get_classifier() -> SkillAlignClassifier:
    global _classifier

    if _classifier is None:
        _classifier = SkillAlignClassifier()

    return _classifier


def analyze_requirements(requirements: list[dict]) -> list[dict]:
    classifier = get_classifier()

    results = []

    for item in requirements:
        result = classifier.predict(
            requirement=item["requirement"],
            evidence=item.get("evidence"),
            requirement_type=item.get("requirement_type"),
            importance=item.get("importance"),
        )

        results.append(result)

    return results
