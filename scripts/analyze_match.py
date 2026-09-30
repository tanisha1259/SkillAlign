from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Allow imports from scripts/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skill_gap_engine import generate_gap


MODEL_PATH = ROOT / "outputs" / "modernbert_skillalign"

LABELS = {
    0: "Unsupported",
    1: "Partial",
    2: "Strong",
}


class SkillAlignClassifier:
    def __init__(self, model_path: str | Path = MODEL_PATH):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Loading SkillAlign classifier from: {model_path}")
        print(f"Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        )

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(
        self,
        requirement: str,
        evidence: str | None = None,
        requirement_type: str | None = None,
        importance: str | None = None,
    ) -> dict:
        evidence = evidence or ""

        # No retrieved evidence means the requirement is unsupported.
        # Do not ask the classifier to classify an empty evidence pair.
        if not evidence.strip():
            status = "Unsupported"

            result = generate_gap(
                requirement=requirement,
                requirement_type=requirement_type,
                importance=importance,
                status=status,
                evidence=None,
            )

            result["confidence"] = 1.0
            result["class_probabilities"] = {
                "Unsupported": 1.0,
                "Partial": 0.0,
                "Strong": 0.0,
            }

            return result

        encoded = self.tokenizer(
            requirement,
            evidence,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        outputs = self.model(**encoded)

        probabilities = torch.softmax(outputs.logits, dim=-1)[0]
        predicted_id = int(torch.argmax(probabilities).item())

        status = LABELS[predicted_id]
        confidence = float(probabilities[predicted_id].item())

        result = generate_gap(
            requirement=requirement,
            requirement_type=requirement_type,
            importance=importance,
            status=status,
            evidence=evidence if evidence else None,
        )

        result["confidence"] = round(confidence, 4)

        result["class_probabilities"] = {
            LABELS[i]: round(float(probabilities[i].item()), 4)
            for i in range(len(LABELS))
        }

        return result


def main():
    classifier = SkillAlignClassifier()

    test_cases = [
        {
            "requirement": "5+ years of business development experience",
            "requirement_type": "experience",
            "importance": "required",
            "evidence": "3 years of business development experience",
        },
        {
            "requirement": "Salesforce experience",
            "requirement_type": "tool",
            "importance": "required",
            "evidence": "",
        },
        {
            "requirement": "Excellent communication skills",
            "requirement_type": "soft_skill",
            "importance": "preferred",
            "evidence": "Excellent communication and interpersonal skills",
        },
    ]

    print("\n" + "=" * 70)
    print("SKILLALIGN — END-TO-END MATCH ANALYSIS")
    print("=" * 70)

    for i, case in enumerate(test_cases, 1):
        result = classifier.predict(**case)

        print(f"\nCASE {i}")
        print("-" * 70)

        print(f"Requirement : {result['requirement']}")
        print(f"Evidence    : {result['evidence']}")
        print(f"Status      : {result['status']}")
        print(f"Confidence  : {result['confidence']}")
        print(f"Gap Type    : {result['gap_type']}")
        print(f"Gap         : {result['gap']}")
        print(f"Priority    : {result['priority']}")

        print(
            "Probabilities:",
            result["class_probabilities"],
        )


if __name__ == "__main__":
    main()
