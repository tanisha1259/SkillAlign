from __future__ import annotations

import re
from typing import Any


def normalize(text: str) -> str:
    return " ".join(str(text).lower().split())


def extract_year_requirement(requirement: str) -> float | None:
    """
    Extract the minimum years required from phrases such as:
    '5+ years', 'at least 3 years', '3-5 years'.
    """
    text = normalize(requirement)

    patterns = [
        r"(\d+(?:\.\d+)?)\s*\+\s*years?",
        r"at\s+least\s+(\d+(?:\.\d+)?)\s+years?",
        r"minimum\s+(?:of\s+)?(\d+(?:\.\d+)?)\s+years?",
        r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s+years?",
    ]

    for i, pattern in enumerate(patterns):
        match = re.search(pattern, text)
        if match:
            if i == 3:
                return float(match.group(1))
            return float(match.group(1))

    return None


def extract_resume_years(evidence: str) -> float | None:
    """
    Extract an explicitly stated experience duration from resume evidence.
    """
    text = normalize(evidence)

    patterns = [
        r"(\d+(?:\.\d+)?)\s*\+\s*years?",
        r"(\d+(?:\.\d+)?)\s+years?",
        r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s+years?",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))

    return None


def classify_gap_type(
    requirement: str,
    requirement_type: str | None = None,
) -> str:
    text = normalize(requirement)
    rtype = normalize(requirement_type or "")

    # Explicit requirement type takes priority.
    if rtype == "experience":
        return "experience_gap"

    if rtype == "education":
        return "education_gap"

    if rtype == "tool":
        return "tool_gap"

    if rtype == "soft_skill":
        return "soft_skill_gap"

    if rtype == "technical_skill":
        return "skill_gap"

    if rtype in {"capability", "responsibility"}:
        return "skill_gap"

    # Fallback inference when requirement_type is unavailable.
    if re.search(
        r"\b(certification|certified|certificate|license|licence)\b",
        text,
    ):
        return "certification_gap"

    if re.search(
        r"\b(bachelor|b\.?s\.?|master|m\.?s\.?|mba|phd|degree|diploma)\b",
        text,
    ):
        return "education_gap"

    if re.search(
        r"\b(crm|salesforce|hubspot|excel|power\s*bi|sql|python|java|"
        r"oracle|sap|tableau|jira|agile|scrum|microsoft office)\b",
        text,
    ):
        return "tool_gap"

    if re.search(r"\b(years?|experience|experienced)\b", text):
        return "experience_gap"

    return "skill_gap"


def determine_priority(
    status: str,
    importance: str | None,
    gap_type: str,
) -> str:
    status = normalize(status)
    importance = normalize(importance or "")

    if status == "strong":
        return "none"

    if importance in {"required", "must_have", "high", "critical"}:
        return "high"

    if status == "unsupported":
        return "high"

    if gap_type in {"experience_gap", "education_gap", "certification_gap"}:
        return "high"

    return "medium"


def generate_gap(
    requirement: str,
    requirement_type: str | None,
    importance: str | None,
    status: str,
    evidence: str | None = None,
) -> dict[str, Any]:

    status_normalized = normalize(status)

    result = {
        "requirement": requirement,
        "requirement_type": requirement_type,
        "importance": importance,
        "status": status,
        "evidence": evidence or None,
        "gap_type": None,
        "gap": None,
        "priority": "none",
    }

    if status_normalized == "strong":
        return result

    gap_type = classify_gap_type(requirement, requirement_type)

    priority = determine_priority(
        status,
        importance,
        gap_type,
    )

    required_years = extract_year_requirement(requirement)
    evidence_years = extract_resume_years(evidence or "")

    # Experience-specific reasoning
    if gap_type == "experience_gap" and required_years is not None:
        if evidence_years is not None and evidence_years < required_years:
            shortfall = required_years - evidence_years

            if shortfall.is_integer():
                shortfall_text = str(int(shortfall))
            else:
                shortfall_text = f"{shortfall:.1f}"

            result["gap"] = (
                f"{shortfall_text} additional year(s) of relevant experience "
                f"may be required"
            )

        elif status_normalized == "unsupported":
            result["gap"] = (
                f"No supporting evidence found for the required "
                f"{int(required_years) if required_years.is_integer() else required_years}+ "
                f"years of experience"
            )
        else:
            result["gap"] = "Experience requirement is only partially supported"

    elif status_normalized == "unsupported":
        result["gap"] = (
            "No supporting evidence found in the resume"
        )

    else:
        result["gap"] = (
            "Requirement has partial supporting evidence; "
            "additional or stronger evidence may be needed"
        )

    result["gap_type"] = gap_type
    result["priority"] = priority

    return result


def analyze_requirements(requirements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert classified requirements into structured skill-gap records.
    """
    results = []

    for item in requirements:
        results.append(
            generate_gap(
                requirement=item.get("requirement", ""),
                requirement_type=item.get("requirement_type"),
                importance=item.get("importance"),
                status=item.get("status", "Unsupported"),
                evidence=item.get("evidence"),
            )
        )

    return results


if __name__ == "__main__":
    examples = [
        {
            "requirement": "5+ years of business development experience",
            "requirement_type": "experience",
            "importance": "required",
            "status": "Partial",
            "evidence": "3 years of business development experience",
        },
        {
            "requirement": "Salesforce experience",
            "requirement_type": "tool",
            "importance": "required",
            "status": "Unsupported",
            "evidence": None,
        },
        {
            "requirement": "Excellent communication skills",
            "requirement_type": "soft_skill",
            "importance": "preferred",
            "status": "Strong",
            "evidence": "Excellent communication and interpersonal skills",
        },
    ]

    for result in analyze_requirements(examples):
        print(result)
