from __future__ import annotations

import re


def clean_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("\r", "\n")

    # Remove bullet characters / markdown bullets
    text = re.sub(r"^\s*[-•▪◦*]\s*", "", text)

    # Remove numbered bullets
    text = re.sub(r"^\s*\d+[\.)]\s*", "", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def split_sentences(text: str) -> list[str]:
    text = clean_text(text)

    # Normalize bullets.
    text = re.sub(r"[•▪●◦]", "\n", text)

    # Put common JD headings on their own line.
    text = re.sub(
        r"\b(?:requirements?|qualifications?|responsibilities?)\s*:",
        "\n",
        text,
        flags=re.IGNORECASE,
    )

    lines = []

    for line in text.split("\n"):
        line = line.strip()

        if not line:
            continue

        # Remove bullet / numbering prefixes.
        line = re.sub(
            r"^(?:[-*+]|\d+[.)]|[a-zA-Z][.)])\s+",
            "",
            line,
        )

        # Remove standalone headings.
        if re.fullmatch(
            r"(?:job\s+description|requirements?|qualifications?|"
            r"responsibilities?)",
            line,
            flags=re.IGNORECASE,
        ):
            continue

        if line:
            lines.append(line)

    sentences = []

    for line in lines:
        parts = re.split(r"(?<=[.!?])\s+", line)

        for part in parts:
            part = part.strip()

            if len(part) >= 15:
                sentences.append(part)

    return sentences


def is_requirement(text: str) -> bool:
    text_lower = text.lower()

    requirement_patterns = [
        r"\brequire",
        r"\brequired\b",
        r"\bmust\b",
        r"\bshould\b",
        r"\bexperience\b",
        r"\byears?\b",
        r"\bproficien",
        r"\bknowledge\b",
        r"\bskills?\b",
        r"\bability\b",
        r"\bstrong\b",
        r"\bexcellent\b",
        r"\bfamiliar\b",
        r"\bdegree\b",
        r"\bbachelor",
        r"\bmaster",
        r"\bmba\b",
        r"\bphd\b",
        r"\bcertif",
        r"\bproven track record\b",
        r"\bexpertise\b",
        r"\bproficient\b",
    ]

    return any(
        re.search(pattern, text_lower)
        for pattern in requirement_patterns
    )


def infer_requirement_type(text: str) -> str:
    t = text.lower()

    # IMPORTANT:
    # Explicit tools/technologies take priority over the generic
    # word "experience".
    if re.search(
        r"\b("
        r"salesforce|hubspot|crm|"
        r"microsoft\s+office|ms\s+office|"
        r"excel|power\s*bi|sql|python|java|"
        r"oracle|sap|tableau|jira|"
        r"agile|scrum"
        r")\b",
        t,
    ):
        return "tool"

    if re.search(
        r"\b(certification|certified|certificate|license|licence)\b",
        t,
    ):
        return "certification"

    if re.search(
        r"\b(bachelor|b\.?s\.?|master|m\.?s\.?|mba|phd|degree|diploma)\b",
        t,
    ):
        return "education"

    if re.search(
        r"\b(years?|experience|experienced)\b",
        t,
    ):
        return "experience"

    if re.search(
        r"\b(communication|leadership|teamwork|"
        r"interpersonal|negotiation|collaboration|"
        r"presentation|problem solving)\b",
        t,
    ):
        return "soft_skill"

    if re.search(
        r"\b(develop|manage|lead|coordinate|"
        r"generate|analyze|forecast|plan|"
        r"build|design|implement)\b",
        t,
    ):
        return "capability"

    return "technical_skill"


def infer_importance(text: str) -> str:
    t = text.lower()

    if re.search(
        r"\b(required|required qualifications|must|mandatory)\b",
        t,
    ):
        return "required"

    if re.search(
        r"\b(preferred|nice to have|plus|bonus)\b",
        t,
    ):
        return "preferred"

    return "required"


def extract_requirements(job_description: str) -> list[dict]:
    candidates = split_sentences(job_description)

    requirements = []
    seen = set()

    for sentence in candidates:
        if not is_requirement(sentence):
            continue

        normalized = " ".join(sentence.lower().split())

        if normalized in seen:
            continue

        seen.add(normalized)

        requirements.append(
            {
                "requirement": sentence,
                "requirement_type": infer_requirement_type(sentence),
                "importance": infer_importance(sentence),
            }
        )

    return requirements
