from __future__ import annotations

import re


SECTION_NAMES = {
    "education",
    "education & qualifications",
    "professional experience",
    "work experience",
    "employment history",
    "experience",
    "skills",
    "technical skills",
    "skills & interests",
    "projects",
    "certifications",
    "achievements",
    "honors & awards",
    "awards",
    "additional experience",
    "additional information",
    "summary",
    "profile",
    "objective",
    "references",
}


def clean_line(line: str) -> str:
    line = str(line).strip()

    # Normalize bullets.
    line = re.sub(r"^[•▪●◦*-]\s*", "", line)

    # Normalize whitespace.
    line = re.sub(r"\s+", " ", line)

    return line.strip()


def is_section_heading(line: str) -> bool:
    normalized = line.lower().rstrip(":").strip()
    return normalized in SECTION_NAMES


def is_metadata(line: str) -> bool:
    return bool(
        re.search(
            r"(address\s*:|cell\s*:|phone\s*:|email\s*:|"
            r"linkedin|github|date of birth|dob\s*:)",
            line,
            re.IGNORECASE,
        )
    )


def looks_like_role_or_header(line: str) -> bool:
    """
    Detect likely experience/education headers.

    Date lines are NOT treated as new blocks because they
    belong to the preceding role/company header.
    """

    lower = line.lower()

    # A standalone date range belongs to the current role.
    if re.fullmatch(
        r"\s*(?:19|20)\d{2}\s*[-–—]\s*"
        r"(?:present|(?:19|20)\d{2})\s*",
        line,
        re.IGNORECASE,
    ):
        return False

    role_keywords = [
        "manager",
        "developer",
        "engineer",
        "analyst",
        "intern",
        "lead",
        "consultant",
        "coordinator",
        "specialist",
        "director",
        "executive",
        "associate",
        "assistant",
        "designer",
    ]

    return any(keyword in lower for keyword in role_keywords)

def build_contextual_chunks(text: str) -> list[str]:
    """
    Build context-preserving resume evidence blocks.

    Keeps a role/company/date header together with the
    responsibilities that follow it.
    """

    text = str(text).replace("\r", "\n")

    raw_lines = [
        clean_line(line)
        for line in text.split("\n")
    ]

    lines = [
        line
        for line in raw_lines
        if line
    ]

    chunks: list[str] = []

    current: list[str] = []
    current_section = ""

    def flush():
        nonlocal current

        if not current:
            return

        content = " ".join(current)
        content = re.sub(r"\s+", " ", content).strip()

        if len(content) >= 30:
            chunks.append(content[:1200])

        current = []

    for i, line in enumerate(lines):

        if re.fullmatch(r"\d{1,3}", line):
            continue

        if is_metadata(line):
            continue

        if re.fullmatch(r"https?://\S+", line):
            continue

        if is_section_heading(line):
            flush()
            current_section = line.rstrip(":")
            continue

        # If this looks like a role title, collect the following
        # company/date lines before treating the block as complete.
        if not current and looks_like_role_or_header(line):
            current.append(line)
            continue

        # If we already have a role title, keep the next few
        # header lines with it.
        if current and len(current) <= 2:
            current.append(line)
            continue

        # Otherwise this is evidence/responsibility content.
        current.append(line)

        if len(" ".join(current)) >= 900:
            flush()

    flush()

    return chunks