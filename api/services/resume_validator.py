from __future__ import annotations

import re
from pathlib import Path

# Signals that commonly appear in resumes.
CONTACT_PATTERNS = [
    r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b",              # email
    r"(?:\+?\d[\d\s().-]{7,}\d)",                 # phone
    r"\blinkedin\.com\b",
    r"\bgithub\.com\b",
    r"\bportfolio\b",
]

EDUCATION_TERMS = [
    "bachelor",
    "master",
    "b.e.",
    "b.e",
    "b.tech",
    "m.e.",
    "m.e",
    "m.tech",
    "b.sc",
    "m.sc",
    "bca",
    "mca",
    "phd",
    "degree",
    "university",
    "college",
    "institute",
    "education",
]

EXPERIENCE_TERMS = [
    "experience",
    "work experience",
    "professional experience",
    "internship",
    "intern",
    "employment",
    "worked",
    "role",
    "position",
    "responsibilities",
]

SKILL_TERMS = [
    "skills",
    "technical skills",
    "programming",
    "python",
    "java",
    "c++",
    "sql",
    "machine learning",
    "deep learning",
    "tensorflow",
    "pytorch",
    "scikit-learn",
    "javascript",
    "react",
    "docker",
    "aws",
    "azure",
    "git",
]

PROJECT_TERMS = [
    "projects",
    "project",
    "developed",
    "built",
    "implemented",
    "designed",
    "created",
    "deployed",
    "developed a",
]

RESUME_SECTION_TERMS = [
    "summary",
    "objective",
    "education",
    "experience",
    "work experience",
    "skills",
    "technical skills",
    "projects",
    "certifications",
    "achievements",
    "leadership",
    "activities",
    "publications",
    "awards",
]

NON_RESUME_TERMS = [
    "invoice",
    "invoice number",
    "subtotal",
    "tax amount",
    "total due",
    "bill to",
    "purchase order",
    "receipt",
    "balance due",
    "terms and conditions",
    "abstract",
    "keywords",
    "references",
    "doi:",
    "table of contents",
    "copyright",
]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _count_matches(text: str, terms: list[str]) -> int:
    return sum(1 for term in terms if term in text)


def _has_contact_signal(text: str) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in CONTACT_PATTERNS)


def validate_resume_document(
    text: str,
    *,
    min_characters: int = 250,
) -> dict:
    """
    Determine whether extracted PDF text is likely to represent a resume.

    Returns:
        {
            "is_resume": bool,
            "confidence": float,
            "reason": str,
            "signals": dict,
        }
    """

    normalized = _normalize(text)

    if len(normalized) < min_characters:
        return {
            "is_resume": False,
            "confidence": 0.0,
            "reason": (
                "We couldn't extract enough text from this PDF. "
                "Please upload a text-readable resume."
            ),
            "signals": {},
        }

    # ---------------------------------------------------------
    # Positive signals
    # ---------------------------------------------------------

    contact_signal = _has_contact_signal(normalized)
    education_hits = _count_matches(normalized, EDUCATION_TERMS)
    experience_hits = _count_matches(normalized, EXPERIENCE_TERMS)
    skill_hits = _count_matches(normalized, SKILL_TERMS)
    project_hits = _count_matches(normalized, PROJECT_TERMS)
    section_hits = _count_matches(normalized, RESUME_SECTION_TERMS)

    # A resume usually contains multiple independent categories.
    category_count = sum(
        [
            contact_signal,
            education_hits > 0,
            experience_hits > 0,
            skill_hits >= 2,
            project_hits > 0,
            section_hits >= 2,
        ]
    )

    # ---------------------------------------------------------
    # Negative signals
    # ---------------------------------------------------------

    non_resume_hits = _count_matches(normalized, NON_RESUME_TERMS)

    # Strong document-type indicators should have more influence.
    score = 0.0

    if contact_signal:
        score += 2.0

    if education_hits > 0:
        score += 2.0

    if experience_hits > 0:
        score += 2.0

    if skill_hits >= 2:
        score += 2.0
    elif skill_hits == 1:
        score += 1.0

    if project_hits > 0:
        score += 1.5

    if section_hits >= 2:
        score += 2.0
    elif section_hits == 1:
        score += 1.0

    score -= non_resume_hits * 2.5

    # ---------------------------------------------------------
    # Explicit resume-like combinations
    # ---------------------------------------------------------

    strong_resume_pattern = (
        contact_signal
        and (education_hits > 0 or experience_hits > 0)
        and (skill_hits >= 2 or project_hits > 0)
    )

    if strong_resume_pattern:
        score += 2.0

    # ---------------------------------------------------------
    # Explicit non-resume protection
    # ---------------------------------------------------------

    # A document dominated by academic-paper signals should not
    # accidentally pass merely because it contains words like
    # "experience" or "projects".
    academic_paper_signal = (
        "abstract" in normalized
        and ("references" in normalized or "doi:" in normalized)
        and "keywords" in normalized
    )

    if academic_paper_signal:
        return {
            "is_resume": False,
            "confidence": 0.98,
            "reason": (
                "This document appears to be an academic paper "
                "rather than a resume."
            ),
            "signals": {
                "contact": contact_signal,
                "education_hits": education_hits,
                "experience_hits": experience_hits,
                "skill_hits": skill_hits,
                "project_hits": project_hits,
                "section_hits": section_hits,
                "non_resume_hits": non_resume_hits,
                "category_count": category_count,
            },
        }

    # Invoice / receipt / financial document protection.
    if non_resume_hits >= 3 and category_count <= 2:
        return {
            "is_resume": False,
            "confidence": 0.98,
            "reason": (
                "This document does not appear to be a resume. "
                "Please upload a candidate resume in PDF format."
            ),
            "signals": {
                "contact": contact_signal,
                "education_hits": education_hits,
                "experience_hits": experience_hits,
                "skill_hits": skill_hits,
                "project_hits": project_hits,
                "section_hits": section_hits,
                "non_resume_hits": non_resume_hits,
                "category_count": category_count,
            },
        }

    # ---------------------------------------------------------
    # Final decision
    # ---------------------------------------------------------

    # Strong multi-signal resume.
    if score >= 7.0 and category_count >= 3:
        is_resume = True
    # Borderline documents are allowed through if they contain
    # enough resume-specific evidence.
    elif score >= 5.0 and category_count >= 3:
        is_resume = True
    else:
        is_resume = False

    confidence = min(max(score / 12.0, 0.0), 1.0)

    if is_resume:
        reason = "Resume document detected."
    else:
        reason = (
            "This file doesn't appear to be a resume. "
            "Please upload a candidate resume in PDF format."
        )

    return {
        "is_resume": is_resume,
        "confidence": round(confidence, 3),
        "reason": reason,
        "signals": {
            "contact": contact_signal,
            "education_hits": education_hits,
            "experience_hits": experience_hits,
            "skill_hits": skill_hits,
            "project_hits": project_hits,
            "section_hits": section_hits,
            "non_resume_hits": non_resume_hits,
            "category_count": category_count,
        },
    }