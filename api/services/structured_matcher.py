from __future__ import annotations
import re
from datetime import datetime
def required_years(requirement: str) -> int | None:
    """
    Extract an explicit experience threshold.
    Examples:
        '5+ years of experience' -> 5
        '3-5 years of experience' -> 3
        'at least 5 years' -> 5
        'experience leading teams' -> None
    """
    text = requirement.lower()
    patterns = [
        r"(\d+)\s\+\syears?",
        r"at least\s+(\d+)\syears?",
        r"minimum\s+(\d+)\syears?",
        r"(\d+)\s[-–]\s(\d+)\syears?",
        r"(\d+)\syears?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None
def extract_years(text: str) -> list[int]:
    """
    Extract actual experience durations from evidence.
    Handles:
        3 years
        5+ years
        2020 - 2025
        2020 - Present
    Does NOT treat arbitrary numbers such as:
        10+ events
        2024 projects
    as experience durations.
    """
    if not text:
        return []
    text = text.lower()
    years = []
    # ---------------------------------------------------------
    # Explicit "X years" expressions
    # ---------------------------------------------------------
    explicit = re.findall(
        r"(\d+)\s\+?\syears?\b",
        text,
    )
    for value in explicit:
        years.append(int(value))
    # ---------------------------------------------------------
    # Date ranges: 2020 - 2025
    # ---------------------------------------------------------
    date_ranges = re.findall(
        r"\b(19\d{2}|20\d{2})\s[-–]\s(19\d{2}|20\d{2})\b",
        text,
    )
    for start, end in date_ranges:
        duration = int(end) - int(start)
        if 0 < duration <= 50:
            years.append(duration)
    # ---------------------------------------------------------
    # Date range ending in Present
    # ---------------------------------------------------------
    current_year = datetime.now().year
    present_ranges = re.findall(
        r"\b(19\d{2}|20\d{2})\s[-–]\s(?:present|current)\b",
        text,
    )
    for start in present_ranges:
        duration = current_year - int(start)
        if 0 < duration <= 50:
            years.append(duration)
    return years
def compare_experience(requirement: str, evidence: str) -> dict:
    """
    Structured comparison for explicit numeric experience requirements.
    Important:
    This function must NOT treat education date ranges as relevant
    experience unless the evidence also contains relevant domain terms.
    """
    required = required_years(requirement)
    # Only numeric experience requirements belong here.
    if required is None:
        return {
            "supported": False,
            "status_hint": "Unsupported",
            "required_years": None,
            "candidate_years": None,
        }
    req = requirement.lower()
    ev = evidence.lower()
    # ---------------------------------------------------------
    # Domain-aware experience matching
    # ---------------------------------------------------------
    domain_aliases = {
        "business development": [
            "business development",
            "sales",
            "sales development",
            "lead generation",
            "client acquisition",
            "account management",
            "client management",
        ],
        "sales": [
            "sales",
            "business development",
            "account management",
            "client acquisition",
        ],
        "marketing": [
            "marketing",
            "digital marketing",
            "sales",
            "business development",
        ],
        "software development": [
            "software development",
            "software engineer",
            "developer",
            "programming",
        ],
        "machine learning": [
            "machine learning",
            "ml",
            "deep learning",
        ],
        "data science": [
            "data science",
            "data scientist",
            "machine learning",
        ],
        "data analysis": [
            "data analysis",
            "data analytics",
            "analyst",
        ],
        "project management": [
            "project management",
            "project manager",
            "program manager",
        ],
        "account management": [
            "account management",
            "account manager",
            "client management",
            "customer management",
        ],
        "customer success": [
            "customer success",
            "client success",
            "account management",
        ],
    }
    relevant_aliases = []
    for domain, aliases in domain_aliases.items():
        if domain in req:
            relevant_aliases.extend(aliases)
    # If the requirement has a recognizable domain,
    # evidence must contain relevant terminology.
    if relevant_aliases:
        if not any(alias in ev for alias in relevant_aliases):
            return {
                "supported": False,
                "status_hint": "Unsupported",
                "required_years": required,
                "candidate_years": None,
            }
    candidate_years = extract_years(evidence)
    if not candidate_years:
        return {
            "supported": False,
            "status_hint": "Unsupported",
            "required_years": required,
            "candidate_years": None,
        }
    candidate = max(candidate_years)
    if candidate >= required:
        status = "Strong"
    else:
        status = "Partial"
    return {
        "supported": True,
        "status_hint": status,
        "required_years": required,
        "candidate_years": candidate,
    }
def compare_education(requirement: str, evidence: str) -> dict:
    """
    Structured education matching.
    Only explicit degree/field evidence is accepted.
    A generic bachelor's degree does not automatically satisfy
    a Business Administration / Marketing / Economics requirement.
    Handles common variants such as:
        AI/ML
        AI / ML
        AI and ML
        AIML
        Artificial Intelligence and Machine Learning
    """
    req = requirement.lower()
    ev = evidence.lower()
    # ---------------------------------------------------------
    # Normalize common formatting variations
    # ---------------------------------------------------------
    req_normalized = re.sub(r"\s+", " ", req).strip()
    ev_normalized = re.sub(r"\s+", " ", ev).strip()
    # Normalize slash spacing:
    # "AI / ML" -> "AI/ML"
    # "AI /ML"  -> "AI/ML"
    # "AI/ ML"  -> "AI/ML"
    req_normalized = re.sub(
        r"\bai\s/\sml\b",
        "ai/ml",
        req_normalized,
    )
    ev_normalized = re.sub(
        r"\bai\s/\sml\b",
        "ai/ml",
        ev_normalized,
    )
    # ---------------------------------------------------------
    # Degree / field aliases
    # ---------------------------------------------------------
    degree_aliases = {
        "business administration": [
            "business administration",
            "bba",
            "b.b.a",
            "business management",
            "management studies",
        ],
        "marketing": [
            "marketing",
            "marketing management",
        ],
        "economics": [
            "economics",
            "economic",
        ],
        "computer science": [
            "computer science",
            "computer engineering",
            "b.sc computer science",
            "b.s. computer science",
        ],
        "information technology": [
            "information technology",
            "information systems",
        ],
        "artificial intelligence": [
            "artificial intelligence",
            "artificial intelligence and machine learning",
            "ai/ml",
            "ai and ml",
            "ai & ml",
            "aiml",
            "machine learning",
        ],
        "data science": [
            "data science",
            "data analytics",
        ],
    }
    requested_aliases = []
    for field, aliases in degree_aliases.items():
        if field in req_normalized:
            requested_aliases.extend(aliases)
    # Explicitly handle AI/ML variants in the requirement.
    if "ai/ml" in req_normalized:
        requested_aliases.extend(
            degree_aliases["artificial intelligence"]
        )
    # Handle "AI & ML"
    if "ai & ml" in req_normalized:
        requested_aliases.extend(
            degree_aliases["artificial intelligence"]
        )
    # Handle "AIML"
    if re.search(r"\baiml\b", req_normalized):
        requested_aliases.extend(
            degree_aliases["artificial intelligence"]
        )
    # ---------------------------------------------------------
    # Related technical fields
    # ---------------------------------------------------------
    # AI/ML is a closely related technical field for CS, IT, and
    # Data Science requirements. Do not apply this relationship to
    # business/marketing/economics requirements.
    technical_related_aliases = degree_aliases["artificial intelligence"]
    if "computer science" in req_normalized:
        requested_aliases.extend(technical_related_aliases)
    if "information technology" in req_normalized:
        requested_aliases.extend(technical_related_aliases)
    if "data science" in req_normalized:
        requested_aliases.extend(technical_related_aliases)
    # Remove duplicates while preserving order.
    requested_aliases = list(dict.fromkeys(requested_aliases))
    # ---------------------------------------------------------
    # No recognized field in the requirement
    # ---------------------------------------------------------
    if not requested_aliases:
        return {
            "supported": False,
            "status_hint": "Unsupported",
            "matched_field": None,
        }
    # ---------------------------------------------------------
    # Look for an explicit matching field in the resume
    # ---------------------------------------------------------
    matched_alias = next(
        (
            alias
            for alias in requested_aliases
            if alias in ev_normalized
        ),
        None,
    )
    if matched_alias is None:
        return {
            "supported": False,
            "status_hint": "Unsupported",
            "matched_field": None,
        }
    return {
        "supported": True,
        "status_hint": "Strong",
        "matched_field": matched_alias,
    }
