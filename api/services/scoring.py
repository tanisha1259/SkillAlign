from __future__ import annotations


STATUS_VALUES = {
    "Strong": 1.0,
    "Partial": 0.5,
    "Unsupported": 0.0,
}


def requirement_weight(importance: str | None) -> float:
    """
    Required requirements contribute fully.
    Preferred requirements contribute half as much.
    """
    if importance == "preferred":
        return 0.5

    return 1.0


def calculate_match_score(results: list[dict]) -> dict:
    """
    Calculate an explainable weighted resume-JD match score.

    Strong       = 1.0
    Partial      = 0.5
    Unsupported  = 0.0

    Required    = weight 1.0
    Preferred   = weight 0.5
    """

    if not results:
        return {
            "score": 0.0,
            "total_weight": 0.0,
            "earned_weight": 0.0,
        }

    total_weight = 0.0
    earned_weight = 0.0

    for result in results:
        importance = result.get("importance")
        status = result.get("status", "Unsupported")

        weight = requirement_weight(importance)
        value = STATUS_VALUES.get(status, 0.0)

        total_weight += weight
        earned_weight += weight * value

    score = (
        (earned_weight / total_weight) * 100
        if total_weight > 0
        else 0.0
    )

    return {
        "score": round(score, 2),
        "total_weight": round(total_weight, 2),
        "earned_weight": round(earned_weight, 2),
    }