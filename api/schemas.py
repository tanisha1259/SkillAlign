from pydantic import BaseModel


class MatchResult(BaseModel):
    requirement: str
    requirement_type: str | None
    importance: str | None
    status: str
    evidence: str | None
    gap_type: str | None
    gap: str | None
    priority: str
    confidence: float
    class_probabilities: dict[str, float]
    retrieved_evidence: list[dict] = []


class AnalysisSummary(BaseModel):
    strong: int
    partial: int
    unsupported: int
    high_priority_gaps: int


class AnalysisResponse(BaseModel):
    requirements_analyzed: int
    match_score: float
    summary: AnalysisSummary
    results: list[MatchResult]
