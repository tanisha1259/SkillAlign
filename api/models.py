from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.db import Base


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    job_title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    job_description: Mapped[str] = mapped_column(Text, nullable=False)
    match_score: Mapped[float] = mapped_column(Float, nullable=False)
    requirements_analyzed: Mapped[int] = mapped_column(Integer, nullable=False)
    strong_count: Mapped[int] = mapped_column(Integer, nullable=False)
    partial_count: Mapped[int] = mapped_column(Integer, nullable=False)
    unsupported_count: Mapped[int] = mapped_column(Integer, nullable=False)
    high_priority_gaps: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    results: Mapped[list["MatchResult"]] = relationship(
        back_populates="analysis",
        cascade="all, delete-orphan",
    )


class MatchResult(Base):
    __tablename__ = "match_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    analysis_id: Mapped[int] = mapped_column(
        ForeignKey("analyses.id"),
        nullable=False,
        index=True,
    )

    requirement: Mapped[str] = mapped_column(Text, nullable=False)
    requirement_type: Mapped[str] = mapped_column(String(100), nullable=False)
    importance: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)

    evidence: Mapped[str | None] = mapped_column(Text, nullable=True)
    gap_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    gap: Mapped[str | None] = mapped_column(Text, nullable=True)
    priority: Mapped[str] = mapped_column(String(50), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    analysis: Mapped["Analysis"] = relationship(
        back_populates="results",
    )