from __future__ import annotations

import re
import sys
from pathlib import Path

from sqlalchemy.orm import Session

from api.models import Analysis, MatchResult
from api.services.scoring import calculate_match_score
from api.services.structured_matcher import (
    compare_experience,
    compare_education,
    required_years,
)
from api.services.evidence_validator import validate_evidence
from api.services.requirements import extract_requirements
from api.services.parser import extract_resume_text
from api.services.semantic_retriever import (
    get_model,
    retrieve_semantic_evidence,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_match import SkillAlignClassifier

from api.services.resume_chunks import build_contextual_chunks


_classifier = None


def get_classifier() -> SkillAlignClassifier:
    global _classifier

    if _classifier is None:
        _classifier = SkillAlignClassifier()

    return _classifier


def analyze_resume_and_jd(
    resume_path: str | Path,
    job_description: str,
    top_k: int = 5,
    db: Session | None = None,
) -> dict:

    # ---------------------------------------------------------
    # 1. Extract resume
    # ---------------------------------------------------------

    resume_text = extract_resume_text(resume_path)

    if not resume_text.strip():
        raise ValueError(
            "No extractable text was found in the resume PDF."
        )

    # ---------------------------------------------------------
    # 2. Extract requirements
    # ---------------------------------------------------------

    requirements = extract_requirements(job_description)

    if not requirements:
        raise ValueError(
            "No job requirements could be extracted."
        )

    # ---------------------------------------------------------
    # 3. Chunk resume
    # ---------------------------------------------------------

    evidence_chunks = build_contextual_chunks(resume_text)

    if not evidence_chunks:
        raise ValueError(
            "No usable resume evidence could be extracted."
        )

    # ---------------------------------------------------------
    # Load semantic retrieval model before initializing
    # the CUDA classifier.
    # ---------------------------------------------------------

    get_model()

    classifier = get_classifier()

    results = []

    # ---------------------------------------------------------
    # 4. Semantic evidence retrieval + classification
    # ---------------------------------------------------------

    for requirement in requirements:

        retrieved = retrieve_semantic_evidence(
            requirement=requirement["requirement"],
            evidence_chunks=evidence_chunks,
            top_k=top_k,
        )

        structured_result = None

        # -----------------------------------------------------
        # Structured education matching
        # -----------------------------------------------------

        if requirement["requirement_type"] == "education":

            education_candidates = retrieve_semantic_evidence(
                requirement["requirement"],
                evidence_chunks,
                top_k=10,
            )

            matched = None

            for item in education_candidates:

                chunk = item["chunk"]

                check = compare_education(
                    requirement["requirement"],
                    chunk,
                )

                if check["supported"]:

                    matched = {
                        "check": check,
                        "evidence": chunk,
                        "score": item["score"],
                    }

                    break

            if matched:

                result = {
                    "requirement": requirement["requirement"],
                    "requirement_type": requirement["requirement_type"],
                    "importance": requirement["importance"],
                    "status": "Strong",
                    "evidence": matched["evidence"],
                    "gap_type": None,
                    "gap": None,
                    "priority": "none",
                    "confidence": 1.0,
                    "class_probabilities": {
                        "Unsupported": 0.0,
                        "Partial": 0.0,
                        "Strong": 1.0,
                    },
                    "retrieved_evidence": [
                        {
                            "text": matched["evidence"],
                            "score": round(
                                matched["score"],
                                4,
                            ),
                        }
                    ],
                }

            else:

                result = {
                    "requirement": requirement["requirement"],
                    "requirement_type": requirement["requirement_type"],
                    "importance": requirement["importance"],
                    "status": "Unsupported",
                    "evidence": None,
                    "gap_type": "education_gap",
                    "gap": (
                        "No supporting evidence found for "
                        "the required educational background"
                    ),
                    "priority": (
                        "high"
                        if requirement["importance"] == "required"
                        else "medium"
                    ),
                    "confidence": 1.0,
                    "class_probabilities": {
                        "Unsupported": 1.0,
                        "Partial": 0.0,
                        "Strong": 0.0,
                    },
                    "retrieved_evidence": [
                        {
                            "text": item["chunk"],
                            "score": round(
                                item["score"],
                                4,
                            ),
                        }
                        for item in education_candidates
                    ],
                }

            results.append(result)
            continue

        # -----------------------------------------------------
        # Structured experience matching
        # -----------------------------------------------------

        if (
            requirement["requirement_type"] == "experience"
            and required_years(requirement["requirement"]) is not None
        ):

            # First use semantic retrieval to find likely relevant
            # experience evidence.
            experience_candidates = retrieve_semantic_evidence(
                requirement["requirement"],
                evidence_chunks,
                top_k=20,
            )

            for item in experience_candidates:

                chunk = item["chunk"]

                check = compare_experience(
                    requirement["requirement"],
                    chunk,
                )

                if check["supported"]:

                    structured_result = {
                        "check": check,
                        "evidence": chunk,
                        "retrieval_score": item["score"],
                    }

                    if check["status_hint"] == "Strong":
                        break

        # -----------------------------------------------------
        # Classify retrieved evidence candidates.
        #
        # We don't blindly use only rank #1.
        # ModernBERT sees the top candidates and we keep the
        # strongest supported interpretation.
        # -----------------------------------------------------

        candidate_results = []

        for item in retrieved:

            candidate_evidence = item["chunk"]

            if not validate_evidence(
                requirement=requirement["requirement"],
                requirement_type=requirement["requirement_type"],
                evidence=candidate_evidence,
            ):
                continue

            prediction = classifier.predict(
                requirement=requirement["requirement"],
                evidence=candidate_evidence,
                requirement_type=requirement["requirement_type"],
                importance=requirement["importance"],
            )

            prediction["_retrieval_score"] = item["score"]
            prediction["_retrieved_text"] = candidate_evidence

            candidate_results.append(prediction)

        # -----------------------------------------------------
        # Structured experience result
        # -----------------------------------------------------

        if (
            requirement["requirement_type"] == "experience"
            and structured_result is not None
        ):

            check = structured_result["check"]
            evidence = structured_result["evidence"]

            result = classifier.predict(
                requirement=requirement["requirement"],
                evidence=evidence,
                requirement_type=requirement["requirement_type"],
                importance=requirement["importance"],
            )

            # Structured numeric reasoning overrides the classifier
            # when an explicit experience threshold exists.
            result["status"] = check["status_hint"]

            if check["status_hint"] == "Strong":

                result["gap_type"] = None
                result["gap"] = None
                result["priority"] = "none"

            elif check["status_hint"] == "Partial":

                result["gap_type"] = "experience_gap"

                required = check["required_years"]
                candidate = check["candidate_years"]

                if candidate is not None:

                    result["gap"] = (
                        f"{candidate} year(s) of relevant experience found; "
                        f"{required}+ year(s) required"
                    )

                else:

                    result["gap"] = (
                        "Relevant experience found, but the "
                        f"{required}+ year requirement could not be fully verified"
                    )

                result["priority"] = "high"

            result["evidence"] = evidence

            result["retrieved_evidence"] = [
                {
                    "text": evidence,
                    "score": 1.0,
                }
            ]

            results.append(result)
            continue

        # -----------------------------------------------------
        # No valid evidence candidates
        # -----------------------------------------------------

        if not candidate_results:

            result = classifier.predict(
                requirement=requirement["requirement"],
                evidence="",
                requirement_type=requirement["requirement_type"],
                importance=requirement["importance"],
            )

            selected_evidence = None

        else:

            # -------------------------------------------------
            # IMPORTANT FIX:
            #
            # Prefer retrieval relevance first.
            #
            # Previously we prioritized:
            #   Strong status
            #   classifier confidence
            #   retrieval score
            #
            # This could select an unrelated resume chunk that
            # ModernBERT happened to classify with high confidence.
            #
            # Now retrieval relevance is the primary signal.
            # -------------------------------------------------

            status_rank = {
                "Strong": 2,
                "Partial": 1,
                "Unsupported": 0,
            }

            selected = max(
                candidate_results,
                key=lambda x: (
                    x["_retrieval_score"],
                    status_rank.get(
                        x["status"],
                        0,
                    ),
                    x["confidence"],
                ),
            )

            result = {
                key: value
                for key, value in selected.items()
                if not key.startswith("_")
            }

            selected_evidence = selected["_retrieved_text"]

        # -----------------------------------------------------
        # Preserve all semantic retrieval candidates.
        # -----------------------------------------------------

        result["retrieved_evidence"] = [
            {
                "text": item["chunk"],
                "score": round(
                    item["score"],
                    4,
                ),
            }
            for item in retrieved
        ]

        # -----------------------------------------------------
        # If the selected result is Unsupported, don't expose
        # irrelevant text as "evidence".
        # -----------------------------------------------------

        if result["status"] == "Unsupported":
            result["evidence"] = None

        results.append(result)

    # ---------------------------------------------------------
    # 5. Summary
    # ---------------------------------------------------------

    strong = sum(
        r["status"] == "Strong"
        for r in results
    )

    partial = sum(
        r["status"] == "Partial"
        for r in results
    )

    unsupported = sum(
        r["status"] == "Unsupported"
        for r in results
    )

    high_priority_gaps = sum(
        r["priority"] == "high"
        for r in results
    )

    score_data = calculate_match_score(results)

    response = {
        "requirements_analyzed": len(results),
        "match_score": score_data["score"],
        "summary": {
            "strong": strong,
            "partial": partial,
            "unsupported": unsupported,
            "high_priority_gaps": high_priority_gaps,
        },
        "results": results,
    }

    # ---------------------------------------------------------
    # 6. Persist analysis
    # ---------------------------------------------------------

    if db is not None:

        analysis = Analysis(
            job_title=None,
            job_description=job_description,
            match_score=score_data["score"],
            requirements_analyzed=len(results),
            strong_count=strong,
            partial_count=partial,
            unsupported_count=unsupported,
            high_priority_gaps=high_priority_gaps,
        )

        db.add(analysis)
        db.flush()

        for result in results:

            db_result = MatchResult(
                analysis_id=analysis.id,
                requirement=result["requirement"],
                requirement_type=result["requirement_type"],
                importance=result["importance"],
                status=result["status"],
                evidence=result.get("evidence"),
                gap_type=result.get("gap_type"),
                gap=result.get("gap"),
                priority=result["priority"],
                confidence=result["confidence"],
            )

            db.add(db_result)

        db.commit()

    return response