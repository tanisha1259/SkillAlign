from pathlib import Path
import tempfile

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from api.services.analyzer import analyze_resume_and_jd
from api.schemas import AnalysisResponse
from sqlalchemy.orm import Session

from api.db import get_db

router = APIRouter(
    prefix="/api",
    tags=["analysis"],
)


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
)
async def analyze(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
    db: Session = Depends(get_db),
):

    if not resume.filename:
        raise HTTPException(
            status_code=400,
            detail="Resume file is required.",
        )

    if not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Resume must be a PDF file.",
        )

    if len(job_description.strip()) < 20:
        raise HTTPException(
            status_code=400,
            detail="Job description is too short.",
        )

    suffix = Path(resume.filename).suffix

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:

            content = await resume.read()

            if not content:
                raise HTTPException(
                    status_code=400,
                    detail="Uploaded resume is empty.",
                )

            temp_file.write(content)
            temp_path = Path(temp_file.name)

        result = analyze_resume_and_jd(
            resume_path=temp_path,
            job_description=job_description,
            db=db,
        )

        return result

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {exc}",
        )

    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)
