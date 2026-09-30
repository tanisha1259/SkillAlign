from fastapi import FastAPI

from api.routes.analysis import router as analysis_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="SkillAlign API",
    version="0.1.0",
    description="Semantic resume-job matching and skill-gap analysis API.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "skillalign",
    }


app.include_router(analysis_router)
