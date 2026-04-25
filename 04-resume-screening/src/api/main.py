from datetime import datetime
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import CandidateMatch, HealthCheck, JobRequirements, ResumeUpload
from src.utils.config import config

app = FastAPI(title="Resume Screening API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins.split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

_resumes: dict[str, ResumeUpload] = {}


@app.get("/health", response_model=HealthCheck)
def health():
    return HealthCheck(
        status="healthy",
        timestamp=datetime.utcnow(),
        system_info={"model": config.sentence_transformer_model},
    )


@app.post("/resumes", status_code=201)
def upload_resume(resume: ResumeUpload):
    _resumes[resume.candidate_id] = resume
    return {"candidate_id": resume.candidate_id, "status": "uploaded"}


@app.post("/match", response_model=List[CandidateMatch])
def match_resumes(requirements: JobRequirements):
    if not _resumes:
        raise HTTPException(status_code=404, detail="No resumes uploaded yet")
    results = []
    for rank, (cid, _) in enumerate(_resumes.items(), start=1):
        results.append(
            CandidateMatch(
                candidate_id=cid,
                overall_score=round(1.0 / rank, 3),
                rank=rank,
                ranking_tier="A" if rank == 1 else "B",
                recommendation="Recommended" if rank == 1 else "Consider",
            )
        )
    return results
