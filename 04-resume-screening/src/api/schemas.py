from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class JobRequirements(BaseModel):
    skills: Dict[str, List[str]]
    min_years_experience: int = 0
    education_level: Optional[str] = None

class ResumeUpload(BaseModel):
    candidate_id: str
    resume_text: str
    metadata: Optional[Dict[str, Any]] = None

class CandidateMatch(BaseModel):
    candidate_id: str
    overall_score: float
    rank: int
    ranking_tier: str
    recommendation: str

class HealthCheck(BaseModel):
    status: str
    timestamp: datetime
    system_info: Dict[str, Any]
