"""Data validation schemas for the resume processing pipeline (Pydantic v2)."""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum


class SkillLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ContactInfo(BaseModel):
    email: str = Field(..., pattern=r"^[^@]+@[^@]+\.[^@]+$")
    phone: Optional[str] = Field(None, pattern=r"^\+?[\d\s\-\(\)]{10,15}$")
    linkedin: Optional[str] = None
    github: Optional[str] = None
    portfolio: Optional[str] = None


class Skill(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    level: Optional[SkillLevel] = None
    years_experience: Optional[int] = Field(None, ge=0, le=50)


class Education(BaseModel):
    degree: str = Field(..., min_length=1)
    institution: str = Field(..., min_length=1)
    graduation_year: Optional[int] = Field(None, ge=1950, le=2030)
    gpa: Optional[float] = Field(None, ge=0.0, le=4.0)
    major: Optional[str] = None
    minor: Optional[str] = None


class Experience(BaseModel):
    title: str = Field(..., min_length=1)
    company: str = Field(..., min_length=1)
    start_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    end_date: Optional[str] = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$|^present$")
    description: str = Field(..., min_length=20, max_length=2000)
    skills_used: Optional[List[str]] = []
    location: Optional[str] = None


class ResumeSchema(BaseModel):
    """Complete resume validation schema."""

    id: Optional[str] = None
    name: str = Field(..., min_length=1, max_length=100)
    contact: ContactInfo
    skills: List[Union[str, Skill]] = Field(..., min_length=1)
    education: List[Education] = Field(..., min_length=1)
    experience: List[Experience] = Field(default_factory=list)
    summary: Optional[str] = Field(None, min_length=50, max_length=1000)
    certifications: Optional[List[str]] = []
    languages: Optional[List[str]] = []

    @field_validator("skills")
    @classmethod
    def validate_skills_count(cls, v: list) -> list:
        if len(v) > 50:
            raise ValueError("Too many skills listed (max 50)")
        return v

    @field_validator("experience")
    @classmethod
    def validate_experience_dates(cls, v: list) -> list:
        for exp in v:
            if exp.end_date and exp.end_date != "present":
                if exp.start_date > exp.end_date:
                    raise ValueError(
                        f"start_date {exp.start_date} is after end_date {exp.end_date}"
                    )
        return v


class DataQualityMetrics(BaseModel):
    """Schema for data quality reporting."""

    total_records: int
    valid_records: int
    invalid_records: int
    completeness_scores: Dict[str, float]
    duplicate_count: int
    duplicate_percentage: float
    validation_errors: List[Dict[str, Any]]
    quality_score: float = 0.0

    @model_validator(mode="after")
    def calculate_quality_score(self) -> "DataQualityMetrics":
        if self.total_records > 0:
            self.quality_score = (self.valid_records / self.total_records) * 100
        return self
