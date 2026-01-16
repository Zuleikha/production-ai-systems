"""Data validation schemas for resume processing pipeline."""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

class SkillLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class ContactInfo(BaseModel):
    email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
    phone: Optional[str] = Field(None, regex=r'^\+?[\d\s\-\(\)]{10,15}$')
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
    start_date: str = Field(..., regex=r'^\d{4}-\d{2}-\d{2}$')
    end_date: Optional[str] = Field(None, regex=r'^\d{4}-\d{2}-\d{2}$|^present$')
    description: str = Field(..., min_length=20, max_length=2000)
    skills_used: Optional[List[str]] = []
    location: Optional[str] = None

class ResumeSchema(BaseModel):
    """Complete resume validation schema."""
    id: Optional[str] = None
    name: str = Field(..., min_length=1, max_length=100)
    contact: ContactInfo
    skills: List[Union[str, Skill]] = Field(..., min_items=1)
    education: List[Education] = Field(..., min_items=1)
    experience: List[Experience] = Field(default_factory=list)
    summary: Optional[str] = Field(None, min_length=50, max_length=1000)
    certifications: Optional[List[str]] = []
    languages: Optional[List[str]] = []
    
    @validator('skills')
    def validate_skills_count(cls, v):
        if len(v) > 50:
            raise ValueError('Too many skills listed (max 50)')
        return v
    
    @validator('experience')
    def validate_experience_dates(cls, v):
        for exp in v:
            if exp.end_date and exp.end_date != 'present':
                if exp.start_date > exp.end_date:
                    raise ValueError(f'Start date {exp.start_date} is after end date {exp.end_date}')
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
    quality_score: float
    
    @validator('quality_score')
    def calculate_quality_score(cls, v, values):
        if 'total_records' in values and values['total_records'] > 0:
            return (values.get('valid_records', 0) / values['total_records']) * 100
        return 0.0