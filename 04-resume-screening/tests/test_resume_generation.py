import pytest
from src.data_generation.resume_generator import ResumeGenerator

def test_resume_generator_initialization():
    """Test resume generator initializes correctly."""
    generator = ResumeGenerator()
    assert generator.fake is not None
    assert len(generator.tech_skills) > 0

def test_generate_candidate_profile():
    """Test candidate profile generation."""
    generator = ResumeGenerator()
    profile = generator.generate_candidate_profile('data_science', 'mid')
    
    assert 'personal_info' in profile
    assert 'experience' in profile
    assert 'skills' in profile
    assert profile['personal_info']['name']
    assert profile['personal_info']['email']

def test_generate_resume_text():
    """Test resume text generation."""
    generator = ResumeGenerator()
    profile = generator.generate_candidate_profile('software_engineering', 'senior')
    resume_text = generator.generate_resume_text(profile)
    
    assert len(resume_text) > 100
    assert profile['personal_info']['name'] in resume_text
    assert 'EXPERIENCE' in resume_text.upper()
