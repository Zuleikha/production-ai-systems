import json
import os
import re
import pandas as pd
from typing import List, Dict
import string

class DataPreprocessor:
    def __init__(self):
        """Initialize preprocessor with text cleaning patterns"""
        self.stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text data"""
        if not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep letters, numbers, spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        return text
    
    def normalize_skills(self, skills: List[str]) -> List[str]:
        """Normalize skill names for consistency"""
        normalized = []
        for skill in skills:
            # Clean skill name
            clean_skill = self.clean_text(skill)
            
            # Standardize common variations
            skill_mappings = {
                'javascript': 'JavaScript',
                'js': 'JavaScript', 
                'python': 'Python',
                'java': 'Java',
                'react': 'React',
                'reactjs': 'React',
                'sql': 'SQL',
                'mysql': 'SQL',
                'postgresql': 'SQL',
                'aws': 'AWS',
                'amazon web services': 'AWS',
                'docker': 'Docker',
                'kubernetes': 'Kubernetes',
                'k8s': 'Kubernetes'
            }
            
            # Apply mapping or keep original (title case)
            normalized_skill = skill_mappings.get(clean_skill, skill.title())
            normalized.append(normalized_skill)
            
        # Remove duplicates while preserving order
        return list(dict.fromkeys(normalized))
    
    def process_resumes(self, resumes_data: List[Dict]) -> List[Dict]:
        """Process resume data for ML pipeline"""
        processed = []
        
        for resume in resumes_data:
            processed_resume = {
                'name': self.clean_text(resume['name']),
                'role_category': self.clean_text(resume['role_category']),
                'skills': self.normalize_skills(resume['skills']),
                'experience_years': int(resume['experience_years']),
                'skills_count': len(resume['skills']),
                'processed': True
            }
            processed.append(processed_resume)
            
        return processed
    
    def process_jobs(self, jobs_data: List[Dict]) -> List[Dict]:
        """Process job description data for ML pipeline"""
        processed = []
        
        for job in jobs_data:
            processed_job = {
                'title': self.clean_text(job['title']),
                'company': self.clean_text(job['company']),
                'required_skills': self.normalize_skills(job['required_skills']),
                'min_experience': int(job['min_experience']),
                'max_experience': int(job['max_experience']),
                'skills_count': len(job['required_skills']),
                'experience_range': job['max_experience'] - job['min_experience'],
                'processed': True
            }
            processed.append(processed_job)
            
        return processed

def main():
    """Main preprocessing pipeline"""
    print("Starting data preprocessing...")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load raw data
    print("Loading raw data...")
    with open('data/synthetic_resumes/resumes.json', 'r') as f:
        resumes_data = json.load(f)
    
    with open('data/jobs.json', 'r') as f:
        jobs_data = json.load(f)
    
    print(f"Loaded {len(resumes_data)} resumes and {len(jobs_data)} jobs")
    
    # Process data
    print("Processing resumes...")
    processed_resumes = preprocessor.process_resumes(resumes_data)
    
    print("Processing job descriptions...")
    processed_jobs = preprocessor.process_jobs(jobs_data)
    
    # Create output directory
    os.makedirs('data/processed', exist_ok=True)
    
    # Save processed data
    print("Saving processed data...")
    with open('data/processed/resumes_processed.json', 'w') as f:
        json.dump(processed_resumes, f, indent=2)
    
    with open('data/processed/jobs_processed.json', 'w') as f:
        json.dump(processed_jobs, f, indent=2)
    
    # Save as CSV for easier analysis
    pd.DataFrame(processed_resumes).to_csv('data/processed/resumes_processed.csv', index=False)
    pd.DataFrame(processed_jobs).to_csv('data/processed/jobs_processed.csv', index=False)
    
    print("✅ Preprocessing complete!")
    print(f"Processed files saved to data/processed/")
    print(f"- {len(processed_resumes)} processed resumes")
    print(f"- {len(processed_jobs)} processed jobs")
    
    # Show sample
    print("\n=== SAMPLE PROCESSED DATA ===")
    print("Sample processed resume:")
    print(json.dumps(processed_resumes[0], indent=2))
    
    print("\nSample processed job:")
    print(json.dumps(processed_jobs[0], indent=2))

if __name__ == "__main__":
    main()