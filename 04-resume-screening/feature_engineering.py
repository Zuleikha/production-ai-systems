import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import os

class FeatureEngineer:
    """Simple feature engineering for resume-job matching"""
    
    def __init__(self):
        """Initialize with basic components"""
        self.tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        self.all_skills = set()
        
    def calculate_skill_match_score(self, resume_skills, job_skills):
        """Calculate what % of job skills the resume has"""
        if not job_skills:
            return 0.0
        
        resume_set = set(skill.lower() for skill in resume_skills)
        job_set = set(skill.lower() for skill in job_skills)
        
        # How many job skills does resume have?
        matches = len(resume_set.intersection(job_set))
        return matches / len(job_set)
    
    def calculate_experience_match(self, resume_exp, job_min_exp, job_max_exp):
        """Score how well experience matches job requirements"""
        if resume_exp < job_min_exp:
            # Under-qualified: score based on how close
            return max(0, resume_exp / job_min_exp)
        elif resume_exp <= job_max_exp:
            # Perfect fit: full score
            return 1.0
        else:
            # Over-qualified: slight penalty
            excess = resume_exp - job_max_exp
            return max(0.7, 1.0 - (excess * 0.1))
    
    def create_skill_vectors(self, resumes_df, jobs_df):
        """Create binary skill vectors for all candidates and jobs"""
        print("Creating skill vectors...")
        
        # Get all unique skills
        all_resume_skills = []
        all_job_skills = []
        
        for skills in resumes_df['skills']:
            if isinstance(skills, str):
                skills = eval(skills)  # Convert string representation back to list
            all_resume_skills.extend([s.lower() for s in skills])
            
        for skills in jobs_df['required_skills']:
            if isinstance(skills, str):
                skills = eval(skills)
            all_job_skills.extend([s.lower() for s in skills])
        
        self.all_skills = sorted(set(all_resume_skills + all_job_skills))
        print(f"Found {len(self.all_skills)} unique skills: {self.all_skills[:5]}...")
        
        # Create binary vectors
        resume_vectors = []
        job_vectors = []
        
        for _, resume in resumes_df.iterrows():
            skills = eval(resume['skills']) if isinstance(resume['skills'], str) else resume['skills']
            resume_skills_lower = [s.lower() for s in skills]
            vector = [1 if skill in resume_skills_lower else 0 for skill in self.all_skills]
            resume_vectors.append(vector)
            
        for _, job in jobs_df.iterrows():
            skills = eval(job['required_skills']) if isinstance(job['required_skills'], str) else job['required_skills']
            job_skills_lower = [s.lower() for s in skills]
            vector = [1 if skill in job_skills_lower else 0 for skill in self.all_skills]
            job_vectors.append(vector)
            
        return np.array(resume_vectors), np.array(job_vectors)
    
    def create_matching_features(self, resumes_df, jobs_df):
        """Create all features for resume-job matching"""
        print("Creating matching features...")
        
        # Get skill vectors
        resume_vectors, job_vectors = self.create_skill_vectors(resumes_df, jobs_df)
        
        # Create matching matrix (each resume vs each job)
        matches = []
        
        for i, (_, resume) in enumerate(resumes_df.iterrows()):
            for j, (_, job) in enumerate(jobs_df.iterrows()):
                
                # Extract skills (handle string representation)
                resume_skills = eval(resume['skills']) if isinstance(resume['skills'], str) else resume['skills']
                job_skills = eval(job['required_skills']) if isinstance(job['required_skills'], str) else job['required_skills']
                
                # Calculate features
                skill_match = self.calculate_skill_match_score(resume_skills, job_skills)
                exp_match = self.calculate_experience_match(
                    resume['experience_years'], 
                    job['min_experience'], 
                    job['max_experience']
                )
                
                # Cosine similarity of skill vectors
                skill_similarity = cosine_similarity([resume_vectors[i]], [job_vectors[j]])[0][0]
                
                # Combined score (simple weighted average)
                overall_score = (skill_match * 0.5 + exp_match * 0.3 + skill_similarity * 0.2)
                
                match = {
                    'resume_id': i,
                    'job_id': j,
                    'resume_name': resume['name'],
                    'job_title': job['title'],
                    'skill_match_score': skill_match,
                    'experience_match_score': exp_match,
                    'skill_similarity': skill_similarity,
                    'overall_match_score': overall_score,
                    'resume_exp': resume['experience_years'],
                    'job_min_exp': job['min_experience'],
                    'job_max_exp': job['max_experience']
                }
                matches.append(match)
        
        return pd.DataFrame(matches)
    
    def create_visualizations(self, features_df):
        """Create plots to understand the features"""
        print("Creating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Distribution of match scores
        axes[0,0].hist(features_df['overall_match_score'], bins=20, alpha=0.7, color='skyblue')
        axes[0,0].set_title('Distribution of Overall Match Scores')
        axes[0,0].set_xlabel('Match Score')
        axes[0,0].set_ylabel('Count')
        
        # 2. Skill vs Experience match correlation
        axes[0,1].scatter(features_df['skill_match_score'], features_df['experience_match_score'], 
                         alpha=0.6, color='orange')
        axes[0,1].set_title('Skill Match vs Experience Match')
        axes[0,1].set_xlabel('Skill Match Score')
        axes[0,1].set_ylabel('Experience Match Score')
        
        # 3. Top matches heatmap (sample)
        top_matches = features_df.nlargest(20, 'overall_match_score')
        pivot_data = top_matches.pivot_table(
            values='overall_match_score', 
            index='resume_name', 
            columns='job_title', 
            fill_value=0
        )
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[1,0])
        axes[1,0].set_title('Top 20 Matches Heatmap')
        
        # 4. Feature correlation
        corr_features = features_df[['skill_match_score', 'experience_match_score', 
                                   'skill_similarity', 'overall_match_score']].corr()
        sns.heatmap(corr_features, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
        axes[1,1].set_title('Feature Correlations')
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs('data/plots', exist_ok=True)
        plt.savefig('data/plots/feature_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig

def main():
    """Main feature engineering pipeline"""
    print("Starting Feature Engineering (Phase 4)")
    print("=" * 50)
    
    # Load processed data
    print("Loading processed data...")
    resumes_df = pd.read_csv('data/processed/resumes_processed.csv')
    jobs_df = pd.read_csv('data/processed/jobs_processed.csv')
    
    print(f"Loaded {len(resumes_df)} resumes and {len(jobs_df)} jobs")
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Create matching features
    features_df = engineer.create_matching_features(resumes_df, jobs_df)
    
    # Create visualizations
    engineer.create_visualizations(features_df)
    
    # Save features
    print("Saving features...")
    os.makedirs('data/features', exist_ok=True)
    features_df.to_csv('data/features/matching_features.csv', index=False)
    
    # Show top matches
    print("\nTOP 10 MATCHES:")
    top_matches = features_df.nlargest(10, 'overall_match_score')
    for _, match in top_matches.iterrows():
        print(f"{match['resume_name']} -> {match['job_title']}: {match['overall_match_score']:.3f}")
    
    # Show feature statistics
    print(f"\nFEATURE SUMMARY:")
    print(f"Total resume-job combinations: {len(features_df)}")
    print(f"Average match score: {features_df['overall_match_score'].mean():.3f}")
    print(f"Best match score: {features_df['overall_match_score'].max():.3f}")
    print(f"Matches above 0.5: {len(features_df[features_df['overall_match_score'] > 0.5])}")
    
    print("\nPhase 4 Complete! Features saved to data/features/")
    print("Visualizations saved to data/plots/")
    print("Ready for Phase 5: Model Development")

if __name__ == "__main__":
    main()