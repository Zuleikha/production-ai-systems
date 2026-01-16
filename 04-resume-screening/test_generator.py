from src.data_generation.resume_generator import ResumeGenerator
import json

# Create generator
g = ResumeGenerator()

print("=== SAMPLE JOB DESCRIPTION ===")
job = g.generate_job()
print(json.dumps(job, indent=2))

print("\n=== SAMPLE RESUME ===")
resume = g.generate_resume()
print(json.dumps(resume, indent=2))