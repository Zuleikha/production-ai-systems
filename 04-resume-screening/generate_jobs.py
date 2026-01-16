from src.data_generation.resume_generator import ResumeGenerator
import json
import os

# Create generator
g = ResumeGenerator()

# Generate 25 job descriptions
jobs = []
for i in range(25):
    job = g.generate_job()
    jobs.append(job)

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Save to data/jobs.json
with open('data/jobs.json', 'w') as f:
    json.dump(jobs, f, indent=2)

print(f"Generated {len(jobs)} job descriptions saved to data/jobs.json")

# Show first 2 as sample
print("\n=== SAMPLE JOBS ===")
for i, job in enumerate(jobs[:2]):
    print(f"\nJob {i+1}:")
    print(json.dumps(job, indent=2))