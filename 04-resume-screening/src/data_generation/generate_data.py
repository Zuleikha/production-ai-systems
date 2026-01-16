# Example usage
if __name__ == "__main__":
    generator = ImprovedResumeGenerator()
    
    # Generate datasets
    resumes = generator.generate_dataset(100)  # More resumes for better testing
    jobs = generator.generate_job_dataset(30)
    
    # Save to files
    generator.save_dataset(resumes)
    generator.save_jobs(jobs)
    
    # Show a sample resume
    print("\n📋 Sample Resume:")
    print(json.dumps(resumes[0], indent=2))
    
    # Show sample text formats
    print("\n📄 Sample Resume Text:")
    print(generator.generate_resume_text(resumes[0]))
    
    print("\n📋 Sample Job Description Text:")
    print(generator.generate_job_description_text(jobs[0]))