import random
import json
from pathlib import Path
from datetime import datetime, timedelta
import uuid

class resumeGenerator:
    def __init__(self):
        """Enhanced resume generator with realistic, diverse data"""
        
        # Expanded diverse names from different backgrounds
        self.first_names = [
            # Western names
            "Alex", "Maria", "James", "Sarah", "Michael", "Jennifer", "David", "Lisa",
            "Robert", "Emily", "William", "Jessica", "John", "Ashley", "Daniel", "Amanda",
            # Hispanic/Latino names  
            "Carlos", "Isabella", "Roberto", "Sofia", "Diego", "Camila", "Luis", "Valentina",
            # Asian names
            "Wei", "Li", "Hiroshi", "Yuki", "Raj", "Priya", "Kenji", "Sakura", "Chen", "Ming",
            # Middle Eastern/Arabic names
            "Ahmed", "Fatima", "Hassan", "Aisha", "Omar", "Layla", "Khalid", "Noor",
            # African names
            "Kwame", "Amara", "Kofi", "Nia", "Sekou", "Zara", "Malik", "Kaia",
            # European names
            "Dmitri", "Anya", "Gustav", "Ingrid", "Marco", "Elena", "Pietro", "Sophia",
            # Additional diverse names
            "Arjun", "Maya", "Olumide", "Grace", "Tariq", "Zoe", "Ravi", "Chloe"
        ]
        
        self.last_names = [
            # Common Western surnames
            "Smith", "Johnson", "Williams", "Brown", "Davis", "Wilson", "Anderson", "Taylor",
            "Thomas", "Jackson", "White", "Thompson", "Robinson", "Lewis", "Walker", "Hall",
            # Hispanic surnames
            "Rodriguez", "Martinez", "Garcia", "Lopez", "Gonzalez", "Hernandez", "Perez", "Sanchez",
            # Asian surnames
            "Kim", "Lee", "Park", "Chen", "Zhang", "Wang", "Liu", "Yang", "Patel", "Singh",
            "Kumar", "Sharma", "Gupta", "Tanaka", "Suzuki", "Sato", "Yamamoto",
            # Middle Eastern/Arabic surnames
            "Hassan", "Ali", "Ahmad", "Mohamed", "Ibrahim", "Mahmoud", "Al-Rashid",
            # African surnames
            "Okafor", "Adebayo", "Mensah", "Diallo", "Kone", "Traore",
            # European surnames
            "Mueller", "Schmidt", "Johansson", "Nielsen", "Rossi", "Ferrari", "O'Connor", "Murphy"
        ]
        
        # Role-specific skills for more realistic matching
        self.skills_by_role = {
            "Software Engineer": {
                "primary": ["Python", "Java", "JavaScript", "C++", "Go", "Rust"],
                "secondary": ["React", "Node.js", "Angular", "Vue.js", "Spring Boot", "Django"],
                "tools": ["Git", "Docker", "Jenkins", "JIRA", "VS Code", "IntelliJ"]
            },
            "Data Scientist": {
                "primary": ["Python", "R", "SQL", "Scala", "Julia"],
                "secondary": ["Pandas", "NumPy", "Scikit-learn", "TensorFlow", "PyTorch", "Keras"],
                "tools": ["Jupyter", "Tableau", "Power BI", "Apache Spark", "Databricks", "MLflow"]
            },
            "DevOps Engineer": {
                "primary": ["Python", "Bash", "Go", "YAML", "Terraform"],
                "secondary": ["AWS", "Azure", "GCP", "Kubernetes", "Docker", "Ansible"],
                "tools": ["Jenkins", "GitLab CI", "Prometheus", "Grafana", "Helm", "Istio"]
            },
            "Frontend Developer": {
                "primary": ["JavaScript", "TypeScript", "HTML", "CSS"],
                "secondary": ["React", "Vue.js", "Angular", "Svelte", "Next.js", "Nuxt.js"],
                "tools": ["Webpack", "Vite", "Sass", "Tailwind CSS", "Figma", "Chrome DevTools"]
            },
            "Backend Developer": {
                "primary": ["Python", "Java", "Node.js", "C#", "PHP", "Go"],
                "secondary": ["Django", "Flask", "Spring Boot", "Express.js", "Laravel", "Gin"],
                "tools": ["PostgreSQL", "MongoDB", "Redis", "RabbitMQ", "Docker", "AWS"]
            },
            "Full Stack Developer": {
                "primary": ["JavaScript", "Python", "TypeScript", "Java"],
                "secondary": ["React", "Node.js", "Django", "PostgreSQL", "MongoDB", "Express.js"],
                "tools": ["Git", "Docker", "AWS", "Heroku", "Netlify", "Vercel"]
            }
        }
        
        # Expanded role categories
        self.roles = list(self.skills_by_role.keys())
        
        # More realistic company names across different industries
        self.companies = [
            # Tech Companies
            "TechCorp", "DataSoft", "CloudInc", "DevCompany", "InnovateLabs",
            "NextGen Systems", "AgileWorks", "ScaleUp Tech", "QuantumSoft",
            "ByteForge", "CodeCraft", "DataFlow Inc", "CloudNine Solutions",
            # Finance/Banking
            "FinanceFirst", "Capital Solutions", "InvestPro", "BankTech Corp",
            "CreditFlow", "MoneyWise Inc", "TradeTech", "SecureBank",
            # Healthcare/Biotech
            "MedTech Solutions", "HealthCare Plus", "BioInnovate", "MedFlow",
            "LifeSciences Corp", "CareConnect", "BioTech Labs", "HealthFirst",
            # E-commerce/Retail
            "ShopTech", "Commerce Cloud", "RetailFlow", "MarketPlace Inc",
            "EcomSolutions", "DigitalMart", "OrderFlow", "RetailTech",
            # Consulting/Services
            "ConsultPro", "Strategy Plus", "BusinessFlow", "ServiceTech",
            "SolutionsGroup", "Advisory Corp", "ClientFirst", "ExpertSystems",
            # Startups/Generic
            "StartupLab", "VentureFlow", "InnovateCorp", "DisruptTech",
            "FastGrow Inc", "LaunchPad", "ScaleTech", "BoostCorp"
        ]
        
        # Education levels and institutions
        self.education_levels = ["Bachelor's", "Master's", "PhD"]
        self.degree_fields = {
            "Software Engineer": ["Computer Science", "Software Engineering", "Information Technology"],
            "Data Scientist": ["Data Science", "Statistics", "Mathematics", "Computer Science"],
            "DevOps Engineer": ["Computer Science", "Information Systems", "Engineering"],
            "Frontend Developer": ["Computer Science", "Web Development", "Design"],
            "Backend Developer": ["Computer Science", "Software Engineering", "Information Technology"],
            "Full Stack Developer": ["Computer Science", "Software Engineering", "Web Development"]
        }
        
        self.universities = [
            "State University", "Tech Institute", "City College", "Metro University",
            "Valley Tech", "Riverside University", "Mountain State", "Coastal College"
        ]
        
        # Expanded cities for better geographic diversity
        self.cities = [
            # US Major Cities
            "San Francisco, CA", "New York, NY", "Seattle, WA", "Boston, MA",
            "Austin, TX", "Denver, CO", "Atlanta, GA", "Chicago, IL",
            "Los Angeles, CA", "Portland, OR", "Miami, FL", "Detroit, MI",
            "Phoenix, AZ", "Dallas, TX", "Philadelphia, PA", "San Diego, CA",
            "Nashville, TN", "Charlotte, NC", "Minneapolis, MN", "Tampa, FL",
            # International Cities
            "Toronto, Canada", "Vancouver, Canada", "London, UK", "Berlin, Germany",
            "Amsterdam, Netherlands", "Sydney, Australia", "Melbourne, Australia",
            "Tokyo, Japan", "Singapore", "Dublin, Ireland", "Zurich, Switzerland",
            "Stockholm, Sweden", "Copenhagen, Denmark", "Tel Aviv, Israel"
        ]
        
        # Job benefits
        self.benefits = [
            "Health Insurance", "401k Matching", "Remote Work", "Flexible Hours",
            "Professional Development", "Stock Options", "Paid Time Off",
            "Gym Membership", "Free Lunch", "Learning Budget"
        ]
    
    def generate_realistic_skills(self, role, experience_years):
        """Generate skills based on role and experience level"""
        role_skills = self.skills_by_role[role]
        
        # More experienced people have more skills
        if experience_years <= 2:
            primary_count = random.randint(1, 2)
            secondary_count = random.randint(1, 2)
            tools_count = random.randint(2, 3)
        elif experience_years <= 5:
            primary_count = random.randint(2, 3)
            secondary_count = random.randint(2, 4)
            tools_count = random.randint(3, 4)
        else:
            primary_count = random.randint(3, 4)
            secondary_count = random.randint(3, 5)
            tools_count = random.randint(4, 6)
        
        skills = []
        skills.extend(random.sample(role_skills["primary"], min(primary_count, len(role_skills["primary"]))))
        skills.extend(random.sample(role_skills["secondary"], min(secondary_count, len(role_skills["secondary"]))))
        skills.extend(random.sample(role_skills["tools"], min(tools_count, len(role_skills["tools"]))))
        
        return skills
    
    def generate_work_history(self, role, experience_years):
        """Generate realistic work history"""
        if experience_years < 1:
            return []
        
        history = []
        remaining_years = experience_years
        current_date = datetime.now()
        
        while remaining_years > 0 and len(history) < 4:  # Max 4 jobs
            # Job duration between 1-4 years, but not more than remaining
            duration = min(random.randint(1, 4), remaining_years)
            
            end_date = current_date
            start_date = current_date - timedelta(days=duration * 365)
            
            # Generate job title (sometimes promotions within same role category)
            if random.random() < 0.7:  # 70% same role category
                job_title = role
            else:  # 30% related role
                job_title = random.choice(self.roles)
            
            # Add seniority based on experience
            if experience_years - remaining_years + duration >= 5:
                seniority = random.choice(["Senior", "Lead", "Principal"])
                job_title = f"{seniority} {job_title}"
            elif experience_years - remaining_years + duration >= 2:
                if random.random() < 0.3:
                    job_title = f"Junior {job_title}"
            
            history.append({
                "title": job_title,
                "company": random.choice(self.companies),
                "start_date": start_date.strftime("%Y-%m"),
                "end_date": end_date.strftime("%Y-%m"),
                "duration_years": duration
            })
            
            remaining_years -= duration
            current_date = start_date - timedelta(days=30)  # Gap between jobs
        
        return history
    
    def generate_resume(self):
        """Generate a comprehensive, realistic resume"""
        
        # Basic info
        first_name = random.choice(self.first_names)
        last_name = random.choice(self.last_names)
        full_name = f"{first_name} {last_name}"
        
        role = random.choice(self.roles)
        experience_years = random.randint(0, 15)
        
        # Generate realistic email
        email_formats = [
            f"{first_name.lower()}.{last_name.lower()}@gmail.com",
            f"{first_name.lower()}{last_name.lower()[:3]}@yahoo.com",
            f"{first_name[:1].lower()}{last_name.lower()}@outlook.com"
        ]
        email = random.choice(email_formats)
        
        # Skills based on role and experience
        skills = self.generate_realistic_skills(role, experience_years)
        
        # Education
        education_level = random.choice(self.education_levels)
        degree_field = random.choice(self.degree_fields[role])
        university = random.choice(self.universities)
        
        # Work history
        work_history = self.generate_work_history(role, experience_years)
        
        # Salary expectation (rough estimates)
        base_salary = {
            "Software Engineer": 85000,
            "Data Scientist": 95000,
            "DevOps Engineer": 90000,
            "Frontend Developer": 80000,
            "Backend Developer": 85000,
            "Full Stack Developer": 88000
        }
        
        salary_multiplier = 1 + (experience_years * 0.08)  # 8% increase per year
        expected_salary = int(base_salary[role] * salary_multiplier * random.uniform(0.9, 1.1))
        
        return {
            "id": str(uuid.uuid4()),
            "personal_info": {
                "name": full_name,
                "email": email,
                "phone": f"({random.randint(200,999)}) {random.randint(200,999)}-{random.randint(1000,9999)}",
                "location": random.choice(self.cities)
            },
            "role_category": role,
            "experience_years": experience_years,
            "skills": skills,
            "education": {
                "level": education_level,
                "field": degree_field,
                "institution": university,
                "graduation_year": datetime.now().year - experience_years - random.randint(0, 4)
            },
            "work_history": work_history,
            "expected_salary": expected_salary,
            "availability": random.choice(["Immediate", "2 weeks", "1 month"]),
            "remote_preference": random.choice(["Remote", "Hybrid", "On-site", "No preference"])
        }
    
    def generate_job(self):
        """Generate a comprehensive, realistic job posting"""
        
        role = random.choice(self.roles)
        company = random.choice(self.companies)
        
        # Experience requirements
        min_exp = random.randint(0, 3)
        max_exp = min_exp + random.randint(2, 8)
        
        # Salary range based on role and experience
        base_salaries = {
            "Software Engineer": (75000, 180000),
            "Data Scientist": (85000, 200000),
            "DevOps Engineer": (80000, 190000),
            "Frontend Developer": (70000, 160000),
            "Backend Developer": (75000, 170000),
            "Full Stack Developer": (78000, 175000)
        }
        
        salary_min, salary_max = base_salaries[role]
        # Adjust for experience requirements
        exp_multiplier = 1 + (min_exp * 0.1)
        job_salary_min = int(salary_min * exp_multiplier)
        job_salary_max = int(salary_max * exp_multiplier * 0.8)  # Max is usually lower than top of range
        
        # Required vs preferred skills
        role_skills = self.skills_by_role[role]
        required_skills = random.sample(
            role_skills["primary"] + role_skills["secondary"][:3], 
            random.randint(3, 5)
        )
        preferred_skills = random.sample(
            role_skills["secondary"] + role_skills["tools"], 
            random.randint(2, 4)
        )
        
        # Job benefits
        num_benefits = random.randint(3, 7)
        job_benefits = random.sample(self.benefits, num_benefits)
        
        # Seniority level
        if min_exp >= 5:
            seniority = random.choice(["Senior", "Lead", "Principal"])
            title = f"{seniority} {role}"
        elif min_exp <= 1:
            title = f"Junior {role}" if random.random() < 0.4 else role
        else:
            title = role
        
        return {
            "id": str(uuid.uuid4()),
            "title": title,
            "company": company,
            "location": random.choice(self.cities),
            "role_category": role,
            "experience_requirements": {
                "min_years": min_exp,
                "max_years": max_exp
            },
            "salary_range": {
                "min": job_salary_min,
                "max": job_salary_max,
                "currency": "USD"
            },
            "required_skills": required_skills,
            "preferred_skills": preferred_skills,
            "employment_type": random.choice(["Full-time", "Contract", "Part-time"]),
            "remote_policy": random.choice(["Remote", "Hybrid", "On-site"]),
            "benefits": job_benefits,
            "posted_date": (datetime.now() - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d"),
            "application_deadline": (datetime.now() + timedelta(days=random.randint(7, 60))).strftime("%Y-%m-%d")
        }
    
    def generate_dataset(self, num_resumes=50):
        """Generate multiple realistic resumes"""
        return [self.generate_resume() for _ in range(num_resumes)]
    
    def generate_job_dataset(self, num_jobs=20):
        """Generate multiple realistic job postings"""
        return [self.generate_job() for _ in range(num_jobs)]
    
    def generate_job_description_text(self, job_data):
        """Convert job data to formatted text template"""
        return f"""
{job_data['title']} at {job_data['company']}
Location: {job_data['location']}

We are seeking a {job_data['role_category']} with {job_data['experience_requirements']['min_years']}-{job_data['experience_requirements']['max_years']} years experience.

Required Skills: {', '.join(job_data['required_skills'])}
Preferred Skills: {', '.join(job_data['preferred_skills'])}

Salary: ${job_data['salary_range']['min']:,} - ${job_data['salary_range']['max']:,}
Benefits: {', '.join(job_data['benefits'])}
        """

    def generate_resume_text(self, resume_data):
        """Convert resume data to formatted text"""
        work_history_text = '\n'.join([
            f"{job['title']} at {job['company']} ({job['start_date']} - {job['end_date']})" 
            for job in resume_data['work_history']
        ])
        
        return f"""
{resume_data['personal_info']['name']}
{resume_data['personal_info']['email']} | {resume_data['personal_info']['phone']}
{resume_data['personal_info']['location']}

PROFESSIONAL SUMMARY
{resume_data['role_category']} with {resume_data['experience_years']} years of experience

SKILLS
{', '.join(resume_data['skills'])}

EDUCATION  
{resume_data['education']['level']} in {resume_data['education']['field']}
{resume_data['education']['institution']} ({resume_data['education']['graduation_year']})

WORK EXPERIENCE
{work_history_text}
        """
    
    def save_dataset(self, resumes, filename="realistic_resumes.json"):
        """Save resumes with better organization"""
        save_path = Path("data/synthetic_resumes")
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        dataset = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_resumes": len(resumes),
                "generator_version": "2.0"
            },
            "resumes": resumes
        }
        
        file_path = save_path / filename
        with open(file_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"Saved {len(resumes)} realistic resumes to {file_path}")
        
        # Generate summary statistics
        self.print_dataset_summary(resumes)
    
    def save_jobs(self, jobs, filename="realistic_jobs.json"):
        """Save job postings with metadata"""
        save_path = Path("data/job_descriptions")
        save_path.mkdir(parents=True, exist_ok=True)
        
        dataset = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_jobs": len(jobs),
                "generator_version": "2.0"
            },
            "jobs": jobs
        }
        
        file_path = save_path / filename
        with open(file_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"Saved {len(jobs)} realistic job postings to {file_path}")
    
    def print_dataset_summary(self, resumes):
        """Print useful statistics about the generated dataset"""
        print("\n Dataset Summary:")
        print(f"Total resumes: {len(resumes)}")
        
        # Role distribution
        roles = {}
        experience_ranges = {"0-2": 0, "3-5": 0, "6-10": 0, "10+": 0}
        
        for resume in resumes:
            role = resume["role_category"]
            roles[role] = roles.get(role, 0) + 1
            
            exp = resume["experience_years"]
            if exp <= 2:
                experience_ranges["0-2"] += 1
            elif exp <= 5:
                experience_ranges["3-5"] += 1
            elif exp <= 10:
                experience_ranges["6-10"] += 1
            else:
                experience_ranges["10+"] += 1
        
        print("\nRole distribution:")
        for role, count in roles.items():
            print(f"  {role}: {count}")
        
        print("\nExperience distribution:")
        for range_name, count in experience_ranges.items():
            print(f"  {range_name} years: {count}")

