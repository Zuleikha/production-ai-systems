# Deployment Guide

## Local Development

1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run API: `uvicorn src.api.main:app --reload`
4. Access docs: http://localhost:8000/docs

## Docker Deployment

```bash
docker-compose up -d
```

## Cloud Deployment

### AWS
- Use ECS with ECR for container registry
- RDS for PostgreSQL database
- ElastiCache for Redis

### GCP  
- Use Cloud Run for serverless deployment
- Cloud SQL for database
- Memorystore for Redis

See terraform configurations in `deployment/terraform/`
