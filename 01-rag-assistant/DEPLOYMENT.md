# RAG Assistant - Deployment Guide

## Overview
This guide covers local Docker deployment and AWS cloud deployment for the RAG Assistant API.

## Prerequisites

### Required Software
- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- Docker Compose v2.0+
- Git
- AWS CLI (for AWS deployment)

### Required Accounts & Keys
- OpenAI API key (from https://platform.openai.com/api-keys)
- AWS account (for cloud deployment)

## Local Docker Deployment

### 1. Environment Setup

Create `.env` file in the project root directory:
```bash
OPENAI_API_KEY=sk-proj-your-key-here
```

**Important:** Never commit `.env` files to Git. Ensure `.env` is in `.gitignore`.

### 2. Build and Run

From the repository root directory:

```bash
# Build the Docker image
docker-compose build rag-assistant

# Start the container
docker-compose up rag-assistant

# Or run in background (detached mode)
docker-compose up rag-assistant -d
```

### 3. Verify Deployment

The API will be available at:
- API Base URL: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

Test the health endpoint:
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "RAG Assistant API",
  "version": "1.0.0",
  "openai_configured": true,
  "rag_system_initialized": true
}
```

### 4. Common Docker Commands

```bash
# View logs
docker-compose logs -f rag-assistant

# Stop container
docker-compose down

# Rebuild without cache
docker-compose build --no-cache rag-assistant

# Restart container
docker-compose restart rag-assistant

# View running containers
docker ps
```

## AWS Deployment

### Architecture
- **Container Registry:** Amazon ECR (Elastic Container Registry)
- **Compute:** AWS App Runner or ECS Fargate
- **Recommended:** App Runner for simplicity and auto-scaling

### Step 1: Push Image to AWS ECR

#### 1.1 Configure AWS CLI
```bash
aws configure
# Enter your AWS Access Key ID
# Enter your AWS Secret Access Key
# Enter your default region (e.g., us-east-1)
```

#### 1.2 Create ECR Repository
```bash
aws ecr create-repository \
    --repository-name rag-assistant \
    --region us-east-1
```

Note the `repositoryUri` from the output (e.g., `123456789.dkr.ecr.us-east-1.amazonaws.com/rag-assistant`)

#### 1.3 Authenticate Docker to ECR
```bash
aws ecr get-login-password --region us-east-1 | \
docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com
```

#### 1.4 Tag and Push Image
```bash
# Build the image
docker-compose build rag-assistant

# Tag for ECR
docker tag ai-engineering-portfolio-rag-assistant:latest \
    123456789.dkr.ecr.us-east-1.amazonaws.com/rag-assistant:latest

# Push to ECR
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/rag-assistant:latest
```

### Step 2: Deploy with AWS App Runner

#### 2.1 Create App Runner Service (via AWS Console)

1. Go to AWS App Runner in the console
2. Click "Create service"
3. Select "Container registry" as source
4. Choose "Amazon ECR" and browse to your image
5. Configure:
   - **Service name:** rag-assistant-api
   - **Port:** 8000
   - **Environment variables:**
     - Key: `OPENAI_API_KEY`
     - Value: Your OpenAI API key
   - **CPU & Memory:** 1 vCPU, 2 GB (adjust based on needs)
   - **Auto scaling:** Min 1, Max 3 instances
6. Click "Create & deploy"

#### 2.2 Create App Runner Service (via CLI)

First, create `apprunner.yaml`:
```yaml
version: 1.0
runtime: python3.11
build:
  commands:
    build:
      - echo "Using pre-built container"
run:
  runtime-version: 3.11
  command: python src/main.py
  network:
    port: 8000
  env:
    - name: OPENAI_API_KEY
      value: "your-key-here"
```

Then deploy:
```bash
aws apprunner create-service \
    --service-name rag-assistant-api \
    --source-configuration '{
        "ImageRepository": {
            "ImageIdentifier": "123456789.dkr.ecr.us-east-1.amazonaws.com/rag-assistant:latest",
            "ImageRepositoryType": "ECR",
            "ImageConfiguration": {
                "Port": "8000",
                "RuntimeEnvironmentVariables": {
                    "OPENAI_API_KEY": "your-key-here"
                }
            }
        },
        "AutoDeploymentsEnabled": true
    }' \
    --instance-configuration '{
        "Cpu": "1024",
        "Memory": "2048"
    }' \
    --region us-east-1
```

### Step 3: Alternative - Deploy with ECS Fargate

#### 3.1 Create ECS Cluster
```bash
aws ecs create-cluster \
    --cluster-name rag-assistant-cluster \
    --region us-east-1
```

#### 3.2 Create Task Definition

Create `ecs-task-definition.json`:
```json
{
  "family": "rag-assistant-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "rag-assistant",
      "image": "123456789.dkr.ecr.us-east-1.amazonaws.com/rag-assistant:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "OPENAI_API_KEY",
          "value": "your-key-here"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/rag-assistant",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

Register the task:
```bash
aws ecs register-task-definition \
    --cli-input-json file://ecs-task-definition.json
```

#### 3.3 Create ECS Service

```bash
aws ecs create-service \
    --cluster rag-assistant-cluster \
    --service-name rag-assistant-service \
    --task-definition rag-assistant-task \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration '{
        "awsvpcConfiguration": {
            "subnets": ["subnet-12345"],
            "securityGroups": ["sg-12345"],
            "assignPublicIp": "ENABLED"
        }
    }' \
    --region us-east-1
```

## Cost Estimates

### AWS App Runner
- **Development/Testing:** ~$5-15/month
- **Production (low traffic):** ~$20-50/month
- Pricing: $0.064/vCPU-hour + $0.007/GB-hour + data transfer

### ECS Fargate
- **Development/Testing:** ~$10-30/month
- **Production (low traffic):** ~$30-80/month
- Pricing depends on vCPU/memory configuration and running time

### Free Alternatives
- **Render.com:** Free tier available, Docker support
- **Railway.app:** $5 free credit/month
- **Fly.io:** Free tier with limitations

## Monitoring & Troubleshooting

### View Application Logs

**Docker:**
```bash
docker-compose logs -f rag-assistant
```

**AWS App Runner:**
```bash
aws logs tail /aws/apprunner/rag-assistant-api --follow
```

**ECS:**
```bash
aws logs tail /ecs/rag-assistant --follow
```

### Common Issues

#### Issue: ModuleNotFoundError for langchain
**Solution:** Ensure you're using the updated imports:
- `from langchain_text_splitters import RecursiveCharacterTextSplitter`
- `from langchain_core.documents import Document`

#### Issue: OpenAI API authentication error
**Solution:** 
- Verify `OPENAI_API_KEY` is set correctly in `.env`
- Check OpenAI account has available credits
- Confirm key hasn't been revoked

#### Issue: Container won't start
**Solution:**
```bash
# Rebuild without cache
docker-compose build --no-cache rag-assistant

# Check logs for specific errors
docker-compose logs rag-assistant
```

#### Issue: Port already in use
**Solution:**
```bash
# Find process using port 8000
lsof -i :8000  # Mac/Linux
netstat -ano | findstr :8000  # Windows

# Stop existing container
docker-compose down
```

## Security Best Practices

1. **Never commit API keys** - Always use `.env` files and `.gitignore`
2. **Use AWS Secrets Manager** for production API keys
3. **Enable HTTPS** for production deployments
4. **Restrict API access** using API keys or authentication
5. **Monitor API usage** to detect unusual patterns
6. **Regularly update dependencies** for security patches

## CI/CD Integration

### GitHub Actions Example

Create `.github/workflows/deploy.yml`:
```yaml
name: Deploy to AWS

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      
      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v1
      
      - name: Build and push Docker image
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          ECR_REPOSITORY: rag-assistant
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG ./01-rag-assistant
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
          docker tag $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG $ECR_REGISTRY/$ECR_REPOSITORY:latest
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest
```

## Performance Optimization

1. **Use production ASGI server:** Uvicorn with multiple workers
2. **Enable caching:** Cache embeddings and vector store
3. **Optimize Docker image:** Use multi-stage builds
4. **Monitor response times:** Set up CloudWatch alarms
5. **Scale horizontally:** Use auto-scaling based on CPU/memory

## Next Steps

1. Add authentication/authorization
2. Implement rate limiting
3. Add request/response logging
4. Set up monitoring and alerts
5. Configure custom domain with SSL
6. Implement backup strategy for vector store

## Support & Resources

- **Docker Documentation:** https://docs.docker.com
- **AWS App Runner:** https://docs.aws.amazon.com/apprunner
- **AWS ECS:** https://docs.aws.amazon.com/ecs
- **OpenAI API:** https://platform.openai.com/docs
- **FastAPI:** https://fastapi.tiangolo.com

---

**Last Updated:** February 2026