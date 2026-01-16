# API Documentation

## Overview
The AI Resume Screening System provides RESTful APIs for intelligent candidate matching and resume analysis.

## Base URL
```
http://localhost:8000
```

## Authentication
Currently no authentication required for development. Production deployment should implement proper auth.

## Endpoints

### Health Check
```http
GET /health
```

### Match Candidates
```http
POST /match
```

### Upload Resume
```http
POST /upload
```

See OpenAPI documentation at `/docs` for detailed schemas.
