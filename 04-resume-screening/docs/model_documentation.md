# Model Documentation

## Overview
The system uses multiple NLP approaches for resume-job matching:

## Models Used

### 1. Sentence Transformers
- **Model**: all-MiniLM-L6-v2
- **Purpose**: Semantic similarity between resumes and job descriptions
- **Performance**: 384-dimensional embeddings

### 2. TF-IDF Vectorizer
- **Purpose**: Keyword-based matching
- **Features**: 5000 max features, 1-2 ngrams

### 3. Skills Extraction
- **Approach**: Rule-based + NER
- **Categories**: Programming languages, frameworks, tools, soft skills

## Performance Metrics
- Semantic similarity: Cosine similarity scores 0-1
- Overall matching: Weighted combination of multiple signals
