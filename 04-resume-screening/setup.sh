#!/bin/bash
# AI Resume Screening System - Environment Setup

echo "Setting up AI Resume Screening System..."

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Copy environment file
cp .env.example .env

echo "Setup complete!"
echo "Next steps:"
echo "   1. Activate virtual environment: source venv/bin/activate"
echo "   2. Run the API: make run"
echo "   3. Generate sample data: make generate-data"
echo "   4. Visit http://localhost:8000/docs"
