# AI Resume Screening System - Makefile

.PHONY: install test run clean docker-build docker-run

# Development setup
install:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm
	pre-commit install

# Testing
test:
	pytest tests/ --cov=src --cov-report=html --cov-report=term

test-watch:
	pytest-watch tests/

# Running the application
run:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-prod:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

# Data generation
generate-data:
	python -m src.data_generation.resume_generator

# Docker operations
docker-build:
	docker build -t resume-screening .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

# Code quality
format:
	black src/ tests/
	
lint:
	flake8 src/ tests/
	
# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .coverage htmlcov/ .pytest_cache/

# Documentation
docs:
	mkdocs serve

# Full development setup
setup: install
	@echo "AI Resume Screening System setup complete!"
	@echo "Read README.md for next steps"
	@echo "Run 'make run' to start the development server"
