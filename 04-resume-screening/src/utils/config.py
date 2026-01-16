from pydantic import BaseSettings
from typing import Dict, Any

class Config(BaseSettings):
    """
    Application-wide configuration settings.
    
    PURPOSE: This centralizes all settings so you can:
    - Change parameters without editing code
    - Use different settings for development vs production
    - Load settings from environment variables or .env file
    - Ensure type safety with pydantic validation
    """
    
    # DATABASE CONFIGURATION
    # Where to store application data (resumes, jobs, user data)
    database_url: str = "sqlite:///./resume_screening.db"
    # Default: Uses SQLite database file in current directory
    # Production might use: "postgresql://user:password@host:port/database"
    
    # MACHINE LEARNING MODEL SETTINGS
    # Which sentence transformer model to use for text embeddings
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    # This model converts text into numerical vectors for similarity matching
    # Options: "all-MiniLM-L6-v2" (fast), "all-mpnet-base-v2" (better quality)
    
    # Maximum length of resume text to process
    max_resume_length: int = 10000
    # Longer resumes get truncated to avoid memory issues
    
    # How many resumes to process at once during training/inference
    batch_size: int = 32
    # Larger = faster but more memory usage
    
    # REST API CONFIGURATION
    # Maximum file size for uploaded resumes (in MB)
    max_file_size_mb: int = 10
    # Prevents users from uploading huge files that crash the system
    
    # Cross-Origin Resource Sharing - which websites can call your API
    cors_origins: str = "*"
    # "*" = allow all origins (development), production should be specific URLs
    
    # Logging detail level
    log_level: str = "INFO"
    # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
    
    # MACHINE LEARNING MODEL HYPERPARAMETERS
    # Parameters for the classification/matching model
    model_params: Dict[str, Any] = {
        'n_estimators': 100,    # Number of decision trees (if using Random Forest)
        'max_depth': 10,        # Maximum depth of each tree
        'random_state': 42      # For reproducible results
    }
    # These get passed to scikit-learn models during training
    
    # FILE PATHS
    # Where to save trained ML models
    model_save_path: str = "data/models/resume_matcher.pkl"
    # Models are saved as pickle files for later loading
    
    class Config:
        # Load additional settings from .env file
        env_file = ".env"
        # Example .env file:
        # DATABASE_URL=postgresql://localhost:5432/resumes
        # SENTENCE_TRANSFORMER_MODEL=all-mpnet-base-v2
        # LOG_LEVEL=DEBUG

# HOW TO USE THIS CONFIG:
"""
# In other files, import and use like this:
from src.utils.config import Config

config = Config()
print(f"Using model: {config.sentence_transformer_model}")
print(f"Database: {config.database_url}")

# The config automatically:
# 1. Uses default values defined above
# 2. Overrides with environment variables if set
# 3. Overrides with .env file values if present
# 4. Validates all types with pydantic
"""