"""Application-wide configuration settings (pydantic-settings v2)."""
from typing import Any, Dict, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Database
    database_url: str = Field(default="sqlite:///./resume_screening.db")

    # ML model
    sentence_transformer_model: str = Field(default="all-MiniLM-L6-v2")
    max_resume_length: int = Field(default=10000, ge=1000)
    batch_size: int = Field(default=32, ge=1)

    # API
    max_file_size_mb: int = Field(default=10, ge=1, le=100)
    cors_origins: str = Field(default="*")
    log_level: str = Field(default="INFO")

    # ML hyperparameters
    model_params: Dict[str, Any] = Field(
        default={"n_estimators": 100, "max_depth": 10, "random_state": 42}
    )

    # Paths
    model_save_path: str = Field(default="data/models/resume_matcher.pkl")


config = Config()
