"""Application settings — validated against environment variables at import time."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # App
    app_name: str = "RAG Assistant"
    app_version: str = "2.0.0"
    debug: bool = False
    log_level: str = "INFO"

    # OpenAI
    openai_api_key: str = Field(..., description="OpenAI API key — required")
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    llm_model: str = "gpt-4o-mini"

    # Retrieval
    retrieval_top_k: int = Field(default=10, ge=1, le=50)
    rerank_top_k: int = Field(default=3, ge=1, le=20)

    # Pinecone
    pinecone_api_key: str = Field(..., description="Pinecone API key — required")
    pinecone_index_name: str = "rag-assistant"

    # Chunking — token-based sizes, not character-based
    chunk_size: int = Field(default=512, ge=64, le=2048)
    chunk_overlap: int = Field(default=64, ge=0, le=256)

    # Generation
    max_tokens: int = Field(default=1024, ge=64, le=4096)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)


settings = Settings()
