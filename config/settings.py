"""Application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the ScholarSync MCP server."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    mongodb_uri: str = Field(default="mongodb://localhost:27017", alias="MONGODB_URI")
    mongodb_database: str = Field(default="academic_assistant", alias="MONGODB_DATABASE")
    default_user_id: str = Field(default="local_user", alias="SCHOLARSYNC_DEFAULT_USER_ID")
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="SCHOLARSYNC_EMBEDDING_MODEL",
    )
    embedding_dimension: int = Field(default=384, alias="SCHOLARSYNC_EMBEDDING_DIMENSION")
    chunk_collection: str = Field(default="academic_chunks", alias="SCHOLARSYNC_CHUNK_COLLECTION")
    chunk_size_words: int = Field(default=420, alias="SCHOLARSYNC_CHUNK_SIZE_WORDS")
    chunk_overlap_words: int = Field(default=80, alias="SCHOLARSYNC_CHUNK_OVERLAP_WORDS")
    vector_search_mode: str = Field(default="local", alias="SCHOLARSYNC_VECTOR_SEARCH_MODE")
    atlas_vector_index_name: str = Field(default="academic_chunks_vector_index", alias="SCHOLARSYNC_ATLAS_VECTOR_INDEX")
    search_mode: str = Field(default="vector", alias="SCHOLARSYNC_SEARCH_MODE")


@lru_cache
def get_settings() -> Settings:
    """Return cached settings."""
    return Settings()
