from functools import lru_cache
from typing import Literal

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    APP_ENV: Literal["development", "staging", "production", "test"] = "development"
    LOG_LEVEL: str = "INFO"

    INTERNAL_HEADERS_ENABLED: bool = True
    INTERNAL_SOURCE_VALUE: str = "django-college-management"
    INTERNAL_ALLOWED_SOURCES: list[str] = Field(default_factory=list)
    INTERNAL_API_SECRET: str = "change-me"

    MAX_UPLOAD_MB: int = 10
    ALLOWED_EXTENSIONS: list[str] = Field(default_factory=lambda: ["pdf", "txt", "docx"])

    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 120
    TOP_K: int = 4

    EMBEDDING_PROVIDER: str = "sentence-transformers"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    LLM_PROVIDER: str = "ollama"
    LLM_MODEL: str = "phi3"
    OLLAMA_URL: str = "http://localhost:11434/api/generate"

    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "study_material"

    REQUEST_TIMEOUT_SECONDS: float = 30.0

    @field_validator("ALLOWED_EXTENSIONS", mode="before")
    @classmethod
    def parse_allowed_extensions(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, list):
            return [item.strip().lower().lstrip(".") for item in value if item and item.strip()]
        if isinstance(value, str):
            return [item.strip().lower().lstrip(".") for item in value.split(",") if item.strip()]
        raise ValueError("ALLOWED_EXTENSIONS must be a list or comma-separated string")

    @field_validator("INTERNAL_ALLOWED_SOURCES", mode="before")
    @classmethod
    def parse_internal_allowed_sources(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, list):
            return [item.strip() for item in value if item and item.strip()]
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        raise ValueError("INTERNAL_ALLOWED_SOURCES must be a list or comma-separated string")

    @field_validator("INTERNAL_SOURCE_VALUE")
    @classmethod
    def validate_internal_source_value(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("INTERNAL_SOURCE_VALUE cannot be empty")
        return value

    @field_validator("MAX_UPLOAD_MB")
    @classmethod
    def validate_max_upload_mb(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("MAX_UPLOAD_MB must be greater than 0")
        return value

    @field_validator("CHUNK_SIZE")
    @classmethod
    def validate_chunk_size(cls, value: int) -> int:
        if value < 100:
            raise ValueError("CHUNK_SIZE must be at least 100")
        return value

    @field_validator("CHUNK_OVERLAP")
    @classmethod
    def validate_chunk_overlap(cls, value: int, info: ValidationInfo) -> int:
        chunk_size = int(info.data.get("CHUNK_SIZE", 800))
        if value < 0:
            raise ValueError("CHUNK_OVERLAP cannot be negative")
        if value >= chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return value

    @field_validator("TOP_K")
    @classmethod
    def validate_top_k(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("TOP_K must be greater than 0")
        return value

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_MB * 1024 * 1024

    @property
    def allowed_internal_sources(self) -> list[str]:
        if self.INTERNAL_ALLOWED_SOURCES:
            return self.INTERNAL_ALLOWED_SOURCES
        return [self.INTERNAL_SOURCE_VALUE]

    @property
    def require_internal_secret(self) -> bool:
        return bool(self.INTERNAL_API_SECRET.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
