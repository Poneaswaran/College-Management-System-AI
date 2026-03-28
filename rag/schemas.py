from typing import Any

from pydantic import BaseModel, Field, field_validator


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Human-readable error message")
    code: str = Field(..., description="Stable error code")
    correlation_id: str | None = Field(default=None, description="Request correlation id")


class HealthResponse(BaseModel):
    status: str


class IngestResponse(BaseModel):
    document_id: str = Field(..., description="Stable vector document id")
    material_id: str
    chunks_indexed: int
    replaced_chunks: int = 0


class QueryFilters(BaseModel):
    material_id: int = Field(..., gt=0)


class QueryRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    filters: QueryFilters

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message cannot be empty")
        return value


class SourceItem(BaseModel):
    chunk_id: str
    snippet: str
    material_id: str
    score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[Any] = Field(default_factory=list)


class DeleteRequest(BaseModel):
    vector_document_id: str = Field(..., min_length=1, max_length=256)

    @field_validator("vector_document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("vector_document_id cannot be empty")
        return value


class DeleteResponse(BaseModel):
    vector_document_id: str
    deleted: bool
    deleted_count: int
    status: str
