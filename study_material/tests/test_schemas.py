import pytest
from pydantic import ValidationError

from rag.schemas import DeleteRequest, QueryRequest


def test_query_schema_valid_payload() -> None:
    payload = QueryRequest.model_validate(
        {
            "message": "  Explain chapter 1 topics  ",
            "filters": {"material_id": 42},
        }
    )
    assert payload.message == "Explain chapter 1 topics"
    assert payload.filters.material_id == 42


def test_query_schema_rejects_blank_message() -> None:
    with pytest.raises(ValidationError):
        QueryRequest.model_validate(
            {
                "message": "   ",
                "filters": {"material_id": 42},
            }
        )


def test_delete_schema_rejects_blank_document_id() -> None:
    with pytest.raises(ValidationError):
        DeleteRequest.model_validate({"vector_document_id": "   "})
