from fastapi import APIRouter, Depends

from api.deps import get_rag_service
from rag.logging import get_logger
from rag.schemas import DeleteRequest, DeleteResponse
from rag.security import validate_internal_request
from services.rag_service import RAGService

router = APIRouter(tags=["study_material"])
logger = get_logger(__name__)


@router.post("/delete", response_model=DeleteResponse)
async def delete_study_material_vectors(
    payload: DeleteRequest,
    _: None = Depends(validate_internal_request),
    rag_service: RAGService = Depends(get_rag_service),
) -> DeleteResponse:
    deleted_count = rag_service.delete(vector_document_id=payload.vector_document_id)
    deleted = deleted_count > 0
    status = "deleted" if deleted else "not_found"
    logger.info("delete success vector_document_id=%s deleted_count=%s", payload.vector_document_id, deleted_count)

    return DeleteResponse(
        vector_document_id=payload.vector_document_id,
        deleted=deleted,
        deleted_count=deleted_count,
        status=status,
    )
