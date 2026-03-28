from fastapi import APIRouter, Depends

from api.deps import get_rag_service
from rag.logging import get_logger
from rag.schemas import QueryRequest, QueryResponse
from rag.security import validate_internal_request
from services.rag_service import RAGService

router = APIRouter(tags=["study_material"])
logger = get_logger(__name__)


@router.post("/query", response_model=QueryResponse)
async def query_study_material(
    payload: QueryRequest,
    _: None = Depends(validate_internal_request),
    rag_service: RAGService = Depends(get_rag_service),
) -> QueryResponse:
    material_id = str(payload.filters.material_id)

    result = await rag_service.query(
        message=payload.message,
        material_id=material_id,
    )
    logger.info("query success material_id=%s source_count=%s", material_id, len(result.sources))
    return QueryResponse(answer=result.answer, sources=result.sources)
