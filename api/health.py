from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from rag.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/health/live", response_model=HealthResponse)
async def live() -> HealthResponse:
    return HealthResponse(status="alive")


@router.get("/health/ready", response_model=HealthResponse)
async def ready(request: Request) -> HealthResponse | JSONResponse:
    rag_service = getattr(request.app.state, "rag_service", None)
    if rag_service is None:
        return JSONResponse(status_code=503, content={"status": "not_ready"})
    return HealthResponse(status="ready")
