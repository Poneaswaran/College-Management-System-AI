from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.delete import router as delete_router
from api.health import router as health_router
from api.ingest import router as ingest_router
from api.query import router as query_router
from rag.chunker import TextChunker
from rag.config import Settings, get_settings
from rag.errors import AppError, build_error_payload
from rag.extractor import TextExtractor
from rag.logging import CORRELATION_ID_HEADER, configure_logging, get_correlation_id, get_logger, set_correlation_id
from rag.vector_store import ChromaVectorStore
from services.embedder import create_embedder
from services.llm_service import create_llm_service
from services.rag_service import RAGService

logger = get_logger(__name__)


def build_rag_service(settings: Settings) -> RAGService:
    extractor = TextExtractor()
    chunker = TextChunker(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
    embedder = create_embedder(settings)
    vector_store = ChromaVectorStore(settings)
    llm_service = create_llm_service(settings)

    return RAGService(
        extractor=extractor,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        llm_service=llm_service,
        top_k=settings.TOP_K,
    )


def create_app(settings: Settings | None = None, rag_service: RAGService | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.LOG_LEVEL)

    app = FastAPI(
        title="Study Material RAG Service",
        version="1.0.0",
        description="FastAPI AI worker for Django College Management System",
    )

    app.state.settings = settings
    app.state.rag_service = rag_service or build_rag_service(settings)

    app.include_router(health_router)
    app.include_router(ingest_router)
    app.include_router(query_router)
    app.include_router(delete_router)

    @app.middleware("http")
    async def correlation_id_middleware(request: Request, call_next):
        correlation_id = request.headers.get(CORRELATION_ID_HEADER) or str(uuid4())
        set_correlation_id(correlation_id)

        response = await call_next(request)
        response.headers[CORRELATION_ID_HEADER] = correlation_id
        return response

    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        logger.warning("request failed code=%s status=%s", exc.code, exc.status_code)
        return JSONResponse(
            status_code=exc.status_code,
            content=build_error_payload(exc.detail, exc.code, correlation_id=get_correlation_id()),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=build_error_payload(
                detail="Request validation failed",
                code="validation_error",
                correlation_id=get_correlation_id(),
            )
            | {"errors": exc.errors()},
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
        detail = str(exc.detail) if exc.detail else "HTTP error"
        return JSONResponse(
            status_code=exc.status_code,
            content=build_error_payload(
                detail=detail,
                code="http_error",
                correlation_id=get_correlation_id(),
            ),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled exception: %s", type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content=build_error_payload(
                detail="Internal server error",
                code="internal_error",
                correlation_id=get_correlation_id(),
            ),
        )

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        llm_service = getattr(app.state.rag_service, "llm_service", None)
        if llm_service and hasattr(llm_service, "aclose"):
            await llm_service.aclose()

    return app


app = create_app()
