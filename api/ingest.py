from functools import partial
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool

from api.deps import get_rag_service
from rag.config import Settings, get_settings
from rag.errors import AppError
from rag.logging import get_logger
from rag.schemas import IngestResponse
from rag.security import validate_internal_request
from services.rag_service import RAGService

router = APIRouter(tags=["study_material"])
logger = get_logger(__name__)

ALLOWED_MIME_TYPES: dict[str, set[str]] = {
    "pdf": {"application/pdf", "application/x-pdf"},
    "txt": {"text/plain"},
    "docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/octet-stream",
    },
}


@router.post("/ingest", response_model=IngestResponse)
async def ingest_study_material(
    file: UploadFile = File(...),
    material_id: str = Form(...),
    subject_id: str = Form(...),
    section_id: str = Form(...),
    faculty_id: str = Form(...),
    _: None = Depends(validate_internal_request),
    settings: Settings = Depends(get_settings),
    rag_service: RAGService = Depends(get_rag_service),
) -> IngestResponse:
    filename = file.filename or ""
    extension = Path(filename).suffix.lower().lstrip(".")

    if not extension:
        raise AppError(status_code=400, detail="Uploaded file must include an extension", code="missing_file_extension")

    if extension not in settings.ALLOWED_EXTENSIONS:
        raise AppError(status_code=400, detail=f"File extension .{extension} is not allowed", code="invalid_file_extension")

    expected_mime_types = ALLOWED_MIME_TYPES.get(extension, set())
    if expected_mime_types and file.content_type and file.content_type not in expected_mime_types:
        raise AppError(status_code=400, detail="Invalid MIME type for uploaded file", code="invalid_mime_type")

    file_bytes = await file.read()
    if not file_bytes:
        raise AppError(status_code=400, detail="Uploaded file is empty", code="empty_file")

    if len(file_bytes) > settings.max_upload_bytes:
        raise AppError(
            status_code=413,
            detail=f"File exceeds max upload size of {settings.MAX_UPLOAD_MB} MB",
            code="file_too_large",
        )

    logger.info("ingestion start material_id=%s", material_id)
    ingest_result = await run_in_threadpool(
        partial(
            rag_service.ingest,
            file_bytes=file_bytes,
            filename=filename,
            material_id=str(material_id),
            subject_id=str(subject_id),
            section_id=str(section_id),
            faculty_id=str(faculty_id),
        )
    )
    logger.info(
        "ingestion success material_id=%s chunks=%s",
        material_id,
        ingest_result.chunks_indexed,
    )

    return IngestResponse(
        document_id=ingest_result.document_id,
        material_id=ingest_result.material_id,
        chunks_indexed=ingest_result.chunks_indexed,
        replaced_chunks=ingest_result.replaced_chunks,
    )
