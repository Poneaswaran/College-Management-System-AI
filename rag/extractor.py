import io
from pathlib import Path

from pypdf import PdfReader

from rag.errors import AppError


class TextExtractor:
    """Extract plain text from supported study material formats."""

    def extract(self, file_bytes: bytes, filename: str) -> str:
        extension = Path(filename).suffix.lower().lstrip(".")

        if extension == "pdf":
            return self._extract_pdf(file_bytes)
        if extension == "txt":
            return file_bytes.decode("utf-8", errors="ignore")
        if extension == "docx":
            return self._extract_docx(file_bytes)

        raise AppError(
            status_code=400,
            detail=f"Unsupported file extension: .{extension}",
            code="unsupported_extension",
        )

    def _extract_pdf(self, file_bytes: bytes) -> str:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    def _extract_docx(self, file_bytes: bytes) -> str:
        try:
            from docx import Document
        except ImportError as exc:
            raise AppError(
                status_code=503,
                detail="DOCX extraction dependency unavailable",
                code="docx_dependency_missing",
            ) from exc

        document = Document(io.BytesIO(file_bytes))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
