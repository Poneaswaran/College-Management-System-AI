import pytest
from fastapi.testclient import TestClient

from rag.config import Settings, get_settings
from rag.schemas import SourceItem
from services.rag_service import IngestResult, QueryResult
from study_material.main import create_app


class FakeRAGService:
    def ingest(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        material_id: str,
        subject_id: str,
        section_id: str,
        faculty_id: str,
    ) -> IngestResult:
        return IngestResult(
            document_id=f"material:{material_id}",
            material_id=material_id,
            chunks_indexed=3,
            replaced_chunks=1,
        )

    async def query(self, *, message: str, material_id: str) -> QueryResult:
        if material_id == "999":
            return QueryResult(answer="I could not find relevant context for this material.", sources=[])

        return QueryResult(
            answer="Mock answer from LLM",
            sources=[
                SourceItem(
                    chunk_id=f"material:{material_id}:0",
                    snippet="Sample source snippet",
                    material_id=material_id,
                    score=0.92,
                )
            ],
        )

    def delete(self, *, vector_document_id: str) -> int:
        return 2 if vector_document_id == "material:1" else 0


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        APP_ENV="test",
        LOG_LEVEL="INFO",
        INTERNAL_SOURCE_VALUE="django-test",
        INTERNAL_API_SECRET="super-secret",
        MAX_UPLOAD_MB=2,
        ALLOWED_EXTENSIONS=["pdf", "txt", "docx"],
        CHUNK_SIZE=200,
        CHUNK_OVERLAP=40,
        TOP_K=3,
        EMBEDDING_PROVIDER="sentence-transformers",
        EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2",
        LLM_PROVIDER="ollama",
        LLM_MODEL="phi3",
        CHROMA_PERSIST_DIR="./tmp_chroma",
        CHROMA_COLLECTION_NAME="study_material_test",
        REQUEST_TIMEOUT_SECONDS=5,
    )


@pytest.fixture
def internal_headers(test_settings: Settings) -> dict[str, str]:
    return {
        "X-Internal-Source": test_settings.INTERNAL_SOURCE_VALUE,
        "X-Internal-Secret": test_settings.INTERNAL_API_SECRET,
    }


@pytest.fixture
def client(test_settings: Settings) -> TestClient:
    app = create_app(settings=test_settings, rag_service=FakeRAGService())
    app.dependency_overrides[get_settings] = lambda: test_settings

    with TestClient(app) as test_client:
        yield test_client
