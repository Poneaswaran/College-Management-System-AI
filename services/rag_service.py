from dataclasses import dataclass

from rag.errors import AppError
from rag.logging import get_logger
from rag.schemas import SourceItem
from rag.chunker import TextChunker
from rag.extractor import TextExtractor
from rag.vector_store import ChromaVectorStore, RetrievalResult
from services.embedder import Embedder


@dataclass(slots=True)
class IngestResult:
    document_id: str
    material_id: str
    chunks_indexed: int
    replaced_chunks: int


@dataclass(slots=True)
class QueryResult:
    answer: str
    sources: list[SourceItem]


class RAGService:
    """Coordinates extraction, chunking, embeddings, retrieval, and generation."""

    def __init__(
        self,
        *,
        extractor: TextExtractor,
        chunker: TextChunker,
        embedder: Embedder,
        vector_store: ChromaVectorStore,
        llm_service,
        top_k: int,
    ) -> None:
        self.extractor = extractor
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.top_k = top_k
        self.logger = get_logger(__name__)

    def _vector_document_id(self, material_id: str) -> str:
        return f"material:{material_id}"

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
        try:
            text = self.extractor.extract(file_bytes=file_bytes, filename=filename)
            if not text.strip():
                raise AppError(
                    status_code=400,
                    detail="No extractable text found in uploaded file",
                    code="empty_document",
                )

            chunks = self.chunker.chunk_text(text)
            if not chunks:
                raise AppError(
                    status_code=400,
                    detail="No valid text chunks produced from uploaded file",
                    code="no_chunks_generated",
                )

            vector_document_id = self._vector_document_id(material_id)
            replaced_chunks = self.vector_store.delete_by_material(material_id)
            embeddings = self.embedder.embed_texts([chunk.text for chunk in chunks])
            indexed_count = self.vector_store.upsert_document(
                vector_document_id=vector_document_id,
                material_id=material_id,
                subject_id=subject_id,
                section_id=section_id,
                faculty_id=faculty_id,
                chunks=chunks,
                embeddings=embeddings,
            )

            return IngestResult(
                document_id=vector_document_id,
                material_id=str(material_id),
                chunks_indexed=indexed_count,
                replaced_chunks=replaced_chunks,
            )
        except AppError:
            raise
        except Exception as exc:
            self.logger.exception("ingest pipeline failed")
            raise AppError(
                status_code=503,
                detail="Ingestion pipeline failed",
                code="ingest_failed",
            ) from exc

    async def query(self, *, message: str, material_id: str) -> QueryResult:
        try:
            query_embedding = self.embedder.embed_query(message)
            matches = self.vector_store.query_by_material(
                query_embedding=query_embedding,
                material_id=material_id,
                top_k=self.top_k,
            )

            if not matches:
                return QueryResult(
                    answer="I could not find relevant context for this material.",
                    sources=[],
                )

            answer = await self.llm_service.generate_answer(
                question=message,
                context_chunks=[match.text for match in matches],
            )
            sources = [self._to_source_item(match) for match in matches]
            return QueryResult(answer=answer, sources=sources)
        except AppError:
            raise
        except Exception as exc:
            self.logger.exception("query pipeline failed")
            raise AppError(
                status_code=503,
                detail="Query pipeline failed",
                code="query_failed",
            ) from exc

    def delete(self, *, vector_document_id: str) -> int:
        try:
            return self.vector_store.delete_by_vector_document_id(vector_document_id)
        except Exception as exc:
            self.logger.exception("delete pipeline failed")
            raise AppError(
                status_code=503,
                detail="Delete pipeline failed",
                code="delete_failed",
            ) from exc

    def _to_source_item(self, match: RetrievalResult) -> SourceItem:
        snippet = match.text[:280].strip()
        return SourceItem(
            chunk_id=match.chunk_id,
            snippet=snippet,
            material_id=str(match.metadata.get("material_id", "")),
            score=match.score,
        )
