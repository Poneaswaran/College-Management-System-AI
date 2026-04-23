from dataclasses import dataclass

import chromadb
from chromadb.api.models.Collection import Collection

from rag.chunker import TextChunk
from rag.config import Settings


@dataclass(slots=True)
class RetrievalResult:
    chunk_id: str
    text: str
    metadata: dict
    score: float | None


class ChromaVectorStore:
    """Vector store adapter for ChromaDB operations."""

    def __init__(self, settings: Settings, collection: Collection | None = None) -> None:
        self.settings = settings
        if collection is not None:
            self.collection = collection
            return

        self.client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        self.collection = self.client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def delete_by_material(self, material_id: str) -> int:
        records = self.collection.get(where={"material_id": str(material_id)}, include=["metadatas"])
        ids = records.get("ids", [])
        if not ids:
            return 0

        self.collection.delete(ids=ids)
        return len(ids)

    def upsert_document(
        self,
        *,
        vector_document_id: str,
        material_id: str,
        subject_id: str,
        section_id: str,
        faculty_id: str,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")

        ids = [f"{vector_document_id}:{chunk.index}" for chunk in chunks]
        documents = [chunk.text for chunk in chunks]

        metadatas = [
            {
                "vector_document_id": vector_document_id,
                "material_id": str(material_id),
                "subject_id": str(subject_id),
                "section_id": str(section_id),
                "faculty_id": str(faculty_id),
                "chunk_index": int(chunk.index),
            }
            for chunk in chunks
        ]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        return len(ids)

    def query_by_material(
        self,
        *,
        query_embedding: list[float],
        material_id: str,
        top_k: int,
    ) -> list[RetrievalResult]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"material_id": str(material_id)},
            include=["documents", "metadatas", "distances"],
        )

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        output: list[RetrievalResult] = []
        for idx, text in enumerate(documents):
            chunk_id = ids[idx] if idx < len(ids) else f"unknown:{idx}"
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            distance = distances[idx] if idx < len(distances) else None
            score = None if distance is None else float(1 / (1 + distance))
            output.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    text=text,
                    metadata=metadata,
                    score=score,
                )
            )

        return output

    def store_memory(
        self,
        *,
        text: str,
        embedding: list[float],
        metadata: dict,
    ) -> str:
        """Store a natural language fact or rule into the vector store."""
        import uuid
        memory_id = f"mem:{uuid.uuid4()}"
        
        self.collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{**metadata, "type": "memory"}],
        )
        return memory_id

    def query(
        self,
        *,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[RetrievalResult]:
        """Generic query with metadata filtering."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        output: list[RetrievalResult] = []
        for idx, text in enumerate(documents):
            chunk_id = ids[idx] if idx < len(ids) else f"unknown:{idx}"
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            distance = distances[idx] if idx < len(distances) else None
            score = None if distance is None else float(1 / (1 + distance))
            output.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    text=text,
                    metadata=metadata,
                    score=score,
                )
            )

        return output

    def delete_by_vector_document_id(self, vector_document_id: str) -> int:
        records = self.collection.get(
            where={"vector_document_id": vector_document_id},
            include=["metadatas"],
        )
        ids = records.get("ids", [])
        if not ids:
            return 0

        self.collection.delete(ids=ids)
        return len(ids)
