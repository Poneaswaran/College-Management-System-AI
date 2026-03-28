from fastapi import Request

from services.rag_service import RAGService


def get_rag_service(request: Request) -> RAGService:
    rag_service = getattr(request.app.state, "rag_service", None)
    if rag_service is None:
        raise RuntimeError("RAG service is not initialized")
    return rag_service
