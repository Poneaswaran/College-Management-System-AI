from typing import Protocol

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from rag.config import Settings


class Embedder(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class SentenceTransformerEmbedder:
    """Embedding adapter for sentence-transformers models."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=retry_if_exception_type(Exception),
    )
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        model = self._get_model()
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        if hasattr(vectors, "tolist"):
            output = vectors.tolist()
        else:
            output = [list(vector) for vector in vectors]

        return [[float(value) for value in vector] for vector in output]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


def create_embedder(settings: Settings) -> Embedder:
    provider = settings.EMBEDDING_PROVIDER.strip().lower()
    if provider in {"sentence-transformers", "local"}:
        return SentenceTransformerEmbedder(settings.EMBEDDING_MODEL)

    raise ValueError(f"Unsupported embedding provider: {settings.EMBEDDING_PROVIDER}")
