from rag.config import Settings
from rag.vector_store import ChromaVectorStore


class DummyCollection:
    def __init__(self) -> None:
        self.last_query_kwargs: dict = {}

    def query(self, **kwargs):
        self.last_query_kwargs = kwargs
        return {
            "ids": [["material:42:0"]],
            "documents": [["chunk text"]],
            "metadatas": [[{"material_id": "42"}]],
            "distances": [[0.2]],
        }


def test_query_uses_material_filter() -> None:
    settings = Settings(
        INTERNAL_SOURCE_VALUE="django",
        INTERNAL_API_SECRET="secret",
    )
    collection = DummyCollection()
    store = ChromaVectorStore(settings=settings, collection=collection)

    results = store.query_by_material(query_embedding=[0.1, 0.2], material_id="42", top_k=3)

    assert collection.last_query_kwargs["where"] == {"material_id": "42"}
    assert collection.last_query_kwargs["n_results"] == 3
    assert len(results) == 1
    assert results[0].metadata["material_id"] == "42"
