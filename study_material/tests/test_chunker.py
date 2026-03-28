from rag.chunker import TextChunker


def test_chunker_is_deterministic() -> None:
    chunker = TextChunker(chunk_size=60, chunk_overlap=10)
    text = " ".join([f"token{i}" for i in range(80)])

    first_run = chunker.chunk_text(text)
    second_run = chunker.chunk_text(text)

    assert [chunk.text for chunk in first_run] == [chunk.text for chunk in second_run]
    assert [chunk.index for chunk in first_run] == list(range(len(first_run)))
    assert all(len(chunk.text) <= 60 for chunk in first_run)
    assert len(first_run) > 1
