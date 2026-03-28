from dataclasses import dataclass


@dataclass(slots=True)
class TextChunk:
    index: int
    text: str


class TextChunker:
    """Deterministic character-based chunker with overlap."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> list[TextChunk]:
        normalized = " ".join(text.split())
        if not normalized:
            return []

        chunks: list[TextChunk] = []
        start = 0
        index = 0
        text_len = len(normalized)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            if end < text_len:
                split_at = normalized.rfind(" ", start, end)
                min_split = start + int(self.chunk_size * 0.6)
                if split_at > min_split:
                    end = split_at

            content = normalized[start:end].strip()
            if content:
                chunks.append(TextChunk(index=index, text=content))
                index += 1

            if end >= text_len:
                break

            start = max(end - self.chunk_overlap, start + 1)

        return chunks
