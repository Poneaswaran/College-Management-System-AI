from pypdf import PdfReader
from .chroma_client import collection

def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def chunk_text(text):
    sections = text.split("\n\n")  # split by paragraphs

    chunks = []
    current_chunk = ""

    for section in sections:
        if len(current_chunk) + len(section) < 500:
            current_chunk += section + "\n\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = section + "\n\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def ingest_pdf(file_path):
    text = load_pdf(file_path)
    chunks = chunk_text(text)

    ids = [f"doc_{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=[{"source": "pdf"} for _ in chunks]
    )

    return len(chunks)