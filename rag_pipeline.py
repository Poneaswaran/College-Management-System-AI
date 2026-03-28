from rag.chroma_client import collection
from services.llm import generate_answer

def ask(query):
    results = collection.query(
        query_texts=[query],
        n_results=3
    )

    context = "\n\n".join(results["documents"][0])

    return generate_answer(query, context)