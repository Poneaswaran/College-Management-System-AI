from rag.chroma_client import collection


from rag.ingest import ingest_pdf

count = ingest_pdf("temp_construction_requirements_chennai.pdf")
print("Inserted:", count)
results = collection.query(
    query_texts=["foundation requirements for 2 floor house"],
    n_results=3
)

for doc in results["documents"][0]:
    print("\n---\n", doc)