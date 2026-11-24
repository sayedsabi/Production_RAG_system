
from pathlib import Path

from src.ingestion.loader import load_directory
from src.chunking.chunk_documents import chunk_document
from src.embeddings.embedder import Embedder


# 1. Load documents
result = load_directory(Path(r"C:\Users\sayed\Projects\genai-engineering\02-production-rag-system\data\raw"))
documents = result["documents"]

print("Documents loaded:", len(documents))
print("Loading failures:", len(result["failures"]))


# 2. Chunk documents
all_chunks = []

for document in documents:
    chunks = chunk_document(
        document,
        chunk_size=500,
        overlap=50
    )

    all_chunks.extend(chunks)

print("Total chunks:", len(all_chunks))


# 3. Generate embeddings
embedder = Embedder()

embedded_chunks = embedder.embed_chunks(all_chunks)

print("Embedded chunks:", len(embedded_chunks))


# 4. Inspect one embedded chunk
if embedded_chunks:
    sample = embedded_chunks[0]

    print("\nSample embedded chunk")
    print("Chunk ID:", sample["chunk_id"])
    print("Source:", sample["source"])
    print("Text preview:", sample["text"][:150])
    print("Vector dimensions:", len(sample["embedding"]))
