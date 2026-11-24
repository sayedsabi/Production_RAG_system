
from src.embeddings.embedder import Embedder

embedder = Embedder()

chunks = [
    {
        "chunk_id": "chunk_001",
        "document_id": "doc_001",
        "source": "sample.txt",
        "chunk_index": 0,
        "text": "Machine learning enables computers to learn from data."
    },
    {
        "chunk_id": "chunk_002",
        "document_id": "doc_001",
        "source": "sample.txt",
        "chunk_index": 1,
        "text": "Deep learning uses neural networks with multiple layers."
    }
]

embedded_chunks = embedder.embed_chunks(chunks)

for chunk in embedded_chunks:
    print("Chunk ID:", chunk["chunk_id"])
    print("Embedding dimensions:", len(chunk["embedding"]))
    print("First five values:", chunk["embedding"][:5])
    print()

query_embedding = embedder.embed_query(
    "How do computers learn from information?"
)

print("Query dimensions:", len(query_embedding))
