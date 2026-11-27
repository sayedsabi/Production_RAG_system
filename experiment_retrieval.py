
from pathlib import Path

from src.ingestion.loader import load_directory
from src.chunking.chunk_documents import chunk_document
from src.embeddings.embedder import Embedder
from src.retrieval.retriever import retrieve


# Load documents
result = load_directory(Path(r"C:\Users\sayed\Projects\genai-engineering\02-production-rag-system\data\raw"))
documents = result["documents"]

# Chunk documents
all_chunks = []

for document in documents:
    chunks = chunk_document(
        document,
        chunk_size=500,
        overlap=50
    )
    all_chunks.extend(chunks)

# Generate embeddings
embedder = Embedder()
embedded_chunks = embedder.embed_chunks(all_chunks)

# Ask a question
query = "How do I train a convolutional neural network?"

# Retrieve relevant chunks
results = retrieve(
    query=query,
    embedded_chunks=embedded_chunks,
    embedder=embedder,
    top_k=3
)

# Display results
for rank, result in enumerate(results, start=1):
    print(f"\nRank: {rank}")
    print(f"Score: {result['score']:.4f}")
    print(f"Source: {result['source']}")
    print(f"Chunk ID: {result['chunk_id']}")
    print(f"Text:\n{result['text'][:300]}")
