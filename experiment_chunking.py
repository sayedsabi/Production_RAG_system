
from pathlib import Path

from src.ingestion.loader import load_directory
from src.chunking.chunk_documents import chunk_document


result = load_directory(Path("data/raw"))

for document in result["documents"]:
    print(f"\nDocument: {document['document_id']}")

    for size, overlap in [(100, 0), (100, 20), (500, 50)]:
        chunks = chunk_document(
            document,
            chunk_size=size,
            overlap=overlap,
        )

        print(
            f"chunk_size={size}, overlap={overlap}: "
            f"{len(chunks)} chunks"
        )

        for chunk in chunks[:2]:
            print("  ", repr(chunk["text"][:100]))
