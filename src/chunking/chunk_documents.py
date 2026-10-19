
from src.chunking.chunker import chunk_text


def chunk_document(
    document: dict,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict]:
    """Split a loaded document into chunks with source metadata."""

    chunks = chunk_text(
        document["text"],
        chunk_size=chunk_size,
        overlap=overlap,
    )

    results = []

    for index, text in enumerate(chunks):
        results.append({
            "chunk_id": f'{document["document_id"]}_{index}',
            "document_id": document["document_id"],
            "source": document["source"],
            "chunk_index": index,
            "text": text,
        })

    return results
