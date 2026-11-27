
import numpy as np


def retrieve(query, embedded_chunks, embedder, top_k=3):
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    if not embedded_chunks:
        return []

    # 1. Generate query embedding
    query_embedding = np.asarray(
        embedder.embed_query(query)
    )

    # 2. Extract stored document embeddings
    document_embeddings = np.asarray([
        chunk["embedding"]
        for chunk in embedded_chunks
    ])

    # 3. Calculate similarity scores
    scores = document_embeddings @ query_embedding

    # 4. Rank chunks by similarity
    ranked_indices = np.argsort(scores)[::-1]

    # 5. Return top-K results
    results = []

    for index in ranked_indices[:top_k]:
        result = embedded_chunks[index].copy()
        result["score"] = float(scores[index])
        results.append(result)

    return results
