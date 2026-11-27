
import json
from pathlib import Path

from src.ingestion.loader import load_directory
from src.chunking.chunk_documents import chunk_document
from src.embeddings.embedder import Embedder
from src.retrieval.retriever import retrieve

from src.evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"

EVALUATION_FILE = (
    PROJECT_ROOT / "data" / "evaluation" / "retrieval_dataset.json"
)

with open(EVALUATION_FILE, "r", encoding="utf-8") as file:
    evaluation_data = json.load(file)

print("Evaluation questions:", len(evaluation_data))


result = load_directory(DATA_DIRECTORY)
documents = result["documents"]

all_chunks = []

for document in documents:
    chunks = chunk_document(
        document,
        chunk_size=500,
        overlap=50
    )

    all_chunks.extend(chunks)

embedder = Embedder()

embedded_chunks = embedder.embed_chunks(all_chunks)

print("Documents:", len(documents))
print("Chunks:", len(embedded_chunks))


K = 3

precision_scores = []
recall_scores = []
rr_scores = []

for item in evaluation_data:

    query = item["query"]
    relevant_ids = item["relevant_chunk_ids"]

    results = retrieve(
        query=query,
        embedded_chunks=embedded_chunks,
        embedder=embedder,
        top_k=K
    )

    retrieved_ids = [
        result["chunk_id"]
        for result in results
    ]

    precision = precision_at_k(
        retrieved_ids,
        relevant_ids,
        K
    )

    recall = recall_at_k(
        retrieved_ids,
        relevant_ids,
        K
    )

    rr = reciprocal_rank(
        retrieved_ids,
        relevant_ids
    )

    precision_scores.append(precision)
    recall_scores.append(recall)
    rr_scores.append(rr)

    print("\nQuery:", query)
    print("Retrieved:", retrieved_ids)
    print("Relevant:", relevant_ids)
    print(f"Precision@{K}: {precision:.3f}")
    print(f"Recall@{K}: {recall:.3f}")
    print(f"Reciprocal Rank: {rr:.3f}")

print("\n========== EVALUATION SUMMARY ==========")

print(
    f"Mean Precision@{K}: "
    f"{sum(precision_scores) / len(precision_scores):.3f}"
)

print(
    f"Mean Recall@{K}: "
    f"{sum(recall_scores) / len(recall_scores):.3f}"
)

print(
    f"MRR: "
    f"{sum(rr_scores) / len(rr_scores):.3f}"
)

