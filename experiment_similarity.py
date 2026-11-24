
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "I love machine learning.",
    "I enjoy studying artificial intelligence.",
    "The weather is very cold today."
]

query = "I like learning about AI."

# Convert documents and query into embeddings
document_embeddings = model.encode(documents)
query_embedding = model.encode([query])

# Calculate similarity between query and every document
scores = cosine_similarity(
    query_embedding,
    document_embeddings
)[0]

# Rank documents by similarity
ranked_results = sorted(
    zip(documents, scores),
    key=lambda item: item[1],
    reverse=True
)

for document, score in ranked_results:
    print(f"Score: {score:.4f} | {document}")
