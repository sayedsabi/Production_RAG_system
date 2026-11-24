
from sentence_transformers import SentenceTransformer

# 1. Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Create sample sentences
sentences = [
    "I love machine learning.",
    "I enjoy studying artificial intelligence.",
    "The weather is very cold today."
]

# 3. Convert sentences into embeddings
embeddings = model.encode(sentences)

# 4. Inspect the output
print("Number of sentences:", len(sentences))
print("Embedding shape:", embeddings.shape)

print("\nFirst sentence embedding:")
print(embeddings[0])