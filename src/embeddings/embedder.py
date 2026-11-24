
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts):
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True
        )

        return embeddings

    def embed_chunks(self, chunks):
        if not chunks:
            return []

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embed_texts(texts)

        embedded_chunks = []

        for chunk, embedding in zip(chunks, embeddings):
            embedded_chunks.append({
                **chunk,
                "embedding": embedding.tolist()
            })

        return embedded_chunks

    def embed_query(self, query):
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        embedding = self.embed_texts([query])[0]

        return embedding.tolist()
