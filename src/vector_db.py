from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# ── Embeddings con MiniLM ──
EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")

class VectorDB:
    def __init__(self, path="chroma_db"):
        self.client = PersistentClient(path)
        self.collection = self.client.get_or_create_collection(
            name="tweets", metadata={"hnsw:space": "cosine"}
        )

    def add(self, ids, texts):
        embeddings = EMBEDDER.encode(texts, batch_size=64).tolist()
        self.collection.add(ids=ids, documents=texts, embeddings=embeddings)

    def query(self, query_text: str, k=30) -> list[str]:
        q_emb = EMBEDDER.encode([query_text]).tolist()
        res = self.collection.query(query_embeddings=q_emb, n_results=k)
        return res["documents"][0]
