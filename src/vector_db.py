import streamlit as st
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer


@st.cache_resource
def load_embedder() -> SentenceTransformer:
    """Carga una sola vez el modelo Mini-LM para todo el proceso."""
    return SentenceTransformer("all-MiniLM-L6-v2")


class VectorDB:
    """
    Wrapper sobre ChromaDB con deduplicación y embeddings cacheados.
    Accepta embeddings precalculados si vienen en el DataFrame.
    """
    def __init__(self, path: str = "chroma_db"):
        self.client = PersistentClient(path)
        self.collection = self.client.get_or_create_collection(
            name="tweets",
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = load_embedder()

    # ── helpers internos ────────────────────────────────────────────
    def _filter_new(self, ids, texts, embeds):
        """Filtra doc_ids ya existentes para evitar ValueError de duplicados."""
        existing = set(self.collection.get(ids=ids, include=[])["ids"])
        new_ids, new_txt, new_emb = [], [], []
        for i, t, e in zip(ids, texts, embeds):
            if i not in existing:
                new_ids.append(i)
                new_txt.append(t)
                new_emb.append(e)
        return new_ids, new_txt, new_emb

    # ── API pública ────────────────────────────────────────────────
    def add(self, ids, texts, embeddings=None):
        """
        Añade documentos.
        - Si embeddings==None, los calcula con Mini-LM.
        - Si se proporcionan (lista de listas), se usan directamente.
        """
        if embeddings is None:
            embeddings = self.embedder.encode(texts, batch_size=64).tolist()

        ids, texts, embeddings = self._filter_new(ids, texts, embeddings)
        if not ids:  # nada nuevo
            return

        self.collection.add(ids=ids, documents=texts, embeddings=embeddings)

    def query(self, query_text: str, k: int = 30):
        q_emb = self.embedder.encode([query_text]).tolist()
        res = self.collection.query(query_embeddings=q_emb, n_results=k)
        return res["documents"][0]
