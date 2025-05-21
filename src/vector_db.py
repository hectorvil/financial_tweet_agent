import streamlit as st
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer


@st.cache_resource
def load_embedder() -> SentenceTransformer:
    """Carga y cachea el modelo Mini-LM."""
    return SentenceTransformer("all-MiniLM-L6-v2")


class VectorDB:
    """
    Wrapper sencillo sobre ChromaDB:
    * Deduplicación de IDs antes de insertar
    * Búsqueda semántica con Mini-LM
    """
    def __init__(self, path: str = "chroma_db"):
        self.client = PersistentClient(path)
        self.collection = self.client.get_or_create_collection(
            name="tweets",
            metadata={"hnsw:space": "cosine"}
        )
        self.embedder = load_embedder()

    # ── helpers internos ────────────────────────────────────────────
    def _filter_new(self, ids: list[str], texts: list[str]) -> tuple[list[str], list[str]]:
        """Devuelve solo los (ids, textos) que aún no existen en la colección."""
        existing = set(self.collection.get(ids=ids, include=[])["ids"])
        new_ids, new_texts = [], []
        for _id, txt in zip(ids, texts):
            if _id not in existing:
                new_ids.append(_id)
                new_texts.append(txt)
        return new_ids, new_texts

    # ── API pública ────────────────────────────────────────────────
    def add(self, ids: list[str], texts: list[str]) -> None:
        """Añade documentos; ignora silenciosamente los IDs duplicados."""
        ids, texts = self._filter_new(ids, texts)
        if not ids:  # nada nuevo
            return
        embeds = self.embedder.encode(texts, batch_size=64).tolist()
        self.collection.add(ids=ids, documents=texts, embeddings=embeds)

    def query(self, query_text: str, k: int = 30) -> list[str]:
        q_emb = self.embedder.encode([query_text]).tolist()
        res = self.collection.query(query_embeddings=q_emb, n_results=k)
        return res["documents"][0]
