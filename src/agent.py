import pandas as pd
import openai

from src.vector_db import VectorDB
from src.data_pipeline import add_labels, clean


class FinancialTweetAgent:
    """
    Orquesta la ingesta de Parquet, gestiona la base vectorial ChromaDB
    y expone utilidades para chat histórico, live search y dashboard.
    """

    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        self.model = model
        self.db = VectorDB()
        self.df = pd.DataFrame()

    # ────────────────────────────────────────────────────────────────
    # Ingesta
    # ────────────────────────────────────────────────────────────────
    def ingest(self, parquet_file):
        """
        Carga un Parquet, añade etiquetas faltantes y sube documentos a Chroma.

        - Si el Parquet YA incluye columnas `sentiment`, `topic`, `tickers`,
          `clean` y opcionalmente `embedding`, no se recalculan.
        - Evita duplicados comparando `doc_id`.
        """
        df = pd.read_parquet(parquet_file)

        # 1) Completar etiquetas solo si faltan
        needs_labels = any(
            col not in df for col in ("sentiment", "topic", "clean", "tickers")
        )
        if needs_labels:
            df = add_labels(df, skip_if_present=True)

        # 2) Clean fallback
        if "clean" not in df:
            df["clean"] = df["text"].map(clean)

        # 3) doc_id obligatorio
        if "doc_id" not in df:
            df["doc_id"] = df.index.astype(str)

        # 4) Añadir a Chroma (usa embeddings precalculados si existen)
        if "embedding" in df:
            self.db.add(
                ids=df["doc_id"].tolist(),
                texts=df["clean"].tolist(),
                embeddings=df["embedding"].tolist(),
            )
        else:
            self.db.add(df["doc_id"].tolist(), df["clean"].tolist())

        # 5) Cache in-memory para pivot
        self.df = pd.concat([self.df, df], ignore_index=True)

    # ────────────────────────────────────────────────────────────────
    # Dashboard helper
    # ────────────────────────────────────────────────────────────────
    def pivot(self, min_m: int = 20) -> pd.DataFrame:
        """Devuelve un DataFrame agregado por ticker y sentimiento."""
        if self.df.empty or "tickers" not in self.df:
            return pd.DataFrame()

        piv = (
            self.df.explode("tickers")
            .query("tickers != ''")
            .groupby(["tickers", "sentiment"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )

        for col in ("positive", "neutral", "negative"):
            if col not in piv:
                piv[col] = 0

        piv["total"] = piv[["positive", "neutral", "negative"]].sum(axis=1)
        piv = piv[piv["total"] >= min_m]
        piv["pos_ratio"] = piv["positive"] / piv["total"]
        piv["neg_ratio"] = piv["negative"] / piv["total"]
        return piv.sort_values("neg_ratio", ascending=False)

    # ────────────────────────────────────────────────────────────────
    # Chat histórico (RAG sobre corpus)
    # ────────────────────────────────────────────────────────────────
    def insight_hist(self, query: str, k: int = 30) -> str:
        docs = self.db.query(query, k)

        # Buscar en self.df los registros originales para obtener sus sentimientos
        subset = self.df[self.df["clean"].isin(docs)]
        pos = (subset["sentiment"] == "positive").sum()
        neu = (subset["sentiment"] == "neutral").sum()
        neg = (subset["sentiment"] == "negative").sum()
        total = max(pos + neu + neg, 1)

        ratios = f"(+ {pos/total:.2f} | = {neu/total:.2f} | − {neg/total:.2f})"

        context = "\n".join(t[:280] for t in docs)

        prompt = f"""
Usa SOLO el contexto siguiente para responder.
Contexto:
{context}

Pregunta: {query}
Responde en español de forma clara, cita tweet_id cuando corresponda y menciona si predomina un tono positivo o negativo.
""".strip()

        response = openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        answer = response.choices[0].message.content.strip()
        return f"{answer}\n\n📊 Sentiment {ratios}"


    # ────────────────────────────────────────────────────────────────
    # Live search (Twitter) + ingest
    # ────────────────────────────────────────────────────────────────
    def live_search(self, query: str, n: int = 30) -> pd.DataFrame:
        from src.twitter_live import search

        live = pd.DataFrame(search(query, n=n))
        if live.empty:
            return pd.DataFrame()

        if "doc_id" not in live:
            live["doc_id"] = live.index.astype(str)

        live = add_labels(live)  # siempre etiqueta porque viene sin procesar
        self.db.add(live["doc_id"].tolist(), live["clean"].tolist())
        self.df = pd.concat([self.df, live], ignore_index=True)
        return live

    def insight_live(self, query: str, n: int = 30) -> str:
        recent = self.live_search(query, n=n)
        if not recent.empty:
            context = "\n".join(recent["clean"].tolist()[:30])
        else:
            context = "\n".join(self.db.query(query, k=30))

        prompt = f"""
Con base en el contexto, responde a la pregunta.
Contexto:
{context}

Pregunta: {query}
""".strip()

        response = openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
