import os
import pandas as pd
import openai

from src.vector_db import VectorDB
from src.data_pipeline import add_labels

class FinancialTweetAgent:
    def __init__(self, model="gpt-4o-mini-2024-07-18"):
        self.model = model
        self.db = VectorDB()
        self.df = pd.DataFrame()

    def ingest(self, parquet_file):
        """Carga un archivo parquet, etiqueta si es necesario, y añade a Chroma."""
        print("📥 Leyendo archivo parquet...")
        df = pd.read_parquet(parquet_file)

        print("🧼 Limpieza y etiquetado...")
        if "sentiment" not in df or "topic" not in df:
            df = add_labels(df)
        else:
            print("✔️ Ya viene etiquetado con sentimiento y tema.")

        if "doc_id" not in df:
            df["doc_id"] = df.index.astype(str)

        print("📦 Añadiendo a la base vectorial...")
        self.db.add(df["doc_id"].tolist(), df["clean"].tolist())

        print("🧠 Cargando en memoria interna del agente...")
        self.df = pd.concat([self.df, df], ignore_index=True)
        print("✅ Ingest terminado")

    def pivot(self, min_m=20):
        df = self.df.copy()
        if "tickers" not in df:
            return pd.DataFrame()

        piv = (
            df.explode("tickers")
              .query("tickers != ''")
              .groupby(["tickers", "sentiment"])
              .size()
              .unstack(fill_value=0)
              .reset_index()
        )

        for col in ["positive", "neutral", "negative"]:
            if col not in piv:
                piv[col] = 0

        piv["total"] = piv[["positive", "neutral", "negative"]].sum(axis=1)
        piv = piv[piv["total"] >= min_m]
        piv["pos_ratio"] = piv["positive"] / piv["total"]
        piv["neg_ratio"] = piv["negative"] / piv["total"]
        return piv.sort_values("neg_ratio", ascending=False)

    def insight_hist(self, query: str, k: int = 30) -> str:
        context = "\n".join(self.db.query(query, k))
        prompt = f"""
Usa solo el siguiente contexto para responder a la consulta.
Contexto:
{context}

Pregunta: {query}
Responde en español, de forma concisa y clara. Cita tickers si los hay.
        """.strip()

        response = openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()

    def live_search(self, query: str, n: int = 30):
        from src.twitter_live import search
        live = pd.DataFrame(search(query, n=n))
        if live.empty:
            return pd.DataFrame()

        if "doc_id" not in live:
            live["doc_id"] = live.index.astype(str)

        print("⚡ Etiquetando tweets en vivo...")
        live = add_labels(live)

        print("📦 Añadiendo nuevos tweets a la base vectorial...")
        self.db.add(live["doc_id"].tolist(), live["clean"].tolist())
        self.df = pd.concat([self.df, live], ignore_index=True)
        return live

    def insight_live(self, query: str, n: int = 30) -> str:
        recent = self.live_search(query, n=n)

        if not recent.empty:
            context = "\n".join(recent["clean"].tolist()[:30])
            fuente = "tweets en vivo"
        else:
            context = "\n".join(self.db.query(query, k=30))
            fuente = "datos históricos"

        prompt = f"""
Con base en el siguiente contexto extraído de {fuente}, responde brevemente.
Contexto:
{context}

Pregunta: {query}
Responde en español, indicando tendencias o tickers si aplica.
        """.strip()

        response = openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
