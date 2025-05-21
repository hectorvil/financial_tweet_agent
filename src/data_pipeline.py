import re
import emoji
from pathlib import Path

import pandas as pd
import torch
import joblib
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── mapeos ─────────────────────────────────────────────────────────
id2label = {0: "negative", 1: "neutral", 2: "positive"}

label_map = [
    "Analyst Update", "Bank", "Buyback", "Dividend", "ECB", "Federal Reserve",
    "Financials", "Forecast", "General News", "Gold", "IPO", "Market Commentary",
    "Mergers & Acquisitions", "Oil", "Politics", "Quarterly Results",
    "Stock Movement", "Tech", "Trade", "USD"
]

# ── carga de clasificador de temas ─────────────────────────────────
topic_path = Path(__file__).with_name("topic_clf.joblib")
try:
    topic_clf = joblib.load(topic_path)
except FileNotFoundError:
    topic_clf = None
    print(f"⚠️  No se encontró {topic_path.name}. Se omitirá la predicción de temas.")

# ── FinBERT cacheado ──────────────────────────────────────────────
@st.cache_resource
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval()
    return tokenizer, model

# ── limpieza y utilidades ─────────────────────────────────────────
def clean(text: str) -> str:
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"http\S+|@\w+|#\w+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def finbert_sentiment(texts: list[str], batch: int = 32) -> list[str]:
    tokenizer, model = load_finbert()
    preds = []
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        toks = tokenizer(chunk, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**toks).logits
        preds.extend(torch.argmax(logits, dim=1).tolist())
    return [id2label[p] for p in preds]


COMMON_WORDS = {"BANK", "GDP", "USA", "FED", "ECB", "USD"}


def extract_tickers(text: str) -> list[str]:
    tickers = [t.lstrip("$") for t in re.findall(r"\$?[A-Z]{2,5}\b", text)]
    return [t for t in tickers if t not in COMMON_WORDS]

# ── pipeline principal ────────────────────────────────────────────
def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["clean"] = df["text"].map(clean)

    # Sentimiento
    df["sentiment"] = finbert_sentiment(df["clean"].tolist())

    # Tickers
    df["tickers"] = df["clean"].map(extract_tickers)

    # Tema
    if "label" in df:
        df["topic"] = df["label"].map(
            lambda x: label_map[x] if 0 <= x < len(label_map) else "Unknown"
        )
    elif topic_clf:
        df["topic"] = topic_clf.predict(df["clean"])
    else:
        df["topic"] = "Unknown"

    return df
