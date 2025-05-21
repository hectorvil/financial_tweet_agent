import re
import emoji
import pandas as pd
import torch
import joblib
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Mapeo de sentimiento ──
id2label = {0: "negative", 1: "neutral", 2: "positive"}

# ── Mapeo de temas ──
label_map = [
    "Analyst Update", "Bank", "Buyback", "Dividend", "ECB", "Federal Reserve",
    "Financials", "Forecast", "General News", "Gold", "IPO", "Market Commentary",
    "Mergers & Acquisitions", "Oil", "Politics", "Quarterly Results",
    "Stock Movement", "Tech", "Trade", "USD"
]

# ── Carga opcional del clasificador de temas ──
try:
    topic_clf = joblib.load("src/topic_clf.joblib")
except:
    topic_clf = None
    print("⚠️ No se encontró topic_clf.joblib. Solo se usarán etiquetas si están disponibles.")

# ── Carga segura de FinBERT (evita errores de event loop y Torch en Streamlit) ──
@st.cache_resource
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval()
    return tokenizer, model

# ── Limpieza de texto ──
def clean(text: str) -> str:
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"http\S+|@\w+|#\w+", "", text)
    return re.sub(r"\s+", " ", text).strip()

# ── Clasificación de sentimiento con FinBERT ──
def finbert_sentiment(texts: list[str]) -> list[str]:
    tokenizer, model = load_finbert()
    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**tokens).logits
    preds = torch.argmax(logits, dim=1).tolist()
    return [id2label[p] for p in preds]

# ── Detección de tickers ──
def extract_tickers(text: str) -> list[str]:
    return re.findall(r"\b[A-Z]{2,5}\b", text)

# ── Función principal: etiquetado completo ──
def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df["clean"] = df["text"].map(clean)
    df["sentiment"] = finbert_sentiment(df["clean"].tolist())
    df["tickers"] = df["clean"].map(extract_tickers)

    if "label" in df:
        # Usa columna 'label' numérica si está disponible
        df["topic"] = df["label"].map(lambda x: label_map[x] if 0 <= x < len(label_map) else "Unknown")
    elif topic_clf:
        # Si no hay label, pero hay clasificador, lo usa
        df["topic"] = topic_clf.predict(df["clean"])
    else:
        # Caso de fallback
        df["topic"] = "Unknown"

    return df
