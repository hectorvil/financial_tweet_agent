import re
import emoji
import pandas as pd
import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Configuración del modelo FinBERT ──
FINBERT_MODEL = "ProsusAI/finbert"
_tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
_model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
_model.eval()
id2label = {0: "negative", 1: "neutral", 2: "positive"}

# ── Carga del clasificador de temas ──
topic_clf = joblib.load("src/topic_clf.joblib")  # entrenado previamente
label_map = [
    "Analyst Update","Bank","Buyback","Dividend","ECB","Federal Reserve",
    "Financials","Forecast","General News","Gold","IPO","Market Commentary",
    "Mergers & Acquisitions","Oil","Politics","Quarterly Results",
    "Stock Movement","Tech","Trade","USD"
]

def clean(text: str) -> str:
    """Limpia texto: emojis, URLs, menciones, hashtags, espacios extra."""
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"http\S+|@\w+|#\w+", "", text)
    return re.sub(r"\s+", " ", text).strip()

def finbert_sentiment(texts: list[str]) -> list[str]:
    """Clasifica sentimiento con FinBERT (batch)."""
    tokens = _tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = _model(**tokens).logits
    preds = torch.argmax(logits, dim=1).tolist()
    return [id2label[p] for p in preds]

def extract_tickers(text: str) -> list[str]:
    """Detecta posibles tickers en mayúsculas (ej. TSLA, NVDA)."""
    return re.findall(r"\b[A-Z]{2,5}\b", text)

def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia y etiqueta sentimiento, tema y tickers."""
    df["clean"] = df["text"].map(clean)
    df["sentiment"] = finbert_sentiment(df["clean"].tolist())
    df["topic"] = topic_clf.predict(df["clean"])
    df["tickers"] = df["clean"].map(extract_tickers)
    return df
