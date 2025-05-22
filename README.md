# 📊 Financial-Tweet Agent

Un agente Streamlit interactivo que analiza y clasifica tweets financieros por **tema** y **sentimiento**, permite consultas históricas y en vivo, y visualiza el comportamiento del mercado en un dashboard simple.

---

## ¿Qué hace?

- **Etiquetado automático** de tweets financieros (sentimiento: `positive`, `neutral`, `negative`; tema: 21 categorías como `Politics`, `IPO`, `Federal Reserve`, etc.)
- **Chat RAG histórico**: puedes preguntar cosas como *"¿Qué se dice sobre NVIDIA?"*
- **Consulta en vivo**: busca en Twitter (X) y te resume lo más reciente
- **Dashboard visual**: muestra sentimiento por ticker (ej. `AAPL`, `TSLA`)

---

## Estructura

Upload / Live fetch

Tweets entran en bruto desde un .parquet o la Twitter API.

Data pipeline

clean() quita URLs, menciones y emojis.

FinBERT → asigna positive / neutral / negative.
FinBERT es un bert-base-uncased afinado por ProsusAI en earnings calls y news headlines. Tiene 110 M parámetros y entiende terminología financiera (“hawkish”, “buyback”).

Topic classifier → SVM lineal entrenada en embeddings MPNet, 20 etiquetas fijas (Dividend, Federal Reserve, M&A, …).

Mini-LM embeddings → vector ℝ^384 para cada tweet; solo se calcula si la columna embedding no existe.

ChromaDB

Guarda doc_id, texto y embedding en un índice HNSW (cosine).

Responde k-NN en < 20 ms.

RAG (Retrieval-Augmented Generation)

La pregunta del usuario se embebe con Mini-LM → se consulta Chroma → se recuperan 30 tweets relevantes.

Se construye un prompt:

makefile
Copy
Edit
Contexto:
• 17-May NVDA beats estimates…
• …
Pregunta: ¿Qué se dice de NVIDIA?
GPT-4o-mini sintetiza la respuesta usando solo ese contexto.

Dashboard

agent.pivot() agrupa por tickers y sentiment, calcula pos_ratio / neg_ratio y Plotly dibuja el ranking


---

## Cómo correrlo localmente

1. Clona el repo
2. Instala dependencias
3. Añade tus claves a `.env`

```bash
git clone https://github.com/tu_usuario/financial-tweet-agent.git
cd financial-tweet-agent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # luego edítalo con tus claves reales
streamlit run app.py
