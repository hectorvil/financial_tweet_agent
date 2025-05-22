# 📊 Financial-Tweet Agent

Un agente Streamlit interactivo que analiza y clasifica tweets financieros por **tema** y **sentimiento**, permite consultas históricas y en vivo, y visualiza el comportamiento del mercado en un dashboard simple.

---

## ¿Qué hace?

- **Etiquetado automático** de tweets financieros mediante FinBert (sentimiento: `positive`, `neutral`, `negative`. Y también etiqueta por tema en 21 categorías como: `Politics`, `IPO`, `Federal Reserve`, `Gold`, etc.)
- **Chat RAG histórico**: puedes preguntar cosas como *"¿Qué se dice sobre NVIDIA?"*
- **Consulta en vivo**: busca en Twitter (X) y te resume lo más reciente
- **Dashboard visual**: muestra sentimiento por ticker (ej. `AAPL`, `TSLA`, `AMZN`, `MSFT`)

---


## 🔄 Flujo de extremo a extremo

| Etapa | Qué ocurre | Detalles técnicos |
|-------|------------|-------------------|
| **1. Upload / Live fetch** | Ingesta de tweets en bruto (archivo **.parquet** histórico o stream desde la **Twitter API**). | — |
| **2. Data pipeline** | Limpieza → etiquetado → embeddings. | **`clean()`** elimina URLs, menciones y emojis. <br> **FinBERT** (`ProsusAI/finbert`, 110 M parámetros) asigna **positive / neutral / negative**. <br> **Topic classifier** (SVM + MPNet) mapea 20 temas fijos — Dividend, Fed, M&A… <br> **Mini-LM** (`all-MiniLM-L6-v2`) produce un vector por tweet; se salta si la columna `embedding` ya existe. |
| **3. ChromaDB** | Persistencia y búsqueda vectorial. | Almacena `doc_id`, texto y embedding en un índice **HNSW** (*cosine*); responde k-NN en **< 20 ms**. |
| **4. RAG (Retrieval-Augmented Generation)** | Contexto + LLM. | 1) La pregunta se embebe con Mini-LM.<br>2) Chroma devuelve los 30 tweets más cercanos.<br>3) Se arma el prompt:<br><code>: ¿Qué se dice de NVIDIA?</code><br>4) GPT-4o-mini responde usando <i>solo</i> ese contexto. |
| **5. Dashboard** | Métricas de sentimiento. | `agent.pivot()` agrupa por **ticker** y **sentiment**, calcula `pos_ratio / neg_ratio`; **Plotly** renderiza el ranking interactivo. |



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
