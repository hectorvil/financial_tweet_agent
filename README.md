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

financial-tweet-agent/
├── app.py
├── requirements.txt
├── .env.example
├── data/
│ └── tweets_fin_2024.parquet
├── src/
│ ├── agent.py
│ ├── data_pipeline.py
│ ├── twitter_live.py
│ ├── vector_db.py
│ └── plotting.py


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
