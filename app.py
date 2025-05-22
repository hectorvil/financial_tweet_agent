import os
import streamlit as st
import pandas as pd
from src.agent import FinancialTweetAgent
from src.plotting import build_sentiment_bar

# ── Configuración inicial ───────────────────────────────────────────────────
st.set_page_config(page_title="Financial-Tweet Agent", layout="wide")
st.title("Financial-Tweet Agent")

# ── Carga de claves desde entorno o st.secrets ─────────────────────────────
openai_key = os.getenv("OPENAI_API_KEY")
twitter_key = os.getenv("TWITTER_BEARER")

if not openai_key:
    st.warning("OPENAI_API_KEY no configurada. GPT no responderá preguntas.")
if not twitter_key:
    st.warning("TWITTER_BEARER no configurada. La búsqueda en vivo no funcionará.")

# ── Inicializar agente si no existe ────────────────────────────────────────
if "agent" not in st.session_state:
    st.session_state.agent = FinancialTweetAgent()

agent = st.session_state.agent

# ── Sidebar: carga de archivo parquet ──────────────────────────────────────
st.sidebar.header("Cargar archivo")
parquet_file = st.sidebar.file_uploader("Sube un archivo .parquet", type="parquet")

if parquet_file:
    st.sidebar.success("✅ Archivo subido")
    with st.spinner("🧠 Procesando: limpiando, clasificando, generando embeddings..."):
        agent.ingest(parquet_file)
elif not agent.df.empty:
    st.sidebar.info("Usando dataset ya cargado en memoria")
else:
    demo_path = "data/tweets_fin_2024.parquet"
    if os.path.exists(demo_path):
        agent.ingest(demo_path)
        st.sidebar.success("Dataset de demo cargado automáticamente")
    else:
        st.stop()

# ── Tabs: interfaz principal ───────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🤖 Chat histórico", "⚡ Live", "📊 Dashboard"])

# ── Tab 1: Chat histórico ──────────────────────────────────────────────────
with tab1:
    st.subheader("Haz una pregunta sobre el corpus histórico")
    question = st.text_input("Pregunta (ej. ¿Qué se dice de NVIDIA?)")
    if question and openai_key:
        with st.spinner("🧠 Pensando..."):
            answer = agent.insight_hist(question)
            st.success(answer)
    elif question:
        st.error("GPT está deshabilitado. Configura tu clave.")

# ── Tab 2: Live Search ─────────────────────────────────────────────────────
with tab2:
    st.subheader("🔍 Buscar tweets en vivo")
    live_query = st.text_input("Consulta live (ej. TSLA OR NVDA)")
    if live_query and twitter_key:
        with st.spinner("Buscando tweets recientes..."):
            live_df = agent.live_search(live_query)
        if not live_df.empty:
            st.write("Resultados en vivo:")
            st.dataframe(live_df[["text", "topic", "sentiment"]])
        else:
            st.info("No se encontraron tweets en vivo.")
    elif live_query:
        st.error("No tienes TWITTER_BEARER configurado.")

# ── Tab 3: Dashboard ───────────────────────────────────────────────────────
with tab3:
    st.subheader("Análisis de sentimiento por ticker")
    min_m = st.slider("Mínimo de menciones por ticker", 10, 300, 50, 10)
    metric = st.selectbox("Métrica a mostrar", ["neg_ratio", "pos_ratio", "total"])
    piv = agent.pivot(min_m)
    if not piv.empty:
        chart = build_sentiment_bar(piv, metric)
        st.plotly_chart(chart, use_container_width=True)
    else:
        st.warning("No hay suficientes datos para mostrar.")
