import pandas as pd
import plotly.express as px

def build_sentiment_bar(pivot_df: pd.DataFrame, metric: str):
    """Crea gráfico de barras por ticker y métrica seleccionada."""
    fig = px.bar(
        pivot_df.sort_values(metric, ascending=False).head(30),
        x="tickers",
        y=metric,
        text_auto=".2f" if "ratio" in metric else True,
        height=450,
        template="plotly_white"
    )
    fig.update_layout(
        xaxis_title="Ticker",
        yaxis_title=metric.replace("_", " ").title(),
        xaxis_tickangle=-45
    )
    return fig
