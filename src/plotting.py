import pandas as pd
import plotly.express as px


def build_sentiment_bar(pivot_df: pd.DataFrame, metric: str):
    """Gráfico de barras limitado a los 30 (o menos) principales tickers."""
    limit = min(30, len(pivot_df))
    fig = px.bar(
        pivot_df.sort_values(metric, ascending=False).head(limit),
        x="tickers",
        y=metric,
        text_auto=".2f" if "ratio" in metric else True,
        height=450,
        template="plotly_white",
    )
    fig.update_layout(
        xaxis_title="Ticker",
        yaxis_title=metric.replace("_", " ").title(),
        xaxis_tickangle=-45,
    )
    return fig
