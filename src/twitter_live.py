import os
import tweepy

# ── Lista de cuentas financieras ──
handles = [
    "Bloomberg","Reuters","FT","WSJ","CNBC","CNNBusiness","BBCBusiness","APBusiness",
    "MarketWatch","YahooFinance","Forbes","TheEconomist","WSJmarkets","markets",
    "Benzinga","Stocktwits","BreakoutStocks","bespokeinvest","nytimesbusiness","IBDinvestors",
    "WSJDealJournal","TheStreet","SeekingAlpha","Investingcom","MorningstarInc",
    "Investopedia","barronsonline","MorganStanley","GoldmanSachs","BlackRock","stlouisfed",
    "FortuneMagazine","BusinessInsider","ElFinanciero_Mx","eleconomista","ExpansionMx",
    "Forbes_Mexico","ReutersLatam","BloombergLinea","Banxico","BMVMercados","larepublica_co",
    "valoreconomico","AmericaEconomia","Gestionpe","Ambitocom","Cronistacom","DFinanciero",
    "IMFNews"
]

# ── Autenticación ──
client = tweepy.Client(
    bearer_token=os.getenv("TWITTER_BEARER"),
    wait_on_rate_limit=True
)

# ── Helper para construir consultas por bloques ──
def chunked_queries(handles_list, max_len=512) -> list[str]:
    """
    Divide la lista de handles en queries válidas para la API (≤ 512 caracteres)
    """
    chunks, cur = [], []
    for h in handles_list:
        test = cur + [h]
        query = "(" + " OR ".join(f"from:{x}" for x in test) + ") -is:retweet"
        if len(query) > max_len:
            chunks.append(cur)
            cur = [h]
        else:
            cur = test
    if cur:
        chunks.append(cur)
    return [ "(" + " OR ".join(f"from:{h}" for h in c) + ") -is:retweet" for c in chunks ]

# ── Función principal de búsqueda ──
def search(query: str = None, n: int = 30) -> list[dict]:
    """
    Busca tweets recientes. Si query es None, usa la lista de handles por defecto.
    Devuelve hasta `n` tweets en total (máximo).
    """
    tweets = []
    try:
        if query:
            # Consulta personalizada del usuario (ej: "TSLA OR NVDA")
            resp = client.search_recent_tweets(
                query=query + " -is:retweet",
                tweet_fields=["id", "text", "created_at", "lang"],
                max_results=min(n, 100)
            )
            tweets = resp.data or []
        else:
            # Consulta usando cuentas por defecto
            collected = []
            for q in chunked_queries(handles.copy()):
                resp = client.search_recent_tweets(
                    query=q,
                    tweet_fields=["id", "text", "created_at", "lang"],
                    max_results=100
                )
                collected.extend(resp.data or [])
                if len(collected) >= n:
                    break
            tweets = collected[:n]
    except Exception as e:
        print(f"[Twitter API error] {e}")
        return []

    return [{
        "doc_id": str(t.id),
        "text": t.text,
        "created_at": t.created_at
    } for t in tweets if t.lang == "en"]
