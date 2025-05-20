import os
import tweepy

# ── Cliente autenticado con token del entorno ──
client = tweepy.Client(
    bearer_token=os.getenv("TWITTER_BEARER"),
    wait_on_rate_limit=True
)

def search(query: str, n: int = 30) -> list[dict]:
    """
    Busca tweets recientes que coincidan con `query` (máx n).
    Retorna una lista de dicts con: doc_id, text, created_at.
    Solo devuelve tweets en inglés.
    """
    try:
        resp = client.search_recent_tweets(
            query=query,
            tweet_fields=["id", "text", "created_at", "lang"],
            max_results=min(n, 100)
        )
    except Exception as e:
        print(f"[Twitter API error] {e}")
        return []

    tweets = []
    for t in resp.data or []:
        if t.lang == "en":
            tweets.append({
                "doc_id": str(t.id),
                "text": t.text,
                "created_at": t.created_at
            })
    return tweets
