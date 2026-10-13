"""
LLM-based relation extraction for co-mentioned ticker pairs.

Relation types: causal-driver, substitute-comparison, co-hype, unrelated-mention.
Uses Anthropic's API if ANTHROPIC_API_KEY is set; otherwise falls back to a
deterministic mock classifier (keyword heuristics) so the rest of the
pipeline (graph construction + evaluation) can be developed/tested without a
live key or approved social-media API access.
"""
import itertools
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ALL_TICKERS, RAW_DIR, PROCESSED_DIR

RELATION_TYPES = ["causal-driver", "substitute-comparison", "co-hype", "unrelated-mention"]
TICKER_PATTERN = re.compile(r"\$?\b(" + "|".join(ALL_TICKERS) + r")\b")

PROMPT_TEMPLATE = """You are classifying the relationship asserted between two \
stock tickers in a social media post about financial markets.

Post: "{text}"
Ticker pair: {ticker_a} and {ticker_b}

Classify the relation as exactly one of: causal-driver, substitute-comparison, \
co-hype, unrelated-mention.
Also give a sentiment_alignment score from -1 (opposite sentiment) to 1 \
(same/aligned sentiment).

Respond with JSON only: {{"relation": "...", "sentiment_alignment": 0.0}}"""


def _mock_classify(text: str, ticker_a: str, ticker_b: str) -> dict:
    """Deterministic keyword heuristic, used when no LLM key is configured."""
    lowered = text.lower()
    causal_kw = ["drives", "driven by", "demand for", "because of", "due to", "supplier"]
    compare_kw = ["vs", "versus", "or", "better than", "instead of", "compared to"]
    if any(k in lowered for k in causal_kw):
        relation = "causal-driver"
        sentiment = 0.5
    elif any(k in lowered for k in compare_kw):
        relation = "substitute-comparison"
        sentiment = 0.0
    elif any(k in lowered for k in ["moon", "rocket", "bull", "hype", "🚀"]):
        relation = "co-hype"
        sentiment = 1.0
    else:
        relation = "unrelated-mention"
        sentiment = 0.0
    return {"relation": relation, "sentiment_alignment": sentiment}


def _llm_classify(client, text: str, ticker_a: str, ticker_b: str) -> dict:
    prompt = PROMPT_TEMPLATE.format(text=text, ticker_a=ticker_a, ticker_b=ticker_b)
    resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return _mock_classify(text, ticker_a, ticker_b)
    if parsed.get("relation") not in RELATION_TYPES:
        return _mock_classify(text, ticker_a, ticker_b)
    return parsed


def extract_pair_relations(posts: list[dict]) -> list[dict]:
    """posts: [{"text": str, "created_at": str, ...}] -> list of pairwise relation records."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = None
    if api_key:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

    records = []
    for post in posts:
        text = post.get("text") or f"{post.get('title', '')} {post.get('selftext', '')}"
        # GDELT posts arrive pre-tagged with ticker symbols (article text says
        # "Nvidia", not "NVDA", so the regex below can't find them there).
        # Reddit/StockTwits posts aren't pre-tagged, so fall back to regex.
        mentioned = sorted(set(post["tickers"])) if post.get("tickers") else sorted(set(TICKER_PATTERN.findall(text)))
        if len(mentioned) < 2:
            continue
        for ticker_a, ticker_b in itertools.combinations(mentioned, 2):
            result = (
                _llm_classify(client, text, ticker_a, ticker_b)
                if client
                else _mock_classify(text, ticker_a, ticker_b)
            )
            records.append(
                {
                    "ticker_a": ticker_a,
                    "ticker_b": ticker_b,
                    "relation": result["relation"],
                    "sentiment_alignment": result["sentiment_alignment"],
                    "timestamp": post.get("created_at") or post.get("timestamp"),
                    "source_text": text,
                }
            )
    return records


SOURCE_CANDIDATES = ["gdelt_posts.jsonl", "reddit_posts.jsonl", "stocktwits_messages.jsonl"]


def _pick_source_path() -> str:
    for name in SOURCE_CANDIDATES:
        path = os.path.join(RAW_DIR, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"None of {SOURCE_CANDIDATES} found under {RAW_DIR}. "
        "Run fetch_gdelt.py + fetch_gdelt_articles.py (recommended, no auth needed), "
        "or fetch_reddit.py / fetch_stocktwits.py (fallback dataset mode) first."
    )


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    raw_path = _pick_source_path()
    print(f"[info] extracting relations from {raw_path}")
    posts = [json.loads(l) for l in open(raw_path) if l.strip()]
    records = extract_pair_relations(posts)
    out_path = os.path.join(PROCESSED_DIR, "pair_relations.jsonl")
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(records)} pairwise relation records -> {out_path}")


if __name__ == "__main__":
    main()
