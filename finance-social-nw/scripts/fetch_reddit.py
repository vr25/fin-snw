"""
Pull ticker-mentioning posts/comments from finance subreddits (r/wallstreetbets,
r/stocks) via PRAW.

STATUS (2026-09-27): Reddit's self-service OAuth registration is closed under
the "Responsible Builder Policy". New apps require manual approval (2-4 week
turnaround, not guaranteed). Free tier once approved: ~100 queries/min per
OAuth client, non-commercial/academic use only. This script raises
RuntimeError until REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET are set. Use
`load_fallback_dataset()` in the meantime.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ALL_TICKERS,
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT,
    RAW_DIR,
    USE_FALLBACK_DATASET,
)

SUBREDDITS = ["wallstreetbets", "stocks", "investing"]
TICKER_PATTERN = re.compile(r"\$?\b(" + "|".join(ALL_TICKERS) + r")\b")


def fetch_via_praw(limit_per_subreddit: int = 500) -> list[dict]:
    import praw  # imported lazily so the module can be inspected without the dep installed

    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )
    posts = []
    for sub_name in SUBREDDITS:
        subreddit = reddit.subreddit(sub_name)
        for submission in subreddit.new(limit=limit_per_subreddit):
            text = f"{submission.title}\n{submission.selftext}"
            mentions = sorted(set(TICKER_PATTERN.findall(text)))
            if len(mentions) >= 2:
                posts.append(
                    {
                        "id": submission.id,
                        "subreddit": sub_name,
                        "created_utc": submission.created_utc,
                        "text": text,
                        "tickers": mentions,
                    }
                )
    return posts


def load_fallback_dataset() -> list[dict]:
    fallback_dir = os.path.join(RAW_DIR, "reddit_fallback")
    if not os.path.isdir(fallback_dir):
        raise FileNotFoundError(
            f"No fallback dataset found at {fallback_dir}. "
            "Place a pre-scraped Reddit finance corpus (e.g. from an academic "
            "archive or HuggingFace dataset) there, one JSON-lines file per "
            "source, before running this script in fallback mode."
        )
    posts = []
    for fname in os.listdir(fallback_dir):
        if fname.endswith(".jsonl"):
            with open(os.path.join(fallback_dir, fname)) as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    text = f"{rec.get('title', '')}\n{rec.get('selftext', rec.get('text', ''))}"
                    mentions = sorted(set(TICKER_PATTERN.findall(text)))
                    if len(mentions) >= 2:
                        rec["tickers"] = mentions
                        posts.append(rec)
    return posts


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    if USE_FALLBACK_DATASET:
        print("[info] Reddit credentials not set — using fallback dataset mode.")
        posts = load_fallback_dataset()
    else:
        posts = fetch_via_praw()

    out_path = os.path.join(RAW_DIR, "reddit_posts.jsonl")
    with open(out_path, "w") as f:
        for p in posts:
            f.write(json.dumps(p) + "\n")
    print(f"Wrote {len(posts)} multi-ticker posts -> {out_path}")


if __name__ == "__main__":
    main()
