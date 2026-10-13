"""
Pull ticker-tagged StockTwits messages for the ticker basket.

STATUS (2026-09-27): StockTwits is not currently accepting new API developer
registrations while it reviews its API/terms. This script will raise
RuntimeError until STOCKTWITS_ACCESS_TOKEN is set. Until then, use
`load_fallback_dataset()` to read a pre-scraped academic corpus placed under
data/raw/stocktwits_fallback/.

Live endpoint (once access is granted): GET /api/2/streams/symbol/{ticker}.json
Rate limits are account-tier dependent; confirm current limits in the
developer dashboard before scheduling bulk pulls.
"""
import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TICKER_BASKET, STOCKTWITS_ACCESS_TOKEN, RAW_DIR, USE_FALLBACK_DATASET

API_BASE = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
REQUEST_DELAY_SEC = 1.5  # conservative default; tighten once real limits are confirmed


def fetch_symbol_stream(ticker: str, access_token: str, max_id: int | None = None) -> dict:
    params = {"access_token": access_token}
    if max_id:
        params["max"] = max_id
    resp = requests.get(API_BASE.format(ticker=ticker), params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_all(tickers: list[str], access_token: str) -> list[dict]:
    all_messages = []
    for ticker in tickers:
        try:
            payload = fetch_symbol_stream(ticker, access_token)
            all_messages.extend(payload.get("messages", []))
        except requests.HTTPError as e:
            print(f"[warn] StockTwits fetch failed for {ticker}: {e}")
        time.sleep(REQUEST_DELAY_SEC)
    return all_messages


def load_fallback_dataset() -> list[dict]:
    fallback_dir = os.path.join(RAW_DIR, "stocktwits_fallback")
    if not os.path.isdir(fallback_dir):
        raise FileNotFoundError(
            f"No fallback dataset found at {fallback_dir}. "
            "Place a pre-scraped StockTwits/finance-social corpus (e.g. from "
            "HuggingFace or Kaggle) there, one JSON-lines file per source, "
            "before running this script in fallback mode."
        )
    messages = []
    for fname in os.listdir(fallback_dir):
        if fname.endswith(".jsonl"):
            with open(os.path.join(fallback_dir, fname)) as f:
                messages.extend(json.loads(line) for line in f if line.strip())
    return messages


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    if USE_FALLBACK_DATASET:
        print("[info] STOCKTWITS_ACCESS_TOKEN not set — using fallback dataset mode.")
        messages = load_fallback_dataset()
    else:
        messages = fetch_all(TICKER_BASKET, STOCKTWITS_ACCESS_TOKEN)

    out_path = os.path.join(RAW_DIR, "stocktwits_messages.jsonl")
    with open(out_path, "w") as f:
        for m in messages:
            f.write(json.dumps(m) + "\n")
    print(f"Wrote {len(messages)} messages -> {out_path}")


if __name__ == "__main__":
    main()
