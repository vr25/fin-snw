"""
Fetch article text for GDELT co-mention hits (from fetch_gdelt.py) and emit
them in the same posts-schema extract_relations.py already consumes:
{"text": ..., "created_at": ..., "tickers": [...], "source": "gdelt", "domain": ...}

Deduplicates by URL first (many co-mention hits can share a source URL across
GKG's 15-min slices re-indexing the same article). Skips articles that fail to
fetch or extract (paywalls, dead links, non-HTML) rather than erroring out —
real-world web text always has partial failures; we log the drop rate instead.
"""
import json
import os
import sys
import time

import requests
import trafilatura

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RAW_DIR

REQUEST_TIMEOUT_SEC = 15
REQUEST_DELAY_SEC = 0.3
USER_AGENT = "Mozilla/5.0 (compatible; narrative-graph-research/0.1; +mailto:research@example.com)"


def _gdelt_date_to_iso(gdelt_date: str) -> str:
    # GKG DATE field is YYYYMMDDHHMMSS
    return f"{gdelt_date[0:4]}-{gdelt_date[4:6]}-{gdelt_date[6:8]}T{gdelt_date[8:10]}:{gdelt_date[10:12]}:{gdelt_date[12:14]}Z"


def fetch_article_text(url: str) -> str | None:
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT_SEC, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
    except requests.RequestException:
        return None
    text = trafilatura.extract(resp.text, include_comments=False, include_tables=False)
    return text


def main():
    import sys as _sys

    hits_filename = _sys.argv[1] if len(_sys.argv) > 1 else "gdelt_comentions.jsonl"
    out_filename = _sys.argv[2] if len(_sys.argv) > 2 else "gdelt_posts.jsonl"
    hits_path = os.path.join(RAW_DIR, hits_filename)
    if not os.path.exists(hits_path):
        raise FileNotFoundError(f"{hits_path} not found. Run fetch_gdelt.py first.")

    hits = [json.loads(l) for l in open(hits_path) if l.strip()]

    by_url = {}
    for h in hits:
        by_url.setdefault(h["url"], []).append(h)
    print(f"[info] {len(hits)} co-mention hits across {len(by_url)} unique URLs", flush=True)

    out_path = os.path.join(RAW_DIR, out_filename)
    n_ok = 0
    n_failed = 0
    with open(out_path, "w") as out_f:
        for i, (url, url_hits) in enumerate(by_url.items()):
            text = fetch_article_text(url)
            if not text or len(text.strip()) < 50:
                n_failed += 1
            else:
                tickers = sorted(set(t for h in url_hits for t in h["tickers"]))
                post = {
                    "text": text,
                    "created_at": _gdelt_date_to_iso(url_hits[0]["gdelt_date"]),
                    "tickers": tickers,
                    "source": "gdelt",
                    "domain": url_hits[0].get("domain", ""),
                    "url": url,
                }
                out_f.write(json.dumps(post) + "\n")
                out_f.flush()
                n_ok += 1
            time.sleep(REQUEST_DELAY_SEC)
            if (i + 1) % 50 == 0:
                print(f"[info] fetched {i + 1}/{len(by_url)} URLs, {n_ok} ok, {n_failed} failed", flush=True)

    print(f"Wrote {n_ok} articles with text -> {out_path} ({n_failed} URLs failed/skipped)")


if __name__ == "__main__":
    main()
