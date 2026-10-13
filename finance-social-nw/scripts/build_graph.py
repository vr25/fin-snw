"""
Build the narrative graph and its three baselines (co-mention, price-correlation,
sentiment-co-occurrence) per monthly window, from pairwise relation records and
price data.
"""
import json
import os
import sys
from collections import defaultdict

import networkx as nx
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RAW_DIR, PROCESSED_DIR, WINDOW_FREQ

RELATION_WEIGHTS = {
    "causal-driver": 1.0,
    "substitute-comparison": 0.6,
    "co-hype": 0.4,
    "unrelated-mention": 0.0,
}


def _window_key(ts, freq=WINDOW_FREQ):
    return pd.to_datetime(ts, utc=True, errors="coerce").to_period("M" if freq == "MS" else freq)


def build_narrative_graphs(records: list[dict]) -> dict:
    """Returns {window: nx.Graph} with edge attrs: weight, per-type counts."""
    by_window = defaultdict(lambda: defaultdict(lambda: {"weight": 0.0, "count": 0, "sentiment_sum": 0.0}))
    for r in records:
        w = _window_key(r["timestamp"])
        if pd.isna(w):
            continue
        key = tuple(sorted((r["ticker_a"], r["ticker_b"])))
        edge = by_window[w][key]
        edge["weight"] += RELATION_WEIGHTS.get(r["relation"], 0.0)
        edge["count"] += 1
        edge["sentiment_sum"] += r.get("sentiment_alignment", 0.0)

    graphs = {}
    for w, edges in by_window.items():
        G = nx.Graph()
        for (a, b), attrs in edges.items():
            G.add_edge(a, b, weight=attrs["weight"], count=attrs["count"],
                       sentiment_avg=attrs["sentiment_sum"] / attrs["count"])
        graphs[w] = G
    return graphs


def build_comention_graphs(records: list[dict]) -> dict:
    """Baseline 1: raw co-occurrence frequency, ignoring relation type/sentiment."""
    by_window = defaultdict(lambda: defaultdict(int))
    for r in records:
        w = _window_key(r["timestamp"])
        if pd.isna(w):
            continue
        key = tuple(sorted((r["ticker_a"], r["ticker_b"])))
        by_window[w][key] += 1

    graphs = {}
    for w, edges in by_window.items():
        G = nx.Graph()
        for (a, b), count in edges.items():
            G.add_edge(a, b, weight=float(count))
        graphs[w] = G
    return graphs


def build_sentiment_coocc_graphs(records: list[dict]) -> dict:
    """Baseline 3: co-occurrence frequency scaled by sentiment alignment (no relation typing)."""
    by_window = defaultdict(lambda: defaultdict(lambda: {"count": 0, "sentiment_sum": 0.0}))
    for r in records:
        w = _window_key(r["timestamp"])
        if pd.isna(w):
            continue
        key = tuple(sorted((r["ticker_a"], r["ticker_b"])))
        edge = by_window[w][key]
        edge["count"] += 1
        edge["sentiment_sum"] += r.get("sentiment_alignment", 0.0)

    graphs = {}
    for w, edges in by_window.items():
        G = nx.Graph()
        for (a, b), attrs in edges.items():
            weight = attrs["count"] * max(attrs["sentiment_sum"] / attrs["count"], 0.0)
            G.add_edge(a, b, weight=weight)
        graphs[w] = G
    return graphs


def build_price_correlation_graphs(prices: pd.DataFrame) -> dict:
    """Baseline 2: Pearson correlation of daily returns within each window."""
    prices = prices.copy()
    prices["Date"] = pd.to_datetime(prices["Date"], utc=True)
    prices["window"] = prices["Date"].dt.to_period("M")
    graphs = {}
    for w, group in prices.groupby("window"):
        pivot = group.pivot_table(index="Date", columns="ticker", values="return")
        corr = pivot.corr()
        G = nx.Graph()
        tickers = corr.columns.tolist()
        for i, a in enumerate(tickers):
            for b in tickers[i + 1:]:
                val = corr.loc[a, b]
                if pd.notna(val):
                    G.add_edge(a, b, weight=float(val))
        graphs[w] = G
    return graphs


def main():
    rel_path = os.path.join(PROCESSED_DIR, "pair_relations.jsonl")
    price_path = os.path.join(RAW_DIR, "prices.parquet")
    if not os.path.exists(rel_path):
        raise FileNotFoundError(f"{rel_path} not found. Run extract_relations.py first.")
    records = [json.loads(l) for l in open(rel_path) if l.strip()]

    narrative = build_narrative_graphs(records)
    comention = build_comention_graphs(records)
    sentiment = build_sentiment_coocc_graphs(records)

    print(f"Built narrative graphs for {len(narrative)} windows")
    print(f"Built co-mention baseline graphs for {len(comention)} windows")
    print(f"Built sentiment-co-occurrence baseline graphs for {len(sentiment)} windows")

    if os.path.exists(price_path):
        prices = pd.read_parquet(price_path)
        price_corr = build_price_correlation_graphs(prices)
        print(f"Built price-correlation baseline graphs for {len(price_corr)} windows")
    else:
        print(f"[warn] {price_path} not found — skipping price-correlation baseline. Run fetch_prices.py first.")

    return narrative, comention, sentiment


if __name__ == "__main__":
    main()
