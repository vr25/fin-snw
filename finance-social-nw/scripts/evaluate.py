"""
Core ablation: does window-w graph edge weight predict window-(w+1) realized
pairwise return co-movement, and does the narrative graph beat the three
baselines (co-mention, price-correlation, sentiment-co-occurrence)?

Residualizes returns against the equal-weighted basket factor before computing
co-movement, so the target is idiosyncratic pairwise co-movement rather than
shared market/sector beta (see paper.tex, Discussion section, for the
motivating confound).
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RAW_DIR, PROCESSED_DIR
from build_graph import (
    build_narrative_graphs,
    build_comention_graphs,
    build_sentiment_coocc_graphs,
    build_price_correlation_graphs,
)


def _residualized_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Subtract the cross-sectional (equal-weighted basket) mean return per day."""
    prices = prices.copy()
    prices["Date"] = pd.to_datetime(prices["Date"], utc=True)
    pivot = prices.pivot_table(index="Date", columns="ticker", values="return")
    market_factor = pivot.mean(axis=1)
    residual = pivot.sub(market_factor, axis=0)
    return residual


def future_comovement_by_window(prices: pd.DataFrame) -> dict:
    """window -> pairwise Pearson correlation matrix of residualized returns."""
    residual = _residualized_returns(prices)
    residual["window"] = residual.index.to_series().dt.tz_convert(None).dt.to_period("M")
    out = {}
    for w, group in residual.groupby("window"):
        corr = group.drop(columns="window").corr()
        out[w] = corr
    return out


def _edge_weight(G, a, b):
    if G.has_edge(a, b):
        return G[a][b].get("weight", 0.0)
    return 0.0


def build_evaluation_frame(graph_variants: dict, future_comovement: dict, tickers: list) -> pd.DataFrame:
    """graph_variants: {variant_name: {window: nx.Graph}}. Aligns window w graph
    weight against window (w+1) realized co-movement, pooled across all windows
    and ticker pairs."""
    rows = []
    all_windows = sorted(future_comovement.keys())
    window_to_idx = {w: i for i, w in enumerate(all_windows)}

    for variant_name, graphs in graph_variants.items():
        for w, G in graphs.items():
            idx = window_to_idx.get(w)
            if idx is None or idx + 1 >= len(all_windows):
                continue
            next_w = all_windows[idx + 1]
            corr_next = future_comovement[next_w]
            for a, b in itertools.combinations(sorted(tickers), 2):
                if a not in corr_next.columns or b not in corr_next.columns:
                    continue
                target = corr_next.loc[a, b]
                if pd.isna(target):
                    continue
                rows.append(
                    {
                        "variant": variant_name,
                        "window": str(w),
                        "ticker_a": a,
                        "ticker_b": b,
                        "edge_weight": _edge_weight(G, a, b),
                        "future_comovement": target,
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    median = df.groupby("variant")["future_comovement"].transform("median")
    df["high_comovement"] = (df["future_comovement"] > median).astype(int)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for variant, group in df.groupby("variant"):
        if group["edge_weight"].nunique() < 2 or group["high_comovement"].nunique() < 2:
            rho, pval, auc = float("nan"), float("nan"), float("nan")
        else:
            rho, pval = spearmanr(group["edge_weight"], group["future_comovement"])
            auc = roc_auc_score(group["high_comovement"], group["edge_weight"])
        results.append({"variant": variant, "n": len(group), "spearman_rho": rho,
                         "spearman_p": pval, "auc": auc})
    return pd.DataFrame(results).sort_values("variant")


def main():
    rel_path = os.path.join(PROCESSED_DIR, "pair_relations.jsonl")
    price_path = os.path.join(RAW_DIR, "prices.parquet")
    if not os.path.exists(rel_path) or not os.path.exists(price_path):
        raise FileNotFoundError(
            "Missing pair_relations.jsonl or prices.parquet — run extract_relations.py "
            "and fetch_prices.py first."
        )

    import json
    records = [json.loads(l) for l in open(rel_path) if l.strip()]
    prices = pd.read_parquet(price_path)
    tickers = sorted(prices["ticker"].unique())

    graph_variants = {
        "narrative": build_narrative_graphs(records),
        "co_mention": build_comention_graphs(records),
        "sentiment_coocc": build_sentiment_coocc_graphs(records),
        "price_correlation": build_price_correlation_graphs(prices),
    }
    future_comovement = future_comovement_by_window(prices)

    df = build_evaluation_frame(graph_variants, future_comovement, tickers)
    if df.empty:
        print("[warn] No overlapping (window, window+1) pairs to evaluate. "
              "Need relation data spanning >= 2 consecutive monthly windows.")
        return

    summary = summarize(df)
    print(summary.to_string(index=False))
    out_path = os.path.join(PROCESSED_DIR, "ablation_results.csv")
    summary.to_csv(out_path, index=False)
    print(f"\nWrote ablation table -> {out_path}")


if __name__ == "__main__":
    main()
