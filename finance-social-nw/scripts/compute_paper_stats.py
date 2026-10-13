"""Recompute every quantitative claim in the paper from raw data.

Writes analysis/metrics/paper_stats.json (single source of truth for paper numbers).
Usage: python scripts/compute_paper_stats.py
"""
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REL = ROOT / "data/processed/pair_relations.jsonl"
POSTS = ROOT / "data/raw/gdelt_posts.jsonl"
PRICES = ROOT / "data/raw/prices.parquet"
OUT = ROOT / "analysis/metrics/paper_stats.json"

LAUNCH, SPLIT = "2022-11-30", "2023-05-01"
SECTORS = {
    "GPU": ["NVDA", "AMD", "AVGO", "SMCI"],
    "Cloud": ["MSFT", "GOOGL", "META", "PLTR"],
    "Speculative": ["BBAI", "IONQ", "RGTI", "SOUN"],
    "Control": ["KO", "PG", "XOM"],
}
SEC_OF = {f: s for s, fs in SECTORS.items() for f in fs}
TYPES = ["causal-driver", "substitute-comparison", "co-hype", "unrelated-mention"]


def r3(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 3)


def period_of(ts):
    return "early" if ts[:10] < SPLIT else "late"


def graph(rels):
    g = nx.Graph()
    g.add_nodes_from(SEC_OF)
    for r in rels:
        a, b = r["ticker_a"], r["ticker_b"]
        w = g[a][b]["weight"] + 1 if g.has_edge(a, b) else 1
        g.add_edge(a, b, weight=w)
    return g


def sector_metrics(g, firms):
    sub = g.subgraph(firms)
    n = len(firms)
    return {"edges": sub.number_of_edges(), "density": r3(nx.density(sub)),
            "avg_clustering": r3(nx.average_clustering(sub)) if n > 2 else None}


def pair_block(rels, s1, s2):
    c = Counter()
    for r in rels:
        sa, sb = SEC_OF[r["ticker_a"]], SEC_OF[r["ticker_b"]]
        if {sa, sb} == {s1, s2} and (s1 != s2 or sa == sb):
            c[r["relation"]] += 1
    return {t: c[t] for t in TYPES} | {"total": sum(c.values())}


def main():
    rels = [json.loads(l) for l in open(REL)]
    posts = [json.loads(l) for l in open(POSTS)]
    S = {}

    urls = [p.get("url") for p in posts]
    S["corpus"] = {
        "articles": len(posts), "unique_urls": len(set(u for u in urls if u)),
        "relations": len(rels),
        "first_relation": min(r["timestamp"] for r in rels)[:10],
        "last_relation": max(r["timestamp"] for r in rels)[:10],
        "tickers_in_relations": sorted({t for r in rels for t in (r["ticker_a"], r["ticker_b"])}),
        "post_keys": sorted(posts[0].keys()),
    }
    ment = Counter()
    for r in rels:
        ment[r["ticker_a"]] += 1
        ment[r["ticker_b"]] += 1
    S["mentions_by_ticker"] = dict(ment.most_common())

    per = {"early": [r for r in rels if period_of(r["timestamp"]) == "early"],
           "late": [r for r in rels if period_of(r["timestamp"]) == "late"],
           "pre_launch": [r for r in rels if r["timestamp"][:10] < LAUNCH]}
    S["relation_distribution"] = {"all": {t: sum(r["relation"] == t for r in rels) for t in TYPES}}
    for k, v in per.items():
        S["relation_distribution"][k] = {t: sum(r["relation"] == t for r in v) for t in TYPES} | {"total": len(v)}

    def sent(v):
        x = np.array([r["sentiment_alignment"] for r in v], dtype=float)
        return {"n": len(x), "mean": r3(x.mean()), "std": r3(x.std(ddof=1)), "median": r3(np.median(x))} if len(x) else None

    S["sentiment"] = {k: sent(per[k]) for k in ("early", "late")}
    S["sentiment_by_relation"] = {t: sent([r for r in rels if r["relation"] == t]) for t in TYPES}
    S["sentiment_by_sector_pair"] = {}
    for s1, s2 in [("GPU", "GPU"), ("Cloud", "Cloud"), ("GPU", "Cloud"), ("GPU", "Speculative"), ("Cloud", "Speculative")]:
        for k in ("early", "late"):
            v = [r for r in per[k] if {SEC_OF[r["ticker_a"]], SEC_OF[r["ticker_b"]]} == {s1, s2}]
            S["sentiment_by_sector_pair"][f"{s1}-{s2}/{k}"] = sent(v)

    G = {k: graph(per[k]) for k in ("early", "late")}
    S["sector_network"] = {s: {k: sector_metrics(G[k], f) for k in G} for s, f in SECTORS.items()}
    S["full_network"] = {}
    for k, g in G.items():
        act = g.subgraph([n for n in g if g.degree(n) > 0])
        bc = nx.betweenness_centrality(act)
        S["full_network"][k] = {"active_nodes": act.number_of_nodes(), "edges": act.number_of_edges(),
                                "density_active": r3(nx.density(act)),
                                "betweenness": {n: r3(v) for n, v in sorted(bc.items(), key=lambda x: -x[1])},
                                "degree": dict(sorted(dict(act.degree()).items(), key=lambda x: -x[1]))}

    S["cross_sector_relations"] = {}
    for s1, s2 in [("GPU", "GPU"), ("Cloud", "Cloud"), ("GPU", "Cloud"), ("GPU", "Speculative"),
                   ("Cloud", "Speculative"), ("GPU", "Control"), ("Cloud", "Control")]:
        for k in ("early", "late"):
            S["cross_sector_relations"][f"{s1}-{s2}/{k}"] = pair_block(per[k], s1, s2)
        S["cross_sector_relations"][f"{s1}-{s2}/all"] = pair_block(rels, s1, s2)

    for k in ("early", "late"):
        b = S["cross_sector_relations"][f"GPU-Cloud/{k}"]
        S.setdefault("homophily_gpu_cloud", {})[k] = {
            "causal": b["causal-driver"], "substitute": b["substitute-comparison"],
            "H=causal/(sub+1)": r3(b["causal-driver"] / (b["substitute-comparison"] + 1))}

    pc = defaultdict(lambda: defaultdict(Counter))
    for r in rels:
        pc[tuple(sorted((r["ticker_a"], r["ticker_b"])))][period_of(r["timestamp"])][r["relation"]] += 1
    S["pairs"] = {f"{a}-{b}": {"early": sum(v["early"].values()), "late": sum(v["late"].values()),
                               "early_causal": v["early"]["causal-driver"], "late_causal": v["late"]["causal-driver"],
                               "late_sub": v["late"]["substitute-comparison"]}
                  for (a, b), v in sorted(pc.items(), key=lambda kv: -sum(sum(c.values()) for c in kv[1].values()))}

    # ---- prices
    p = pd.read_parquet(PRICES)
    p["Date"] = pd.to_datetime(p["Date"])
    ret = p.pivot(index="Date", columns="ticker", values="return").sort_index()
    close = p.pivot(index="Date", columns="ticker", values="close").sort_index()
    S["price_window"] = [str(ret.index.min().date()), str(ret.index.max().date()), int(len(ret))]
    S["total_return_pct"] = {t: r3((close[t].dropna().iloc[-1] / close[t].dropna().iloc[0] - 1) * 100) for t in close}
    d23 = close.loc[:"2023-12-31"]
    S["return_to_end2023_pct"] = {t: r3((d23[t].dropna().iloc[-1] / d23[t].dropna().iloc[0] - 1) * 100) for t in d23}

    def block_corr(C, s1, s2):
        a, b = SECTORS[s1], SECTORS[s2]
        vals = [C.loc[x, y] for x, y in combinations(a, 2)] if s1 == s2 else [C.loc[x, y] for x in a for y in b]
        return r3(np.nanmean(vals))

    blocks = [("GPU", "GPU"), ("Cloud", "Cloud"), ("Speculative", "Speculative"), ("Control", "Control"),
              ("GPU", "Cloud"), ("GPU", "Speculative"), ("Cloud", "Speculative"), ("GPU", "Control"),
              ("Cloud", "Control"), ("Speculative", "Control")]
    C = ret.corr()
    S["corr_full_period"] = {f"{a}-{b}": block_corr(C, a, b) for a, b in blocks}
    S["corr_by_period"] = {}
    for k, sl in {"early": ret.loc[:"2023-04-30"], "late": ret.loc["2023-05-01":]}.items():
        Ck = sl.corr()
        S["corr_by_period"][k] = {f"{a}-{b}": block_corr(Ck, a, b) for a, b in blocks}
    order = [f for s in SECTORS.values() for f in s]
    S["corr_matrix"] = {a: {b: r3(C.loc[a, b]) for b in order} for a in order}
    S["corr_log_returns"] = {}
    lr = np.log(close).diff()
    Cl = lr.corr()
    S["corr_log_returns"] = {f"{a}-{b}": block_corr(Cl, a, b) for a, b in blocks}
    S["corr_rolling_mean"] = {}
    for w in (30, 60, 90):
        mats = []
        for start in range(0, len(ret) - w + 1, w):
            mats.append(ret.iloc[start:start + w].corr())
        S["corr_rolling_mean"][str(w)] = {f"{a}-{b}": r3(np.nanmean([block_corr(m, a, b) for m in mats]))
                                          for a, b in blocks}

    # ---- matched 6-month windows (controls for unequal period lengths)
    starts = pd.date_range("2022-11-01", "2026-05-01", freq="6MS")
    S["six_month_windows"] = []
    for st in starts:
        en = st + pd.DateOffset(months=6)
        v = [r for r in rels if str(st.date()) <= r["timestamp"][:10] < str(en.date())]
        if not v:
            continue
        g = graph(v)
        gc = pair_block(v, "GPU", "Cloud")
        rsl = ret.loc[st:en - pd.Timedelta(days=1)]
        Cw = rsl.corr()
        S["six_month_windows"].append({
            "start": str(st.date()), "n": len(v),
            "causal_share": r3(sum(r["relation"] == "causal-driver" for r in v) / len(v)),
            "gpu_density": sector_metrics(g, SECTORS["GPU"])["density"],
            "cloud_density": sector_metrics(g, SECTORS["Cloud"])["density"],
            "gpu_cloud_causal": gc["causal-driver"], "gpu_cloud_sub": gc["substitute-comparison"],
            "gpu_cloud_total": gc["total"],
            "nvda_mentions": sum(r["ticker_a"] == "NVDA" or r["ticker_b"] == "NVDA" for r in v),
            "corr_gpu_cloud": block_corr(Cw, "GPU", "Cloud"), "corr_gpu": block_corr(Cw, "GPU", "GPU"),
        })

    # ---- pair-level: do narratively linked pairs co-move more? (late period)
    from scipy.stats import mannwhitneyu
    Cl8 = ret.loc[SPLIT:].corr()
    late_rel = per["late"]
    linked = {tuple(sorted((r["ticker_a"], r["ticker_b"]))) for r in late_rel}
    causal = {tuple(sorted((r["ticker_a"], r["ticker_b"]))) for r in late_rel if r["relation"] == "causal-driver"}
    allp = list(combinations(sorted(ret.columns), 2))
    grp = {"linked": [Cl8.loc[a, b] for a, b in allp if (a, b) in linked],
           "unlinked": [Cl8.loc[a, b] for a, b in allp if (a, b) not in linked],
           "causal": [Cl8.loc[a, b] for a, b in allp if (a, b) in causal],
           "linked_noncausal": [Cl8.loc[a, b] for a, b in allp if (a, b) in linked and (a, b) not in causal]}
    S["pair_level_late"] = {k: {"n": len(v), "mean_corr": r3(np.mean(v))} for k, v in grp.items()}
    S["pair_level_late"]["p_linked_vs_unlinked"] = float(mannwhitneyu(grp["linked"], grp["unlinked"]).pvalue)
    S["pair_level_late"]["p_causal_vs_linked_noncausal"] = float(
        mannwhitneyu(grp["causal"], grp["linked_noncausal"]).pvalue)
    # ---- sector control: sector-pair (block) fixed effects + within-block permutation
    import statsmodels.formula.api as smf
    D = pd.DataFrame([{"corr": Cl8.loc[a, b], "linked": int((a, b) in linked), "causal": int((a, b) in causal),
                       "block": "-".join(sorted([SEC_OF[a], SEC_OF[b]]))} for a, b in allp])
    m_raw = smf.ols("corr ~ linked", D).fit(cov_type="HC1")
    m_fe = smf.ols("corr ~ linked + block", D).fit(cov_type="HC1")
    DL = D[D.linked == 1].copy()
    m_c = smf.ols("corr ~ causal + block", DL).fit(cov_type="HC1")
    rng = np.random.default_rng(0)
    null = []
    for _ in range(5000):
        E = DL.copy()
        E["causal"] = E.groupby("block")["causal"].transform(lambda x: rng.permutation(x.values))
        null.append(smf.ols("corr ~ causal + block", E).fit().params["causal"])
    S["pair_level_sector_fe"] = {
        "linked_coef_raw": r3(m_raw.params["linked"]),
        "linked_coef_fe": r3(m_fe.params["linked"]), "linked_p_fe": float(m_fe.pvalues["linked"]),
        "r2_blocks_only": r3(smf.ols("corr ~ block", D).fit().rsquared),
        "n_pairs_linked_varies_within_block": int(sum(
            min(g.linked.sum(), len(g) - g.linked.sum()) for _, g in D.groupby("block"))),
        "causal_coef_fe": r3(m_c.params["causal"]), "causal_p_fe": float(m_c.pvalues["causal"]),
        "causal_perm_p": float((np.sum(np.array(null) >= m_c.params["causal"]) + 1) / 5001),
    }
    # ---- framing contrast: causal vs substitute, cross-sector vs within-sector (late period)
    from scipy.stats import chi2_contingency, fisher_exact
    csr = S["cross_sector_relations"]
    gc = [csr["GPU-Cloud/late"]["causal-driver"], csr["GPU-Cloud/late"]["substitute-comparison"]]
    cc = [csr["Cloud-Cloud/late"]["causal-driver"], csr["Cloud-Cloud/late"]["substitute-comparison"]]
    gg = [csr["GPU-GPU/late"]["causal-driver"], csr["GPU-GPU/late"]["substitute-comparison"]]
    chi2, p, _, _ = chi2_contingency([gc, cc])
    S["framing_tests_late"] = {
        "gpu_cloud_vs_cloud_cloud": {"chi2": r3(chi2), "p": float(p), "odds_ratio": r3((gc[0] / gc[1]) / (cc[0] / cc[1]))},
        "gpu_cloud_vs_gpu_gpu": {"p_fisher": float(fisher_exact([gc, gg])[1]),
                                 "odds_ratio": r3((gc[0] / gc[1]) / (gg[0] / gg[1]))},
    }
    abl = ROOT / "data/processed/ablation_results.csv"
    if abl.exists():
        S["ablation"] = pd.read_csv(abl).round(4).to_dict(orient="records")

    # ---- robustness: full GKG co-mention stream (29,690 hits, before article sampling)
    gkg = ROOT / "data/raw/gdelt_comentions_full.jsonl"
    if gkg.exists():
        sec_of = {t: s for s, ts in SECTORS.items() for t in ts}
        hits = [json.loads(l) for l in open(gkg) if l.strip()]

        def gper(d):
            return "pre" if d < "20221130" else ("early" if d < "20230501" else "late")
        G = {p: {"hits": 0, "nvda_hits": 0, "pairs": Counter()} for p in ["pre", "early", "late"]}
        W = {}
        for h in hits:
            d = h["gdelt_date"]
            ts = sorted(set(h["tickers"]))
            g = G[gper(d)]
            g["hits"] += 1
            g["nvda_hits"] += "NVDA" in ts
            idx = ((int(d[:4]) - 2022) * 12 + int(d[4:6]) - 11) // 6
            W.setdefault(idx, Counter())
            for a, b in combinations(ts, 2):
                k = "-".join(sorted([sec_of[a], sec_of[b]]))
                g["pairs"][k] += 1
                W[idx][k] += 1
        S["gkg_full"] = {"n_hits": len(hits)}
        for p, g in G.items():
            tot = sum(g["pairs"].values())
            S["gkg_full"][p] = {"hits": g["hits"], "nvda_share": r3(g["nvda_hits"] / g["hits"]),
                                "gpu_cloud_share": r3(g["pairs"].get("Cloud-GPU", 0) / tot),
                                "gpu_gpu_share": r3(g["pairs"].get("GPU-GPU", 0) / tot),
                                "spec_any_share": r3(sum(v for k, v in g["pairs"].items() if "Spec" in k) / tot)}
        S["gkg_full"]["windows_gpu_cloud_share"] = [
            {"window": i, "pairs": sum(W[i].values()), "gpu_cloud_share": r3(W[i]["Cloud-GPU"] / sum(W[i].values()))}
            for i in sorted(W)]

    OUT.write_text(json.dumps(S, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
