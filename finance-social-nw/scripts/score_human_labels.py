"""Score completed human annotation sheets against the LLM runs.

Usage: python scripts/score_human_labels.py data/validation/human/annotation_sheet_*.csv
Reports inter-annotator kappa (if >=2 sheets), and each model's accuracy, kappa and per-class F1
against each human sheet and the human majority (ties excluded), overall and by stratum.
Writes analysis/metrics/human_agreement.json.
"""
import csv
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

from sklearn.metrics import cohen_kappa_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
TYPES = ["causal-driver", "substitute-comparison", "co-hype", "unrelated-mention"]
MODELS = ["final_haiku", "gpt_5_4_mini", "haiku_repeat", "gpt_6_sol"]
AGREEMENT_N = 240
SECTORS = {"GPU": ["NVDA", "AMD", "AVGO", "SMCI"], "Cloud": ["MSFT", "GOOGL", "META", "PLTR"],
           "Speculative": ["BBAI", "IONQ", "RGTI", "SOUN"], "Control": ["KO", "PG", "XOM"]}


def weighted_scores(gold, pred, ids, wt):
    n = sum(wt[i] for i in ids)
    po = sum(wt[i] for i in ids if gold[i] == pred[i]) / n
    pg = {t: sum(wt[i] for i in ids if gold[i] == t) / n for t in TYPES}
    pp = {t: sum(wt[i] for i in ids if pred[i] == t) / n for t in TYPES}
    pe = sum(pg[t] * pp[t] for t in TYPES)
    prec, rec = {}, {}
    for t in TYPES:
        tp = sum(wt[i] for i in ids if gold[i] == t and pred[i] == t)
        prec[t] = round(tp / (pp[t] * n), 3) if pp[t] else None
        rec[t] = round(tp / (pg[t] * n), 3) if pg[t] else None
    return {"acc": round(po, 3), "kappa": round((po - pe) / (1 - pe), 3), "precision": prec, "recall": rec}


def load_sheet(path):
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        lab = r["relation"].strip().lower()
        if lab in TYPES:
            out[r["item_id"].zfill(4)] = lab
        elif lab:
            print(f"  {Path(path).name}: row {r['row']} has invalid label {lab!r}")
    return out


def score(gold, pred, ids):
    ids = [i for i in ids if i in gold and i in pred]
    if not ids:
        return None
    g, p = [gold[i] for i in ids], [pred[i] for i in ids]
    return {"n": len(ids), "acc": round(sum(a == b for a, b in zip(g, p)) / len(ids), 3),
            "kappa": round(cohen_kappa_score(g, p), 3),
            "f1": dict(zip(TYPES, [round(x, 3) for x in f1_score(g, p, labels=TYPES, average=None, zero_division=0)])),
            "macro_f1": round(f1_score(g, p, labels=TYPES, average="macro", zero_division=0), 3)}


def main():
    key = json.load(open(ROOT / "data/validation/human/key.json"))
    sheets = {Path(p).stem.replace("annotation_sheet_", ""): load_sheet(p) for p in sys.argv[1:]}
    res = {"coverage": {k: len(v) for k, v in sheets.items()}}
    for a, b in combinations(sheets, 2):
        common = [i for i in key if i in sheets[a] and i in sheets[b]]
        res[f"human_{a}__{b}"] = {"n": len(common), "kappa": round(cohen_kappa_score(
            [sheets[a][i] for i in common], [sheets[b][i] for i in common]), 3)}
    golds = dict(sheets)
    if len(sheets) >= 2:
        maj = {}
        for i in key:
            c = Counter(s[i] for s in sheets.values() if i in s).most_common()
            if c and (len(c) == 1 or c[0][1] > c[1][1]):
                maj[i] = c[0][0]
        golds["human_majority"] = maj
    strata = {"all": list(key), **{s: [i for i in key if key[i]["stratum"] == s] for s in ("disputed", "unanimous")}}
    for gname, gold in golds.items():
        for m in MODELS:
            pred = {i: key[i][m] for i in key}
            for sname, ids in strata.items():
                r = score(gold, pred, ids)
                if r:
                    res[f"{m}_vs_{gname}_{sname}"] = r
    # Reweight to the 240-item agreement set: all disputed items are sheeted, unanimous ones are a sample.
    n_unan = AGREEMENT_N - sum(1 for i in key if key[i]["stratum"] == "disputed")
    n_unan_sheet = sum(1 for i in key if key[i]["stratum"] == "unanimous")
    wt = {i: (n_unan / n_unan_sheet if key[i]["stratum"] == "unanimous" else 1.0) for i in key}
    sec = {t: s for s, ts in SECTORS.items() for t in ts}
    tick = {r["item_id"].zfill(4): (r["ticker_a"], r["ticker_b"]) for r in
            csv.DictReader(open(ROOT / "data/validation/human/annotation_sheet_original.csv", encoding="utf-8-sig"))}
    for gname, gold in golds.items():
        ids = [i for i in key if i in gold]
        for m in MODELS:
            res[f"{m}_vs_{gname}_weighted240"] = weighted_scores(gold, {i: key[i][m] for i in key}, ids, wt)
        blocks = {}
        for i in ids:
            b = "-".join(sorted({sec.get(t, "Other") for t in tick[i]}))
            for src, lab in [("human", gold[i])] + [(m, key[i][m]) for m in MODELS]:
                blocks.setdefault(b, {}).setdefault(src, Counter())[lab] += 1
        res[f"sector_pair_counts_{gname}"] = {b: {k: dict(v) for k, v in d.items()} for b, d in blocks.items()}
    out = ROOT / "analysis/metrics/human_agreement.json"
    out.write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        if k == "coverage" or "_all" in k or "weighted240" in k or k.startswith("human_"):
            print(k, {x: v[x] for x in v if x != "f1"} if isinstance(v, dict) else v)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
