"""Inter-model agreement on the 240-instance validation sample.

Compares: original labels (claude-haiku-4-5, prompt v1, data/processed/pair_relations_haiku_v1.jsonl),
gpt-5.4-mini with prompt v1 (data/validation/out_*.jsonl), and claude-haiku-4.5 / gpt-5.4-mini
with the definitions prompt (data/validation/prompt_v2.txt) under the verified one-item-at-a-time
protocol (data/validation/v3_{haiku,gpt}_*.jsonl; lines failing scripts/validate_labels.py are dropped),
gpt-6-sol under the same protocol (gpt6_v3), and the final full-corpus relabel (haiku_final, data/processed/pair_relations.jsonl; same model, prompt and protocol).
Writes analysis/metrics/label_agreement.json.
"""
import glob
import json
from itertools import combinations
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / "data/validation"
TYPES = ["causal-driver", "substitute-comparison", "co-hype", "unrelated-mention"]


def load(pattern):
    out = {}
    for f in sorted(glob.glob(str(V / pattern))):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                out[int(r["id"])] = r["relation"]
    return out


def load_v3(model):
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from validate_labels import check
    out = {}
    for f in sorted(glob.glob(str(V / f"v3_{model}_*.jsonl"))):
        tag = f[-8:-6]
        good, _, _ = check(f, str(V / f"b12_{tag}.txt"))
        out.update({int(i): r["relation"] for i, r in good.items()})
    return out


def main():
    rels = [json.loads(l) for l in open(ROOT / "data/processed/pair_relations_haiku_v1.jsonl")]
    final = [json.loads(l) for l in open(ROOT / "data/processed/pair_relations.jsonl")]
    ids = json.load(open(V / "sample_ids.json"))
    sets = {"haiku_v1": {i: rels[i]["relation"] for i in ids},
            "gpt_v1": load("out_*.jsonl"),
            "haiku_v3": load_v3("haiku"),
            "gpt_v3": load_v3("gpt"),
            "haiku_final": {i: final[i]["relation"] for i in ids},
            "gpt6_v3": load_v3("gpt6")}
    sets = {k: v for k, v in sets.items() if v and all(x in TYPES for x in v.values())}
    res = {"n": len(ids), "coverage": {k: len(v) for k, v in sets.items()},
           "label_counts": {k: {t: sum(x == t for x in v.values()) for t in TYPES} for k, v in sets.items()}}
    for a, b in combinations(sets, 2):
        common = [i for i in ids if i in sets[a] and i in sets[b]]
        x, y = [sets[a][i] for i in common], [sets[b][i] for i in common]
        xb, yb = [v != "unrelated-mention" for v in x], [v != "unrelated-mention" for v in y]
        res[f"{a}__{b}"] = {"n": len(common), "agree": round(sum(p == q for p, q in zip(x, y)) / len(common), 3),
                            "kappa": round(cohen_kappa_score(x, y), 3),
                            "kappa_typed_vs_unrelated": round(cohen_kappa_score(xb, yb), 3)}
    if "gpt6_v3" in sets:
        F, G, J = sets["haiku_final"], sets["gpt_v3"], sets["gpt6_v3"]
        dis = [i for i in ids if i in J and F[i] != G[i]]
        res["gpt6_adjudication_final_vs_gpt"] = {
            "n_disagreements": len(dis),
            "sides_with_final": sum(J[i] == F[i] for i in dis),
            "sides_with_gpt_v3": sum(J[i] == G[i] for i in dis),
            "neither": sum(J[i] not in (F[i], G[i]) for i in dis)}
        agreed = [i for i in ids if i in J and F[i] == G[i]]
        res["gpt6_on_final_gpt_agreed"] = {"n": len(agreed), "gpt6_agrees": sum(J[i] == F[i] for i in agreed)}
    out = ROOT / "analysis/metrics/label_agreement.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
