"""Merge verified relabel outputs (data/relabel/out_*.jsonl) into data/processed/pair_relations.jsonl.

The original labels are preserved once in data/processed/pair_relations_haiku_v1.jsonl.
Every output line must pass scripts/validate_labels.py (id, tickers and first words match the item).
Usage: python scripts/merge_relabel.py   (fails if any instance is missing)
"""
import glob
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_labels import check  # noqa: E402

REL = ROOT / "data/processed/pair_relations.jsonl"
V1 = ROOT / "data/processed/pair_relations_haiku_v1.jsonl"


def main():
    if not V1.exists():
        shutil.copy(REL, V1)
    rels = [json.loads(l) for l in open(V1)]
    new = {}
    for f in sorted(glob.glob(str(ROOT / "data/relabel/out_*.jsonl"))):
        tag = Path(f).stem.split("_")[1]
        good, _, _ = check(f, str(ROOT / f"data/relabel/b20_{tag}.txt"))
        new.update({int(i): r for i, r in good.items()})
    missing = [i for i in range(len(rels)) if i not in new]
    if missing:
        sys.exit(f"{len(missing)} instances missing, e.g. {missing[:10]}")
    with open(REL, "w") as out:
        for i, r in enumerate(rels):
            r = dict(r, relation=new[i]["relation"], sentiment_alignment=float(new[i]["sentiment_alignment"]))
            out.write(json.dumps(r) + "\n")
    print(f"wrote {len(rels)} relabelled instances -> {REL}")


if __name__ == "__main__":
    main()
