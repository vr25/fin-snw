"""Validate v3 annotator outputs: each line's id/tickers/first_words must match its item file.

Usage: python scripts/validate_labels.py 'data/validation/v3_haiku_*.jsonl' [items_dir]
Prints per-file status; exits non-zero if any file is incomplete or misaligned.
"""
import glob
import json
import re
import sys
from pathlib import Path

TYPES = {"causal-driver", "substitute-comparison", "co-hype", "unrelated-mention"}
norm = lambda s: re.sub(r"\W+", " ", s).strip().lower()


def item_meta(path):
    t = open(path).read()
    a, b, _, body = t.split("\n", 3)
    return a.split(": ", 1)[1].strip(), b.split(": ", 1)[1].strip(), body


def check(out_file, batch_file):
    exp = [Path(p).stem for p in open(batch_file).read().split()]
    items_dir = Path(open(batch_file).read().split()[0]).parent
    good, bad = {}, []
    for line in open(out_file):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            bad.append(("json", line[:60])); continue
        i = str(r.get("id", "")).zfill(4)
        if i not in exp or r.get("relation") not in TYPES:
            bad.append((i, "id/relation")); continue
        a, b, body = item_meta(items_dir / f"{i}.txt")
        fw = norm(r.get("first_words", ""))
        if {r.get("ticker_a"), r.get("ticker_b")} != {a, b} or not fw or fw[:25] not in norm(body[:1000]):
            bad.append((i, "mismatch")); continue
        good[i] = r
    missing = [i for i in exp if i not in good]
    return good, bad, missing


if __name__ == "__main__":
    ok = True
    for f in sorted(glob.glob(sys.argv[1])):
        tag = re.search(r"_(\d+)\.jsonl$", f).group(1)
        prefix = sys.argv[2] if len(sys.argv) > 2 else "data/validation/b12_"
        good, bad, missing = check(f, f"{prefix}{tag}.txt")
        if bad or missing:
            ok = False
            print(f, "good", len(good), "bad", bad, "missing", missing)
    print("ALL OK" if ok else "PROBLEMS")
    sys.exit(0 if ok else 1)
