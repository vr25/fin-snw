"""Build a blind human-annotation sheet from the 240-instance agreement sample.

Selects every instance on which the four verified LLM runs do not all agree, plus a random
set of unanimous instances (as an accuracy spot-check), shuffles them, and writes:
  data/validation/human/annotation_sheet.csv   (given to annotators; no model labels)
  data/validation/human/key.json               (strata and model labels; NOT for annotators)
Usage: python scripts/make_human_sheet.py [n_unanimous=50]
"""
import csv
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import label_agreement as L  # noqa: E402

OUT = ROOT / "data/validation/human"
NAMES = {"NVDA": ["Nvidia"], "AMD": ["AMD", "Advanced Micro"], "AVGO": ["Broadcom"],
         "SMCI": ["Super Micro", "Supermicro"], "MSFT": ["Microsoft"], "GOOGL": ["Google", "Alphabet"],
         "META": ["Meta", "Facebook", "Instagram", "WhatsApp"], "PLTR": ["Palantir"], "IONQ": ["IonQ"]}
DISPLAY = {"NVDA": "Nvidia", "AMD": "AMD", "AVGO": "Broadcom", "SMCI": "Super Micro Computer",
           "MSFT": "Microsoft", "GOOGL": "Alphabet/Google", "META": "Meta Platforms", "PLTR": "Palantir", "IONQ": "IonQ"}


def excerpt(text, a, b, limit=1800):
    """Sentences mentioning either firm (both-firm sentences marked), with one sentence of context."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    pat = {t: re.compile(r"\b(" + "|".join(map(re.escape, NAMES.get(t, [t]) + [t])) + r")\b", re.I) for t in (a, b)}
    hit = [(bool(pat[a].search(s)), bool(pat[b].search(s))) for s in sents]
    keep = set()
    for k, (ha, hb) in enumerate(hit):
        if ha or hb:
            keep.update({k - 1, k, k + 1})
    order = sorted(k for k in keep if 0 <= k < len(sents))
    both = [k for k in order if all(hit[k])]
    priority = sorted(set(both) | {k + d for k in both for d in (-1, 1)} & set(order))
    rest = [k for k in order if k not in priority]
    out, n = [], 0
    for k in sorted(priority) + rest:
        if n + len(sents[k]) > limit:
            break
        out.append(k); n += len(sents[k])
    parts, prev = [], None
    for k in sorted(out):
        if prev is not None and k != prev + 1:
            parts.append("[...]")
        parts.append((">> " if all(hit[k]) else "") + sents[k])
        prev = k
    return " ".join(parts) if parts else text[:limit]


def main():
    n_unan = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    final = [json.loads(l) for l in open(ROOT / "data/processed/pair_relations.jsonl")]
    posts = {p["text"]: p for p in map(json.loads, open(ROOT / "data/raw/gdelt_posts.jsonl"))}
    ids = json.load(open(ROOT / "data/validation/sample_ids.json"))
    runs = {"final_haiku": {i: final[i]["relation"] for i in ids}, "gpt_5_4_mini": L.load_v3("gpt"),
            "haiku_repeat": L.load_v3("haiku"), "gpt_6_sol": L.load_v3("gpt6")}
    disputed = [i for i in ids if len({runs[k][i] for k in runs}) > 1]
    unanimous = [i for i in ids if i not in disputed]
    rng = random.Random(0)
    chosen = [(i, "disputed") for i in disputed] + [(i, "unanimous") for i in rng.sample(unanimous, n_unan)]
    rng.shuffle(chosen)
    OUT.mkdir(parents=True, exist_ok=True)
    key, rows = {}, []
    for n, (i, stratum) in enumerate(chosen, 1):
        r = final[i]
        a, b = r["ticker_a"], r["ticker_b"]
        body = open(ROOT / f"data/validation/items/{i:04d}.txt").read().split("\n", 3)[3]
        url = posts.get(r["source_text"], {}).get("url", "")
        rows.append({"row": n, "item_id": f"{i:04d}", "ticker_a": a, "company_a": DISPLAY.get(a, a),
                     "ticker_b": b, "company_b": DISPLAY.get(b, b), "date": r["timestamp"][:10], "url": url,
                     "full_text_file": f"data/validation/items/{i:04d}.txt", "excerpt": excerpt(body, a, b),
                     "relation": "", "sentiment_alignment": "", "confidence": "", "needed_full_text": "", "notes": ""})
        key[f"{i:04d}"] = {"stratum": stratum, **{k: runs[k][i] for k in runs}}
    with open(OUT / "annotation_sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    json.dump(key, open(OUT / "key.json", "w"), indent=1)
    print(f"{len(rows)} rows ({len(disputed)} disputed, {n_unan} unanimous) -> {OUT}")


if __name__ == "__main__":
    main()
