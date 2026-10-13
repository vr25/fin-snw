"""
Scan GDELT's public GKG (Global Knowledge Graph) bulk files for co-mentions of
AI-hype basket companies, across the full study window. No auth required —
this is the replacement for the blocked StockTwits/Reddit live collection.

GDELT publishes a GKG file every 15 minutes at:
  https://data.gdeltproject.org/gdeltv2/{YYYYMMDDHHMMSS}.gkg.csv.zip
Full backfill at 15-min granularity across ~4 years is ~140k files, far more
than needed for monthly-window graph construction, so we subsample a fixed
number of slices per day (default 4) spread across the study window.

NOTE: GKG records give organization mentions + aggregate tone + source URL,
but not full article text. Use fetch_gdelt_articles.py afterwards to pull
article text for the co-mention hits found here — that's the input the
relation-extraction step (extract_relations.py) actually needs.
"""
import io
import json
import os
import re
import sys
import time
import zipfile

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import START_DATE, END_DATE, RAW_DIR

# Map basket tickers -> company name strings as they appear in GKG's Organizations field.
TICKER_TO_COMPANY = {
    "NVDA": ["Nvidia"],
    "MSFT": ["Microsoft"],
    "GOOGL": ["Google", "Alphabet"],
    "META": ["Meta", "Facebook"],
    "SMCI": ["Super Micro Computer", "Supermicro"],
    "AMD": ["Advanced Micro Devices", "AMD"],
    "AVGO": ["Broadcom"],
    "PLTR": ["Palantir"],
    "SOUN": ["SoundHound"],
    "BBAI": ["BigBear.ai"],
    "IONQ": ["IonQ"],
    "RGTI": ["Rigetti"],
}

GKG_URL_TMPL = "https://data.gdeltproject.org/gdeltv2/{ts}.gkg.csv.zip"
SLICES_PER_DAY = 4  # e.g. 00:00, 06:00, 12:00, 18:00 UTC
REQUEST_DELAY_SEC = 0.5
MAX_RETRIES = 2


def _daily_timestamps(start_date: str, end_date: str, slices_per_day: int = SLICES_PER_DAY):
    import pandas as pd

    hours = [int(24 / slices_per_day * i) for i in range(slices_per_day)]
    for day in pd.date_range(start_date, end_date, freq="D"):
        for h in hours:
            yield day.strftime("%Y%m%d") + f"{h:02d}0000"


def _compile_patterns():
    pats = {}
    for ticker, names in TICKER_TO_COMPANY.items():
        pats[ticker] = re.compile("|".join(re.escape(n) for n in names), re.I)
    return pats


def _fetch_gkg_slice(ts: str) -> list[str]:
    url = GKG_URL_TMPL.format(ts=ts)
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=20)
            if resp.status_code == 404:
                return []  # slice not published (rare gaps in GDELT's archive)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                name = zf.namelist()[0]
                with zf.open(name) as f:
                    return f.read().decode("utf-8", errors="replace").splitlines()
        except (requests.RequestException, zipfile.BadZipFile) as e:
            if attempt == MAX_RETRIES:
                print(f"[warn] failed to fetch {ts}: {e}")
                return []
            time.sleep(1)
    return []


def scan_window(start_date: str, end_date: str) -> list[dict]:
    patterns = _compile_patterns()
    hits = []
    timestamps = list(_daily_timestamps(start_date, end_date))
    print(f"[info] scanning {len(timestamps)} GKG slices from {start_date} to {end_date}")

    for i, ts in enumerate(timestamps):
        lines = _fetch_gkg_slice(ts)
        for line in lines:
            fields = line.split("\t")
            if len(fields) < 16:
                continue
            date_str, orgs_v1, orgs_v2, tone_field = fields[1], fields[13], fields[14], fields[15]
            url = fields[4] if len(fields) > 4 else ""
            org_blob = f"{orgs_v1} {orgs_v2}"
            mentioned = [t for t, p in patterns.items() if p.search(org_blob)]
            mentioned = sorted(set(mentioned))
            if len(mentioned) < 2:
                continue
            try:
                tone = float(tone_field.split(",")[0])
            except (ValueError, IndexError):
                tone = None
            hits.append(
                {
                    "gdelt_date": date_str,
                    "url": url,
                    "domain": fields[3] if len(fields) > 3 else "",
                    "tickers": mentioned,
                    "tone": tone,
                }
            )
        time.sleep(REQUEST_DELAY_SEC)
        if (i + 1) % 50 == 0:
            print(f"[info] scanned {i + 1}/{len(timestamps)} slices, {len(hits)} co-mention hits so far")

    return hits


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    hits = scan_window(START_DATE, END_DATE)
    out_path = os.path.join(RAW_DIR, "gdelt_comentions.jsonl")
    with open(out_path, "w") as f:
        for h in hits:
            f.write(json.dumps(h) + "\n")
    print(f"Wrote {len(hits)} co-mention hits -> {out_path}")


if __name__ == "__main__":
    main()
