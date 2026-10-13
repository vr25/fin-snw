"""
Scan GDELT GKG for co-mentions in the PRE-ChatGPT period: Nov 2021 - Oct 2022.
Mirrors fetch_gdelt.py but with custom date range.
"""
import io
import json
import os
import re
import sys
import time
import zipfile
from datetime import datetime, timedelta

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RAW_DIR

# Ticker mappings (same as fetch_gdelt.py)
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
SLICES_PER_DAY = 4
REQUEST_DELAY_SEC = 0.5
MAX_RETRIES = 2

# PRE-ChatGPT date range
PRE_START = "2021-11-01"
PRE_END = "2022-10-31"


def _daily_timestamps(start_date, end_date, slices_per_day=SLICES_PER_DAY):
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


def _fetch_gkg_slice(ts):
    url = GKG_URL_TMPL.format(ts=ts)
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=20)
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                name = zf.namelist()[0]
                with zf.open(name) as f:
                    return f.read().decode("utf-8", errors="replace").splitlines()
        except (requests.RequestException, zipfile.BadZipFile) as e:
            if attempt < MAX_RETRIES:
                time.sleep(1)
            else:
                return []


def scan_gkg(patterns):
    hits = []
    timestamps = list(_daily_timestamps(PRE_START, PRE_END, SLICES_PER_DAY))
    print(f"[info] scanning {len(timestamps)} GKG slices ({PRE_START} to {PRE_END})")
    
    for i, ts in enumerate(timestamps):
        if i % 50 == 0:
            print(f"[info] slice {i}/{len(timestamps)}")
        
        lines = _fetch_gkg_slice(ts)
        if not lines:
            continue
        
        for line in lines:
            parts = line.split("\t")
            if len(parts) < 17:
                continue
            
            date_str = parts[1]
            orgs = parts[15]
            source_url = parts[12]
            
            # Find co-mentions
            found_tickers = {}
            for ticker, pat in patterns.items():
                if pat.search(orgs):
                    found_tickers[ticker] = True
            
            if len(found_tickers) >= 2:
                hits.append({
                    "date": date_str,
                    "tickers": list(found_tickers.keys()),
                    "url": source_url,
                })
        
        time.sleep(REQUEST_DELAY_SEC)
    
    return hits


def main():
    patterns = _compile_patterns()
    hits = scan_gkg(patterns)
    
    out_path = os.path.join(RAW_DIR, "gdelt_comentions_pre_chatgpt.jsonl")
    with open(out_path, "w") as f:
        for h in hits:
            f.write(json.dumps(h) + "\n")
    
    print(f"\n[info] found {len(hits)} co-mention hits")
    print(f"[info] wrote -> {out_path}")


if __name__ == "__main__":
    main()
