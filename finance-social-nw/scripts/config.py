"""
Central config for narrative-graph data collection.

Access notes (as of 2026-09-27):
- StockTwits: API is closed to new developer registrations pending an internal
  review. Live collection via `fetch_stocktwits.py` will fail until access is
  granted (contact developers@stocktwits.com). Use FALLBACK_DATASET mode below
  in the meantime.
- Reddit: self-service OAuth app registration is closed under Reddit's
  "Responsible Builder Policy". New apps require manual approval (2-4 week
  turnaround, not guaranteed). Free tier once approved: ~100 queries/min per
  OAuth client for non-commercial/academic use.

Set credentials via environment variables, never hard-code them here.
"""
import os

# --- Ticker basket -----------------------------------------------------
MEGA_CAP_AI = ["NVDA", "MSFT", "GOOGL", "META"]
PURE_PLAY_AI_HW = ["SMCI", "AMD", "AVGO"]
SPECULATIVE_AI = ["PLTR", "SOUN", "BBAI", "IONQ", "RGTI"]
CONTROL_NON_AI = ["KO", "PG", "XOM"]  # non-AI names for the beta-control check

TICKER_BASKET = MEGA_CAP_AI + PURE_PLAY_AI_HW + SPECULATIVE_AI
ALL_TICKERS = TICKER_BASKET + CONTROL_NON_AI

# --- Time window ---------------------------------------------------------
START_DATE = "2022-11-01"  # ChatGPT launch month
END_DATE = "2026-09-27"    # today
WINDOW_FREQ = "MS"          # monthly windows (pandas offset alias)

# --- Credentials (env-driven; None until approved/available) ------------
STOCKTWITS_ACCESS_TOKEN = os.environ.get("STOCKTWITS_ACCESS_TOKEN")

REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.environ.get("REDDIT_USER_AGENT", "narrative-graph-research/0.1")

# --- Fallback mode ---------------------------------------------------------
# If live API creds are unavailable, scripts fall back to reading a local
# academic/pre-scraped dataset (HuggingFace/Kaggle) placed under data/raw/.
USE_FALLBACK_DATASET = not (STOCKTWITS_ACCESS_TOKEN or (REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
