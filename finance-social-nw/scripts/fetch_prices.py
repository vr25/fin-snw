"""
Pull daily OHLCV for the ticker basket via yfinance (no auth required).

Usage:
    python fetch_prices.py
"""
import os
import sys

import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ALL_TICKERS, START_DATE, END_DATE, RAW_DIR


def fetch_prices() -> pd.DataFrame:
    frames = []
    for ticker in ALL_TICKERS:
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        if df.empty:
            print(f"[warn] no price data returned for {ticker}")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Close"]].rename(columns={"Close": "close"})
        df["ticker"] = ticker
        df["return"] = df["close"].pct_change()
        frames.append(df.reset_index())
    return pd.concat(frames, ignore_index=True)


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    prices = fetch_prices()
    out_path = os.path.join(RAW_DIR, "prices.parquet")
    prices.to_parquet(out_path, index=False)
    print(f"Wrote {len(prices)} rows across {prices['ticker'].nunique()} tickers -> {out_path}")


if __name__ == "__main__":
    main()
