import json
import time
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
import yfinance as yf

from const import DATA_DIR, BASE_DIR, VARS_TO_LAG, RESAMPLE_MEAN_COLS, RESAMPLE_LAST_EXCLUDE
from src.transform.features import (
    ompute_technical_indicators
    calculate_returns,
    get_fama_french_betas
    )
from src.transform.validation import handle_ticker_changes, validate_and_clean_tickers
from src.utils.logger import setup_logger

logger = setup_logger("etl")

_DOWNLOAD_RETRIES = 3
_DOWNLOAD_RETRY_WAIT = 5  # seconds
_HISTORY_YEARS = 10


def _download_raw_prices(tickers: list[str]) -> Optional[pd.DataFrame]:
    """
    Downloads adjusted OHLCV data from Yahoo Finance with retry logic.
    Returns a stacked (date, ticker) multi-index DataFrame, or None on failure.
    """
    end   = (datetime.today() + pd.DateOffset(days=1)).strftime("%Y-%m-%d")
    start = (pd.Timestamp.today() - pd.DateOffset(years=_HISTORY_YEARS)).strftime("%Y-%m-%d")

    logger.info(f"Downloading market data ({start} → {end}) for {len(tickers)} assets...")

    for attempt in range(1, _DOWNLOAD_RETRIES + 1):
        try:
            df = yf.download(
                tickers,
                start=start,
                end=end,
                progress=False,
                auto_adjust=False,
                threads=True,
            )
            if not df.empty:
                logger.info(f"Download successful (attempt {attempt})")
                return df
            logger.warning(f"Empty response (attempt {attempt})")
        except Exception as exc:
            logger.warning(f"Download error (attempt {attempt}): {exc}")
            time.sleep(_DOWNLOAD_RETRY_WAIT)

    logger.error("Failed to download data after all retries.")
    return None


def _resample_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resamples a daily (date, ticker) DataFrame to business-month-end frequency.
    euro_volume is averaged; all other kept columns use last observation.
    """
    last_cols = [c for c in df.columns if c not in RESAMPLE_LAST_EXCLUDE]

    monthly = pd.concat(
        [
            df.unstack("ticker")[RESAMPLE_MEAN_COLS[0]]
            .resample("BM").mean()
            .stack("ticker")
            .to_frame(RESAMPLE_MEAN_COLS[0]),
            df.unstack()[last_cols].resample("BM").last().stack("ticker"),
        ],
        axis=1,
    ).dropna()

    return monthly


def get_data_pipeline() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Full ETL pipeline:
      1. Resolve ticker changes / delistings
      2. Download raw daily prices
      3. Compute technical indicators
      4. Resample to monthly + momentum returns + Fama-French betas
      5. Add lagged features
      6. Persist artifacts to DATA_DIR
    Returns (df_daily, df_monthly), or (None, None) on failure.
    """
    ticker_changes, delisted = handle_ticker_changes()
    active_tickers = [
        ticker_changes.get(t, t)
        for t in __import__("src.utils.config_loader", fromlist=["TICKERS"]).TICKERS
        if t not in delisted
    ]

    raw = _download_raw_prices(active_tickers)
    if raw is None:
        return None, None

    df = raw.stack(future_stack=True)
    df.index.names = ["date", "ticker"]
    df.columns = df.columns.str.lower()
    if "adj close" not in df.columns and "close" in df.columns:
        df["adj close"] = df["close"]

    df, valid_tickers, alerts = validate_and_clean_tickers(df, active_tickers)

    with open(BASE_DIR / "ticker_validation.json", "w") as fh:
        json.dump(
            {"date": str(datetime.now()), "alerts": alerts, "valid_tickers": len(valid_tickers)},
            fh,
            indent=2,
        )

    logger.info(f"Saving raw data to {DATA_DIR}...")
    df.to_parquet(DATA_DIR / "daily_raw.parquet", compression="gzip")

    df = compute_technical_indicators(df)

    logger.info("Resampling to monthly frequency...")
    df_monthly = _resample_to_monthly(df)
    df_monthly = df_monthly.groupby(level=1, group_keys=False).apply(calculate_returns)
    df_monthly = get_fama_french_betas(df_monthly)

    for col in VARS_TO_LAG:
        if col in df_monthly.columns:
            df_monthly[f"{col}_lag1"] = df_monthly.groupby("ticker")[col].shift(1)

    logger.info("Saving monthly features...")
    df_monthly.to_parquet(DATA_DIR / "monthly_features.parquet", compression="gzip")

    return df, df_monthly