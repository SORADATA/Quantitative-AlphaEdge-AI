import json
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd

from const import DATA_DIR, BASE_DIR
from src.extract.extractor import MarketExtractor
from src.transform.processor import MarketDataProcessor
from src.transform.ticker_manager import handle_ticker_changes
from src.utils.logger import setup_logger

logger = setup_logger("etl")


def get_data_pipeline(market_config: dict) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    market_name = market_config["market_name"]
    tickers = market_config["tickers"]

    ticker_changes, delisted = handle_ticker_changes()
    active_tickers = [
        ticker_changes.get(t, t) for t in tickers if t not in delisted
    ]

    extractor = MarketExtractor(market_name=market_name, tickers=active_tickers)
    raw = extractor.fetch_market_data()
    if raw is None or raw.empty:
        logger.error(f"Abandon du pipeline pour {market_name} : aucune donnée extraite.")
        return None, None

    processor = MarketDataProcessor(active_tickers=active_tickers)
    df_daily, df_monthly, alerts = processor.process(raw)

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    with open(BASE_DIR / f"{market_name}_ticker_validation.json", "w") as fh:
        json.dump(
            {
                "date":          str(datetime.now()),
                "alerts":        alerts,
                "valid_tickers": len(active_tickers) - len(alerts),
            },
            fh, indent=2,
        )

    processed_dir = DATA_DIR / "processed" / market_name
    processed_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Sauvegarde des données dans {processed_dir}...")
    df_daily.to_parquet(processed_dir / "daily_raw.parquet", compression="gzip")
    df_monthly.to_parquet(processed_dir / "monthly_features.parquet", compression="gzip")

    return df_daily, df_monthly


if __name__ == "__main__":
    from pathlib import Path

    config_dir = Path("config/markets")
    for config_file in sorted(config_dir.glob("*.json")):
        with open(config_file) as f:
            market_config = json.load(f)
        market_name = market_config.get("market_name", "UNKNOWN")
        logger.info(f"Lancement ETL — {market_name}...")
        df_daily, df_monthly = get_data_pipeline(market_config)
        if df_daily is not None:
            logger.info(f" Succès ! Shape daily: {df_daily.shape}, Shape monthly: {df_monthly.shape}")
        else:
            logger.error(f" Échec ETL pour {market_name}.")