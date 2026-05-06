import json
import warnings
from datetime import datetime
from huggingface_hub import HfApi
import io
import pandas as pd

from src.utils.logger import setup_logger
from src.utils.market_utils import build_export_df
from src.pipeline.etl import get_data_pipeline, load_models
from src.backtest.backtest import (
    backtest_strategy_with_rebalancing,
    get_optimal_weights,
    get_topk_equal_weights,
)
from src.strategy.signals import AlphaSignal
from src.utils.settings import SETTINGS

warnings.filterwarnings("ignore")

BASE_DIR = SETTINGS["BASE_DIR"]
TARGET_CLUSTER = SETTINGS["TARGET_CLUSTER"]
PROBA_THRESHOLD = SETTINGS["PROBA_THRESHOLD"]
FEATURE_COLS = SETTINGS["FEATURE_COLS"]
HF_TOKEN = SETTINGS["HF_TOKEN"]
REPO_ID = SETTINGS["REPO_ID"]
END_TIME = SETTINGS["END_TIME"]
TRADING_DAYS_YEAR = SETTINGS["TRADING_DAYS_YEAR"]
MARKET_NAME = SETTINGS.get("MARKET_NAME", "CAC40")
MARKET_SUFFIX = SETTINGS.get("MARKET_SUFFIX", "cac40")

logger = setup_logger("DailyRun")
logger.info(f"Pipeline running for period up to: {END_TIME}")


def upload_to_hub(df: pd.DataFrame, filename: str):
    if HF_TOKEN is None:
        logger.info(f"Local mode: token unavailable, skip upload for {filename}")
        return
    try:
        api = HfApi()
        parquet_buffer = io.BytesIO()
        df.to_parquet(parquet_buffer, index=True, compression="gzip")
        parquet_buffer.seek(0)

        api.upload_file(
            path_or_fileobj=parquet_buffer,
            path_in_repo=f"data/{filename}.parquet",
            repo_id=REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN,
        )
        logger.info(f"Sync successful on HF: {filename}.parquet")
    except Exception as e:
        logger.error(f"Sync failed for {filename}: {e}")


def run_pipeline():
    start_time = datetime.now()

    logger.info("-" * 60)
    logger.info(f"STARTING PIPELINE | MARKET: {MARKET_NAME}")
    logger.info("-" * 60)

    try:
        xgb_model, kmeans_model = load_models()
        df_daily, df_monthly = get_data_pipeline()   # <- sans config_file

        if df_daily is None or df_monthly is None:
            raise RuntimeError(f"Data Pipeline failure for {MARKET_NAME}")

        logger.info("Generating signal cache (vectorized)...")
        signal_generator = AlphaSignal.from_xgboost_kmeans(
            df_monthly, xgb_model, kmeans_model, FEATURE_COLS
        )

        last_date = df_monthly.index.get_level_values("date").max()
        today_data = df_monthly.xs(last_date, level="date").copy()

        today_signals = signal_generator.get_signal(last_date)
        today_data["proba_upside"] = today_signals["proba_upside"]
        today_data["cluster"] = today_signals["cluster"]

        selected = today_data[
            (today_data["cluster"] == TARGET_CLUSTER)
            & (today_data["proba_upside"] > PROBA_THRESHOLD)
        ]

        final_alloc_markowitz = {}
        final_alloc_topk = {}

        if not selected.empty:
            sel_tickers = selected.index.tolist()
            prices_subset = (
                df_daily["adj close"]
                .unstack()[sel_tickers]
                .loc[:last_date]
                .iloc[-TRADING_DAYS_YEAR:]
                .dropna(axis=1)
            )

            if not prices_subset.empty and len(prices_subset.columns) >= 3:
                weights, success = get_optimal_weights(prices_subset)
                final_alloc_markowitz = (
                    weights if success else {t: 1.0 / len(sel_tickers) for t in sel_tickers}
                )

            final_alloc_topk = get_topk_equal_weights(
                selected=selected,
                topk=10,
                score_col="proba_upside",
            )

        logger.info("Running Markowitz backtest...")
        hist_marko, rebal_marko = backtest_strategy_with_rebalancing(
            df_daily=df_daily,
            signal_generator=signal_generator,
            get_optimal_weights_fn=get_optimal_weights,
            portfolio_method="markowitz",
        )

        logger.info("Running Top-k backtest...")
        hist_topk, rebal_topk = backtest_strategy_with_rebalancing(
            df_daily=df_daily,
            signal_generator=signal_generator,
            get_optimal_weights_fn=get_optimal_weights,
            portfolio_method="topk",
            topk=10,
        )

        export_df_markowitz = build_export_df(today_data, final_alloc_markowitz)
        export_df_topk = build_export_df(today_data, final_alloc_topk)

        upload_to_hub(export_df_markowitz, f"latest_signals_markowitz_{MARKET_SUFFIX}")
        upload_to_hub(export_df_topk, f"latest_signals_topk_{MARKET_SUFFIX}")
        upload_to_hub(hist_marko, f"portfolio_history_markowitz_{MARKET_SUFFIX}")
        upload_to_hub(rebal_marko, f"rebalance_history_markowitz_{MARKET_SUFFIX}")
        upload_to_hub(hist_topk, f"portfolio_history_topk_{MARKET_SUFFIX}")
        upload_to_hub(rebal_topk, f"rebalance_history_topk_{MARKET_SUFFIX}")

        metadata = {
            "market_name": MARKET_NAME,
            "last_update": datetime.now().isoformat(),
            "current_allocation": {
                "markowitz": final_alloc_markowitz,
                "topk": final_alloc_topk,
            },
            "metrics": {
                "markowitz_final_value": float(hist_marko["Strategy"].iloc[-1]) if not hist_marko.empty else 0.0,
                "topk_final_value": float(hist_topk["Strategy"].iloc[-1]) if not hist_topk.empty else 0.0,
            },
        }

        with open(BASE_DIR / f"metadata_{MARKET_SUFFIX}.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"SUCCESS: {MARKET_NAME} processed in {duration:.1f}s")

    except Exception as e:
        logger.error(f"FAILURE for {MARKET_NAME}: {e}")


if __name__ == "__main__":
    run_pipeline()