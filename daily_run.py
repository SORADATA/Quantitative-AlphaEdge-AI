import sys
import os
import json
import warnings
from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi

from const import (
    BASE_DIR,
    TARGET_CLUSTER,
    PROBA_THRESHOLD,
    FEATURE_COLS
)
from src.utils.logger import setup_logger
from src.utils.config_loader import TICKERS, BENCHMARK_TICKER, MARKET_NAME
from src.utils.market_utils import build_export_df

from src.pipeline.etl import get_data_pipeline, load_models
from src.pipeline.backtest import backtest_strategy_with_rebalancing, get_optimal_weights


# =============================================================================
# INITIALIZATION
# =============================================================================

warnings.filterwarnings('ignore')
logger = setup_logger("DailyRun")

HF_REPO_ID = os.getenv("HF_REPO_ID", "soradata/alphaedge-data")
HF_BRANCH = os.getenv("HF_DATA_BRANCH", "main")
HF_TOKEN = os.getenv("HF_TOKEN")
hf_api = HfApi()


# =============================================================================
# HF UPLOAD HELPER
# =============================================================================

def upload_to_hf(local_path: Path, hf_filename: str):
    """Upload un fichier local vers HuggingFace dans data/{HF_BRANCH}/"""
    try:
        hf_api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=f"data/{hf_filename}",
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN,
        )
        logger.info(f"Uploaded to HF: data/{HF_BRANCH}/{hf_filename}")
    except Exception as e:
        logger.error(f"HF upload failed for {hf_filename}: {e}")


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def run_pipeline():
    start_time = datetime.now()
    logger.info("-" * 60)
    logger.info(f"STARTING DAILY PIPELINE | MARKET: {MARKET_NAME}")
    logger.info(f"ASSETS: {len(TICKERS)} | BENCHMARK: {BENCHMARK_TICKER}")
    logger.info(f"HF TARGET: {HF_REPO_ID}/data/{HF_BRANCH}/")
    logger.info("-" * 60)

    try:
        # 1. LOAD MODELS
        xgb_model, kmeans_model = load_models()
        if xgb_model is None:
            raise RuntimeError("ML Models not found in src/models/")

        # 2. ETL & FEATURE ENGINEERING
        df_daily, df_monthly = get_data_pipeline()
        if df_daily is None or df_monthly is None:
            raise RuntimeError("Data Pipeline failure.")

        # 3. GENERATE CURRENT SIGNALS
        last_date = df_monthly.index.get_level_values('date').max()
        logger.info(f"Generating signals for: {last_date.date()}")

        today_data = df_monthly.xs(last_date, level=0).copy()
        today_data['cluster'] = kmeans_model.predict(today_data[['rsi']].fillna(50))
        today_data['proba_upside'] = xgb_model.predict_proba(
            today_data[FEATURE_COLS].fillna(0))[:, 1]
        selected = today_data[
            (today_data['cluster'] == TARGET_CLUSTER) &
            (today_data['proba_upside'] > PROBA_THRESHOLD)
        ]
        final_alloc = {}
        if not selected.empty:
            tickers = selected.index.tolist()
            prices_subset = df_daily['adj close'].unstack()[tickers].iloc[-252:].dropna(axis=1)
            weights, success = get_optimal_weights(prices_subset)
            final_alloc = weights if success else {t: 1.0/len(tickers) for t in tickers}

        # 4. EXPORT DAILY SIGNALS → HF
        export_df = build_export_df(today_data, final_alloc)
        signals_path = BASE_DIR / 'latest_signals.parquet'
        export_df.to_parquet(signals_path, index=False)
        upload_to_hf(signals_path, "latest_signals.parquet")
        logger.info(f"Signals exported: {len(selected)} BUY signals.")

        # 5. BACKTESTING & MONITORING → HF
        logger.info("Executing strategy backtest...")
        hist_df, rebal_df = backtest_strategy_with_rebalancing(
            df_daily,
            df_monthly,
            xgb_model,
            kmeans_model,
            get_optimal_weights
        )
        hist_path = BASE_DIR / 'portfolio_history.parquet'
        rebal_path = BASE_DIR / 'rebalance_history.parquet'
        hist_df.to_parquet(hist_path)
        rebal_df.to_parquet(rebal_path)
        upload_to_hf(hist_path,  "portfolio_history.parquet")
        upload_to_hf(rebal_path, "rebalance_history.parquet")

        # 6. METADATA UPDATE → HF
        metadata = {
            'market_name':       MARKET_NAME,
            'last_update':       datetime.now().isoformat(),
            'n_assets_tracked':  len(TICKERS),
            'current_allocation': final_alloc,
            'hf_branch':         HF_BRANCH,
        }
        meta_path = BASE_DIR / 'data_metadata.json'
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        upload_to_hf(meta_path, "data_metadata.json")

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"PIPELINE COMPLETED SUCCESSFULLY in {duration:.1f}s")

    except Exception as e:
        logger.critical(f"CRITICAL FAILURE: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run_pipeline()