import os
import json
import warnings
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

from src.utils.logger import setup_logger
from src.pipeline.etl import get_data_pipeline
from src.pipeline.backtest import backtest_strategy_with_rebalancing
from src.models.model_loader import load_champion
from const import BACKTEST_YEARS

# =============================================================================
# CONFIGURATION GLOBALE
# =============================================================================
warnings.filterwarnings("ignore")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = os.getenv("HF_REPO_ID", "soradata/alphaedge-data")
hf_api = HfApi()


def upload_to_hf(local_path: Path, hf_filename: str, market_name: str):
    """Upload silencieux vers le repo Hugging Face."""
    if not HF_TOKEN:
        return
    try:
        hf_api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=f"data/{market_name}/{hf_filename}",
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN,
        )
    except Exception as e:
        print(f"Erreur Upload HF pour {market_name}: {e}")


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================
def run_pipeline(market_config: dict):
    market_name = market_config.get("market_name", "UNKNOWN")
    logger = setup_logger(f"Pipeline_{market_name}")
    logger.info(f" STARTING PIPELINE | MARKET: {market_name}")

    try:
        # 1. Chargement du modèle champion depuis MLflow
        model = load_champion(market_name)
        logger.info(f"Modèle champion prêt pour {market_name}")

        # 2. ETL
        df_daily, df_monthly = get_data_pipeline(market_config)
        if df_daily is None or df_monthly is None:
            raise ValueError("L'ETL n'a renvoyé aucune donnée.")

        cutoff = pd.Timestamp.now() - pd.DateOffset(years=BACKTEST_YEARS)
        df_daily_bt = df_daily[df_daily.index.get_level_values("date") >= cutoff]
        df_monthly_bt = df_monthly[df_monthly.index.get_level_values("date") >= cutoff]

        if df_daily_bt.empty or df_monthly_bt.empty:
            raise ValueError(f"Pas de données sur les {BACKTEST_YEARS} dernières années.")

        logger.info(
            f"Backtest window : {cutoff.date()} → aujourd'hui "
            f"({len(df_monthly_bt.index.get_level_values('date').unique())} mois)"
        )

        # 4. Backtest
        logger.info(f"Executing backtest for {market_name}...")
        hist_df, rebal_df, metrics = backtest_strategy_with_rebalancing(
            df_daily_bt,
            df_monthly_bt,
            model,
            benchmark_ticker=market_config.get("benchmark_ticker", "^FCHI"),
        )

        # 5. Sauvegarde locale
        base_dir = Path(f"data/processed/{market_name}")
        base_dir.mkdir(parents=True, exist_ok=True)
        hist_path = base_dir / "portfolio_history.parquet"
        rebal_path = base_dir / "rebalance_history.parquet"
        hist_df.to_parquet(hist_path)
        rebal_df.to_parquet(rebal_path)

        # 6. Synchronisation Cloud
        upload_to_hf(hist_path,  "portfolio_history.parquet",  market_name)
        upload_to_hf(rebal_path, "rebalance_history.parquet", market_name)
        logger.info(f" Pipeline terminé | Sharpe: {metrics.get('Sharpe', 'N/A')}")

    except Exception as e:
        logger.critical(f"CRITICAL FAILURE {market_name}: {e}", exc_info=True)


# =============================================================================
# ORCHESTRATEUR
# =============================================================================
if __name__ == "__main__":
    config_dir = Path("config/markets")
    for config_file in sorted(config_dir.glob("*.json")):
        with open(config_file) as f:
            market_config = json.load(f)
        run_pipeline(market_config)