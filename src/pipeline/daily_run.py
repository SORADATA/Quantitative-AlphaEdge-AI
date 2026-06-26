import os
import json
import warnings
from pathlib import Path
from huggingface_hub import HfApi

from src.utils.logger import setup_logger
from src.pipeline.etl import get_data_pipeline
from src.pipeline.backtest import backtest_strategy_with_rebalancing
from src.models.ensemble import AlphaEdgeEnsemble

# =============================================================================
# CONFIGURATION GLOBALE
# =============================================================================
warnings.filterwarnings('ignore')
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
    market_name = market_config.get('market_name', 'UNKNOWN')
    logger = setup_logger(f"Pipeline_{market_name}")
    logger.info(f"🚀 STARTING PIPELINE | MARKET: {market_name}")

    try:
        # 1. Chargement du modèle unifié (AlphaEdgeEnsemble)
        model_path = Path("src/models/ensemble_model.pkl")
        if not model_path.exists():
            raise FileNotFoundError(f"Modèle introuvable à {model_path}")
        model = AlphaEdgeEnsemble.load(model_path)

        # 2. ETL : On passe bien 'market_config' entier, pas juste les tickers !
        df_daily, df_monthly = get_data_pipeline(market_config)
        
        if df_daily is None or df_monthly is None:
            raise ValueError("L'ETL n'a renvoyé aucune donnée.")

        # 3. Backtest intégré (le modèle génère ses signaux à l'intérieur)
        logger.info(f"Executing backtest for {market_name}...")
        hist_df, rebal_df, metrics = backtest_strategy_with_rebalancing(
            df_daily, 
            df_monthly, 
            model,
            benchmark_ticker=market_config.get('benchmark_ticker', '^FCHI')
        )

        # 4. Sauvegarde locale
        base_dir = Path(f"data/processed/{market_name}")
        base_dir.mkdir(parents=True, exist_ok=True)
        hist_path = base_dir / 'portfolio_history.parquet'
        rebal_path = base_dir / 'rebalance_history.parquet'
        hist_df.to_parquet(hist_path)
        rebal_df.to_parquet(rebal_path)

        # 5. Synchronisation Cloud
        upload_to_hf(hist_path, "portfolio_history.parquet", market_name)
        upload_to_hf(rebal_path, "rebalance_history.parquet", market_name)
        logger.info(f"✅ Pipeline terminé avec succès | Sharpe: {metrics.get('Sharpe', 'N/A')}")

    except Exception as e:
        logger.critical(f"CRITICAL FAILURE {market_name}: {e}", exc_info=True)

# =============================================================================
# ORCHESTRATEUR
# =============================================================================


if __name__ == "__main__":
    config_dir = Path("config/markets")
    if not config_dir.exists():
        config_dir = Path("config/markets")
    for config_file in config_dir.glob("*.json"):
        with open(config_file, 'r') as f:
            market_config = json.load(f)
        run_pipeline(market_config)
