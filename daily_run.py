import sys
import os
import json
import warnings
from datetime import datetime
from pathlib import Path
from huggingface_hub import HfApi

# Import de tes modules existants
from src.utils.logger import setup_logger
from src.pipeline.etl import get_data_pipeline, load_models
from src.pipeline.backtest import backtest_strategy_with_rebalancing, get_optimal_weights
from src.utils.market_utils import build_export_df

# Import des constantes pour les valeurs par défaut
from const import FEATURE_COLS, TARGET_CLUSTER, PROBA_THRESHOLD

# =============================================================================
# CONFIGURATION & HELPERS
# =============================================================================
warnings.filterwarnings('ignore')
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = os.getenv("HF_REPO_ID", "soradata/alphaedge-data")
hf_api = HfApi()

def upload_to_hf(local_path: Path, hf_filename: str, market_name: str):
    """Upload un fichier vers HuggingFace dans data/{market_name}/"""
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
# CŒUR DU PIPELINE
# =============================================================================
def run_pipeline(market_config: dict):
    market_name = market_config['market_name']
    tickers = market_config['tickers']
    logger = setup_logger(f"Pipeline_{market_name}")
    logger.info(f"🚀 STARTING PIPELINE | MARKET: {market_name}")

    # Récupération dynamique avec fallback sur const.py
    target_cluster = market_config.get('target_cluster', TARGET_CLUSTER)
    proba_threshold = market_config.get('proba_threshold', PROBA_THRESHOLD)
    feature_cols = market_config.get('feature_cols', FEATURE_COLS)

    try:
        # 1. LOAD MODELS
        xgb_model, kmeans_model = load_models()
        
        # 2. ETL
        df_daily, df_monthly = get_data_pipeline(tickers)
        
        # 3. SIGNALS & ALLOCATION
        last_date = df_monthly.index.get_level_values('date').max()
        today_data = df_monthly.xs(last_date, level=0).copy()
        
        today_data['cluster'] = kmeans_model.predict(today_data[['rsi']].fillna(50))
        today_data['proba_upside'] = xgb_model.predict_proba(today_data[feature_cols].fillna(0))[:, 1]
        
        selected = today_data[
            (today_data['cluster'] == target_cluster) &
            (today_data['proba_upside'] > proba_threshold)
        ]
        
        final_alloc = {}
        if not selected.empty:
            prices_subset = df_daily['adj close'].unstack()[selected.index.tolist()].iloc[-252:].dropna(axis=1)
            weights, success = get_optimal_weights(prices_subset)
            final_alloc = weights if success else {t: 1.0/len(selected) for t in selected.index}

        # 4. EXPORT & UPLOAD SIGNALS
        base_dir = Path(f"data/processed/{market_name}")
        base_dir.mkdir(parents=True, exist_ok=True)
        
        export_df = build_export_df(today_data, final_alloc)
        signals_path = base_dir / 'latest_signals.parquet'
        export_df.to_parquet(signals_path, index=False)
        upload_to_hf(signals_path, "latest_signals.parquet", market_name)
        
        # 5. BACKTESTING
        logger.info(f"Executing backtest for {market_name}...")
        hist_df, rebal_df = backtest_strategy_with_rebalancing(
            df_daily, 
            df_monthly, 
            xgb_model, 
            kmeans_model, 
            get_optimal_weights,
            benchmark_ticker=market_config.get('benchmark_ticker', '^FCHI')
        )
        
        hist_path = base_dir / 'portfolio_history.parquet'
        rebal_path = base_dir / 'rebalance_history.parquet'
        hist_df.to_parquet(hist_path)
        rebal_df.to_parquet(rebal_path)
        
        upload_to_hf(hist_path, "portfolio_history.parquet", market_name)
        upload_to_hf(rebal_path, "rebalance_history.parquet", market_name)
        
        logger.info(f"✅ Pipeline terminé avec succès pour {market_name}")

    except Exception as e:
        logger.critical(f"CRITICAL FAILURE {market_name}: {e}", exc_info=True)

# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================
if __name__ == "__main__":
    config_dir = Path("config/markets")
    for config_file in config_dir.glob("*.json"):
        with open(config_file, 'r') as f:
            market_config = json.load(f)
        run_pipeline(market_config)