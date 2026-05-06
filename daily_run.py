import json
import warnings
import os
from datetime import date, datetime
from huggingface_hub import HfApi
import io
import pandas as pd
from src.utils.logger import setup_logger
from src.utils.config_loader import load_market_config
from src.utils.market_utils import build_export_df
from src.pipeline.etl import get_data_pipeline, load_models
from src.backtest.backtest import backtest_strategy_with_rebalancing, get_optimal_weights
from src.strategy.signals import AlphaSignal
from src.utils.settings import SETTINGS

warnings.filterwarnings('ignore')
logger = setup_logger("DailyRun")
logger.info(f"Pipeline running for period up to: {END_TIME}")


BASE_DIR = SETTINGS["BASE_DIR"]
CONFIG_DIR = SETTINGS["CONFIG_DIR"]
TARGET_CLUSTER = SETTINGS["TARGET_CLUSTER"]
PROBA_THRESHOLD = SETTINGS["PROBA_THRESHOLD"]
FEATURE_COLS = SETTINGS["FEATURE_COLS"]
HF_TOKEN = SETTINGS["HF_TOKEN"]
REPO_ID = SETTINGS["REPO_ID"]
END_TIME = SETTINGS["END_TIME"]

# =============================================================================
# SYNC HUGGING FACE
# =============================================================================


def upload_to_hub(df: pd.DataFrame, filename: str):
    """Envoie le dataframe sur Hugging Face datasets au format parquet"""
    if HF_TOKEN is None:
        logger.info(f"Local mode : token unavailable, skip upload for {filename}")
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
            token=HF_TOKEN
        )
        logger.info(f"Sync successful on HF : {filename}.parquet")
    except Exception as e:
        logger.error(f"Sync failed for {filename}: {e}")

# =============================================================================
# MARKET PIPELINE
# =============================================================================


def run_pipeline_for_config(config_file: str):
    start_time = datetime.now()
    config = load_market_config(config_file)
    market_name = config.get('market_name', 'Unknown')
    logger.info("-" * 60)
    logger.info(f"STARTING PIPELINE | MARKET: {market_name}")
    logger.info("-" * 60)

    try:
        # 1. Chargement Modèles et Données
        xgb_model, kmeans_model = load_models()
        df_daily, df_monthly = get_data_pipeline(config_file)
        
        if df_daily is None or df_monthly is None:
            raise RuntimeError(f"Data Pipeline failure for {market_name}")

        # 2. GÉNÉRATION DES SIGNAUX (Style Qlib - Vectorisé)
        logger.info("Generating signal cache (vectorized)...")
        signal_generator = AlphaSignal.from_xgboost_kmeans(
            df_monthly, xgb_model, kmeans_model, FEATURE_COLS
        )

        # 3. RÉCUPÉRATION ET FUSION DES DONNÉES POUR L'EXPORT
        # On récupère les indicateurs techniques complets (RSI, etc.) pour la dernière date
        last_date = df_monthly.index.get_level_values('date').max()
        today_data = df_monthly.xs(last_date, level='date').copy()
        
        # On fusionne avec les probas calculées par le cache
        today_signals = signal_generator.get_signal(last_date)
        today_data['proba_upside'] = today_signals['proba_upside']
        today_data['cluster'] = today_signals['cluster']
        
        # Filtrage pour l'allocation
        selected = today_data[
            (today_data['cluster'] == TARGET_CLUSTER) &
            (today_data['proba_upside'] > PROBA_THRESHOLD)
        ]

        # 4. ALLOCATION OPTIMALE (Markowitz)
        final_alloc = {}
        if not selected.empty:
            sel_tickers = selected.index.tolist()
            prices_subset = df_daily['adj close'].unstack()[sel_tickers].iloc[-252:].dropna(axis=1)
            
            if len(prices_subset.columns) >= 3:
                weights, success = get_optimal_weights(prices_subset)
                final_alloc = weights if success else {t: 1.0/len(sel_tickers) for t in sel_tickers}

        # 5. BACKTEST HISTORIQUE (Réaliste avec Turnover Friction)
        logger.info("Running realistic backtest...")
        hist_df, rebal_df = backtest_strategy_with_rebalancing(
            df_daily, 
            signal_generator, 
            get_optimal_weights
        )

        # 6. EXPORT ET SYNCHRONISATION
        suffix = config_file.replace('.json', '')
        # build_export_df reçoit maintenant today_data qui contient RSI + Probas
        export_df = build_export_df(today_data, final_alloc)
        
        upload_to_hub(export_df, f"latest_signals_{suffix}")
        upload_to_hub(hist_df, f"portfolio_history_{suffix}")
        upload_to_hub(rebal_df, f"rebalance_history_{suffix}")

        # Metadata locales
        metadata = {
            'market_name': market_name,
            'last_update': datetime.now().isoformat(),
            'current_allocation': final_alloc,
            'metrics': {
                'final_value': float(hist_df['Strategy'].iloc[-1]) if not hist_df.empty else 0
            }
        }
        with open(BASE_DIR / f'metadata_{suffix}.json', 'w') as f:
            json.dump(metadata, f, indent=4)

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"SUCCESS: {market_name} processed in {duration:.1f}s")

    except Exception as e:
        logger.error(f"FAILURE for {market_name}: {e}")

# =============================================================================
# GLOBAL EXECUTION
# =============================================================================

def run_all_pipelines():
    if not CONFIG_DIR.exists():
        logger.error(f"Config directory not found at {CONFIG_DIR}")
        return

    config_files = [f for f in os.listdir(CONFIG_DIR) if f.endswith('.json')]
    if not config_files:
        logger.warning("No JSON config files found.")
        return

    logger.info(f"Found {len(config_files)} markets to process.")
    for config_file in config_files:
        run_pipeline_for_config(config_file)

if __name__ == "__main__":
    run_all_pipelines()