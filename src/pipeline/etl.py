import json
import pickle
from datetime import datetime
from typing import Optional, Tuple, Any

import pandas as pd

from const import DATA_DIR, BASE_DIR
from src.extract.extractor import MarketExtractor
from src.transform.processor import MarketDataProcessor
from src.transform.ticker_manager import handle_ticker_changes
from src.utils.logger import setup_logger

logger = setup_logger("etl")


def get_data_pipeline(market_config: dict) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Orchestrateur global du pipeline ETL : Extract -> Process -> Load."""
    market_name = market_config['market_name']
    tickers = market_config['tickers']
    # Résolution des tickers (changement de noms, delisting)
    ticker_changes, delisted = handle_ticker_changes()
    active_tickers = [
        ticker_changes.get(t, t) for t in tickers if t not in delisted
    ]

    # ==========================================
    # 1. EXTRACT (Extraction)
    # ==========================================
    extractor = MarketExtractor(market_name=market_name, tickers=active_tickers)
    raw = extractor.fetch_market_data()
    if raw is None or raw.empty:
        logger.error(f"Abandon du pipeline pour {market_name} : Aucune donnée extraite.")
        return None, None

    # ==========================================
    # 2. TRANSFORM (Processing)
    # ==========================================
    processor = MarketDataProcessor(active_tickers=active_tickers)
    df_daily, df_monthly, alerts = processor.process(raw)

    # ==========================================
    # 3. LOAD
    # ==========================================
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    with open(BASE_DIR / f"{market_name}_ticker_validation.json", "w") as fh:
        json.dump(
            {
                "date": str(datetime.now()),
                "alerts": alerts,
                "valid_tickers": len(active_tickers) - len(alerts)
            },
            fh, indent=2,
        )

    processed_dir = DATA_DIR / "processed" / market_name
    processed_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Sauvegarde des données dans {processed_dir}...")
    df_daily.to_parquet(processed_dir / "daily_raw.parquet", compression="gzip")
    df_monthly.to_parquet(processed_dir / "monthly_features.parquet", compression="gzip")

    return df_daily, df_monthly


def load_models() -> Tuple[Optional[Any], Optional[Any]]:
    """Charge les modèles pré-entraînés XGBoost et KMeans depuis MODEL_DIR."""
    from const import MODEL_DIR
    logger.info(f"Loading ML models from {MODEL_DIR}...")
    try:
        xgb_path = MODEL_DIR / 'xgboost_model.pkl'
        kmeans_path = MODEL_DIR / 'kmeans_model.pkl'
        if not xgb_path.exists() or not kmeans_path.exists():
            logger.error("Modèles introuvables dans MODEL_DIR.")
            return None, None

        with open(xgb_path, 'rb') as f:
            xgb = pickle.load(f)
        with open(kmeans_path, 'rb') as f:
            kmeans = pickle.load(f)
        return xgb, kmeans
    except Exception as e:
        logger.error(f"Erreur lors du chargement des modèles : {e}")
        return None, None
# ==========================================
# BLOC DE TEST LOCAL
# ==========================================
if __name__ == "__main__":
    # On simule une configuration basique pour tester
    test_config = {
        "market_name": "CAC40_Test",
        "tickers": ["AI.PA", "AIR.PA", "OR.PA"]  # Air Liquide, Airbus, L'Oréal
    }
    
    print("🛠️ Lancement du test local de l'ETL...")
    df_daily, df_monthly = get_data_pipeline(test_config)
    
    if df_daily is not None:
        print(f"✅ Test réussi ! Shape daily: {df_daily.shape}, Shape monthly: {df_monthly.shape}")
    else:
        print("❌ Le test a échoué.")