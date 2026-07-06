import os
import pickle
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from dotenv import load_dotenv

from const import MODEL_DIR
from src.utils.logger import setup_logger

load_dotenv()
logger = setup_logger("model_loader")

HF_TOKEN = os.getenv("HF_TOKEN")
USE_MLFLOW = bool(HF_TOKEN)

if USE_MLFLOW:
    os.environ["MLFLOW_TRACKING_USERNAME"] = "SORADATA"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = HF_TOKEN
    mlflow.set_tracking_uri("https://soradata-alphaedge-registry.hf.space")


def _load_champion_from_mlflow(market_name: str) -> Any | None:
    """
    Charge le modèle aliasé 'champion' depuis le MLflow Model Registry.
    Retourne None en cas d'échec (fallback local ensuite).
    """
    registered_model_name = f"AlphaEdge_Ensemble_{market_name}"
    try:
        model_uri = f"models:/{registered_model_name}@champion"
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"[{market_name}] Champion chargé depuis MLflow : {model_uri}")
        return model
    except MlflowException as e:
        logger.warning(f"[{market_name}] Impossible de charger le champion MLflow : {e}")
        return None


def _load_champion_from_local(market_name: str) -> Any | None:
    """
    Fallback : charge le dernier modèle sauvegardé localement (pickle).
    Utilisé si MLflow est indisponible ou HF_TOKEN absent.
    """
    local_path = MODEL_DIR / market_name / "ensemble_model.pkl"
    if not local_path.exists():
        logger.error(f"[{market_name}] Aucun modèle local trouvé à {local_path}")
        return None
    try:
        with open(local_path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"[{market_name}] Modèle chargé depuis le fallback local : {local_path}")
        return model
    except Exception as e:
        logger.error(f"[{market_name}] Échec du chargement local : {e}")
        return None


def load_champion(market_name: str) -> Any:
    """
    Point d'entrée unique utilisé par run_pipeline.py (daily run).
    Priorité : champion MLflow -> fallback local.
    Lève une exception si aucun modèle n'est disponible (le pipeline
    ne doit jamais tourner sans modèle).
    """
    model = None
    if USE_MLFLOW:
        model = _load_champion_from_mlflow(market_name)

    if model is None:
        model = _load_champion_from_local(market_name)

    if model is None:
        raise RuntimeError(
            f"[{market_name}] Aucun modèle champion disponible "
            "(ni MLflow, ni local). Impossible de générer les signaux."
        )

    return model