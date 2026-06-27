import os
import pickle
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from pathlib import Path

from const import MODEL_DIR
from src.utils.logger import setup_logger

logger = setup_logger("model_loader")


def load_champion(market_name: str):
    """
    Charge le modèle champion depuis MLflow registry pour un marché donné.
    Fallback 1 : dernière version enregistrée si l'alias 'champion' est absent.
    Fallback 2 : .pkl local si MLflow est totalement indisponible.

    Args:
        market_name: Nom du marché tel que défini dans config/markets/*.json
                     (ex: "CAC40", "NASDAQ", "SP500")
    """
    hf_token = os.getenv("HF_TOKEN")
    registered_name = f"AlphaEdge_Ensemble_{market_name}"

    if hf_token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = "SORADATA"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = hf_token
        mlflow.set_tracking_uri("https://soradata-alphaedge-registry.hf.space")

        # Tentative 1 : alias champion
        try:
            model_uri = f"models:/{registered_name}@champion"
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"[{market_name}] Champion chargé depuis MLflow : {model_uri}")
            return model
        except Exception as e:
            logger.warning(f"[{market_name}] Alias 'champion' introuvable ({e}), tentative sur dernière version...")

        # Tentative 2 : dernière version enregistrée
        try:
            client = MlflowClient()
            versions = client.get_latest_versions(registered_name)
            if versions:
                latest = sorted(versions, key=lambda v: int(v.version))[-1]
                model_uri = f"models:/{registered_name}/{latest.version}"
                model = mlflow.sklearn.load_model(model_uri)
                logger.warning(
                    f"[{market_name}] Pas de champion — version {latest.version} chargée : {model_uri}"
                    )
                return model
            else:
                logger.warning(f"[{market_name}] Aucune version enregistrée pour {registered_name}.")
        except Exception as e:
            logger.warning(f"[{market_name}] MLflow indisponible ({e}), fallback disque local.")

    # Fallback final : .pkl local par marché
    pkl_path = MODEL_DIR / market_name / "ensemble_model.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"[{market_name}] Aucun modèle disponible — MLflow inaccessible et pas de .pkl local à {pkl_path}"
        )

    with open(pkl_path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"[{market_name}] Champion chargé depuis disque : {pkl_path}")
    return model