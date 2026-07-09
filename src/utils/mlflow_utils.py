"""
MLflow Utils
=============
Récupération des métriques du modèle "champion" pour un marché donné.

Stratégie en cascade (alignée sur train.py) :
  1. MLflow Model Registry via l'alias "champion"
  2. Fallback sur le model_card.json sauvegardé localement par train.py

Fonctions :
  - get_champion_metrics() : renvoie {source, metrics, version, run_id, promoted, error}
"""

import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

from src.utils.logger import setup_logger

logger = setup_logger("mlflow_utils")


def get_champion_metrics(
    market: str,
    model_dir: Path,
    mlflow_enabled: bool = False,
    model_name_prefix: str = "AlphaEdge_Ensemble",
) -> dict:
    """
    Récupère les métriques du modèle "champion" pour un marché donné.

    1) Essaie MLflow via l'alias 'champion' (si mlflow_enabled=True et
       mlflow.set_tracking_uri déjà configuré par l'appelant).
    2) Si MLflow est indisponible ou qu'aucun alias 'champion' n'existe
       encore, retombe sur model_dir/<market>/model_card.json.

    Parameters
    ----------
    market : str — marché ciblé (ex: "CAC40")
    model_dir : Path — répertoire racine des model_card.json locaux
    mlflow_enabled : bool — active la tentative MLflow (nécessite un token)
    model_name_prefix : str — préfixe du nom du modèle enregistré

    Returns
    -------
    dict — {source, metrics, version, run_id, promoted, error}
    """
    result = {
        "source": None,
        "metrics": {},
        "version": None,
        "run_id": None,
        "promoted": None,
        "error": None,
    }
    registered_model_name = f"{model_name_prefix}_{market}"

    if mlflow_enabled:
        try:
            client = MlflowClient()
            mv = client.get_model_version_by_alias(registered_model_name, "champion")
            run = client.get_run(mv.run_id)
            result["source"] = "mlflow"
            result["metrics"] = run.data.metrics
            result["version"] = mv.version
            result["run_id"] = mv.run_id
            result["promoted"] = True
            logger.info(f"Champion MLflow trouvé pour {market} : v{mv.version}")
            return result
        except MlflowException as e:
            result["error"] = f"MLflow: {e}"
            logger.warning(f"Alias 'champion' introuvable pour {registered_model_name} : {e}")
        except Exception as e:
            result["error"] = f"MLflow: {e}"
            logger.warning(f"Erreur MLflow pour {registered_model_name} : {e}")

    card_path = model_dir / market / "model_card.json"
    if card_path.exists():
        try:
            with open(card_path, "r") as f:
                card = json.load(f)
            result["source"] = "local"
            metrics = {}
            for k, v in card.get("metrics_ml", {}).items():
                metrics[f"ml_{k}"] = v
            for k, v in card.get("metrics_fin", {}).items():
                metrics[f"fin_{k}"] = v
            result["metrics"] = metrics
            result["promoted"] = card.get("mlflow", {}).get("promoted", False)
            result["run_id"] = card.get("mlflow", {}).get("run_id")
        except Exception as e:
            result["error"] = (result["error"] + " | " if result["error"] else "") + f"model_card.json: {e}"
            logger.error(f"Lecture model_card.json échouée pour {market} : {e}")
    elif result["source"] is None:
        logger.warning(f"Aucun model_card.json trouvé pour {market} ({card_path})")

    return result