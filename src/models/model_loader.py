"""
Chargement du modèle "champion" pour l'inférence (daily run / backtest).

Stratégie de résolution, par ordre de priorité :
  1. MLflow Model Registry (alias "champion") si HF_TOKEN est configuré.
  2. Fallback local (pickle) si MLflow est indisponible, désactivé, ou
     qu'aucun alias "champion" n'existe encore.

Un cache en mémoire évite de recharger le même modèle plusieurs fois au
sein d'un même processus (ex: backtest + génération de signaux dans le
même run quotidien).
"""

from __future__ import annotations

import os
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import mlflow
from dotenv import load_dotenv
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from const import MODEL_DIR
from src.utils.logger import setup_logger

load_dotenv()
logger = setup_logger("model_loader")


# =============================================================================
# CONFIGURATION
# =============================================================================

MLFLOW_TRACKING_URI = "https://soradata-alphaedge-registry.hf.space"
MLFLOW_USERNAME = "SORADATA"
CHAMPION_ALIAS = "champion"
LOCAL_MODEL_FILENAME = "ensemble_model.pkl"

HF_TOKEN = os.getenv("HF_TOKEN")
USE_MLFLOW = bool(HF_TOKEN)

if USE_MLFLOW:
    os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = HF_TOKEN
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(f"MLflow activé pour le chargement du champion — tracking URI : {MLFLOW_TRACKING_URI}")
else:
    logger.warning("HF_TOKEN absent — chargement en mode local uniquement.")


# =============================================================================
# CHARGEMENT DEPUIS MLFLOW
# =============================================================================

def _load_champion_from_mlflow(market_name: str) -> Optional[Any]:
    """
    Charge le modèle aliasé 'champion' depuis le MLflow Model Registry.
    Retourne None en cas d'échec (le fallback local prend alors le relais).
    """
    registered_model_name = f"AlphaEdge_Ensemble_{market_name}"
    model_uri = f"models:/{registered_model_name}@{CHAMPION_ALIAS}"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"[{market_name}] Champion chargé depuis MLflow : {model_uri}")
        return model
    except MlflowException as exc:
        logger.warning(f"[{market_name}] Impossible de charger le champion MLflow ({model_uri}) : {exc}")
        return None


# =============================================================================
# CHARGEMENT DEPUIS LE FALLBACK LOCAL
# =============================================================================

def _local_model_path(market_name: str) -> Path:
    return MODEL_DIR / market_name / LOCAL_MODEL_FILENAME


def _load_champion_from_local(market_name: str) -> Optional[Any]:
    """
    Fallback : charge le dernier modèle sauvegardé localement (pickle).
    Utilisé si MLflow est indisponible, désactivé (HF_TOKEN absent),
    ou qu'aucun alias 'champion' n'a encore été promu.
    """
    local_path = _local_model_path(market_name)
    if not local_path.exists():
        logger.error(f"[{market_name}] Aucun modèle local trouvé à {local_path}")
        return None

    try:
        with open(local_path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"[{market_name}] Modèle chargé depuis le fallback local : {local_path}")
        return model
    except (pickle.UnpicklingError, EOFError, AttributeError, ModuleNotFoundError) as exc:
        # Ces erreurs signalent typiquement un fichier corrompu ou une
        # incompatibilité de version entre l'environnement d'entraînement
        # et celui d'inférence (classe déplacée/renommée, version sklearn...).
        logger.error(f"[{market_name}] Fichier pickle illisible ou incompatible ({local_path}) : {exc}")
        return None


# =============================================================================
# POINT D'ENTRÉE PUBLIC
# =============================================================================

@lru_cache(maxsize=None)
def load_champion(market_name: str) -> Any:
    """
    Point d'entrée unique utilisé par run_pipeline.py (daily run).

    Priorité : champion MLflow -> fallback local.
    Le résultat est mis en cache par marché pour la durée du processus,
    afin d'éviter des appels réseau MLflow redondants si le pipeline
    (backtest + signaux live) charge le champion plusieurs fois.

    Lève une exception si aucun modèle n'est disponible : le pipeline
    ne doit jamais tourner sans modèle.
    """
    model = _load_champion_from_mlflow(market_name) if USE_MLFLOW else None

    if model is None:
        model = _load_champion_from_local(market_name)

    if model is None:
        raise RuntimeError(
            f"[{market_name}] Aucun modèle champion disponible "
            "(ni MLflow, ni local). Impossible de générer les signaux."
        )

    return model


def clear_champion_cache(market_name: Optional[str] = None) -> None:
    """
    Vide le cache de load_champion.

    Utile après une nouvelle promotion (le champion vient de changer sur
    MLflow) ou dans les tests, pour forcer un rechargement.
    Note : lru_cache ne permet pas d'invalider une seule clé nativement,
    donc on vide tout le cache quel que soit `market_name` fourni.
    """
    load_champion.cache_clear()
    if market_name:
        logger.info(f"[{market_name}] Cache du champion invalidé.")
    else:
        logger.info("Cache du champion invalidé pour tous les marchés.")