import json
from pathlib import Path
from typing import Dict, Any
from src.utils.logger import setup_logger

logger = setup_logger("ConfigLoader")


def load_market_config(config_path: Path) -> Dict[str, Any]:
    """Charge une configuration spécifique depuis un fichier JSON donné."""
    if not config_path.exists():
        logger.error(f"Fichier de config introuvable : {config_path}")
        return {}
        
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Erreur JSON dans {config_path}: {e}")
        return {}