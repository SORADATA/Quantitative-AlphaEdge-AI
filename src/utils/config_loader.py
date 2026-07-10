import json
from pathlib import Path
from typing import Dict, Any
from src.utils.logger import setup_logger
import pandas as pd


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


def get_ticker_names(market: str, base_dir: Path) -> dict:
    """
    Charge le mapping ticker -> nom complet en scannant config/markets/*.json
    et en matchant sur le champ interne "market_name" (insensible à la casse),
    plutôt que sur le nom de fichier — évite les soucis d'incohérence de
    nommage (ex: NASDAQ100 vs nasdaq.json).
    Retourne un dict vide si aucun fichier ne correspond ou si la clé
    ticker_names est absente (fallback silencieux : le ticker brut sera
    affiché à la place).
    """
    markets_dir = base_dir / "config" / "markets"
    if not markets_dir.exists():
        logger.warning(f"Dossier de config introuvable : {markets_dir}")
        return {}

    for config_path in markets_dir.glob("*.json"):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            if config.get("market_name", "").strip().upper() == market.strip().upper():
                return config.get("ticker_names", {})
        except Exception as e:
            logger.warning(f"Lecture échouée pour {config_path} : {e}")
            continue

    logger.warning(f"Aucune config trouvée pour le marché '{market}'")
    return {}

def apply_ticker_names(df: pd.DataFrame, ticker_names: dict, ticker_col: str = "Ticker", name_col: str = "Name") -> pd.DataFrame:
    """
    Ajoute une colonne `name_col` juste après `ticker_col` avec le nom complet
    de chaque ticker (fallback sur le ticker brut si absent du mapping).
    """
    if df.empty or ticker_col not in df.columns:
        return df
    df = df.copy()
    df[name_col] = df[ticker_col].map(ticker_names).fillna(df[ticker_col])
    cols = df.columns.tolist()
    cols.remove(name_col)
    insert_at = cols.index(ticker_col) + 1
    cols.insert(insert_at, name_col)
    return df[cols]