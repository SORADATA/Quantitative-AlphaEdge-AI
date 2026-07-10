"""
Data Loader
============
Chargement des données portfolio (historique, signaux, rebalancing)
depuis le dataset Hugging Face du dashboard.

Fonctions :
  - load_all_data() : charge portfolio_history / latest_signals / rebalance_history
"""

from datetime import datetime

import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("data_loader")


def load_all_data(market: str, hf_repo_id: str) -> tuple:
    """
    Charge portfolio_history / latest_signals / rebalance_history pour un marché
    depuis le dataset Hugging Face hf_repo_id.

    Chargement défensif (safe_load) : une erreur sur un fichier n'empêche pas
    le chargement des autres. Contrôles métier inclus (colonnes manquantes,
    fraîcheur des données).

    Parameters
    ----------
    market : str — nom du marché (ex: "CAC40")
    hf_repo_id : str — repo HF dataset (ex: "soradata/alphaedge-data")

    Returns
    -------
    tuple — (df_hist, df_signals, df_rebalance, errors)
    """
    clean_market = str(market).strip()
    base_url = f"https://huggingface.co/datasets/{hf_repo_id}/resolve/main/data/{clean_market}"

    errors = []

    def safe_load(url, key):
        try:
            df = pd.read_parquet(url)
            if key in ["hist", "rebal"]:
                df.index = pd.to_datetime(df.index, errors="coerce")
                df = df[df.index.notna()].sort_index(ascending=(key == "hist"))
            return df
        except Exception as e:
            msg = f"Error loading {key}: {e}"
            errors.append(msg)
            logger.warning(msg)
            return pd.DataFrame()

    df_hist = safe_load(f"{base_url}/portfolio_history.parquet", "hist")
    df_signals = safe_load(f"{base_url}/latest_signals.parquet", "signals")
    df_rebalance = safe_load(f"{base_url}/rebalance_history.parquet", "rebal")

    if not df_hist.empty:
        days_old = (datetime.now() - df_hist.index[-1]).days
        if days_old > 7:
            errors.append(f"Portfolio data is {days_old} days old")

    if not df_signals.empty:
        missing = [c for c in ["Ticker", "Signal"] if c not in df_signals.columns]
        if missing:
            errors.append(f"Missing columns in signals: {missing}")

    return df_hist, df_signals, df_rebalance, errors