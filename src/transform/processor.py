"""
MarketDataProcessor
====================
Pipeline de transformation complète :
  1. Validation & nettoyage des tickers
  2. Indicateurs techniques (daily)
  3. Agrégation mensuelle
  4. Returns winsorisés (mensuel)
  5. Betas Fama-French rolling (mensuel)
  6. Alpha features
  7. Lag des variables (anti-leakage)
"""

import pandas as pd
from typing import Tuple, List

from const import VARS_TO_LAG, RESAMPLE_MEAN_COLS, RESAMPLE_LAST_EXCLUDE
# processor.py — imports corrects
from src.features.alpha_features import (
    compute_technical_indicators,
    calculate_returns,
    get_fama_french_betas,
)
from src.features.alpha_features import add_all_features
from src.transform.ticker_manager import validate_and_clean_tickers
from src.utils.logger import setup_logger


logger = setup_logger("processor")


class MarketDataProcessor:
    """
    Encapsule toute la logique de transformation et de nettoyage des données.

    Parameters
    ----------
    active_tickers : list[str]
        Liste des tickers actifs sur le marché considéré.
    """

    def __init__(self, active_tickers: List[str]):
        self.active_tickers = active_tickers

    # ── Agrégation mensuelle
    def _resample_to_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Agrège les données journalières en mensuel (Business Month End).

        - euro_volume : moyenne mensuelle (RESAMPLE_MEAN_COLS)
        - autres cols : dernière valeur du mois (last)
        """
        last_cols = [c for c in df.columns if c not in RESAMPLE_LAST_EXCLUDE]

        mean_part = (
            df.unstack("ticker")[RESAMPLE_MEAN_COLS[0]]
            .resample("BME").mean()
            .stack("ticker")
            .to_frame(RESAMPLE_MEAN_COLS[0])
        )
        last_part = (
            df.unstack()[last_cols]
            .resample("BME").last()
            .stack("ticker", future_stack=True)
        )

        monthly = pd.concat([mean_part, last_part], axis=1).dropna(how="all")

        # S'assurer que l'index est bien ordonné (date, ticker)
        monthly = monthly.sort_index()

        return monthly

    # Lag des variables macro/volume
    def _apply_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Lag d'une période pour toutes les variables dans VARS_TO_LAG.
        
        Note : les lags des indicateurs TA (rsi, macd, bb_*, atr, cluster)
        sont gérés directement dans alpha_features._lag_ta_indicators()
        pour éviter la duplication.
        """
        for col in VARS_TO_LAG:
            if col in df.columns:
                df[f"{col}_lag1"] = df.groupby(level="ticker")[col].shift(1)
        return df

    # Pipeline principale
    def process(self, raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """
        Exécute la pipeline de transformation complète.

        Parameters
        ----------
        raw_df : pd.DataFrame
            Données brutes multi-index (date, ticker) issues de l'extracteur.

        Returns
        -------
        df_daily   : pd.DataFrame — données journalières enrichies
        df_monthly : pd.DataFrame — données mensuelles avec toutes les features
        alerts     : dict         — alertes de validation des tickers
        """
        logger.info("Début du processing des données...")
        df = raw_df.copy()
        if "adj close" not in df.columns and "close" in df.columns:
            df["adj close"] = df["close"]
            logger.warning("adj close absent — utilisation de close comme proxy.")
        df, valid_tickers, alerts = validate_and_clean_tickers(df, self.active_tickers)

        df = compute_technical_indicators(df)

        logger.info("Agrégation à la fréquence mensuelle...")
        df_monthly = self._resample_to_monthly(df)

        df_monthly = df_monthly.groupby(
            level=1, group_keys=False
        ).apply(calculate_returns)

        df_monthly = get_fama_french_betas(df_monthly)

        # alpha features
        df_monthly = add_all_features(df_monthly)

        df_monthly = self._apply_lags(df_monthly)

        # Final verif
        n_features = df_monthly.shape[1]
        n_obs = len(df_monthly)
        logger.info(
            f"Processing terminé avec succès. "
            f"Monthly shape : ({n_obs}, {n_features})"
        )

        return df, df_monthly, alerts
