import pandas as pd
from typing import Tuple, List

from const import VARS_TO_LAG, RESAMPLE_MEAN_COLS, RESAMPLE_LAST_EXCLUDE
from src.transform.features import (
    compute_technical_indicators,
    calculate_returns,
    get_fama_french_betas
)
from src.transform.ticker_manager import validate_and_clean_tickers
from src.utils.logger import setup_logger

logger = setup_logger("processor")


class MarketDataProcessor:
    """Encapsule toute la logique de transformation et de nettoyage des données."""
    def __init__(self, active_tickers: List[str]):
        self.active_tickers = active_tickers

    def _resample_to_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Agrége les données journalières en mensuel (Business Month End)."""
        last_cols = [c for c in df.columns if c not in RESAMPLE_LAST_EXCLUDE]

        monthly = pd.concat(
            [
                df.unstack("ticker")[RESAMPLE_MEAN_COLS[0]]
                .resample("BM").mean()
                .stack("ticker")
                .to_frame(RESAMPLE_MEAN_COLS[0]),
                df.unstack()[last_cols].resample("BM").last().stack("ticker"),
            ],
            axis=1,
        ).dropna()
        return monthly

    def process(self, raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Exécute la pipeline de transformation complète."""
        logger.info("Début du processing des données...")
        df = raw_df.copy()
        # Sécurité sur les colonnes de base
        if "adj close" not in df.columns and "close" in df.columns:
            df["adj close"] = df["close"]

        # Validation et nettoyage
        df, valid_tickers, alerts = validate_and_clean_tickers(df, self.active_tickers)
        df = compute_technical_indicators(df)
        logger.info("Agrégation à la fréquence mensuelle...")
        df_monthly = self._resample_to_monthly(df)
        df_monthly = df_monthly.groupby(level=1, group_keys=False).apply(calculate_returns)
        df_monthly = get_fama_french_betas(df_monthly)
        for col in VARS_TO_LAG:
            if col in df_monthly.columns:
                df_monthly[f"{col}_lag1"] = df_monthly.groupby("ticker")[col].shift(1)

        logger.info("Processing terminé avec succès.")
        return df, df_monthly, alerts
