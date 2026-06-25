"""
MarketExtractor
================
Gère le téléchargement des données boursières brutes via yfinance.

Stratégie :
  - Premier run   : téléchargement complet (history_years)
  - Runs suivants : delta uniquement (depuis last_date - 5 jours)
  - Retry automatique (3 tentatives, 5s entre chaque)
  - Validation des données avant sauvegarde
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from const import DATA_DIR
from src.utils.logger import setup_logger

logger = setup_logger("extractor")

_DOWNLOAD_RETRIES = 3
_DOWNLOAD_RETRY_WAIT = 5
_DELTA_OVERLAP_DAYS = 5


class MarketExtractor:
    """
    Gère uniquement le téléchargement des données boursières brutes.

    Parameters
    ----------
    market_name   : str  — nom du marché (ex: 'CAC40')
    tickers       : list[str] — liste des tickers yfinance
    history_years : int  — années d'historique pour le premier téléchargement
    """

    def __init__(
        self,
        market_name: str,
        tickers: list[str],
        history_years: int = 10,
    ):
        self.market_name = market_name
        self.tickers = tickers
        self.history_years = history_years

        self.raw_data_path = DATA_DIR / "raw" / self.market_name
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.file_path = self.raw_data_path / f"{self.market_name}_raw.csv"

        self.data = pd.DataFrame()
        logger.info(
            f"Extracteur initialisé : {self.market_name} ({len(tickers)} tickers)"
        )

    # Helpers privés

    def _load_existing(self) -> Optional[pd.DataFrame]:
        """Charge le CSV existant avec typage strict."""
        if not self.file_path.exists():
            return None
        try:
            df = pd.read_csv(
                self.file_path,
                sep=";",
                index_col=["date", "ticker"],
                parse_dates=True,
                date_format="%Y-%m-%d",
            )
            return df
        except Exception as e:
            logger.warning(f"Impossible de lire le fichier existant ({e}). Full download.")
            return None

    def _validate_download(self, df: pd.DataFrame) -> bool:
        """
        Vérifie que le téléchargement est utilisable.

        Critères :
          - Non vide
          - Au moins 1 ticker valide (non entièrement NaN)
          - Colonne 'adj close' présente
        """
        if df.empty:
            logger.warning("DataFrame téléchargé est vide.")
            return False

        if "adj close" not in df.columns:
            logger.warning(
                f"Colonne 'adj close' absente. Colonnes disponibles : {df.columns.tolist()}"
                )
            return False

        # Vérifier qu'au moins un ticker a des données réelles
        valid_tickers = (
            df["adj close"]
            .unstack("ticker")
            .dropna(how="all", axis=1)
            .columns
            .tolist()
        )
        if not valid_tickers:
            logger.warning("Tous les tickers sont entièrement NaN.")
            return False

        n_expected = len(self.tickers)
        n_valid = len(valid_tickers)
        if n_valid < n_expected:
            logger.warning(
                f" Only {n_valid}/{n_expected} tickers avec des données valides."
            )

        return True

    def _parse_yfinance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise le DataFrame yfinance vers le format (date, ticker) MultiIndex.

        Gère les deux formats yfinance :
          - MultiIndex colonnes (Price, Ticker)  → stack level=1
          - Index simple (un seul ticker)         → ajout colonne ticker
        """
        if isinstance(df.columns, pd.MultiIndex):
            df = df.stack(level=1, future_stack=True)
        elif len(self.tickers) == 1:
            df["ticker"] = self.tickers[0]
            df = df.reset_index().set_index(["Date", "ticker"])

        df.index.names = ["date", "ticker"]
        df.columns = df.columns.str.lower()

        return df

    def _merge_with_existing(
        self,
        existing_df: pd.DataFrame,
        new_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Fusionne les données existantes avec le delta téléchargé.
        En cas de doublon sur (date, ticker), garde la valeur la plus récente (keep='last').
        """
        merged = pd.concat([existing_df, new_df])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        return merged

    # Interface publique

    def fetch_market_data(self) -> Optional[pd.DataFrame]:
        """
        Télécharge les données brutes.
        - Delta si fichier existant (depuis last_date - 5 jours)
        - Téléchargement complet sinon

        Returns
        -------
        pd.DataFrame avec MultiIndex (date, ticker) ou None si échec.
        """
        existing_df = self._load_existing()

        if existing_df is not None:
            last_date = existing_df.index.get_level_values("date").max()
            start_date = (last_date - pd.Timedelta(days=_DELTA_OVERLAP_DAYS)).strftime("%Y-%m-%d")
            logger.info(f"🕒 [{self.market_name}] Mise à jour depuis le {last_date.date()}...")
        else:
            start_date = (
                datetime.today() - pd.DateOffset(days=365 * self.history_years)
            ).strftime("%Y-%m-%d")
            logger.info(f"📥 [{self.market_name}] Premier téléchargement complet...")

        end_date = (datetime.today() + pd.DateOffset(days=1)).strftime("%Y-%m-%d")

        for attempt in range(1, _DOWNLOAD_RETRIES + 1):
            try:
                raw = yf.download(
                    self.tickers,
                    start=start_date,
                    end=end_date,
                    auto_adjust=False,   # garde Adj Close séparé de Close
                    progress=False,
                    threads=True,
                )

                if raw.empty:
                    logger.warning(
                        f"Réponse vide de YFinance (tentative {attempt}/{_DOWNLOAD_RETRIES})"
                        )
                    time.sleep(_DOWNLOAD_RETRY_WAIT)
                    continue

                # Normalisation du format yfinance
                df = self._parse_yfinance(raw)

                # Validation avant fusion
                if not self._validate_download(df):
                    logger.warning(f"Validation échouée (tentative {attempt}/{_DOWNLOAD_RETRIES})")
                    time.sleep(_DOWNLOAD_RETRY_WAIT)
                    continue

                # Fusion avec l'historique existant
                if existing_df is not None:
                    self.data = self._merge_with_existing(existing_df, df)
                else:
                    self.data = df

                # Sauvegarde
                self.data.to_csv(self.file_path, sep=";")
                logger.info(
                    f"[{self.market_name}] Base brute enregistrée : "
                    f"{self.data.shape[0]} lignes, "
                    f"{self.data.index.get_level_values('ticker').nunique()} tickers."
                )

                return self.data

            except Exception as exc:
                logger.warning(
                    f"Erreur de téléchargement (tentative {attempt}/{_DOWNLOAD_RETRIES}): {exc}"
                )
                if attempt < _DOWNLOAD_RETRIES:
                    time.sleep(_DOWNLOAD_RETRY_WAIT)

        logger.error(
            f" Échec du téléchargement pour {self.market_name} "
            f"après {_DOWNLOAD_RETRIES} tentatives."
        )
        return None

    def __repr__(self) -> str:
        n_rows = len(self.data) if not self.data.empty else 0
        return (
            f"MarketExtractor(market={self.market_name}, "
            f"tickers={len(self.tickers)}, rows={n_rows})"
        )
