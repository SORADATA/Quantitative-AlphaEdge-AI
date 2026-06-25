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


class MarketExtractor:
    """Gère uniquement le téléchargement des données boursières brutes."""
    
    def __init__(self, market_name: str, tickers: list[str], history_years: int = 10):
        self.market_name = market_name
        self.tickers = tickers
        self.history_years = history_years
        
        # Définition du chemin de stockage brut
        self.raw_data_path = DATA_DIR / "raw" / self.market_name
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.file_path = self.raw_data_path / f"{self.market_name}_raw.csv"
        
        self.data = pd.DataFrame()
        logger.info(f"Extracteur initialisé : {self.market_name} ({len(tickers)} tickers)")

    def fetch_market_data(self) -> Optional[pd.DataFrame]:
        """Télécharge les données brutes (Delta si fichier existant, complet sinon)."""
        
        if self.file_path.exists():
            existing_df = pd.read_csv(self.file_path, sep=';', index_col=['date', 'ticker'], parse_dates=True)
            last_date = existing_df.index.get_level_values('date').max()
            start_date = (last_date - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            logger.info(f"🕒 [{self.market_name}] Mise à jour depuis le {last_date.date()}...")
        else:
            start_date = (datetime.today() - pd.DateOffset(days=365 * self.history_years)).strftime('%Y-%m-%d')
            logger.info(f"📥 [{self.market_name}] Premier téléchargement complet...")

        end_date = (datetime.today() + pd.DateOffset(days=1)).strftime("%Y-%m-%d")

        for attempt in range(1, _DOWNLOAD_RETRIES + 1):
            try:
                df = yf.download(
                    self.tickers, 
                    start=start_date, 
                    end=end_date,
                    auto_adjust=False, 
                    progress=False,
                    threads=True
                )
                
                if df.empty:
                    logger.warning(f"Réponse vide de YFinance (tentative {attempt})")
                    time.sleep(_DOWNLOAD_RETRY_WAIT)
                    continue

                # Gestion du MultiIndex yfinance (les nouvelles versions empilent les tickers en colonnes)
                if isinstance(df.columns, pd.MultiIndex):
                    df = df.stack(level=1, future_stack=True)
                    
                df.index.names = ['date', 'ticker']
                df.columns = df.columns.str.lower()
                
                # Fusion avec l'historique si la logique Delta s'applique
                if self.file_path.exists():
                    # self.data = pd.concat([existing_df, df]).drop_duplicates().sort_index()
                    self.data = pd.concat([existing_df, df])
                    self.data = self.data[~self.data.index.duplicated(keep='last')].sort_index()
                else:
                    self.data = df
                    
                # Sauvegarde physique des données brutes
                self.data.to_csv(self.file_path, sep=';')
                logger.info(f"✅ [{self.market_name}] Base brute enregistrée : {self.data.shape[0]} lignes.")
                
                return self.data

            except Exception as exc:
                logger.warning(f"Erreur de téléchargement (tentative {attempt}): {exc}")
                time.sleep(_DOWNLOAD_RETRY_WAIT)
                
        logger.error(f"❌ Échec du téléchargement pour {self.market_name} après {_DOWNLOAD_RETRIES} tentatives.")
        return None