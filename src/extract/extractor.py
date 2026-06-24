import os
import warnings
from pathlib import Path
import yfinance as yf
import pandas as pd

# Désactivation des avertissements
warnings.filterwarnings('ignore')

class MarketExtractor:
    def __init__(self, market_name: str, tickers: list):
        """
        Initialise l'extracteur pour un marché spécifique de manière dynamique.
        """
        self.market_name = market_name
        self.tickers = tickers
        
        # 1. Chemins dynamiques : On remonte à la racine du projet peu importe la machine
        # (__file__ est le chemin de ce script, parent.parent remonte de src/ à la racine)
        self.root_dir = Path(__file__).resolve().parent.parent
        self.raw_data_path = self.root_dir / "data" / "raw" / self.market_name
        self.processed_data_path = self.root_dir / "data" / "processed" / self.market_name
        
        # 2. Création automatique des dossiers par marché (ex: data/raw/CAC40/)
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Extracteur initialisé pour le marché : {self.market_name}")

    def fetch_data(self, period_years=10):
        """
        Télécharge les données et les sauvegarde dans le bon dossier.
        """
        print(f"📥 Téléchargement des données pour {len(self.tickers)} actifs...")
        
        # ... Ton code yfinance actuel ...
        # df = yf.download(self.tickers, ...)
        
        # Sauvegarde dynamique
        # file_path = self.raw_data_path / f"{self.market_name}_raw.csv"
        # df.to_csv(file_path)
        # print(f"💾 Données sauvegardées dans {file_path}")

    def compute_features(self):
        """
        Calcule le RSI, MACD, etc.
        """
        print(f"⚙️ Calcul des indicateurs techniques pour {self.market_name}...")
        # Ton code avec la librairie 'ta'
        pass