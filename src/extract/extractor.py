import os
import warnings
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
from sklearn.cluster import KMeans
from huggingface_hub import HfApi
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import MACD

warnings.filterwarnings('ignore')


class MarketExtractor:
    def __init__(self, config_dict: dict, history_years: int = 10):
        self.market_name = config_dict['market_name']
        self.tickers = config_dict['tickers']
        self.ff_region = config_dict.get('ff_region')
        self.history_years = history_years
        # Chemins dynamiques (racine du projet)
        self.root_dir = Path(__file__).resolve().parent.parent
        self.raw_data_path = self.root_dir / "data" / "raw" / self.market_name
        self.processed_data_path = self.root_dir / "data" / "processed" / self.market_name
        
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        self.data = pd.DataFrame()
        self.monthly_data = pd.DataFrame()
        
        print(f" Extracteur initialisé : {self.market_name}")

    def fetch_market_data(self):
        file_path = self.raw_data_path / f"{self.market_name}_raw.csv"
        
        # Logique Delta : si le fichier existe, on ne récupère que les jours manquants
        if file_path.exists():
            existing_df = pd.read_csv(file_path, sep=';', index_col=['date', 'ticker'], parse_dates=True)
            last_date = existing_df.index.get_level_values('date').max()
            start_date = (last_date - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            print(f"🕒 [{self.market_name}] Mise à jour depuis le {last_date.date()}...")
        else:
            start_date = (datetime.today() - pd.DateOffset(days=365 * self.history_years)).strftime('%Y-%m-%d')
            print(f"📥 [{self.market_name}] Premier téléchargement complet...")

        df = yf.download(self.tickers, start=start_date, auto_adjust=False, progress=False).stack()
        df.index.names = ['date', 'ticker']
        df.columns = df.columns.str.lower()
        
        # Fusionner avec les anciennes données si nécessaire
        if file_path.exists():
            self.data = pd.concat([existing_df, df]).drop_duplicates().sort_index()
        else:
            self.data = df
            
        self.data.to_csv(file_path, sep=';')
        print(f" [{self.market_name}] Base brute : {self.data.shape[0]} observations.")

    def compute_features(self):
        print(f"⚙️ [{self.market_name}] Calcul des indicateurs...")
        df = self.data.copy()
        df['garman_klass_vol'] = ((np.log(df['high']/df['low'])**2)/2 - (2*np.log(2)-1)*((np.log(df['adj close']/df['open']))**2))
        df['rsi'] = df.groupby('ticker', group_keys=False)['adj close'].apply(
            lambda x: RSIIndicator(close=x, window=20).rsi() if len(x) > 20 else pd.Series(np.nan, index=x.index)
        )
        df['euro_volume'] = (df['adj close'] * df['volume']) / 1e6
        self.data = df.dropna(subset=['rsi', 'garman_klass_vol'])

    def aggregate_and_cluster(self):
        print(f" [{self.market_name}] Agrégation et Clustering...")
        df_indexed = self.data.reset_index().set_index(['date', 'ticker']).sort_index()
        self.monthly_data = df_indexed.resample('BME', level='date').last() 
        kmeans = KMeans(n_clusters=4, init=np.array([[30], [45], [55], [70]]), n_init=1, random_state=42)
        self.monthly_data['cluster'] = kmeans.fit_predict(self.monthly_data[['rsi']].fillna(50))

    def upload_to_hf(self, repo_id="soradata/alphaedge-data"):
        print(f" [{self.market_name}] Upload HF : {repo_id}")
        api = HfApi(token=os.getenv("HF_TOKEN"))
        file_path = self.processed_data_path / f"{self.market_name}_final.csv"
        self.monthly_data.to_csv(file_path)
        
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=f"{self.market_name}/data.csv",
            repo_id=repo_id,
            repo_type="dataset"
        )

    def run_all(self):
        self.fetch_market_data()
        self.compute_features()
        self.aggregate_and_cluster()
        self.upload_to_hf()