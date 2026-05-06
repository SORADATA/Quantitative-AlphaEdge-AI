# TopK DropOut

# src/strategy/signals.py
import pandas as pd
from typing import Any, List


class AlphaSignal:
    def __init__(self, predictions_df: pd.DataFrame):
        # C'est ici que l'attribut est créé
        self.signal_cache = predictions_df.sort_index()

    @classmethod
    def from_xgboost_kmeans(cls, df_features: pd.DataFrame, xgb_model: Any, kmeans_model: Any, feature_cols: List[str]):
        df_preds = df_features.copy()
        
        # Prédictions vectorisées
        if "rsi" in df_preds.columns:
            df_preds["cluster"] = kmeans_model.predict(df_preds[["rsi"]].fillna(50))
            
        df_preds["proba_upside"] = xgb_model.predict_proba(df_preds[feature_cols].fillna(0))[:, 1]
        
        # CRITIQUE : On retourne une instance de la classe
        # Cela va appeler le __init__ et créer signal_cache
        return cls(df_preds[["proba_upside", "cluster"]])

    def get_signal(self, target_date: pd.Timestamp) -> pd.DataFrame:
        try:
            return self.signal_cache.xs(target_date, level="date")
        except (KeyError, AttributeError):
            return pd.DataFrame()