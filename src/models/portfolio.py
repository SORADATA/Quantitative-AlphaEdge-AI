import os
import pandas as pd
import numpy as np
import mlflow.xgboost
from pypfopt import EfficientFrontier, risk_models, expected_returns

# Configuration
os.environ["MLFLOW_TRACKING_USERNAME"] = "SORADATA"
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("HF_TOKEN")
mlflow.set_tracking_uri("https://soradata-alphaedge-registry.hf.space")


def generate_portfolio_allocations(market_name="CAC40"):
    print(f"💰 Calcul des allocations pour {market_name}...")

    # 1. Charger les données récentes
    file_path = f"data/processed/{market_name}/{market_name}_final.csv"
    df = pd.read_csv(file_path, index_col=['date'], parse_dates=True)
    
    # 2. Charger le dernier modèle depuis MLflow
    # On récupère la dernière run de l'expérience AlphaEdge_XGBoost_Production
    model_uri = f"runs:/{mlflow.search_runs(experiment_ids=['1']).iloc[0].run_id}/model"
    model = mlflow.xgboost.load_model(model_uri)
    
    # 3. Prédictions
    features = ['rsi', 'macd', 'bb_low', 'bb_high', 'atr', 'euro_volume_lag1', 'cluster']
    latest_data = df.iloc[[-1]] # On prend la dernière ligne
    proba = model.predict_proba(latest_data[features])[:, 1]
    
    # 4. Stratégie : Sélection (Proba > 0.65)
    # Dans un vrai scénario, on compare les probas de tous les tickers de l'univers
    # Ici, on simplifie par marché
    if proba[0] > 0.65:
        print(f"✅ Signal d'achat détecté pour {market_name} !")
        # Ton code Markowitz ici...
        # weights, perf = optimize_portfolio_markowitz(...)
    else:
        print(f"⚠️ Signal faible, maintien en Cash.")
        weights = {"CASH": 1.0}

    # 5. Export pour ton application (Superset/Streamlit)
    alloc_df = pd.DataFrame.from_dict(weights, orient='index', columns=['Weight'])
    alloc_df.to_csv(f"data/processed/{market_name}/current_allocation.csv")
    print(f"🚀 Allocation {market_name} exportée.")

if __name__ == "__main__":
    generate_portfolio_allocations("CAC40")