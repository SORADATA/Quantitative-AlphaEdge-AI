import os
import warnings
import pickle
import pandas as pd
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import mlflow
import mlflow.xgboost
from dotenv import load_dotenv

# Import de tes constantes globales
from const import DATA_DIR, MODEL_DIR, FEATURE_COLS

load_dotenv()
warnings.filterwarnings('ignore')

# ── Credentials & MLflow ──────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError("❌ HF_TOKEN non défini dans le fichier .env")

os.environ["MLFLOW_TRACKING_USERNAME"] = "SORADATA"
os.environ["MLFLOW_TRACKING_PASSWORD"] = HF_TOKEN

mlflow.set_tracking_uri("https://soradata-alphaedge-registry.hf.space")
mlflow.set_experiment("AlphaEdge_XGBoost_Production")

# ── Training ──────────────────────────────────────────────────
def train_pipeline(market_name: str = "CAC40"):
    # 1. Chargement des données traitées par l'ETL
    data_path = DATA_DIR / "processed" / market_name / "monthly_features.parquet"
    
    if not data_path.exists():
        raise FileNotFoundError(f"❌ Données introuvables : {data_path}. Lance d'abord l'ETL.")
        
    print(f"📂 Chargement des données : {data_path}")
    df = pd.read_parquet(data_path)

    # 2. Création de la Target (Prédiction : Le rendement du mois SUIVANT est-il positif ?)
    # Sur des données mensuelles, pct_change(1).shift(-1) donne le rendement de T+1
    df['target'] = (
        df.groupby(level='ticker')['adj close']
        .pct_change(1)
        .shift(-1)
        .gt(0).astype(int)
    )

    # Sécurité : on retire la dernière ligne de chaque ticker car on ne connaît pas son futur (target = NaN)
    df = df.dropna(subset=['target', 'rsi']) 

    # 3. Split temporel (pas de fuite de données)
    dates = df.index.get_level_values('date')
    split_date = dates.max() - pd.DateOffset(months=6)

    train_mask = dates <= split_date
    df_train = df[train_mask].copy()
    df_test = df[~train_mask].copy()

    print(f"📊 Période train : {dates[train_mask].min().date()} → {dates[train_mask].max().date()}")
    print(f"📊 Période test  : {dates[~train_mask].min().date()} → {dates[~train_mask].max().date()}")

    # 4. Entraînement et sauvegarde du modèle KMeans
    print("🧩 Entraînement du modèle KMeans (RSI Clustering)...")
    import numpy as np
    kmeans = KMeans(
        n_clusters=4, 
        init=np.array([[30], [45], [55], [70]]), 
        n_init=1, 
        random_state=42
    )
    
    # On fit uniquement sur le Train Set pour éviter le data leakage
    df_train['cluster'] = kmeans.fit_predict(df_train[['rsi']].fillna(50))
    df_test['cluster'] = kmeans.predict(df_test[['rsi']].fillna(50))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / 'kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
    print("✅ Modèle KMeans sauvegardé.")

    # 5. Préparation X et y pour XGBoost
    # On s'assure que toutes les features requises par `const.FEATURE_COLS` sont présentes
    missing_cols = [c for c in FEATURE_COLS if c not in df_train.columns]
    if missing_cols:
        raise ValueError(f"❌ Features manquantes dans le dataset : {missing_cols}")

    df_train = df_train.dropna(subset=FEATURE_COLS)
    df_test = df_test.dropna(subset=FEATURE_COLS)

    X_train, y_train = df_train[FEATURE_COLS], df_train['target']
    X_test,  y_test  = df_test[FEATURE_COLS],  df_test['target']

    # 6. MLflow run & XGBoost Training
    with mlflow.start_run(run_name=f"Training_{market_name}") as run:
        print(f"🤖 Entraînement XGBoost pour {market_name}...")

        param_grid = {
            'max_depth':     [3, 4, 5],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators':  [100, 200, 300],
            'subsample':     [0.8, 1.0],
        }

        xgb_base = xgb.XGBClassifier(
            eval_metric='auc',
            random_state=42,
            n_jobs=-1
        )
        tscv = TimeSeriesSplit(n_splits=5)

        grid_search = GridSearchCV(
            xgb_base, param_grid,
            scoring='roc_auc',
            cv=tscv,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # 7. Sauvegarde locale du XGBoost (Pour daily_run.py)
        with open(MODEL_DIR / 'xgboost_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        print("✅ Modèle XGBoost sauvegardé en local.")

        # 8. Métriques et Log MLflow
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        final_auc    = roc_auc_score(y_test, y_pred_proba)
        cv_auc       = grid_search.best_score_

        print(f"📈 AUC CV (train) : {cv_auc:.4f}")
        print(f"📈 AUC Test       : {final_auc:.4f}")

        mlflow.log_params(grid_search.best_params_)
        mlflow.log_param("market", market_name)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size",  len(X_test))
        mlflow.log_metric("AUC_CV",   cv_auc)
        mlflow.log_metric("AUC_Test", final_auc)

        mlflow.xgboost.log_model(xgb_model=best_model, name="model")
        mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/model",
            name=f"AlphaEdge_XGBoost_{market_name}"
        )

        print(f"✅ Modèle enregistré sur Hugging Face (MLflow) !")

if __name__ == "__main__":
    import json
    
    # On lit la config pour savoir quel marché entraîner (ex: CAC40)
    config_path = Path("config/markets/cac40.json") 
    if config_path.exists():
        with open(config_path, 'r') as f:
            market_config = json.load(f)
        train_pipeline(market_config['market_name'])
    else:
        # Fallback de sécurité
        train_pipeline("CAC40")