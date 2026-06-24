import os
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import roc_auc_score
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import MACD
import xgboost as xgb
import mlflow
import mlflow.xgboost
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings('ignore')

# ── Credentials & MLflow ──────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError("❌ HF_TOKEN non défini")

os.environ["MLFLOW_TRACKING_USERNAME"] = "SORADATA"
os.environ["MLFLOW_TRACKING_PASSWORD"] = HF_TOKEN

mlflow.set_tracking_uri("https://soradata-alphaedge-registry.hf.space")
mlflow.set_experiment("AlphaEdge_XGBoost_Production")

try:
    mlflow.tracking.MlflowClient().search_experiments()
    print("✅ MLflow connecté")
except Exception as e:
    print(f"❌ MLflow inaccessible : {e}")


# ── Feature Engineering ───────────────────────────────────────
def _add_features(g: pd.DataFrame) -> pd.DataFrame:
    close = g['adj close']
    g['rsi']              = RSIIndicator(close, window=20).rsi()
    g['macd']             = MACD(close).macd()
    bb                    = BollingerBands(close)
    g['bb_low']           = bb.bollinger_lband()
    g['bb_high']          = bb.bollinger_hband()
    g['atr']              = AverageTrueRange(g['high'], g['low'], close).average_true_range()
    g['euro_volume']      = (close * g['volume']) / 1e6
    g['euro_volume_lag1'] = g['euro_volume'].shift(1)
    # Target : rendement J+1 positif ?
    g['target']           = (close.pct_change(1).shift(-1) > 0).astype(int)
    return g


def _prepare_data(market_name: str) -> pd.DataFrame:
    raw_path = f"data/raw/{market_name.lower()}_dataset.csv"
    out_path = Path(f"data/processed/{market_name}/{market_name}_final.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"⚙️  Génération des features depuis {raw_path}...")
    df = pd.read_csv(raw_path, index_col=['date', 'ticker'], parse_dates=True)

    # Features calculées par ticker
    df = df.groupby('ticker', group_keys=False).apply(_add_features)

    # Clustering RSI
    kmeans = KMeans(
        n_clusters=4,
        init=np.array([[30], [45], [55], [70]]),
        n_init=1,
        random_state=42
    )
    df['cluster'] = kmeans.fit_predict(df[['rsi']].fillna(50))

    df = df.dropna(subset=['rsi', 'target'])
    df.to_csv(out_path)
    print(f"✅ Fichier généré : {out_path} ({df.shape})")
    return df


# ── Training ──────────────────────────────────────────────────
def train_model(market_name: str = "CAC40"):
    out_path = f"data/processed/{market_name}/{market_name}_final.csv"

    # Génère les features si absent
    if not os.path.exists(out_path):
        df = _prepare_data(market_name)
    else:
        print(f"📂 Chargement : {out_path}")
        df = pd.read_csv(out_path, index_col=['date', 'ticker'], parse_dates=True)

    # ── Préparation features / target ────────────────────────
    feature_cols = ['rsi', 'macd', 'bb_low', 'bb_high', 'atr', 'euro_volume_lag1', 'cluster']
    df = df.dropna(subset=feature_cols + ['target'])

    # Recalcule target si colonne absente (sécurité)
    if 'target' not in df.columns:
        df['target'] = (
            df.groupby('ticker')['adj close']
            .pct_change(1).shift(-1).gt(0).astype(int)
        )

    X = df[feature_cols]
    y = df['target']

    # ── Split temporel (pas de fuite de données) ─────────────
    # On coupe sur la date uniquement (level=0 du MultiIndex)
    dates = df.index.get_level_values('date')
    split_date = dates.max() - pd.DateOffset(months=6)

    train_mask = dates <= split_date
    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[~train_mask], y[~train_mask]

    print(f"📊 Train : {X_train.shape} | Test : {X_test.shape}")
    print(f"   Période train : {dates[train_mask].min().date()} → {dates[train_mask].max().date()}")
    print(f"   Période test  : {dates[~train_mask].min().date()} → {dates[~train_mask].max().date()}")

    # ── MLflow run ───────────────────────────────────────────
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
            n_jobs=-1,
            use_label_encoder=False
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

        # Métriques
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        final_auc    = roc_auc_score(y_test, y_pred_proba)
        cv_auc       = grid_search.best_score_

        print(f"   AUC CV (train) : {cv_auc:.4f}")
        print(f"   AUC Test       : {final_auc:.4f}")

        # Log MLflow
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_param("market", market_name)
        mlflow.log_param("train_size", int(train_mask.sum()))
        mlflow.log_param("test_size",  int((~train_mask).sum()))
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_metric("AUC_CV",   cv_auc)
        mlflow.log_metric("AUC_Test", final_auc)

        # Log & register model
        mlflow.xgboost.log_model(xgb_model=best_model, name="model")

        mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/model",
            name=f"AlphaEdge_XGBoost_{market_name}"
        )

        print(f"✅ Modèle {market_name} enregistré ! AUC Test: {final_auc:.4f}")


if __name__ == "__main__":
    train_model("CAC40")