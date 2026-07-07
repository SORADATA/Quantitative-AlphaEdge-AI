import os
import json
import warnings
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models.signature import infer_signature

from const import DATA_DIR, MODEL_DIR, CONFIG_DIR, SHARPE_THRESHOLD, MAX_DD_THRESHOLD
from src.utils.metrics import calculate_financial_metrics
from src.features.alpha_features import add_all_features
from src.models.ensemble import AlphaEdgeEnsemble, FEATURE_GROUPS
from src.utils.logger import setup_logger

load_dotenv()
warnings.filterwarnings("ignore")
logger = setup_logger("train")

HF_TOKEN = os.getenv("HF_TOKEN")
USE_MLFLOW = bool(HF_TOKEN)

if USE_MLFLOW:
    os.environ["MLFLOW_TRACKING_USERNAME"] = "SORADATA"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = HF_TOKEN
    mlflow.set_tracking_uri("https://soradata-alphaedge-registry.hf.space")
    mlflow.set_experiment("AlphaEdge_Ensemble_Production")
    logger.info(f"MLflow activé — tracking URI : {mlflow.get_tracking_uri()}")
else:
    logger.warning("HF_TOKEN absent — MLflow désactivé.")

def _check_registry_available(client: MlflowClient) -> bool:
    try:
        client.search_registered_models(max_results=1)
        return True
    except MlflowException:
        return False

class AlphaEdgePyFunc(mlflow.pyfunc.PythonModel):
    def __init__(self, model_instance):
        self.model = model_instance

    def predict(self, context, model_input, params=None):
        return self.model.predict_proba(model_input)[:, 1]

def train_pipeline(market_name: str) -> tuple[AlphaEdgeEnsemble, dict]:
    logger.info(f"Début entraînement — {market_name}")

    data_path = DATA_DIR / "processed" / market_name / "monthly_features.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Source introuvable : {data_path}")

    df = pd.read_parquet(data_path)
    df["future_return"] = df.groupby(level="ticker")["adj close"].pct_change(1).shift(-1)
    df["target"] = df["future_return"].gt(0).astype(int)
    df = df.dropna(subset=["target", "future_return"])

    dates = df.index.get_level_values("date")
    split_date = dates.max() - pd.DateOffset(months=6)

    df_train = add_all_features(df[dates <= split_date].copy())
    df_test = add_all_features(df[dates > split_date].copy())

    model = AlphaEdgeEnsemble(n_optuna_trials=50)
    model.fit(df_train, df_train["target"])

    proba = model.predict_proba(df_test)[:, 1]
    final_auc = roc_auc_score(df_test["target"], proba)
    final_apr = average_precision_score(df_test["target"], proba)
    fin_metrics = calculate_financial_metrics(df_test, probas=proba, threshold=0.5)

    market_model_dir = MODEL_DIR / market_name
    market_model_dir.mkdir(parents=True, exist_ok=True)
    model.save(market_model_dir / "ensemble_model.pkl")

    model_card = {
        "market": market_name,
        "metrics_ml": {"auc_test": round(final_auc, 4), "apr_test": round(final_apr, 4)},
        "metrics_fin": fin_metrics,
        "mlflow": {"success": False, "run_id": None, "promoted": False},
    }

    if USE_MLFLOW:
        registered_model_name = f"AlphaEdge_Ensemble_{market_name}"
        client = MlflowClient()

        try:
            with mlflow.start_run(run_name=f"Ensemble_{market_name}") as run:
                mlflow.log_metrics({
                    "AUC_Test": final_auc,
                    "Sharpe_Ratio": fin_metrics["sharpe"],
                    "Max_Drawdown": fin_metrics["max_drawdown"]
                })
                
                pyfunc_model = AlphaEdgePyFunc(model_instance=model)
                mlflow.pyfunc.log_model("ensemble_model", python_model=pyfunc_model)

                model_card["mlflow"].update({"success": True, "run_id": run.info.run_id})

                mv = mlflow.register_model(f"runs:/{run.info.run_id}/ensemble_model", registered_model_name)
                
                champion_sharpe = -999.0
                try:
                    current_champ = client.get_model_version_by_alias(registered_model_name, "champion")
                    champion_sharpe = client.get_run(current_champ.run_id).data.metrics.get("Sharpe_Ratio", -999.0)
                except:
                    logger.info(f"[{market_name}] Premier modèle détecté.")

                if fin_metrics["sharpe"] >= SHARPE_THRESHOLD and fin_metrics["max_drawdown"] >= MAX_DD_THRESHOLD:
                    if fin_metrics["sharpe"] >= champion_sharpe:
                        client.set_registered_model_alias(registered_model_name, "champion", mv.version)
                        model_card["mlflow"]["promoted"] = True
                        logger.info(f"[{market_name}] Promotion réussie : v{mv.version}")
                    else:
                        logger.warning(f"[{market_name}] Challenger battu par Champion.")
                else:
                    logger.warning(f"[{market_name}] Seuils sécurité non atteints.")

        except Exception as e:
            logger.error(f"Erreur MLflow : {e}")

    with open(market_model_dir / "model_card.json", "w") as f:
        json.dump(model_card, f, indent=2)

    return model, model_card

if __name__ == "__main__":
    config_dir = CONFIG_DIR / "markets"
    for config_file in sorted(config_dir.glob("*.json")):
        with open(config_file) as f:
            market = json.load(f).get("market_name")
            if market:
                train_pipeline(market)
