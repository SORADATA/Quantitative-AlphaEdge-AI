import os
import json
import warnings
from pathlib import Path
from typing import Any

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
    logger.warning("HF_TOKEN absent — MLflow désactivé, entraînement local uniquement.")


# =============================================================================
# HELPERS & VALIDATION
# =============================================================================
class AlphaEdgePyFunc(mlflow.pyfunc.PythonModel):
    def __init__(self, model_instance):
        self.model = model_instance

    def predict(self, context, model_input, params=None):
        return self.model.predict_proba(model_input)[:, 1]

def walk_forward_eval(
    df: pd.DataFrame,
    n_windows: int = 4,
    test_months: int = 3,
    n_optuna_trials: int = 20,
) -> pd.DataFrame:
    """Évaluation robuste par fenêtres glissantes pour valider la stabilité."""
    dates = df.index.get_level_values("date").unique().sort_values()
    results = []

    for i in range(n_windows):
        test_end = dates[-(i * test_months + 1)]
        test_start = dates[-(i * test_months + test_months)]
        train_end = test_start - pd.DateOffset(months=1)

        df_tr = df[df.index.get_level_values("date") <= train_end]
        df_te = df[
            (df.index.get_level_values("date") >= test_start)
            & (df.index.get_level_values("date") <= test_end)
        ]

        if len(df_tr) < 50 or len(df_te) < 5 or len(df_te["target"].unique()) < 2:
            continue

        model = AlphaEdgeEnsemble(n_optuna_trials=n_optuna_trials)
        model.fit(df_tr, df_tr["target"])
        proba = model.predict_proba(df_te)[:, 1]

        results.append({
            "window": i + 1,
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "auc": round(roc_auc_score(df_te["target"], proba), 4),
            "apr": round(average_precision_score(df_te["target"], proba), 4),
            "n_test": len(df_te),
        })
        logger.info(f"WF Window {i+1} | AUC: {results[-1]['auc']:.4f} | APR: {results[-1]['apr']:.4f}")

    return pd.DataFrame(results)


# =============================================================================
# PIPELINE D'ENTRAÎNEMENT
# =============================================================================
def train_pipeline(market_name: str) -> tuple[AlphaEdgeEnsemble, dict]:
    logger.info(f"Début de l'entraînement — {market_name}")

    data_path = DATA_DIR / "processed" / market_name / "monthly_features.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Fichier source introuvable : {data_path}")

    # 1. Préparation des données et de la target
    df = pd.read_parquet(data_path)
    df["future_return"] = df.groupby(level="ticker")["adj_close" if "adj_close" in df.columns else "adj close"].pct_change(1).shift(-1)
    df["target"] = df["future_return"].gt(0).astype(int)
    df = df.dropna(subset=["target", "future_return"])

    dates = df.index.get_level_values("date")
    split_date = dates.max() - pd.DateOffset(months=6)

    df_train = add_all_features(df[dates <= split_date].copy())
    df_test = add_all_features(df[dates > split_date].copy())

    if len(df_train) < 100:
        raise ValueError(f"Volume de données insuffisant pour {market_name}: {len(df_train)} lignes.")

    # 2. Entraînement du modèle
    model = AlphaEdgeEnsemble(n_optuna_trials=50)
    model.fit(df_train, df_train["target"])

    # 3. Évaluation sur Test Set récent
    proba = model.predict_proba(df_test)[:, 1]
    final_auc = roc_auc_score(df_test["target"], proba)
    final_apr = average_precision_score(df_test["target"], proba)
    fin_metrics = calculate_financial_metrics(df_test, probas=proba, threshold=0.5)

    logger.info(f"[{market_name}] Test Set -> AUC: {final_auc:.4f} | Sharpe: {fin_metrics['sharpe']:.2f}")

    # 4. Walk-Forward (pour valider la stabilité)
    df_full = pd.concat([df_train, df_test])
    wf_results = walk_forward_eval(df_full, n_windows=4, test_months=3, n_optuna_trials=20)
    wf_auc_mean = wf_results["auc"].mean() if not wf_results.empty else final_auc

    # 5. Sauvegarde Locale
    market_model_dir = MODEL_DIR / market_name
    market_model_dir.mkdir(parents=True, exist_ok=True)
    model.save(market_model_dir / "ensemble_model.pkl")

    model_card = {
        "market": market_name,
        "trained_at": pd.Timestamp.now().isoformat(),
        "metrics_ml": {"auc_test": round(final_auc, 4), "apr_test": round(final_apr, 4)},
        "metrics_fin": fin_metrics,
        "walk_forward_auc_mean": round(wf_auc_mean, 4),
        "mlflow": {"success": False, "run_id": None, "promoted": False},
    }

    # 6. MLOps : MLflow & Promotion
    if USE_MLFLOW:
        registered_model_name = f"AlphaEdge_Ensemble_{market_name}"
        client = MlflowClient()

        try:
            with mlflow.start_run(run_name=f"Ensemble_{market_name}") as run:
                mlflow.log_metrics({
                    "AUC_Test": final_auc,
                    "WF_AUC_Mean": wf_auc_mean,
                    "Sharpe_Ratio": fin_metrics["sharpe"],
                    "Max_Drawdown": fin_metrics["max_drawdown"]
                })
                
                # Sauvegarde du modèle au format standard PyFunc
                pyfunc_model = AlphaEdgePyFunc(model_instance=model)
                input_example = df_test[model.features_].head(3)
                signature = infer_signature(input_example, pyfunc_model.predict(None, input_example))
                
                mlflow.pyfunc.log_model(
                    "ensemble_model", 
                    python_model=pyfunc_model, 
                    signature=signature,
                    input_example=input_example
                )

                model_card["mlflow"].update({"success": True, "run_id": run.info.run_id})

                # Enregistrement dans le Registry
                mv = mlflow.register_model(f"runs:/{run.info.run_id}/ensemble_model", registered_model_name)
                
                # Le Match : Champion vs Challenger
                champion_sharpe = -999.0
                try:
                    current_champ = client.get_model_version_by_alias(registered_model_name, "champion")
                    champion_sharpe = client.get_run(current_champ.run_id).data.metrics.get("Sharpe_Ratio", -999.0)
                except:
                    logger.info(f"[{market_name}] Aucun champion trouvé. C'est le premier modèle !")

                if fin_metrics["sharpe"] >= SHARPE_THRESHOLD and fin_metrics["max_drawdown"] >= MAX_DD_THRESHOLD:
                    if fin_metrics["sharpe"] >= champion_sharpe:
                        client.set_registered_model_alias(registered_model_name, "champion", mv.version)
                        model_card["mlflow"]["promoted"] = True
                        logger.info(f"[{market_name}] 🏆 PROMOTION RÉUSSIE : v{mv.version} est le nouveau Champion !")
                    else:
                        logger.warning(f"[{market_name}] ❌ CHALLENGER BATTU : {fin_metrics['sharpe']:.2f} < {champion_sharpe:.2f}")
                else:
                    logger.warning(f"[{market_name}] ❌ SÉCURITÉ : Seuils minimums non atteints.")

        except Exception as e:
            logger.error(f"[{market_name}] Erreur durant le flux MLflow : {e}")

    with open(market_model_dir / "model_card.json", "w") as f:
        json.dump(model_card, f, indent=2)

    return model, model_card


# =============================================================================
# ORCHESTRATEUR
# =============================================================================
if __name__ == "__main__":
    config_dir = CONFIG_DIR / "markets"
    failures = []
    
    if not config_dir.exists():
        logger.error(f"Dossier de configs introuvable : {config_dir}")
        raise SystemExit(1)

    for config_file in sorted(config_dir.glob("*.json")):
        with open(config_file) as f:
            market_cfg = json.load(f)
            market = market_cfg.get("market_name")
            
        if market:
            try:
                train_pipeline(market)
            except Exception as e:
                logger.critical(f"[{market}] Échec complet de l'entraînement : {e}", exc_info=True)
                failures.append(market)

    if failures:
        logger.error(f"Marchés en échec : {failures}")
        raise SystemExit(1)
