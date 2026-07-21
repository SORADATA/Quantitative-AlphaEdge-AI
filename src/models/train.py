"""
Pipeline d'entraînement AlphaEdge Ensemble.

Entraîne un modèle par marché, l'évalue (test set + walk-forward),
puis gère la promotion "champion" dans le MLflow Model Registry selon
des seuils de sécurité absolus et une comparaison relative au champion
actuel.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import average_precision_score, roc_auc_score

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

from const import CONFIG_DIR, DATA_DIR, MAX_DD_THRESHOLD, MODEL_DIR, SHARPE_THRESHOLD
from src.features.alpha_features import add_all_features
from src.models.ensemble import AlphaEdgeEnsemble
from src.utils.logger import setup_logger
from src.utils.metrics import calculate_financial_metrics
from huggingface_hub import HfApi


load_dotenv()
warnings.filterwarnings("ignore")
logger = setup_logger("train")


# =============================================================================
# CONFIGURATION
# =============================================================================

MLFLOW_TRACKING_URI = "https://soradata-alphaedge-registry.hf.space"
MLFLOW_EXPERIMENT_NAME = "AlphaEdge_Ensemble_Production"
MLFLOW_USERNAME = "SORADATA"

TEST_SET_MONTHS = 6
MIN_TRAIN_ROWS = 100
N_OPTUNA_TRIALS_FINAL = 50

WF_N_WINDOWS = 4
WF_TEST_MONTHS = 3
WF_N_OPTUNA_TRIALS = 20
WF_MIN_TRAIN_ROWS = 50
WF_MIN_TEST_ROWS = 5

# Tolérance de dégradation du Max Drawdown acceptée pour un challenger
# par rapport au champion actuel, avant rejet automatique.
CHALLENGER_DD_TOLERANCE = 0.02
NO_CHAMPION_SENTINEL = -999.0

HF_TOKEN = os.getenv("HF_TOKEN")
USE_MLFLOW = bool(HF_TOKEN)

if USE_MLFLOW:
    os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = HF_TOKEN
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger.info(f"MLflow activé — tracking URI : {mlflow.get_tracking_uri()}")
else:
    logger.warning("HF_TOKEN absent — MLflow désactivé, entraînement local uniquement.")


# =============================================================================
# STRUCTURES DE DONNÉES
# =============================================================================

@dataclass
class ChampionStats:
    """Métriques du modèle champion actuellement enregistré (ou sentinelles si absent)."""
    sharpe: float = NO_CHAMPION_SENTINEL
    sortino: float = NO_CHAMPION_SENTINEL
    max_drawdown: float = NO_CHAMPION_SENTINEL

    @property
    def exists(self) -> bool:
        return self.sharpe != NO_CHAMPION_SENTINEL


@dataclass
class TrainingResult:
    """Résultat consolidé d'un cycle d'entraînement pour un marché."""
    market: str
    auc_test: float
    apr_test: float
    fin_metrics: dict
    wf_auc_mean: float
    mlflow_success: bool = False
    mlflow_run_id: Optional[str] = None
    promoted: bool = False

    def to_model_card(self) -> dict:
        return {
            "market": self.market,
            "trained_at": pd.Timestamp.now().isoformat(),
            "metrics_ml": {
                "auc_test": round(self.auc_test, 4),
                "apr_test": round(self.apr_test, 4),
            },
            "metrics_fin": self.fin_metrics,
            "walk_forward_auc_mean": round(self.wf_auc_mean, 4),
            "mlflow": {
                "success": self.mlflow_success,
                "run_id": self.mlflow_run_id,
                "promoted": self.promoted,
            },
        }


class AlphaEdgePyFunc(mlflow.pyfunc.PythonModel):
    """Wrapper PyFunc pour exposer AlphaEdgeEnsemble via l'API MLflow standard."""

    def __init__(self, model_instance: AlphaEdgeEnsemble):
        self.model = model_instance

    def predict(self, context, model_input: pd.DataFrame, params: Optional[dict] = None):
        return self.model.predict_proba(model_input)[:, 1]


# =============================================================================
# CHARGEMENT & PRÉPARATION DES DONNÉES
# =============================================================================

def _load_market_dataset(market_name: str) -> pd.DataFrame:
    """Charge le parquet mensuel d'un marché et construit la target binaire."""
    data_path = DATA_DIR / "processed" / market_name / "monthly_features.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Fichier source introuvable : {data_path}")

    df = pd.read_parquet(data_path)
    price_col = "adj_close" if "adj_close" in df.columns else "adj close"
    df["future_return"] = df.groupby(level="ticker")[price_col].pct_change(1).shift(-1)
    df["target"] = df["future_return"].gt(0).astype(int)
    return df.dropna(subset=["target", "future_return"])


def _train_test_split_by_date(
    df: pd.DataFrame, test_months: int, market_config: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split temporel (pas de shuffle) : les `test_months` derniers mois servent de test set."""
    dates = df.index.get_level_values("date")
    split_date = dates.max() - pd.DateOffset(months=test_months)
    df_train = add_all_features(df[dates <= split_date].copy(), market_config)
    df_test = add_all_features(df[dates > split_date].copy(), market_config)
    return df_train, df_test


# =============================================================================
# ÉVALUATION
# =============================================================================

def walk_forward_eval(
    df: pd.DataFrame,
    n_windows: int = WF_N_WINDOWS,
    test_months: int = WF_TEST_MONTHS,
    n_optuna_trials: int = WF_N_OPTUNA_TRIALS,
) -> pd.DataFrame:
    """Évaluation par fenêtres glissantes pour valider la stabilité temporelle du modèle."""
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

        if len(df_tr) < WF_MIN_TRAIN_ROWS or len(df_te) < WF_MIN_TEST_ROWS or df_te["target"].nunique() < 2:
            logger.debug(f"WF window {i + 1} ignorée (données insuffisantes).")
            continue

        model = AlphaEdgeEnsemble(n_optuna_trials=n_optuna_trials)
        model.fit(df_tr, df_tr["target"])
        proba = model.predict_proba(df_te)[:, 1]

        result = {
            "window": i + 1,
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "auc": round(roc_auc_score(df_te["target"], proba), 4),
            "apr": round(average_precision_score(df_te["target"], proba), 4),
            "n_test": len(df_te),
        }
        results.append(result)
        logger.info(f"WF Window {result['window']} | AUC: {result['auc']:.4f} | APR: {result['apr']:.4f}")

    return pd.DataFrame(results)


def _evaluate_test_set(model: AlphaEdgeEnsemble, df_test: pd.DataFrame) -> tuple[float, float, dict]:
    """Calcule AUC, APR et métriques financières sur le test set."""
    proba = model.predict_proba(df_test)[:, 1]
    auc = roc_auc_score(df_test["target"], proba)
    apr = average_precision_score(df_test["target"], proba)
    fin_metrics = calculate_financial_metrics(df_test, probas=proba, threshold=0.5)
    return auc, apr, fin_metrics


# =============================================================================
# MLFLOW : PROMOTION DU CHAMPION
# =============================================================================

def _fetch_champion_stats(client: MlflowClient, registered_model_name: str, market_name: str) -> ChampionStats:
    """Récupère les métriques du champion actuel, ou des sentinelles s'il n'existe pas encore."""
    try:
        current_champ = client.get_model_version_by_alias(registered_model_name, "champion")
        champ_metrics = client.get_run(current_champ.run_id).data.metrics
        return ChampionStats(
            sharpe=champ_metrics.get("Sharpe_Ratio", NO_CHAMPION_SENTINEL),
            sortino=champ_metrics.get("Sortino_Ratio", NO_CHAMPION_SENTINEL),
            max_drawdown=champ_metrics.get("Max_Drawdown", NO_CHAMPION_SENTINEL),
        )
    except MlflowException:
        logger.info(f"[{market_name}] Aucun champion trouvé. Déploiement initial.")
        return ChampionStats()


def _should_promote(challenger_sharpe: float, challenger_sortino: float, challenger_max_dd: float,
                    champion: ChampionStats) -> tuple[bool, str]:
    """
    Détermine si le challenger doit être promu champion.

    Règles :
      1. Sécurité absolue : Sharpe et Max Drawdown doivent dépasser les seuils configurés.
      2. Comparaison relative : le Sortino doit être strictement meilleur que celui du
         champion, et le Drawdown ne doit pas se dégrader de plus de CHALLENGER_DD_TOLERANCE.

    Retourne (promu: bool, raison: str).
    """
    passes_safety = (challenger_sharpe >= SHARPE_THRESHOLD) and (challenger_max_dd >= MAX_DD_THRESHOLD)
    if not passes_safety:
        return False, "Sharpe ou Drawdown sous les seuils de sécurité absolus"

    safer_dd = challenger_max_dd >= (champion.max_drawdown - CHALLENGER_DD_TOLERANCE)
    better_sortino = challenger_sortino > champion.sortino

    if not safer_dd:
        return False, "Drawdown trop dégradé par rapport au champion"
    if not better_sortino:
        return False, "Sortino insuffisant par rapport au champion"
    return True, f"Sortino amélioré ({challenger_sortino:.2f} > {champion.sortino:.2f})"


def _log_and_promote_to_mlflow(
    market_name: str,
    model: AlphaEdgeEnsemble,
    df_test: pd.DataFrame,
    result: TrainingResult,
) -> None:
    """Log le run MLflow, enregistre le modèle dans le Registry, et gère la promotion."""
    registered_model_name = f"AlphaEdge_Ensemble_{market_name}"
    client = MlflowClient()
    fin_metrics = result.fin_metrics

    try:
        with mlflow.start_run(run_name=f"Ensemble_{market_name}") as run:
            mlflow.log_metrics({
                "AUC_Test": result.auc_test,
                "WF_AUC_Mean": result.wf_auc_mean,
                "Sharpe_Ratio": fin_metrics.get("sharpe", 0.0),
                "Sortino_Ratio": fin_metrics.get("sortino", 0.0),
                "Calmar_Ratio": fin_metrics.get("calmar", 0.0),
                "Max_Drawdown": fin_metrics.get("max_drawdown", -1.0),
            })

            pyfunc_model = AlphaEdgePyFunc(model_instance=model)
            input_example = df_test[model.features_].head(3)
            signature = infer_signature(input_example, pyfunc_model.predict(None, input_example))

            mlflow.pyfunc.log_model(
                "ensemble_model",
                python_model=pyfunc_model,
                signature=signature,
                input_example=input_example,
            )

            result.mlflow_success = True
            result.mlflow_run_id = run.info.run_id

            model_version = mlflow.register_model(
                f"runs:/{run.info.run_id}/ensemble_model", registered_model_name
            )

            champion = _fetch_champion_stats(client, registered_model_name, market_name)
            promote, reason = _should_promote(
                challenger_sharpe=fin_metrics.get("sharpe", 0.0),
                challenger_sortino=fin_metrics.get("sortino", 0.0),
                challenger_max_dd=fin_metrics.get("max_drawdown", -1.0),
                champion=champion,
            )

            if promote:
                # Promote Mlflow ui
                client.set_registered_model_alias(registered_model_name, "champion", model_version.version)
                result.promoted = True
                logger.info(f"[{market_name}] PROMOTION v{model_version.version} — {reason}")
                try:
                    api = HfApi()
                    # Save on HF
                    local_model_path = MODEL_DIR / market_name / "ensemble_model.pkl"

                    api.upload_file(
                        path_or_fileobj=str(local_model_path),
                        path_in_repo=f"models/{market_name}/champion.pkl",
                        repo_id="soradata/alphaedge-data",
                        repo_type="dataset",
                        token=HF_TOKEN
                    )
                    logger.info(f"[{market_name}] Modèle persistant sauvegardé sur HF Hub ( soradata/alphaedge-data) ")
                except Exception as e:
                    logger.error(f"[{market_name}] Sync faillure on Hf Hub : {e}")

            else:
                logger.warning(f"[{market_name}] CHALLENGER REJETÉ — {reason}")

    except MlflowException as exc:
        logger.error(f"[{market_name}] Erreur durant le flux MLflow : {exc}", exc_info=True)


# =============================================================================
# PIPELINE D'ENTRAÎNEMENT
# =============================================================================

def train_pipeline(market_config: dict) -> tuple[AlphaEdgeEnsemble, dict]:
    """Entraîne, évalue et (le cas échéant) promeut le modèle d'un marché donné."""
    market_name = market_config["market_name"]
    logger.info(f"Début de l'entraînement — {market_name}")

    df = _load_market_dataset(market_name)
    df_train, df_test = _train_test_split_by_date(df, TEST_SET_MONTHS, market_config)

    if len(df_train) < MIN_TRAIN_ROWS:
        raise ValueError(f"Volume de données insuffisant pour {market_name} : {len(df_train)} lignes.")

    model = AlphaEdgeEnsemble(n_optuna_trials=N_OPTUNA_TRIALS_FINAL)
    model.fit(df_train, df_train["target"])

    final_auc, final_apr, fin_metrics = _evaluate_test_set(model, df_test)
    max_dd_pct = fin_metrics.get("max_drawdown", -1.0) * 100
    logger.info(
        f"[{market_name}] Test Set -> AUC: {final_auc:.4f} | "
        f"Sortino: {fin_metrics.get('sortino', 0.0):.2f} | Max DD: {max_dd_pct:.1f}%"
    )

    df_full = pd.concat([df_train, df_test])
    wf_results = walk_forward_eval(df_full)
    wf_auc_mean = wf_results["auc"].mean() if not wf_results.empty else final_auc

    market_model_dir = MODEL_DIR / market_name
    market_model_dir.mkdir(parents=True, exist_ok=True)
    model.save(market_model_dir / "ensemble_model.pkl")

    result = TrainingResult(
        market=market_name,
        auc_test=final_auc,
        apr_test=final_apr,
        fin_metrics=fin_metrics,
        wf_auc_mean=wf_auc_mean,
    )

    if USE_MLFLOW:
        _log_and_promote_to_mlflow(market_name, model, df_test, result)

    model_card = result.to_model_card()
    with open(market_model_dir / "model_card.json", "w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2)

    return model, model_card


# =============================================================================
# ORCHESTRATEUR
# =============================================================================

def _load_configured_markets(config_dir: Path) -> list[dict]:
    """Lit les fichiers de configuration JSON et retourne les configs complètes."""
    configs = []
    for config_file in sorted(config_dir.glob("*.json")):
        with open(config_file, encoding="utf-8") as f:
            market_cfg = json.load(f)
        if not market_cfg.get("market_name"):
            logger.warning(f"Fichier de config sans 'market_name' ignoré : {config_file}")
            continue
        if not market_cfg.get("ff_region"):
            logger.warning(f"Fichier de config sans 'ff_region' ignoré : {config_file}")
            continue
        configs.append(market_cfg)
    return configs


def main() -> None:
    config_dir = CONFIG_DIR / "markets"
    if not config_dir.exists():
        logger.error(f"Dossier de configs introuvable : {config_dir}")
        raise SystemExit(1)

    market_configs = _load_configured_markets(config_dir)
    if not market_configs:
        logger.error(f"Aucun marché configuré trouvé dans {config_dir}")
        raise SystemExit(1)

    failures = []
    for market_config in market_configs:
        market = market_config["market_name"]
        try:
            train_pipeline(market_config)
        except Exception:
            logger.critical(f"[{market}] Échec complet de l'entraînement", exc_info=True)
            failures.append(market)

    if failures:
        logger.error(f"Marchés en échec : {failures}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()