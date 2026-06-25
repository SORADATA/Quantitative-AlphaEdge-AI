"""
Train Pipeline — AlphaEdge Ensemble
=====================================
XGBoost + LightGBM + Ridge → LogisticRegression (stacking).
Optuna Bayesian HPO + PurgedTimeSeriesSplit + Walk-Forward Validation.
"""

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import mlflow

from const import DATA_DIR, MODEL_DIR, CONFIG_DIR
from src.features.alpha_features import add_all_features
from src.models.ensemble import AlphaEdgeEnsemble, FEATURE_GROUPS
from src.utils.logger import setup_logger

load_dotenv()
warnings.filterwarnings("ignore")
logger = setup_logger("train")

HF_TOKEN   = os.getenv("HF_TOKEN")
USE_MLFLOW = bool(HF_TOKEN)

if USE_MLFLOW:
    os.environ["MLFLOW_TRACKING_USERNAME"] = "SORADATA"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = HF_TOKEN
    mlflow.set_tracking_uri("https://soradata-alphaedge-registry.hf.space")
    mlflow.set_experiment("AlphaEdge_Ensemble_Production")


# ══════════════════════════════════════════════════════════════════
# WALK-FORWARD VALIDATION
# ══════════════════════════════════════════════════════════════════

def walk_forward_eval(
    df: pd.DataFrame,
    n_windows: int = 4,
    test_months: int = 3,
    n_optuna_trials: int = 20,
) -> pd.DataFrame:
    """
    Évalue la robustesse sur N fenêtres glissantes out-of-sample.
    Chaque fenêtre entraîne un modèle fresh sur train[: window_start]
    et évalue sur les test_months suivants.
    """
    dates   = df.index.get_level_values("date").unique().sort_values()
    results = []

    for i in range(n_windows):
        test_end   = dates[-(i * test_months + 1)]
        test_start = dates[-(i * test_months + test_months)]
        train_end  = test_start - pd.DateOffset(months=1)

        df_tr = df[df.index.get_level_values("date") <= train_end]
        df_te = df[
            (df.index.get_level_values("date") >= test_start) &
            (df.index.get_level_values("date") <= test_end)
        ]

        if len(df_tr) < 50 or len(df_te) < 5:
            continue
        if len(df_te["target"].unique()) < 2:
            continue

        model = AlphaEdgeEnsemble(n_optuna_trials=n_optuna_trials)
        model.fit(df_tr, df_tr["target"])
        proba = model.predict_proba(df_te)[:, 1]

        results.append({
            "window":     i + 1,
            "test_start": str(test_start.date()),
            "test_end":   str(test_end.date()),
            "auc":        round(roc_auc_score(df_te["target"], proba), 4),
            "apr":        round(average_precision_score(df_te["target"], proba), 4),
            "n_test":     len(df_te),
        })
        logger.info(
            f"   Window {i+1} | AUC: {results[-1]['auc']:.4f} "
            f"| APR: {results[-1]['apr']:.4f} | n={results[-1]['n_test']}"
        )

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════

def train_pipeline(market_name: str = "CAC40") -> tuple[AlphaEdgeEnsemble, dict]:
    logger.info(f"\n{'='*60}")
    logger.info(f"  AlphaEdge Training — {market_name}")
    logger.info(f"{'='*60}")

    # ── 1. Chargement ─────────────────────────────────────────
    data_path = DATA_DIR / "processed" / market_name / "monthly_features.parquet"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Données introuvables : {data_path}\n"
            f"Lance d'abord : python src/pipeline/etl.py"
        )

    logger.info(f"Chargement : {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"Shape brute : {df.shape} | Tickers : {df.index.get_level_values('ticker').nunique()}")

    # ── 2. Alpha features ─────────────────────────────────────
    df = add_all_features(df)

    # ── 3. Target (rendement T+1 positif ?) ───────────────────
    df["target"] = (
        df.groupby(level="ticker")["adj close"]
        .pct_change(1).shift(-1).gt(0).astype(int)
    )
    df = df.dropna(subset=["target"])

    # ── 4. Split temporel ─────────────────────────────────────
    dates      = df.index.get_level_values("date")
    split_date = dates.max() - pd.DateOffset(months=6)

    df_train = df[dates <= split_date].copy()
    df_test  = df[dates > split_date].copy()

    logger.info(
        f"Train : {dates[dates <= split_date].min().date()} → {split_date.date()} "
        f"({len(df_train)} obs)"
    )
    logger.info(
        f"Test  : {dates[dates > split_date].min().date()} → {dates.max().date()} "
        f"({len(df_test)} obs)"
    )

    if len(df_train) < 100:
        raise ValueError(f"Trop peu de données d'entraînement : {len(df_train)}")

    # ── 5. Baseline ───────────────────────────────────────────
    all_feats = [f for g in FEATURE_GROUPS.values() for f in g]
    available = [f for f in all_feats if f in df_train.columns]

    dummy        = DummyClassifier(strategy="most_frequent").fit(df_train[available], df_train["target"])
    baseline_auc = roc_auc_score(
        df_test["target"],
        dummy.predict_proba(df_test[available])[:, 1],
    )
    logger.info(f"Baseline AUC : {baseline_auc:.4f}")

    # ── 6. Entraînement ensemble ───────────────────────────────
    model = AlphaEdgeEnsemble(n_optuna_trials=50)
    model.fit(df_train, df_train["target"])

    # ── 7. Métriques test ─────────────────────────────────────
    proba     = model.predict_proba(df_test)[:, 1]
    final_auc = roc_auc_score(df_test["target"], proba)
    final_apr = average_precision_score(df_test["target"], proba)
    lift      = final_auc - baseline_auc

    logger.info(f"AUC Test : {final_auc:.4f} | APR : {final_apr:.4f} | Lift : +{lift:.4f}")

    # ── 8. Walk-forward validation ────────────────────────────
    logger.info("Walk-forward validation (4 fenêtres)...")
    wf_results = walk_forward_eval(df, n_windows=4, test_months=3, n_optuna_trials=20)

    if not wf_results.empty:
        logger.info(f"\n{wf_results.to_string(index=False)}")
        logger.info(
            f"WF AUC : {wf_results['auc'].mean():.4f} ± {wf_results['auc'].std():.4f}"
        )

    # ── 9. Sauvegarde locale ──────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_DIR / "ensemble_model.pkl")

    model_card = {
        "market":        market_name,
        "trained_at":    pd.Timestamp.now().isoformat(),
        "architecture":  "XGBoost + LightGBM + Ridge → LogisticRegression (stacking)",
        "auc_test":      round(final_auc, 4),
        "apr_test":      round(final_apr, 4),
        "baseline_auc":  round(baseline_auc, 4),
        "lift":          round(lift, 4),
        "model_weights": model.get_model_weights(),
        "walk_forward":  wf_results.to_dict(orient="records") if not wf_results.empty else [],
        "n_features":    len(model.features_),
        "train_size":    len(df_train),
        "test_size":     len(df_test),
    }
    with open(MODEL_DIR / "model_card.json", "w") as f:
        json.dump(model_card, f, indent=2)

    logger.info(f"Modèles sauvegardés dans : {MODEL_DIR}")

    # ── 10. MLflow ────────────────────────────────────────────
    if USE_MLFLOW:
        with mlflow.start_run(run_name=f"Ensemble_{market_name}"):
            mlflow.log_params({
                "architecture": "XGB+LGB+Ridge→LR",
                "market":       market_name,
                "n_features":   len(model.features_),
                "train_size":   len(df_train),
                "test_size":    len(df_test),
            })
            mlflow.log_metrics({
                "AUC_Test":    final_auc,
                "APR_Test":    final_apr,
                "Baseline_AUC": baseline_auc,
                "Lift":        lift,
                "WF_AUC_mean": wf_results["auc"].mean() if not wf_results.empty else 0.0,
                "WF_AUC_std":  wf_results["auc"].std()  if not wf_results.empty else 0.0,
            })
            mlflow.log_dict(model_card, "model_card.json")
            mlflow.sklearn.log_model(model, name="ensemble_model")
            mlflow.register_model(
                model_uri=f"runs:/{mlflow.active_run().info.run_id}/ensemble_model",
                name=f"AlphaEdge_Ensemble_{market_name}",
            )
        logger.info("Métriques loggées sur MLflow.")

    return model, model_card


# ══════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    config_path = CONFIG_DIR / "markets" / "cac40.json"
    market = (
        json.load(open(config_path))["market_name"]
        if config_path.exists()
        else "CAC40"
    )
    train_pipeline(market)