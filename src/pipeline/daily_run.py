import os
import json
import warnings
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from src.utils.logger import setup_logger
from src.pipeline.etl import get_data_pipeline
from src.pipeline.backtest import backtest_strategy_with_rebalancing, generate_live_signals
from src.models.model_loader import load_champion
from const import BACKTEST_YEARS


# =============================================================================
# CONFIGURATION GLOBALE
# =============================================================================
warnings.filterwarnings("ignore")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = os.getenv("HF_REPO_ID", "soradata/alphaedge-data")
hf_api = HfApi()


def upload_to_hf(local_path: Path, hf_filename: str, market_name: str) -> bool:
    """Upload vers le repo Hugging Face, sous data/{market_name}/{hf_filename}."""
    if not HF_TOKEN:
        return False
    try:
        hf_api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=f"data/{market_name}/{hf_filename}",
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN,
        )
        return True
    except Exception as e:
        print(f"Erreur Upload HF ({market_name}/{hf_filename}): {e}")
        return False


def load_rebalance_history_from_hf(market_name: str, local_fallback: Path) -> pd.DataFrame:
    """
    Récupère l'historique de rebalancing existant (source de vérité pour
    savoir si un nouveau rebalancing mensuel doit être déclenché).
    Fallback sur le fichier local, puis sur un DataFrame vide.
    """
    empty = pd.DataFrame(columns=["N_Stocks", "Optim_Method", "Allocation", "Top_Ticker"])
    empty.index.name = "Date"

    if HF_TOKEN:
        try:
            path = hf_hub_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                filename=f"data/{market_name}/rebalance_history.parquet",
                token=HF_TOKEN,
            )
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            print(f"Impossible de charger rebalance_history depuis HF ({market_name}): {e}")

    if local_fallback.exists():
        try:
            df = pd.read_parquet(local_fallback)
            df.index = pd.to_datetime(df.index)
            return df
        except Exception:
            pass

    return empty


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================
def run_pipeline(market_config: dict) -> None:
    market_name = market_config.get("market_name", "UNKNOWN")
    logger = setup_logger(f"Pipeline_{market_name}")
    logger.info(f"STARTING PIPELINE | MARKET: {market_name}")

    try:
        # 1. Chargement du modèle champion
        model = load_champion(market_name)
        logger.info(f"Modèle champion prêt pour {market_name}")

        # 2. ETL — récupère la séance N sur yfinance
        df_daily, df_monthly = get_data_pipeline(market_config)
        if df_daily is None or df_monthly is None:
            raise ValueError("L'ETL n'a renvoyé aucune donnée.")

        cutoff = pd.Timestamp.now() - pd.DateOffset(years=BACKTEST_YEARS)
        df_daily_bt = df_daily[df_daily.index.get_level_values("date") >= cutoff]
        df_monthly_bt = df_monthly[df_monthly.index.get_level_values("date") >= cutoff]

        if df_daily_bt.empty or df_monthly_bt.empty:
            raise ValueError(f"Pas de données sur les {BACKTEST_YEARS} dernières années.")

        last_session = df_daily_bt.index.get_level_values("date").max()
        logger.info(
            f"Dernière séance récupérée : {last_session.date()} | "
            f"Backtest window : {cutoff.date()} → aujourd'hui"
        )

        base_dir = Path(f"data/processed/{market_name}")
        base_dir.mkdir(parents=True, exist_ok=True)
        hist_path = base_dir / "portfolio_history.parquet"
        rebal_path = base_dir / "rebalance_history.parquet"
        signals_path = base_dir / "latest_signals.parquet"
        metadata_path = base_dir / "data_metadata.json"

        # 3. Backtest historique (référence de performance)
        logger.info(f"Executing backtest for {market_name}...")
        hist_df, rebal_df_backtest, metrics = backtest_strategy_with_rebalancing(
            df_daily_bt,
            df_monthly_bt,
            model,
            benchmark_ticker=market_config.get("benchmark_ticker", "^FCHI"),
        )

        # 4. Signaux live (séance N) — proba quotidienne, allocation gelée
        #    jusqu'au prochain rebalancing mensuel réel.
        logger.info(f"Generating daily live signals for {market_name} (session {last_session.date()})...")
        rebalance_history = load_rebalance_history_from_hf(market_name, rebal_path)
        daily_prices = df_daily_bt["adj close"].unstack().ffill()

        signals_df, rebalance_history_updated = generate_live_signals(
            df_daily_bt, daily_prices, model, rebalance_history,
        )

        # 5. Sauvegarde locale
        hist_df.to_parquet(hist_path)
        rebalance_history_updated.to_parquet(rebal_path)
        signals_df.to_parquet(signals_path)

        metadata = {
            "market_name":       market_name,
            "last_run_utc":      datetime.now(timezone.utc).isoformat(),
            "last_session_date": str(last_session.date()),
            "n_signals":         len(signals_df),
            "n_buy_signals":     int((signals_df["Signal"] == "BUY").sum()) if not signals_df.empty else 0,
            "last_rebalance":    str(rebalance_history_updated.index.max().date())
                                  if not rebalance_history_updated.empty else None,
            "metrics":           metrics,
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Fichiers sauvegardés localement dans {base_dir}")

        # 6. Synchronisation Cloud (Hugging Face)
        upload_results = {
            "portfolio_history": upload_to_hf(hist_path, "portfolio_history.parquet", market_name),
            "rebalance_history": upload_to_hf(rebal_path, "rebalance_history.parquet", market_name),
            "latest_signals":    upload_to_hf(signals_path, "latest_signals.parquet", market_name),
            "data_metadata":     upload_to_hf(metadata_path, "data_metadata.json", market_name),
        }

        failed_uploads = [k for k, ok in upload_results.items() if not ok]
        if not HF_TOKEN:
            logger.warning("HF_TOKEN absent — synchronisation Cloud ignorée.")
        elif failed_uploads:
            logger.warning(f"Échecs d'upload HF : {failed_uploads}")
        else:
            logger.info("Synchronisation Hugging Face terminée avec succès.")

        logger.info(
            f"Pipeline terminé | Sharpe: {metrics.get('Sharpe', 'N/A')} | "
            f"Signaux BUY: {metadata['n_buy_signals']}/{metadata['n_signals']} | "
            f"Dernier rebalancing: {metadata['last_rebalance']}"
        )

    except Exception as e:
        logger.critical(f"CRITICAL FAILURE {market_name}: {e}", exc_info=True)
        raise


# =============================================================================
# ORCHESTRATEUR
# =============================================================================
if __name__ == "__main__":
    config_dir = Path("config/markets")
    failures = []

    for config_file in sorted(config_dir.glob("*.json")):
        with open(config_file) as f:
            market_config = json.load(f)
        market = market_config.get("market_name", config_file.stem)
        try:
            run_pipeline(market_config)
        except Exception:
            failures.append(market)

    if failures:
        print(f"Marchés en échec : {failures}")
        raise SystemExit(1)