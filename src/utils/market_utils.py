"""
Market Utils
=============
Utilitaires pour la récupération du benchmark et l'export des signaux.

Fonctions :
  - get_benchmark_returns() : télécharge et reindex les rendements du benchmark
  - build_export_df()       : formate le snapshot journalier pour l'export HF
"""

import time
import pandas as pd
import yfinance as yf

from src.utils.logger import setup_logger

logger = setup_logger("market_utils")

_BENCHMARK_RETRIES = 3
_BENCHMARK_RETRY_WAIT = 5


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def _strip_timezone(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Supprime le timezone d'un DatetimeIndex de manière sécurisée.
    - Si timezone-aware  → convertit en UTC puis supprime le tz
    - Si timezone-naive  → retourne tel quel (pas de crash)
    """
    if index.tz is not None:
        return index.tz_convert("UTC").tz_localize(None)
    return index


# ══════════════════════════════════════════════════════════════════
# BENCHMARK
# ══════════════════════════════════════════════════════════════════

def get_benchmark_returns(
    benchmark_ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    reindex_to: pd.DatetimeIndex,
) -> pd.Series:
    """
    Télécharge les rendements journaliers du benchmark et les reindex
    sur le calendrier de la stratégie.

    Fallback progressif :
      1. Retry 3 fois avec 5s d'attente
      2. Si échec total → série de zéros + warning

    Parameters
    ----------
    benchmark_ticker : str — ticker yfinance (ex: '^FCHI' pour CAC40)
    start            : pd.Timestamp — date de début
    end              : pd.Timestamp — date de fin
    reindex_to       : pd.DatetimeIndex — calendrier de la stratégie

    Returns
    -------
    pd.Series — rendements journaliers reindexés, NaN → 0
    """
    # Normaliser le calendrier cible une seule fois
    reindex_clean = _strip_timezone(reindex_to)

    for attempt in range(1, _BENCHMARK_RETRIES + 1):
        try:
            raw = yf.download(
                benchmark_ticker,
                start=start,
                end=end + pd.DateOffset(days=1),
                progress=False,
                auto_adjust=False,
                threads=False,
            )

            if raw.empty:
                logger.warning(
                    f"Réponse vide pour {benchmark_ticker} "
                    f"(tentative {attempt}/{_BENCHMARK_RETRIES})"
                )
                time.sleep(_BENCHMARK_RETRY_WAIT)
                continue

            # Extraction du prix de clôture
            if isinstance(raw.columns, pd.MultiIndex):
                prices = raw["Adj Close"].iloc[:, 0]   # Adj Close > Close pour benchmark
            else:
                prices = raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"]

            # Normalisation timezone
            prices.index = _strip_timezone(prices.index)

            # Rendements journaliers reindexés sur le calendrier stratégie
            bench_returns = (
                prices.pct_change()
                .reindex(reindex_clean, method="ffill")
                .fillna(0.0)
            )

            logger.info(
                f" Benchmark {benchmark_ticker} chargé : "
                f"{len(bench_returns)} jours | "
                f"Rendement total : {(1 + bench_returns).prod() - 1:.2%}"
            )

            return bench_returns

        except Exception as e:
            logger.warning(
                f"Erreur benchmark {benchmark_ticker} "
                f"(tentative {attempt}/{_BENCHMARK_RETRIES}) : {e}"
            )
            if attempt < _BENCHMARK_RETRIES:
                time.sleep(_BENCHMARK_RETRY_WAIT)

    logger.error(
        f" Impossible de charger le benchmark {benchmark_ticker} "
        f"après {_BENCHMARK_RETRIES} tentatives. Série de zéros utilisée."
    )
    return pd.Series(0.0, index=reindex_clean)


# ══════════════════════════════════════════════════════════════════
# EXPORT DES SIGNAUX
# ══════════════════════════════════════════════════════════════════

def build_export_df(
    today_data: pd.DataFrame,
    final_alloc: dict,
) -> pd.DataFrame:
    """
    Formate le snapshot journalier des signaux ML pour l'export vers HF Space.

    Colonnes de sortie :
      Ticker, Proba_Hausse (%), RSI, Return_3M, Allocation (%), Signal

    Compatible avec les deux versions du pipeline :
      - Colonnes laggées  (rsi_lag1, return_3m_lag1) — nouveau pipeline
      - Colonnes directes (rsi, return_3m)           — ancien pipeline

    Parameters
    ----------
    today_data : pd.DataFrame — slice mensuelle du jour (index = ticker)
    final_alloc : dict        — {ticker: weight} issu de get_optimal_weights()

    Returns
    -------
    pd.DataFrame — propre et prêt pour l'export CSV / Streamlit
    """
    df = today_data.copy()

    # Résolution des colonnes (laggées en priorité, directes en fallback)
    rsi_col = "rsi_lag1" if "rsi_lag1" in df.columns else "rsi"
    return3m_col = "return_3m" if "return_3m" in df.columns else None
    cluster_col = "cluster_lag1" if "cluster_lag1" in df.columns else "cluster"

    # Colonnes à exporter
    cols_to_keep = ["proba_upside"]
    if rsi_col:
        cols_to_keep.append(rsi_col)
    if return3m_col:
        cols_to_keep.append(return3m_col)
    if cluster_col in df.columns:
        cols_to_keep.append(cluster_col)

    export = (
        df[cols_to_keep]
        .reset_index()
        .rename(columns={"ticker": "Ticker"})
    )

    # Renommage dynamique
    rename_map = {"proba_upside": "Proba_Hausse (%)"}
    if rsi_col in export.columns:
        rename_map[rsi_col] = "RSI"
    if return3m_col and return3m_col in export.columns:
        rename_map[return3m_col] = "Return_3M (%)"
    if cluster_col in export.columns:
        rename_map[cluster_col] = "Cluster"

    export = export.rename(columns=rename_map)

    # Formatage
    export["Proba_Hausse (%)"] = (export["Proba_Hausse (%)"] * 100).round(2)
    if "Return_3M (%)" in export.columns:
        export["Return_3M (%)"] = (export["Return_3M (%)"] * 100).round(2)

    # Allocation & Signal
    export["Allocation (%)"] = (
        export["Ticker"].map(final_alloc).fillna(0.0) * 100
    ).round(2)
    export["Signal"] = export["Allocation (%)"].apply(
        lambda w: "🟢 BUY" if w > 0 else "⚪ NEUTRAL"
    )

    # Tri par probabilité décroissante
    export = export.sort_values("Proba_Hausse (%)", ascending=False).reset_index(drop=True)

    return export
