"""
Moteur de backtest et de génération de signaux live pour AlphaEdge.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions

from const import (
    TRADING_DAYS_YEAR,
    RISK_FREE_RATE,
    TRANSACTION_COST,
    MIN_STOCKS_OPTIM,
    MAX_STOCKS_SELECT,
    PROBA_MIN,
    WEIGHT_BOUNDS,
)
from src.features.alpha_features import add_all_features
from src.utils.logger import setup_logger
from src.utils.market_utils import get_benchmark_returns

logger = setup_logger("backtest")

MAX_PRICE_FFILL_DAYS = 5
MAX_ACCEPTABLE_MISSING_RATIO = 0.05

# Nombre minimum de mois d'historique requis par ticker avant que les features
# a fenetre glissante (jusqu'a 12 mois pour realized_vol_12m, sharpe_6m,
# calmar_proxy, hist_var_5pct, cvar_5pct...) soient jugees fiables. Les mois
# en-dessous de ce seuil sont scores sur des features encore incompletes,
# masquees a zero par le fillna(0) interne d'add_all_features -> cela produit
# une periode plate artificielle en debut de backtest si on ne filtre pas.
MIN_FEATURE_HISTORY = 12


# =============================================================================
# 1. CONSTRUCTION DE PORTEFEUILLE & HELPERS
# =============================================================================

def get_optimal_weights(prices_df: pd.DataFrame, risk_free_rate: float = RISK_FREE_RATE) -> Tuple[Dict[str, float], str]:
    n_assets = prices_df.shape[1]
    if n_assets < MIN_STOCKS_OPTIM:
        return {t: 1.0 / n_assets for t in prices_df.columns}, "equal_weight"
    try:
        mu = expected_returns.ema_historical_return(prices_df, frequency=TRADING_DAYS_YEAR, span=252)
        cov = risk_models.CovarianceShrinkage(prices_df, frequency=TRADING_DAYS_YEAR).ledoit_wolf()
        ef = EfficientFrontier(mu, cov, weight_bounds=WEIGHT_BOUNDS)
        ef.add_objective(objective_functions.L2_reg, gamma=0.1)
        ef.max_sharpe(risk_free_rate=risk_free_rate)
        return dict(ef.clean_weights()), "max_sharpe"
    except Exception as exc:
        logger.warning(f"Max Sharpe a échoué ({exc}) -> fallback.")
        return {t: 1.0 / n_assets for t in prices_df.columns}, "equal_weight"


def _score_with_model(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Scoring robuste avec injection automatique des features manquantes."""
    expected_cols = _resolve_expected_features(model)
    # On force l'alignement : si features manquantes, elles sont créées à 0 ici
    x_input = features.reindex(columns=expected_cols).fillna(0)

    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_input)[:, 1]
    return np.asarray(model.predict(x_input)).ravel()


def _resolve_expected_features(model: Any) -> Optional[List[str]]:
    """Résout la liste de features réellement attendue par le modèle (v5 ou v6)."""
    try:
        inner = model._model_impl.python_model
        if hasattr(inner, "features_"): return list(inner.features_)
    except Exception: pass
    try:
        input_schema = model.metadata.get_input_schema()
        if input_schema: return [c.name for c in input_schema.inputs]
    except Exception: pass
    return None


def _build_price_matrix(df_daily: pd.DataFrame, ffill_limit: int = MAX_PRICE_FFILL_DAYS) -> pd.DataFrame:
    col = "adj_close" if "adj_close" in df_daily.columns else "adj close"
    return df_daily[col].unstack().ffill(limit=ffill_limit)


def _build_daily_snapshot(df_daily: pd.DataFrame, market_config: dict) -> Tuple[pd.DataFrame, pd.Timestamp]:
    last_date = df_daily.index.get_level_values("date").max()
    lookback_df = df_daily.iloc[-252:].copy()
    df_feat = add_all_features(lookback_df, market_config)
    return df_feat.xs(last_date, level="date").copy(), last_date


def _filter_warmup_period(df_monthly: pd.DataFrame, min_history: int = MIN_FEATURE_HISTORY) -> pd.Series:
    """
    Calcule un masque booleen (indexe comme df_monthly) qui exclut, pour
    chaque ticker, les tout premiers mois n'ayant pas assez d'historique pour
    que les features a fenetre glissante soient completes. Le calcul se base
    sur le nombre de lignes deja observees par ticker (donnees brutes, avant
    tout calcul de features), donc independant du fillna(0) applique plus
    tard par add_all_features.
    """
    history_count = df_monthly.groupby(level="ticker").cumcount() + 1
    return history_count > min_history


# =============================================================================
# 2. MOTEUR DE SIMULATION & API
# =============================================================================

def _simulate_period(allocation, drifted_allocation, trading_days, daily_returns, benchmark_returns, portfolio_value, benchmark_value):
    turnover = sum(abs(allocation.get(t, 0.0) - drifted_allocation.get(t, 0.0)) for t in set(allocation)|set(drifted_allocation)) / 2.0
    portfolio_value -= (portfolio_value * turnover * TRANSACTION_COST)
    tickers = list(allocation.keys())
    if not tickers:
        return pd.DataFrame({"Strategy": portfolio_value, "Benchmark": benchmark_value, "N_Stocks": 0}, index=trading_days), portfolio_value, benchmark_value, {}
    weights = np.array([allocation.get(t, 0) for t in tickers])
    rets = daily_returns.reindex(index=trading_days, columns=tickers).fillna(0.0).to_numpy()
    growth = np.cumprod(1.0 + rets, axis=0)
    strategy_values = (portfolio_value * weights[np.newaxis, :] * growth).sum(axis=1) + (portfolio_value * (1 - sum(weights)))
    bench_values = benchmark_value * np.cumprod(1.0 + benchmark_returns.reindex(trading_days).fillna(0.0))
    final_total = float(strategy_values[-1])
    new_drifted = dict(zip(tickers, (portfolio_value * weights * growth[-1, :] / final_total)))
    return pd.DataFrame({"Strategy": strategy_values, "Benchmark": bench_values, "N_Stocks": len(tickers)}, index=trading_days), final_total, float(bench_values[-1]), new_drifted


def backtest_strategy_with_rebalancing(df_daily, df_monthly, model, benchmark_ticker, market_config):
    # Masque de warm-up calcule AVANT le calcul des features (sur les donnees
    # brutes), pour ne pas dependre du fillna(0) interne a add_all_features.
    valid_history_mask = _filter_warmup_period(df_monthly)

    df_monthly_feat = add_all_features(df_monthly.copy(), market_config)
    df_monthly_feat = df_monthly_feat[valid_history_mask.reindex(df_monthly_feat.index, fill_value=False)]

    n_dropped = int((~valid_history_mask).sum())
    if n_dropped:
        logger.info(f"Warm-up : {n_dropped} lignes exclues (historique < {MIN_FEATURE_HISTORY} mois).")

    daily_prices = _build_price_matrix(df_daily)
    daily_returns = daily_prices.pct_change().fillna(0)
    bench_rets = get_benchmark_returns(benchmark_ticker, df_daily.index.get_level_values("date").min(), df_daily.index.get_level_values("date").max(), daily_prices.index)

    portfolio_value, benchmark_value, drifted_allocation, period_frames, rebalance_log = 100.0, 100.0, {}, [], []
    monthly_dates = df_monthly_feat.index.get_level_values("date").unique().sort_values()

    if len(monthly_dates) < 2:
        raise ValueError(
            f"Pas assez de mois exploitables après filtrage du warm-up "
            f"({len(monthly_dates)} mois restants, {MIN_FEATURE_HISTORY} mois requis). "
            f"Augmentez la fenêtre de backtest (BACKTEST_YEARS) ou réduisez MIN_FEATURE_HISTORY."
        )

    for i, month_date in enumerate(monthly_dates[:-1]):
        month_data = df_monthly_feat.xs(month_date, level="date").copy()
        month_data["proba_upside"] = _score_with_model(model, month_data)
        tickers = month_data[month_data["proba_upside"] >= PROBA_MIN].sort_values("proba_upside", ascending=False).head(MAX_STOCKS_SELECT).index.tolist()
        allocation = {}
        if tickers:
            prices_subset = daily_prices[tickers].loc[:month_date].iloc[-TRADING_DAYS_YEAR:].dropna(axis=1, thresh=int(TRADING_DAYS_YEAR * 0.8))
            if not prices_subset.empty:
                weights, _ = get_optimal_weights(prices_subset)
                allocation = {t: w for t, w in weights.items() if w > 1e-4}
        trading_days = daily_prices.index[(daily_prices.index >= month_date) & (daily_prices.index < monthly_dates[i + 1])]
        # Le benchmark_value retourne par _simulate_period est reinjecte a
        # l'iteration suivante (au lieu d'etre code en dur a 100.0), afin que
        # la courbe Benchmark compose reellement ses rendements mois apres mois.
        period_df, portfolio_value, benchmark_value, drifted_allocation = _simulate_period(
            allocation, drifted_allocation, trading_days, daily_returns, bench_rets,
            portfolio_value, benchmark_value
        )
        period_frames.append(period_df)
        rebalance_log.append({"Date": month_date, "Allocation": allocation})
    return pd.concat(period_frames), pd.DataFrame(rebalance_log).set_index("Date"), {}


def generate_live_signals(df_daily, daily_prices, model, rebalance_history, market_config):
    snapshot, last_date = _build_daily_snapshot(df_daily, market_config)
    snapshot["proba_upside"] = _score_with_model(model, snapshot)
    tickers = snapshot[snapshot["proba_upside"] >= PROBA_MIN].sort_values("proba_upside", ascending=False).head(MAX_STOCKS_SELECT).index.tolist()
    allocation = {}
    if tickers:
        prices_subset = daily_prices[tickers].loc[:last_date].iloc[-TRADING_DAYS_YEAR:].dropna(axis=1, thresh=int(TRADING_DAYS_YEAR * 0.8))
        if not prices_subset.empty:
            weights, _ = get_optimal_weights(prices_subset)
            allocation = {t: w for t, w in weights.items() if w > 1e-4}
    out = snapshot.reset_index().rename(columns={"ticker": "Ticker"})
    out["Allocation"] = out["Ticker"].map(allocation).fillna(0.0)
    out["Signal"] = np.where(out["Allocation"] > 0, "BUY", "NEUTRAL")
    out["Proba_Hausse"] = (out["proba_upside"] * 100).round(1)
    return out[["Ticker", "Signal", "Allocation", "Proba_Hausse"]], rebalance_history