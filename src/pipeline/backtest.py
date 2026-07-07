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

# =============================================================================
# 1. CONSTRUCTION DE PORTEFEUILLE & HELPERS
# =============================================================================

def get_optimal_weights(
    prices_df: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
) -> Tuple[Dict[str, float], str]:
    n_assets = prices_df.shape[1]
    if n_assets < MIN_STOCKS_OPTIM:
        return {t: 1.0 / n_assets for t in prices_df.columns}, "equal_weight"

    try:
        mu = expected_returns.ema_historical_return(prices_df, frequency=TRADING_DAYS_YEAR, span=252)
        cov = risk_models.CovarianceShrinkage(prices_df, frequency=TRADING_DAYS_YEAR).ledoit_wolf()

        ef = EfficientFrontier(mu, cov, weight_bounds=WEIGHT_BOUNDS)
        ef.add_objective(objective_functions.L2_reg, gamma=0.1)
        ef.max_sharpe(risk_free_rate=risk_free_rate)

        weights = dict(ef.clean_weights())
        _log_concentration(weights)
        return weights, "max_sharpe"
    except Exception as exc:
        logger.warning(f"Max Sharpe a échoué ({exc}) -> fallback.")
        return {t: 1.0 / n_assets for t in prices_df.columns}, "equal_weight"

def _log_concentration(weights: Dict[str, float]) -> None:
    active = np.array([w for w in weights.values() if w > 1e-6])
    if active.size > 0:
        hhi = float(np.sum(active ** 2))
        logger.debug(f"HHI: {hhi:.3f} sur {active.size} positions.")

def _score_with_model(model: Any, features: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        expected_cols = getattr(model, "features_", None)
        x_input = features.reindex(columns=expected_cols).fillna(0)
        return model.predict_proba(x_input)[:, 1]
    
    # Fallback pour modèles sklearn simples
    preds = model.predict(features.fillna(0))
    return np.asarray(preds).ravel()

def _build_price_matrix(df_daily: pd.DataFrame, ffill_limit: int = MAX_PRICE_FFILL_DAYS) -> pd.DataFrame:
    if "adj_close" in df_daily.columns:
        col = "adj_close"
    elif "adj close" in df_daily.columns:
        col = "adj close"
    else:
        raise KeyError("Colonnes de prix non trouvées.")
    return df_daily[col].unstack().ffill(limit=ffill_limit)

def _build_daily_snapshot(df_daily: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """Prend une fenêtre de 252 jours pour calculer les features sans NaN."""
    last_date = df_daily.index.get_level_values("date").max()
    # Fenêtre glissante pour calcul technique robuste
    lookback_df = df_daily.iloc[-252:].copy() 
    df_feat = add_all_features(lookback_df)
    snapshot = df_feat.xs(last_date, level="date").copy()
    return snapshot, last_date

# =============================================================================
# 2. MOTEUR DE SIMULATION & ANALYTICS
# =============================================================================

def _simulate_period(
    allocation: Dict[str, float],
    drifted_allocation: Dict[str, float],
    trading_days: pd.DatetimeIndex,
    daily_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    portfolio_value: float,
    benchmark_value: float,
) -> Tuple[pd.DataFrame, float, float, Dict[str, float]]:
    
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

# =============================================================================
# 3. API PUBLIQUE
# =============================================================================

def backtest_strategy_with_rebalancing(
    df_daily: pd.DataFrame, df_monthly: pd.DataFrame, model: Any, benchmark_ticker: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    
    df_monthly_feat = add_all_features(df_monthly.copy())
    daily_prices = _build_price_matrix(df_daily)
    daily_returns = daily_prices.pct_change().fillna(0)
    
    bench_rets = get_benchmark_returns(benchmark_ticker, df_daily.index.get_level_values("date").min(), df_daily.index.get_level_values("date").max(), daily_prices.index)
    
    portfolio_value, drifted_allocation, period_frames, rebalance_log = 100.0, {}, [], []
    monthly_dates = df_monthly_feat.index.get_level_values("date").unique().sort_values()

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
        period_df, portfolio_value, _, drifted_allocation = _simulate_period(allocation, drifted_allocation, trading_days, daily_returns, bench_rets, portfolio_value, 100.0)
        period_frames.append(period_df)
        rebalance_log.append({"Date": month_date, "Allocation": allocation})

    hist_df = pd.concat(period_frames)
    return hist_df, pd.DataFrame(rebalance_log).set_index("Date"), {}

def generate_live_signals(df_daily: pd.DataFrame, daily_prices: pd.DataFrame, model: Any, rebalance_history: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    snapshot, last_date = _build_daily_snapshot(df_daily)
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
