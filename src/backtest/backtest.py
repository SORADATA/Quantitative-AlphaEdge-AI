from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns

# Assure-toi que ce fichier existe bien !
from src.strategy.signals import AlphaSignal

from const import (
    TARGET_CLUSTER,
    PROBA_THRESHOLD,
    TRADING_DAYS_YEAR,
    RISK_FREE_RATE
)
from src.utils.logger import setup_logger
from src.utils.config_loader import BENCHMARK_TICKER
from src.utils.market_utils import get_benchmark_returns

logger = setup_logger("backtest")


def calculate_turnover_friction(old_weights: dict, new_weights: dict, fee_bps: float = 0.0020):
    total_turnover = 0.0
    all_tickers = set(old_weights.keys()).union(set(new_weights.keys()))
    for ticker in all_tickers:
        old_w = old_weights.get(ticker, 0.0)
        new_w = new_weights.get(ticker, 0.0)
        total_turnover += abs(new_w - old_w)
    true_turnover = total_turnover / 2.0
    friction_cost_pct = true_turnover * fee_bps
    return friction_cost_pct, true_turnover


def get_optimal_weights(prices_df: pd.DataFrame) -> Tuple[Dict[str, float], bool]:
    try:
        mu = expected_returns.mean_historical_return(prices_df, frequency=TRADING_DAYS_YEAR)
        S = risk_models.CovarianceShrinkage(prices_df, frequency=TRADING_DAYS_YEAR).ledoit_wolf()
        ef = EfficientFrontier(mu, S, weight_bounds=(0.02, 0.25))
        ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
        return dict(ef.clean_weights()), True
    except Exception as e:
        logger.warning(f"Optimization failed: {e}")
        return {}, False


def _simulate_daily_returns(
    allocation: Dict[str, float],
    trading_days: pd.DatetimeIndex,
    daily_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    portfolio_value: float,
    benchmark_value: float,
) -> Tuple[list, float, float]:
    records = []
    for date in trading_days:
        bench_ret = benchmark_returns.get(date, 0.0)
        strat_ret = 0.0
        if allocation:
            weights = list(allocation.values())
            rets = [
                daily_returns.loc[date, t] if t in daily_returns.columns and date in daily_returns.index else 0.0
                for t in allocation
            ]
            strat_ret = float(np.average(pd.Series(rets).fillna(0), weights=weights))
        portfolio_value *= (1 + strat_ret)
        benchmark_value *= (1 + bench_ret)
        records.append({"Date": date, "Strategy": portfolio_value, "Benchmark": benchmark_value})
    return records, portfolio_value, benchmark_value


# >>> NOUVEAU : La signature a changé (plus de modèles, plus de df_monthly !)
def backtest_strategy_with_rebalancing(
    df_daily: pd.DataFrame,
    signal_generator: AlphaSignal,
    get_optimal_weights_fn: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Starting backtest with Turnover Friction & Cached Signals...")

    daily_prices = df_daily["adj close"].unstack().ffill()
    daily_returns = daily_prices.pct_change().fillna(0)

    date_min = df_daily.index.get_level_values("date").min()
    date_max = df_daily.index.get_level_values("date").max()

    benchmark_returns = get_benchmark_returns(
        BENCHMARK_TICKER, date_min, date_max, daily_prices.index
    )

    portfolio_value, benchmark_value = 100.0, 100.0
    all_records, rebalance_log = [], []

    # On récupère les dates mensuelles directement depuis le cache des signaux !
    monthly_dates = signal_generator.signal_cache.index.get_level_values("date").unique().sort_values()

    previous_allocation = {}

    for i, month_date in enumerate(monthly_dates[:-1]):
 
        # >>> NOUVEAU : L'appel au cache ultra-rapide remplace l'ancienne fonction
        month_data = signal_generator.get_signal(month_date)

        allocation: Dict[str, float] = {}
        if not month_data.empty:
            selected = month_data[
                (month_data["cluster"] == TARGET_CLUSTER) & 
                (month_data["proba_upside"] > PROBA_THRESHOLD)
            ]

            if not selected.empty:
                tickers = selected.index.tolist()
                # prices_subset = daily_prices[tickers].iloc[-TRADING_DAYS_YEAR:].dropna(axis=1)
                prices_subset = daily_prices[tickers].loc[:month_date].iloc[-TRADING_DAYS_YEAR:].dropna(axis=1)
                if not prices_subset.empty and len(prices_subset.columns) >= 3:
                    weights, success = get_optimal_weights_fn(prices_subset)
                    allocation = weights if success else {t: 1.0 / len(tickers) for t in tickers}

        cost_pct, turnover = calculate_turnover_friction(previous_allocation, allocation, fee_bps=0.0020)
        portfolio_value *= (1 - cost_pct)
        previous_allocation = allocation.copy()

        next_month = monthly_dates[i + 1]
        trading_days = daily_prices.index[
            (daily_prices.index >= month_date) & (daily_prices.index < next_month)
        ]

        day_records, portfolio_value, benchmark_value = _simulate_daily_returns(
            allocation, trading_days, daily_returns,
            benchmark_returns, portfolio_value, benchmark_value,
        )
        all_records.extend(day_records)
  
        rebalance_log.append({
            "Date": month_date,
            "N_Stocks": len(allocation),
            "Turnover": turnover,
            "Cost_Pct": cost_pct,
            "Allocation": allocation
        })

    logger.info(f"Backtest complete. Final value: {portfolio_value:.2f}")
    return pd.DataFrame(all_records).set_index("Date"), pd.DataFrame(rebalance_log).set_index("Date")
