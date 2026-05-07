from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import setup_logger
from src.utils.config_loader import BENCHMARK_TICKER
from src.utils.market_utils import get_benchmark_returns


logger = setup_logger("backtest")


# =============================================================================
# COSTS / TURNOVER
# =============================================================================

def calculate_turnover_friction(
    old_weights: Dict[str, float],
    new_weights: Dict[str, float],
    fee_bps: float = 0.0020,
) -> Tuple[float, float]:
    """
    Calcule:
    - le coût de friction en pourcentage du portefeuille
    - le turnover réel
    """
    total_turnover = 0.0
    all_tickers = set(old_weights.keys()).union(set(new_weights.keys()))

    for ticker in all_tickers:
        old_w = old_weights.get(ticker, 0.0)
        new_w = new_weights.get(ticker, 0.0)
        total_turnover += abs(new_w - old_w)

    true_turnover = total_turnover / 2.0
    friction_cost_pct = true_turnover * fee_bps

    return friction_cost_pct, true_turnover


# =============================================================================
# PORTFOLIO CONSTRUCTION
# =============================================================================

def get_topk_equal_weights(
    selected: pd.DataFrame,
    top_k: int = 5,
    score_col: str = "proba_upside",
) -> Dict[str, float]:
    """
    Prend les top_k meilleurs titres selon score_col
    et attribue des poids égaux.
    """
    if selected is None or selected.empty:
        return {}

    if score_col not in selected.columns:
        logger.warning(f"Column '{score_col}' not found in selected dataframe.")
        return {}

    top_selected = selected.sort_values(score_col, ascending=False).head(top_k)

    n_assets = len(top_selected)
    if n_assets == 0:
        return {}

    weight = 1.0 / n_assets
    return {ticker: weight for ticker in top_selected.index.tolist()}


# =============================================================================
# DAILY PATH SIMULATION
# =============================================================================

def _simulate_daily_returns(
    allocation: Dict[str, float],
    trading_days: pd.DatetimeIndex,
    daily_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    portfolio_value: float,
    benchmark_value: float,
) -> Tuple[List[dict], float, float]:
    """
    Simule l'évolution quotidienne du portefeuille et du benchmark
    entre deux dates de rebalancement.
    """
    records: List[dict] = []

    for current_date in trading_days:
        bench_ret = float(benchmark_returns.get(current_date, 0.0))
        strat_ret = 0.0

        if allocation:
            weights = list(allocation.values())
            asset_returns = []

            for ticker in allocation.keys():
                if ticker in daily_returns.columns and current_date in daily_returns.index:
                    asset_returns.append(daily_returns.loc[current_date, ticker])
                else:
                    asset_returns.append(0.0)

            asset_returns = pd.Series(asset_returns).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            strat_ret = float(np.average(asset_returns, weights=weights))

        portfolio_value *= (1 + strat_ret)
        benchmark_value *= (1 + bench_ret)

        records.append(
            {
                "Date": current_date,
                "Strategy": portfolio_value,
                "Benchmark": benchmark_value,
            }
        )

    return records, portfolio_value, benchmark_value


# =============================================================================
# EMPTY OUTPUT HELPERS
# =============================================================================

def _empty_history_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["Strategy", "Benchmark"])


def _empty_rebalance_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "Method",
            "Selected_Count",
            "N_Stocks",
            "Turnover",
            "Cost_Pct",
            "Allocation",
        ]
    )


# =============================================================================
# MAIN BACKTEST
# =============================================================================

def backtest_strategy_with_rebalancing(
    df_daily: pd.DataFrame,
    signal_generator,
    top_k: int = 5,
    target_cluster: int = 1,
    proba_threshold: float = 0.55,
    fee_bps: float = 0.0020,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backtest Top-k avec:
    - filtrage cluster + seuil de proba
    - pondération égale
    - coûts de turnover
    - benchmark de marché
    """
    logger.info("Starting backtest | method=topk")

    if df_daily is None or df_daily.empty:
        logger.warning("df_daily is empty. Returning empty backtest outputs.")
        return _empty_history_df(), _empty_rebalance_df()

    if "adj close" not in df_daily.columns:
        logger.error("Missing 'adj close' column in df_daily.")
        return _empty_history_df(), _empty_rebalance_df()

    try:
        daily_prices = df_daily["adj close"].unstack().sort_index().ffill()
    except Exception as e:
        logger.error(f"Failed to reshape daily prices: {e}")
        return _empty_history_df(), _empty_rebalance_df()

    if daily_prices.empty:
        logger.warning("daily_prices is empty after unstack.")
        return _empty_history_df(), _empty_rebalance_df()

    daily_returns = (
        daily_prices.pct_change()
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    date_min = df_daily.index.get_level_values("date").min()
    date_max = df_daily.index.get_level_values("date").max()

    benchmark_returns = get_benchmark_returns(
        BENCHMARK_TICKER,
        date_min,
        date_max,
        daily_prices.index,
    )

    if benchmark_returns is None or len(benchmark_returns) == 0:
        benchmark_returns = pd.Series(0.0, index=daily_prices.index)
    else:
        benchmark_returns = (
            benchmark_returns.reindex(daily_prices.index)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

    monthly_dates = (
        signal_generator.signal_cache.index.get_level_values("date")
        .unique()
        .sort_values()
    )

    if len(monthly_dates) < 2:
        logger.warning("Not enough monthly dates to run backtest.")
        return _empty_history_df(), _empty_rebalance_df()

    portfolio_value = 100.0
    benchmark_value = 100.0
    all_records: List[dict] = []
    rebalance_log: List[dict] = []
    previous_allocation: Dict[str, float] = {}

    for month_date, next_month in zip(monthly_dates[:-1], monthly_dates[1:]):
        month_data = signal_generator.get_signal(month_date)

        allocation: Dict[str, float] = {}
        selected_count = 0

        if month_data is not None and not month_data.empty:
            required_cols = {"cluster", "proba_upside"}
            if required_cols.issubset(month_data.columns):
                selected = month_data[
                    (month_data["cluster"] == target_cluster) &
                    (month_data["proba_upside"] >= proba_threshold)
                ].copy()

                selected_count = len(selected)

                if not selected.empty:
                    allocation = get_topk_equal_weights(
                        selected=selected,
                        top_k=top_k,
                        score_col="proba_upside",
                    )
            else:
                logger.warning(
                    f"Signal dataframe missing required columns: {required_cols - set(month_data.columns)}"
                )

        cost_pct, turnover = calculate_turnover_friction(
            previous_allocation,
            allocation,
            fee_bps=fee_bps,
        )

        portfolio_value *= (1 - cost_pct)
        previous_allocation = allocation.copy()

        trading_days = daily_prices.index[
            (daily_prices.index >= month_date) & (daily_prices.index < next_month)
        ]

        if len(trading_days) > 0:
            day_records, portfolio_value, benchmark_value = _simulate_daily_returns(
                allocation=allocation,
                trading_days=trading_days,
                daily_returns=daily_returns,
                benchmark_returns=benchmark_returns,
                portfolio_value=portfolio_value,
                benchmark_value=benchmark_value,
            )
            all_records.extend(day_records)

        rebalance_log.append(
            {
                "Date": month_date,
                "Method": "topk",
                "Selected_Count": selected_count,
                "N_Stocks": len(allocation),
                "Turnover": turnover,
                "Cost_Pct": cost_pct,
                "Allocation": allocation,
            }
        )

    logger.info(f"Backtest complete. Final value: {portfolio_value:.2f}")

    hist_df = (
        pd.DataFrame(all_records).set_index("Date").sort_index()
        if all_records
        else _empty_history_df()
    )

    rebal_df = (
        pd.DataFrame(rebalance_log).set_index("Date").sort_index()
        if rebalance_log
        else _empty_rebalance_df()
    )

    return hist_df, rebal_df