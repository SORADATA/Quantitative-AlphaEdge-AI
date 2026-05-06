from typing import Any, Dict, Tuple
import math
import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns

from src.strategy.signals import AlphaSignal
from const import (
    TARGET_CLUSTER,
    PROBA_THRESHOLD,
    TRADING_DAYS_YEAR,
    RISK_FREE_RATE,
)
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

def get_optimal_weights(
    prices_df: pd.DataFrame,
    weight_bounds: Tuple[float, float] = (0.02, 0.25),
) -> Tuple[Dict[str, float], bool]:
    try:
        if prices_df.empty or prices_df.shape[1] == 0:
            return {}, False

        mu = expected_returns.mean_historical_return(
            prices_df,
            frequency=TRADING_DAYS_YEAR,
        )
        S = risk_models.CovarianceShrinkage(
            prices_df,
            frequency=TRADING_DAYS_YEAR,
        ).ledoit_wolf()

        ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
        ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)

        weights = dict(ef.clean_weights())
        weights = {k: v for k, v in weights.items() if v > 0}

        if not weights:
            return {}, False

        return weights, True

    except Exception as e:
        logger.warning(f"Optimization failed: {e}")
        return {}, False


def get_topk_equal_weights(
    selected: pd.DataFrame,
    topk: int = 10,
    score_col: str = "proba_upside",
) -> Dict[str, float]:
    if selected.empty:
        return {}

    top_selected = selected.sort_values(score_col, ascending=False).head(topk)
    n = len(top_selected)

    if n == 0:
        return {}

    w = 1.0 / n
    return {ticker: w for ticker in top_selected.index.tolist()}


def get_topk_dropout_weights(
    selected: pd.DataFrame,
    previous_allocation: Dict[str, float],
    topk: int = 10,
    n_drop: int = 2,
    score_col: str = "proba_upside",
) -> Dict[str, float]:
    if selected.empty:
        return {}

    ranked = selected.sort_values(score_col, ascending=False)
    target_universe = ranked.index.tolist()

    if not previous_allocation:
        initial = target_universe[:topk]
        if not initial:
            return {}
        w = 1.0 / len(initial)
        return {ticker: w for ticker in initial}

    current_holdings = list(previous_allocation.keys())
    current_scored = ranked.reindex(current_holdings).dropna(subset=[score_col])

    keep = current_scored.sort_values(score_col, ascending=False)
    keep_names = keep.index.tolist()

    stocks_to_sell = keep_names[-min(n_drop, len(keep_names)):] if keep_names else []
    remaining = [t for t in current_holdings if t not in stocks_to_sell]

    candidates = [t for t in target_universe if t not in remaining]
    slots_to_fill = max(0, topk - len(remaining))
    new_buys = candidates[:slots_to_fill]

    final_names = remaining + new_buys
    final_names = final_names[:topk]

    if not final_names:
        return {}

    w = 1.0 / len(final_names)
    return {ticker: w for ticker in final_names}


def get_equal_weight_fallback(tickers: list[str]) -> Dict[str, float]:
    if not tickers:
        return {}
    w = 1.0 / len(tickers)
    return {t: w for t in tickers}


def build_markowitz_allocation(
    selected: pd.DataFrame,
    daily_prices: pd.DataFrame,
    month_date: pd.Timestamp,
    get_optimal_weights_fn: Any,
    lookback_days: int = TRADING_DAYS_YEAR,
    weight_bounds: Tuple[float, float] = (0.02, 0.25),
) -> Dict[str, float]:
    if selected.empty:
        return {}

    selected_tickers = selected.index.tolist()
    prices_subset = (
        daily_prices[selected_tickers]
        .loc[:month_date]
        .iloc[-lookback_days:]
        .dropna(axis=1)
    )

    if prices_subset.empty:
        return {}

    valid_tickers = prices_subset.columns.tolist()
    if not valid_tickers:
        return {}

    min_w, max_w = weight_bounds
    min_assets_required = math.ceil(1.0 / max_w) if max_w > 0 else 999999

    if len(valid_tickers) < min_assets_required:
        logger.info(
            f"Markowitz fallback to equal weight on {len(valid_tickers)} assets "
            f"(need at least {min_assets_required} for max_weight={max_w:.2f})"
        )
        return get_equal_weight_fallback(valid_tickers)

    weights, success = get_optimal_weights_fn(prices_subset)
    if success and weights:
        return weights

    logger.info("Markowitz optimization failed, fallback to equal weight.")
    return get_equal_weight_fallback(valid_tickers)


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
) -> Tuple[list, float, float]:
    records = []

    for date in trading_days:
        bench_ret = benchmark_returns.get(date, 0.0)
        strat_ret = 0.0

        if allocation:
            weights = list(allocation.values())
            rets = [
                daily_returns.loc[date, ticker]
                if ticker in daily_returns.columns and date in daily_returns.index
                else 0.0
                for ticker in allocation
            ]
            strat_ret = float(np.average(pd.Series(rets).fillna(0.0), weights=weights))

        portfolio_value *= (1 + strat_ret)
        benchmark_value *= (1 + bench_ret)

        records.append(
            {
                "Date": date,
                "Strategy": portfolio_value,
                "Benchmark": benchmark_value,
            }
        )

    return records, portfolio_value, benchmark_value


# =============================================================================
# MAIN BACKTEST
# =============================================================================

def backtest_strategy_with_rebalancing(
    df_daily: pd.DataFrame,
    signal_generator: AlphaSignal,
    get_optimal_weights_fn: Any,
    portfolio_method: str = "markowitz",   # "markowitz", "topk", "topk_dropout"
    topk: int = 10,
    n_drop: int = 2,
    fee_bps: float = 0.0020,
    weight_bounds: Tuple[float, float] = (0.02, 0.25),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info(f"Starting backtest | method={portfolio_method}")

    daily_prices = df_daily["adj close"].unstack().ffill()
    daily_returns = daily_prices.pct_change().fillna(0.0)

    date_min = df_daily.index.get_level_values("date").min()
    date_max = df_daily.index.get_level_values("date").max()

    benchmark_returns = get_benchmark_returns(
        BENCHMARK_TICKER,
        date_min,
        date_max,
        daily_prices.index,
    )

    portfolio_value, benchmark_value = 100.0, 100.0
    all_records, rebalance_log = [], []

    monthly_dates = (
        signal_generator.signal_cache.index.get_level_values("date")
        .unique()
        .sort_values()
    )

    previous_allocation: Dict[str, float] = {}

    for i, month_date in enumerate(monthly_dates[:-1]):
        month_data = signal_generator.get_signal(month_date)

        allocation: Dict[str, float] = {}
        selected_count = 0

        if not month_data.empty:
            selected = month_data[
                (month_data["cluster"] == TARGET_CLUSTER)
                & (month_data["proba_upside"] > PROBA_THRESHOLD)
            ].copy()

            selected_count = len(selected)

            if not selected.empty:
                if portfolio_method == "markowitz":
                    allocation = build_markowitz_allocation(
                        selected=selected,
                        daily_prices=daily_prices,
                        month_date=month_date,
                        get_optimal_weights_fn=get_optimal_weights_fn,
                        lookback_days=TRADING_DAYS_YEAR,
                        weight_bounds=weight_bounds,
                    )

                elif portfolio_method == "topk":
                    allocation = get_topk_equal_weights(
                        selected=selected,
                        topk=topk,
                        score_col="proba_upside",
                    )

                elif portfolio_method == "topk_dropout":
                    allocation = get_topk_dropout_weights(
                        selected=selected,
                        previous_allocation=previous_allocation,
                        topk=topk,
                        n_drop=n_drop,
                        score_col="proba_upside",
                    )

                else:
                    raise ValueError(f"Unknown portfolio_method: {portfolio_method}")

        cost_pct, turnover = calculate_turnover_friction(
            previous_allocation,
            allocation,
            fee_bps=fee_bps,
        )
        portfolio_value *= (1 - cost_pct)
        previous_allocation = allocation.copy()

        next_month = monthly_dates[i + 1]
        trading_days = daily_prices.index[
            (daily_prices.index >= month_date) & (daily_prices.index < next_month)
        ]

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
                "Method": portfolio_method,
                "Selected_Count": selected_count,
                "N_Stocks": len(allocation),
                "Turnover": turnover,
                "Cost_Pct": cost_pct,
                "Allocation": allocation,
            }
        )

    logger.info(f"Backtest complete. Final value: {portfolio_value:.2f}")

    hist_df = pd.DataFrame(all_records).set_index("Date")
    rebal_df = pd.DataFrame(rebalance_log).set_index("Date")
    return hist_df, rebal_df