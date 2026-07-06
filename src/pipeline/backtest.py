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
    WEIGHT_BOUNDS
)
from src.features.alpha_features import add_all_features
from src.utils.logger import setup_logger
from src.utils.market_utils import get_benchmark_returns

logger = setup_logger("backtest")


def get_optimal_weights(prices_df: pd.DataFrame, risk_free_rate: float = RISK_FREE_RATE) -> Tuple[Dict[str, float], str]:
    if prices_df.shape[1] < MIN_STOCKS_OPTIM:
        n = prices_df.shape[1]
        return {t: 1.0 / n for t in prices_df.columns}, "equal_weight"

    try:
        mu = expected_returns.ema_historical_return(prices_df, frequency=TRADING_DAYS_YEAR, span=252)
        S = risk_models.CovarianceShrinkage(prices_df, frequency=TRADING_DAYS_YEAR).ledoit_wolf()

        ef = EfficientFrontier(mu, S, weight_bounds=WEIGHT_BOUNDS)
        ef.add_objective(objective_functions.L2_reg, gamma=0.1)
        ef.max_sharpe(risk_free_rate=risk_free_rate)

        return dict(ef.clean_weights()), "max_sharpe"

    except Exception as e1:
        logger.warning(f"Max Sharpe failed ({e1}) -> min_volatility fallback.")
        try:
            mu = expected_returns.ema_historical_return(prices_df, frequency=TRADING_DAYS_YEAR, span=252)
            S = risk_models.CovarianceShrinkage(prices_df, frequency=TRADING_DAYS_YEAR).ledoit_wolf()

            ef2 = EfficientFrontier(mu, S, weight_bounds=WEIGHT_BOUNDS)
            ef2.min_volatility()

            return dict(ef2.clean_weights()), "min_vol"

        except Exception as e2:
            logger.warning(f"Min vol failed ({e2}) -> equal_weight fallback.")
            n = prices_df.shape[1]
            return {t: 1.0 / n for t in prices_df.columns}, "equal_weight"


def _score_with_model(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Scoring robuste avec reindex pour gérer les lags macro manquants le jour J."""
    if hasattr(model, "predict_proba"):
        if hasattr(model, "features_"):
            X_input = X.reindex(columns=model.features_).copy()
        else:
            X_input = X.copy()
        X_input = X_input.fillna(0)
        return model.predict_proba(X_input)[:, 1]

    if hasattr(model, "predict"):
        try:
            input_schema = model.metadata.get_input_schema()
            expected_cols = [c.name for c in input_schema.inputs] if input_schema else None
        except Exception:
            expected_cols = None

        if expected_cols:
            X_input = X.reindex(columns=expected_cols).copy()
        else:
            X_input = X.copy()
        X_input = X_input.fillna(0)
        preds = model.predict(X_input)
        return np.asarray(preds).ravel()

    raise TypeError(f"Type de modèle non supporté : {type(model)}")


def _generate_monthly_signals(month_data: pd.DataFrame, model: Any) -> pd.DataFrame:
    if month_data.empty:
        return pd.DataFrame()
    try:
        month_data = month_data.copy()
        month_data["proba_upside"] = _score_with_model(model, month_data)
    except Exception as e:
        logger.error(f"Scoring échoué : {e}", exc_info=True)
        return pd.DataFrame()
    return month_data


def _select_tickers(month_data: pd.DataFrame, proba_min: float = PROBA_MIN, max_stocks: int = MAX_STOCKS_SELECT) -> List[str]:
    if "proba_upside" not in month_data.columns:
        return []
    selected = month_data[month_data["proba_upside"] >= proba_min]
    if selected.empty:
        return []
    return selected.sort_values(by="proba_upside", ascending=False).head(max_stocks).index.tolist()


def _compute_turnover(new_alloc: Dict[str, float], old_alloc: Dict[str, float]) -> float:
    all_tickers = set(new_alloc) | set(old_alloc)
    return sum(abs(new_alloc.get(t, 0.0) - old_alloc.get(t, 0.0)) for t in all_tickers) / 2.0


def _simulate_daily_returns(
    allocation: Dict[str, float],
    drifted_allocation: Dict[str, float],
    trading_days: pd.DatetimeIndex,
    daily_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    portfolio_value: float,
    benchmark_value: float,
) -> Tuple[list, float, float, Dict[str, float]]:
    records = []
    is_first_day = True
    stock_values = {t: portfolio_value * w for t, w in allocation.items()} if allocation else {}

    for date in trading_days:
        bench_ret = benchmark_returns.get(date, 0.0)
        if is_first_day:
            turnover = _compute_turnover(allocation, drifted_allocation)
            transaction_fees = portfolio_value * turnover * TRANSACTION_COST
            portfolio_value -= transaction_fees
            stock_values = {t: portfolio_value * w for t, w in allocation.items()}
            is_first_day = False

        daily_portfolio_pnl = 0.0
        if stock_values:
            for t in list(stock_values.keys()):
                ret = daily_returns.loc[date, t] if (t in daily_returns.columns and date in daily_returns.index) else 0.0
                pnl = stock_values[t] * ret
                stock_values[t] += pnl
                daily_portfolio_pnl += pnl

        portfolio_value += daily_portfolio_pnl
        benchmark_value *= (1 + bench_ret)
        records.append({"Date": date, "Strategy": portfolio_value, "Benchmark": benchmark_value, "N_Stocks": len(stock_values)})

    new_drifted_allocation = {}
    if portfolio_value > 0 and stock_values:
        new_drifted_allocation = {t: val / portfolio_value for t, val in stock_values.items()}

    return records, portfolio_value, benchmark_value, new_drifted_allocation


def compute_performance_metrics(results_df: pd.DataFrame, rebalance_log: pd.DataFrame, risk_free_rate: float = RISK_FREE_RATE) -> Dict[str, float]:
    strat = results_df["Strategy"]
    bench = results_df["Benchmark"]
    strat_ret = strat.pct_change().dropna()
    bench_ret = bench.pct_change().dropna()
    
    n_years = len(strat_ret) / TRADING_DAYS_YEAR
    cagr = (strat.iloc[-1] / strat.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else 0.0
    vol = strat_ret.std() * np.sqrt(TRADING_DAYS_YEAR)
    excess = strat_ret - risk_free_rate / TRADING_DAYS_YEAR
    sharpe = (excess.mean() / strat_ret.std()) * np.sqrt(TRADING_DAYS_YEAR) if strat_ret.std() != 0 else 0.0
    
    downside = strat_ret[strat_ret < 0].std() * np.sqrt(TRADING_DAYS_YEAR)
    sortino = (cagr - risk_free_rate) / downside if downside > 0 else np.nan
    
    rolling_max = strat.cummax()
    drawdown = (strat - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan
    align_df = pd.concat([strat_ret, bench_ret], axis=1).dropna()
    beta = np.cov(align_df.iloc[:, 0], align_df.iloc[:, 1])[0, 1] / np.var(align_df.iloc[:, 1]) if len(align_df) > 1 else np.nan
    alpha = (cagr - risk_free_rate) - beta * ((bench.iloc[-1] / bench.iloc[0]) ** (1 / n_years) - 1 - risk_free_rate)
    
    metrics = {
        "CAGR": round(cagr, 4), "Volatility": round(vol, 4), "Sharpe": round(sharpe, 4),
        "Sortino": round(sortino, 4), "Calmar": round(calmar, 4), "Max_Drawdown": round(max_dd, 4),
        "Alpha": round(alpha, 4), "Beta": round(beta, 4), "Final_Value": round(strat.iloc[-1], 2),
    }
    return metrics


def backtest_strategy_with_rebalancing(df_daily, df_monthly, model, benchmark_ticker="^FCHI", proba_min=PROBA_MIN, max_stocks=MAX_STOCKS_SELECT):
    df_monthly_feat = add_all_features(df_monthly.copy())
    daily_prices = df_daily["adj close"].unstack().ffill()
    daily_returns = daily_prices.pct_change().fillna(0)
    benchmark_returns = get_benchmark_returns(benchmark_ticker, df_daily.index.get_level_values("date").min(), df_daily.index.get_level_values("date").max(), daily_prices.index)
    
    portfolio_value, benchmark_value = 100.0, 100.0
    all_records, rebalance_log, drifted_allocation = [], [], {}
    monthly_dates = df_monthly_feat.index.get_level_values("date").unique().sort_values()

    for i, month_date in enumerate(monthly_dates[:-1]):
        month_data = _generate_monthly_signals(df_monthly_feat.xs(month_date, level="date").copy(), model)
        allocation = {}
        if not month_data.empty:
            tickers = _select_tickers(month_data, proba_min, max_stocks)
            if tickers:
                prices_subset = daily_prices[[t for t in tickers if t in daily_prices.columns]].loc[:month_date].iloc[-TRADING_DAYS_YEAR:].dropna(axis=1, thresh=int(TRADING_DAYS_YEAR * 0.8))
                if not prices_subset.empty:
                    weights, _ = get_optimal_weights(prices_subset)
                    allocation = {t: w for t, w in weights.items() if w > 1e-4}
        
        trading_days = daily_prices.index[(daily_prices.index >= month_date) & (daily_prices.index < monthly_dates[i+1])]
        day_records, portfolio_value, benchmark_value, drifted_allocation = _simulate_daily_returns(allocation, drifted_allocation, trading_days, daily_returns, benchmark_returns, portfolio_value, benchmark_value)
        all_records.extend(day_records)
        rebalance_log.append({"Date": month_date, "N_Stocks": len(allocation), "Allocation": allocation})

    return pd.DataFrame(all_records).set_index("Date"), pd.DataFrame(rebalance_log).set_index("Date"), {}


def generate_live_signals(df_daily, daily_prices, model, rebalance_history, proba_min=PROBA_MIN, max_stocks=MAX_STOCKS_SELECT):
    snapshot, last_date = _build_daily_snapshot(df_daily)
    snapshot_scored = _generate_monthly_signals(snapshot, model)
    
    # 1. Allocation dynamique quotidienne
    tickers = _select_tickers(snapshot_scored, proba_min, max_stocks)
    allocation = {}
    if tickers:
        prices_subset = daily_prices[[t for t in tickers if t in daily_prices.columns]].loc[:last_date].iloc[-TRADING_DAYS_YEAR:].dropna(axis=1, thresh=int(TRADING_DAYS_YEAR * 0.8))
        if not prices_subset.empty:
            weights, _ = get_optimal_weights(prices_subset)
            allocation = {t: w for t, w in weights.items() if w > 1e-4}
            
    # 2. Log mensuel pour ton dashboard
    last_rebalance_date = rebalance_history.index.max() if not rebalance_history.empty else None
    if last_rebalance_date is None or (last_date.year, last_date.month) != (last_rebalance_date.year, last_rebalance_date.month):
        new_row = pd.DataFrame([{"N_Stocks": len(allocation), "Allocation": allocation}], index=[last_date])
        new_row.index.name = "Date"
        rebalance_history = pd.concat([rebalance_history, new_row]).sort_index()

    out = snapshot_scored.reset_index().rename(columns={"ticker": "Ticker"})
    out["Allocation"] = out["Ticker"].map(allocation).fillna(0.0)
    out["Signal"] = np.where(out["Allocation"] > 0, "BUY", "NEUTRAL")
    out["Proba_Hausse"] = (out["proba_upside"] * 100).round(1)
    
    return out[["Ticker", "Signal", "Allocation", "Proba_Hausse"]], rebalance_history

def _build_daily_snapshot(df_daily):
    last_date = df_daily.index.get_level_values("date").max()
    df_feat = add_all_features(df_daily.copy())
    return df_feat[df_feat.index.get_level_values("date") == last_date].copy(), last_date
