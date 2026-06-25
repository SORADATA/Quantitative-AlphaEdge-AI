"""
Backtest Engine
========================================
Moteur de backtest avec rebalancement mensuel et simulation journalière.

"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions

from const import (
    FEATURE_COLS,
    TRADING_DAYS_YEAR,
    RISK_FREE_RATE,
)
from src.utils.logger import setup_logger
from src.utils.market_utils import get_benchmark_returns

logger = setup_logger("backtest")

# ══════════════════════════════════════════════════════════════════
# CONSTANTES BACKTEST
# ══════════════════════════════════════════════════════════════════
TRANSACTION_COST = 0.0010   # 10 bps par rebalancement (institutionnel réaliste)
MIN_STOCKS_OPTIM = 3        # Minimum de titres pour lancer l'optimiseur
MAX_STOCKS_SELECT = 15       # Maximum de titres sélectionnés par le signal
PROBA_MIN = 0.55     # Seuil minimal de probabilité (plus bas = plus inclusif)
WEIGHT_BOUNDS = (0.02, 0.20)  # Min 2% / Max 20% par titre


# ══════════════════════════════════════════════════════════════════
# OPTIMISATION DE PORTEFEUILLE
# ══════════════════════════════════════════════════════════════════

def get_optimal_weights(
    prices_df: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
) -> Tuple[Dict[str, float], str]:
    """
    Optimisation Max Sharpe avec fallbacks progressifs.

    Cascade :
      1. Max Sharpe + L2 regularization  (préféré)
      2. Min Volatilité                  (si Max Sharpe échoue)
      3. Equal Weight                    (fallback final)

    Parameters
    ----------
    prices_df : pd.DataFrame — prix mensuels, colonnes = tickers
    risk_free_rate : float

    Returns
    -------
    weights : dict {ticker: weight}
    method  : str  — méthode utilisée ('max_sharpe' | 'min_vol' | 'equal_weight')
    """
    if prices_df.shape[1] < MIN_STOCKS_OPTIM:
        n = prices_df.shape[1]
        return {t: 1.0 / n for t in prices_df.columns}, "equal_weight"

    try:
        # EMA returns — pondère davantage les rendements récents
        mu = expected_returns.ema_historical_return(
            prices_df, frequency=TRADING_DAYS_YEAR, span=252
        )
        S = risk_models.CovarianceShrinkage(
            prices_df, frequency=TRADING_DAYS_YEAR
        ).ledoit_wolf()

        ef = EfficientFrontier(mu, S, weight_bounds=WEIGHT_BOUNDS)
        # L2 regularization → évite la concentration extrême
        ef.add_objective(objective_functions.L2_reg, gamma=0.1)
        ef.max_sharpe(risk_free_rate=risk_free_rate)

        return dict(ef.clean_weights()), "max_sharpe"

    except Exception as e1:
        logger.warning(f"Max Sharpe failed ({e1}) → trying min volatility...")
        try:
            mu = expected_returns.ema_historical_return(
                prices_df, frequency=TRADING_DAYS_YEAR, span=252
            )
            S = risk_models.CovarianceShrinkage(
                prices_df, frequency=TRADING_DAYS_YEAR
            ).ledoit_wolf()

            ef2 = EfficientFrontier(mu, S, weight_bounds=WEIGHT_BOUNDS)
            ef2.min_volatility()

            return dict(ef2.clean_weights()), "min_vol"

        except Exception as e2:
            logger.warning(f"Min vol also failed ({e2}) → equal weight fallback.")
            n = prices_df.shape[1]
            return {t: 1.0 / n for t in prices_df.columns}, "equal_weight"


# ══════════════════════════════════════════════════════════════════
# GÉNÉRATION DES SIGNAUX
# ══════════════════════════════════════════════════════════════════

def _generate_monthly_signals(
    month_data: pd.DataFrame,
    model: Any,
) -> pd.DataFrame:
    """
    Génère les signaux ML pour un mois donné.

    Compatible avec :
      - AlphaEdgeEnsemble  (nouveau — recommandé)
      - XGBClassifier seul (ancien — rétrocompatible)

    Le cluster n'est plus calculé ici — il est dans les features
    en tant que cluster_lag1 (calculé dans alpha_features.py).

    Parameters
    ----------
    month_data : pd.DataFrame — slice mensuelle (1 date, N tickers)
    model      : modèle entraîné avec .predict_proba()

    Returns
    -------
    pd.DataFrame avec colonne 'proba_upside' ajoutée.
    """
    available_features = [c for c in FEATURE_COLS if c in month_data.columns]
    missing = [c for c in FEATURE_COLS if c not in month_data.columns]

    if missing:
        logger.warning(f"Features manquantes dans month_data : {missing}")

    if len(available_features) == 0:
        return pd.DataFrame()

    X = month_data[available_features].fillna(0)

    try:
        month_data = month_data.copy()
        month_data["proba_upside"] = model.predict_proba(X)[:, 1]
    except Exception as e:
        logger.error(f"predict_proba failed : {e}")
        return pd.DataFrame()

    return month_data


# ══════════════════════════════════════════════════════════════════
# SÉLECTION DES TITRES
# ══════════════════════════════════════════════════════════════════

def _select_tickers(
    month_data: pd.DataFrame,
    proba_min: float = PROBA_MIN,
    max_stocks: int = MAX_STOCKS_SELECT,
) -> List[str]:
    """
    Sélectionne les N meilleurs titres par probabilité ML.

    Remplace le double filtre (cluster == TARGET_CLUSTER & proba > 0.6)
    qui était trop restrictif et laissait souvent 0 titres sélectionnés.

    Parameters
    ----------
    month_data : pd.DataFrame avec colonne 'proba_upside'
    proba_min  : seuil minimal de probabilité
    max_stocks : nombre maximum de titres à sélectionner

    Returns
    -------
    list[str] — tickers sélectionnés, triés par proba décroissante
    """
    if "proba_upside" not in month_data.columns:
        return []

    selected = month_data[month_data["proba_upside"] >= proba_min]

    if selected.empty:
        # Fallback : prendre les 3 meilleurs même sous le seuil
        selected = month_data.nlargest(3, "proba_upside")
        logger.debug("Aucun ticker au-dessus du seuil — fallback top-3.")

    # Limiter à MAX_STOCKS et trier par proba décroissante
    selected = selected.nlargest(max_stocks, "proba_upside")

    return selected.index.tolist()


# ══════════════════════════════════════════════════════════════════
# SIMULATION JOURNALIÈRE
# ══════════════════════════════════════════════════════════════════

def _compute_turnover(
    new_alloc: Dict[str, float],
    old_alloc: Dict[str, float],
) -> float:
    """
    Calcule le turnover entre deux allocations.
    Turnover = somme des changements de poids absolus / 2
    """
    all_tickers = set(new_alloc) | set(old_alloc)
    return sum(
        abs(new_alloc.get(t, 0.0) - old_alloc.get(t, 0.0))
        for t in all_tickers
    ) / 2.0


def _simulate_daily_returns(
    allocation: Dict[str, float],
    prev_allocation: Dict[str, float],
    trading_days: pd.DatetimeIndex,
    daily_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    portfolio_value: float,
    benchmark_value: float,
) -> Tuple[list, float, float]:
    """
    Simule les rendements journaliers pour un mois donné.

    Inclut les transaction costs au premier jour de chaque période
    (lors du rebalancement).

    Parameters
    ----------
    allocation      : dict {ticker: weight} — nouvelle allocation
    prev_allocation : dict {ticker: weight} — allocation précédente
    trading_days    : index des jours de trading du mois
    daily_returns   : pd.DataFrame — rendements journaliers (colonnes = tickers)
    benchmark_returns : pd.Series — rendements journaliers du benchmark
    portfolio_value : float — valeur courante du portefeuille
    benchmark_value : float — valeur courante du benchmark

    Returns
    -------
    records         : list[dict]
    portfolio_value : float — valeur mise à jour
    benchmark_value : float — valeur mise à jour
    """
    records = []
    is_first_day = True

    for date in trading_days:
        bench_ret = benchmark_returns.get(date, 0.0)
        strat_ret = 0.0

        if allocation:
            rets = [
                daily_returns.loc[date, t]
                if (t in daily_returns.columns and date in daily_returns.index)
                else 0.0
                for t in allocation
            ]
            strat_ret = float(
                np.average(pd.Series(rets).fillna(0), weights=list(allocation.values()))
            )

        # Transaction costs au jour de rebalancement
        if is_first_day and allocation:
            turnover = _compute_turnover(allocation, prev_allocation)
            strat_ret -= turnover * TRANSACTION_COST
            is_first_day = False

        portfolio_value *= (1 + strat_ret)
        benchmark_value *= (1 + bench_ret)

        records.append({
            "Date":      date,
            "Strategy":  portfolio_value,
            "Benchmark": benchmark_value,
            "N_Stocks":  len(allocation),
        })

    return records, portfolio_value, benchmark_value


# ══════════════════════════════════════════════════════════════════
# MÉTRIQUES DE PERFORMANCE
# ══════════════════════════════════════════════════════════════════

def compute_performance_metrics(
    results_df: pd.DataFrame,
    rebalance_log: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
) -> Dict[str, float]:
    """
    Calcule les métriques de performance institutionnelles.

    Returns
    -------
    dict avec : Sharpe, Sortino, Calmar, Max Drawdown, Volatilité,
                CAGR, Hit Rate, Avg N_Stocks, méthode optim breakdown
    """
    strat = results_df["Strategy"]
    bench = results_df["Benchmark"]

    # Rendements journaliers
    strat_ret = strat.pct_change().dropna()
    bench_ret = bench.pct_change().dropna()

    # CAGR
    n_years = len(strat_ret) / TRADING_DAYS_YEAR
    cagr = (strat.iloc[-1] / strat.iloc[0]) ** (1 / n_years) - 1

    # Volatilité annualisée
    vol = strat_ret.std() * np.sqrt(TRADING_DAYS_YEAR)

    # Sharpe ratio
    excess = strat_ret - risk_free_rate / TRADING_DAYS_YEAR
    sharpe = (excess.mean() / strat_ret.std()) * np.sqrt(TRADING_DAYS_YEAR)

    # Sortino ratio
    downside = strat_ret[strat_ret < 0].std() * np.sqrt(TRADING_DAYS_YEAR)
    sortino = (cagr - risk_free_rate) / downside if downside > 0 else np.nan

    # Max Drawdown
    rolling_max = strat.cummax()
    drawdown = (strat - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # Calmar ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

    # Alpha & Beta vs benchmark
    cov_matrix = np.cov(strat_ret.values, bench_ret.values)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else np.nan
    alpha = (cagr - risk_free_rate) - beta * (
        (bench.iloc[-1] / bench.iloc[0]) ** (1 / n_years) - 1 - risk_free_rate
    )

    # Hit Rate mensuel (% de mois positifs)
    monthly_strat = strat.resample("BME").last().pct_change().dropna()
    hit_rate = (monthly_strat > 0).mean()

    # Nombre moyen de titres en portefeuille
    avg_stocks = rebalance_log["N_Stocks"].mean() if "N_Stocks" in rebalance_log.columns else np.nan

    metrics = {
        "CAGR":          round(cagr, 4),
        "Volatility":    round(vol, 4),
        "Sharpe":        round(sharpe, 4),
        "Sortino":       round(sortino, 4),
        "Calmar":        round(calmar, 4),
        "Max_Drawdown":  round(max_dd, 4),
        "Alpha":         round(alpha, 4),
        "Beta":          round(beta, 4),
        "Hit_Rate":      round(hit_rate, 4),
        "Avg_N_Stocks":  round(avg_stocks, 1),
        "Final_Value":   round(strat.iloc[-1], 2),
    }

    logger.info("=" * 50)
    logger.info("📊 PERFORMANCE METRICS")
    logger.info(f"   CAGR          : {cagr:.2%}")
    logger.info(f"   Sharpe Ratio  : {sharpe:.3f}")
    logger.info(f"   Sortino Ratio : {sortino:.3f}")
    logger.info(f"   Calmar Ratio  : {calmar:.3f}")
    logger.info(f"   Max Drawdown  : {max_dd:.2%}")
    logger.info(f"   Alpha         : {alpha:.2%}")
    logger.info(f"   Beta          : {beta:.3f}")
    logger.info(f"   Hit Rate      : {hit_rate:.1%}")
    logger.info("=" * 50)

    return metrics


# ══════════════════════════════════════════════════════════════════
# BACKTEST PRINCIPAL
# ══════════════════════════════════════════════════════════════════

def backtest_strategy_with_rebalancing(
    df_daily: pd.DataFrame,
    df_monthly: pd.DataFrame,
    model: Any,
    benchmark_ticker: str = "^FCHI",
    proba_min: float = PROBA_MIN,
    max_stocks: int = MAX_STOCKS_SELECT,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Backtest complet avec rebalancement mensuel et simulation journalière.

    Parameters
    ----------
    df_daily         : pd.DataFrame — MultiIndex (date, ticker), OHLCV
    df_monthly       : pd.DataFrame — MultiIndex (date, ticker), features mensuelles
    model            : AlphaEdgeEnsemble ou XGBClassifier
    benchmark_ticker : str — ticker du benchmark (défaut: ^FCHI = CAC40)
    proba_min        : float — seuil minimal de probabilité ML
    max_stocks       : int — nombre max de titres en portefeuille

    Returns
    -------
    results_df    : pd.DataFrame — évolution Strategy vs Benchmark
    rebalance_log : pd.DataFrame — log des rebalancements mensuels
    metrics       : dict — métriques de performance complètes
    """
    logger.info(f"Starting backtest | Benchmark: {benchmark_ticker}")

    # ── Préparation des données
    daily_prices = df_daily["adj close"].unstack().ffill()
    daily_returns = daily_prices.pct_change().fillna(0)

    date_min = df_daily.index.get_level_values("date").min()
    date_max = df_daily.index.get_level_values("date").max()

    benchmark_returns = get_benchmark_returns(
        benchmark_ticker, date_min, date_max, daily_prices.index
    )

    #  Initialisation
    portfolio_value = 100.0
    benchmark_value = 100.0
    all_records = []
    rebalance_log = []
    prev_allocation: Dict[str, float] = {} 

    monthly_dates = df_monthly.index.get_level_values("date").unique().sort_values()

    # Boucle de rebalancement mensuel
    for i, month_date in enumerate(monthly_dates[:-1]):

        # 1. Générer les signaux ML
        month_data = _generate_monthly_signals(
            df_monthly.xs(month_date, level="date").copy(),
            model,
        )

        # Sélection des titres
        allocation: Dict[str, float] = {}
        optim_method = "no_signal"

        if not month_data.empty:
            tickers = _select_tickers(month_data, proba_min, max_stocks)

            if tickers:
                # Prix historiques pour l'optimisation (1 an glissant)
                available_tickers = [t for t in tickers if t in daily_prices.columns]
                prices_subset = (
                    daily_prices[available_tickers]
                    .loc[:month_date]
                    .iloc[-TRADING_DAYS_YEAR:]
                    .dropna(axis=1, thresh=int(TRADING_DAYS_YEAR * 0.8))
                )

                if not prices_subset.empty:
                    weights, optim_method = get_optimal_weights(prices_subset)
                    # Filtrer les poids nuls
                    allocation = {t: w for t, w in weights.items() if w > 1e-4}

        # Simulation journalière du mois
        next_month = monthly_dates[i + 1]
        trading_days = daily_prices.index[
            (daily_prices.index >= month_date) &
            (daily_prices.index < next_month)
        ]

        day_records, portfolio_value, benchmark_value = _simulate_daily_returns(
            allocation, prev_allocation,
            trading_days, daily_returns,
            benchmark_returns, portfolio_value, benchmark_value,
        )
        all_records.extend(day_records)

        # 4. Log du rebalancement
        rebalance_log.append({
            "Date":         month_date,
            "N_Stocks":     len(allocation),
            "Optim_Method": optim_method,
            "Allocation":   allocation,
            "Top_Ticker":   max(allocation, key=allocation.get) if allocation else None,
        })

        prev_allocation = allocation.copy()

    # Résultats finaux
    results_df = pd.DataFrame(all_records).set_index("Date")
    rebalance_df = pd.DataFrame(rebalance_log).set_index("Date")

    metrics = compute_performance_metrics(results_df, rebalance_df)

    logger.info(f"✅ Backtest complete | Final value: {portfolio_value:.2f}")

    return results_df, rebalance_df, metrics