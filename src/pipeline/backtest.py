"""
backtest.py
===========

Moteur de backtest et de génération de signaux live pour AlphaEdge.

Architecture du module
-----------------------
1. Construction de portefeuille  : sélection des tickers + optimisation Markowitz
2. Moteur de simulation          : vectorisé (numpy), pas de boucle Python par jour
3. Analytics de performance      : Sharpe, Sortino, Calmar, Alpha/Beta, Information Ratio,
                                    Tracking Error, coût de turnover
4. API publique                  : backtest_strategy_with_rebalancing / generate_live_signals

Notes de conception
--------------------
- Les positions non allouées (somme des poids < 1) sont conservées en cash à
  rendement nul, explicitement comptabilisées — pas de "cash fantôme".
- Les coûts de transaction sont appliqués à chaque rebalancement, proportionnellement
  au turnover réel entre l'allocation cible et l'allocation *drifted* (dérivée
  des variations de prix depuis le dernier rebalancement).
- Aucune métrique de performance passée ne constitue une garantie de résultats
  futurs. Ce module sert à l'analyse quantitative, pas à du conseil en investissement.
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

# Nombre de jours de forward-fill toléré pour un prix manquant avant de
# considérer le titre comme illiquide/délisté et de l'exclure de la fenêtre.
MAX_PRICE_FFILL_DAYS = 5

# Seuil de features manquantes (moyenne sur toutes les colonnes) au-delà
# duquel on log un warning explicite plutôt que d'imputer silencieusement.
MAX_ACCEPTABLE_MISSING_RATIO = 0.05


# =============================================================================
# 1. CONSTRUCTION DE PORTEFEUILLE
# =============================================================================

def get_optimal_weights(
    prices_df: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
) -> Tuple[Dict[str, float], str]:
    """
    Optimise les poids du portefeuille (Max Sharpe -> Min Vol -> Equal Weight en cascade).

    Args:
        prices_df: Prix ajustés, colonnes = tickers, index = dates.
        risk_free_rate: Taux sans risque annualisé utilisé pour Max Sharpe.

    Returns:
        (weights, method) où method indique la stratégie effectivement utilisée,
        utile pour le monitoring de la robustesse de l'optimiseur en production.
    """
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
        logger.warning(f"Max Sharpe a échoué ({exc}) -> fallback min_volatility.")
        try:
            mu = expected_returns.ema_historical_return(prices_df, frequency=TRADING_DAYS_YEAR, span=252)
            cov = risk_models.CovarianceShrinkage(prices_df, frequency=TRADING_DAYS_YEAR).ledoit_wolf()

            ef_minvol = EfficientFrontier(mu, cov, weight_bounds=WEIGHT_BOUNDS)
            ef_minvol.min_volatility()

            weights = dict(ef_minvol.clean_weights())
            _log_concentration(weights)
            return weights, "min_vol"

        except Exception as exc2:
            logger.warning(f"Min Vol a également échoué ({exc2}) -> fallback equal_weight.")
            return {t: 1.0 / n_assets for t in prices_df.columns}, "equal_weight"


def _log_concentration(weights: Dict[str, float]) -> None:
    """Log l'indice de Herfindahl-Hirschman du portefeuille (mesure de concentration)."""
    active = np.array([w for w in weights.values() if w > 1e-6])
    if active.size == 0:
        return
    hhi = float(np.sum(active ** 2))
    logger.debug(f"Concentration du portefeuille (HHI) : {hhi:.3f} sur {active.size} positions.")


def _score_with_model(model: Any, features: pd.DataFrame) -> np.ndarray:
    """
    Scoring robuste du modèle avec reindex pour gérer les colonnes manquantes
    (ex : lags macro non disponibles le jour J).

    Toute imputation par zéro est loggée si elle dépasse un seuil raisonnable,
    afin de détecter une dérive de la qualité de données en production plutôt
    que de la masquer silencieusement.
    """
    if hasattr(model, "predict_proba"):
        expected_cols = getattr(model, "features_", None)
        x_input = features.reindex(columns=expected_cols).copy() if expected_cols else features.copy()
        _warn_if_missing(x_input)
        return model.predict_proba(x_input.fillna(0))[:, 1]

    if hasattr(model, "predict"):
        expected_cols = None
        try:
            input_schema = model.metadata.get_input_schema()
            expected_cols = [c.name for c in input_schema.inputs] if input_schema else None
        except Exception:
            pass

        x_input = features.reindex(columns=expected_cols).copy() if expected_cols else features.copy()
        _warn_if_missing(x_input)
        preds = model.predict(x_input.fillna(0))
        return np.asarray(preds).ravel()

    raise TypeError(f"Type de modèle non supporté pour le scoring : {type(model)}")


def _warn_if_missing(x_input: pd.DataFrame) -> None:
    if x_input.empty:
        return
    missing_ratio = float(x_input.isna().mean().mean())
    if missing_ratio > MAX_ACCEPTABLE_MISSING_RATIO:
        logger.warning(
            f"Scoring : {missing_ratio:.1%} de valeurs manquantes en moyenne dans les "
            "features avant imputation à 0 — vérifier la fraîcheur des données amont."
        )


def _generate_monthly_signals(month_data: pd.DataFrame, model: Any) -> pd.DataFrame:
    """Score un snapshot (un ticker par ligne) et ajoute la colonne 'proba_upside'."""
    if month_data.empty:
        return pd.DataFrame()
    try:
        scored = month_data.copy()
        scored["proba_upside"] = _score_with_model(model, scored)
        return scored
    except Exception:
        logger.error("Scoring du snapshot échoué.", exc_info=True)
        return pd.DataFrame()


def _select_tickers(
    month_data: pd.DataFrame,
    proba_min: float = PROBA_MIN,
    max_stocks: int = MAX_STOCKS_SELECT,
) -> List[str]:
    """
    Sélectionne les tickers dont la probabilité de hausse dépasse le seuil,
    triés par conviction décroissante, plafonnés à max_stocks.

    IMPORTANT : `month_data.index` doit être un index simple de tickers
    (pas un MultiIndex ticker/date), sous peine de retourner des tuples
    qui ne matcheront jamais les colonnes d'un DataFrame de prix en aval.
    """
    if "proba_upside" not in month_data.columns:
        return []
    if isinstance(month_data.index, pd.MultiIndex):
        raise ValueError(
            "_select_tickers attend un index simple de tickers ; reçu un MultiIndex. "
            "Utiliser .xs(date, level='date') en amont pour aplatir l'index."
        )
    selected = month_data[month_data["proba_upside"] >= proba_min]
    if selected.empty:
        return []
    return selected.sort_values("proba_upside", ascending=False).head(max_stocks).index.tolist()


def _blend_with_conviction(
    weights: Dict[str, float],
    proba_by_ticker: pd.Series,
    conviction_tilt: float = 0.0,
) -> Dict[str, float]:
    """
    Incline les poids risk-based (Markowitz) vers les tickers à plus forte
    conviction (proba de hausse plus élevée), dans l'esprit d'un overlay
    "signal-weighted" utilisé par de nombreux quant funds.

    conviction_tilt = 0.0 -> comportement inchangé (poids Markowitz purs).
    conviction_tilt = 1.0 -> poids entièrement proportionnels à la conviction.
    """
    if conviction_tilt <= 0 or not weights:
        return weights

    proba_aligned = proba_by_ticker.reindex(weights.keys())
    if proba_aligned.isna().any():
        proba_aligned = proba_aligned.fillna(proba_aligned.mean())
    if proba_aligned.sum() <= 0:
        return weights

    conviction_weights = (proba_aligned / proba_aligned.sum()).to_dict()
    blended = {
        t: (1 - conviction_tilt) * weights[t] + conviction_tilt * conviction_weights[t]
        for t in weights
    }
    total = sum(blended.values())
    return {t: w / total * sum(weights.values()) for t, w in blended.items()} if total > 0 else weights


def _compute_turnover(new_alloc: Dict[str, float], old_alloc: Dict[str, float]) -> float:
    """Turnover one-way = demi-somme des variations absolues de poids."""
    all_tickers = set(new_alloc) | set(old_alloc)
    return sum(abs(new_alloc.get(t, 0.0) - old_alloc.get(t, 0.0)) for t in all_tickers) / 2.0


def _build_price_matrix(df_daily: pd.DataFrame, ffill_limit: int = MAX_PRICE_FFILL_DAYS) -> pd.DataFrame:
    """
    Construit la matrice de prix ticker x date, avec un forward-fill *borné*.

    Un ffill illimité masquerait indéfiniment un titre délisté ou suspendu
    (prix figé recopié à l'infini) ; on limite la tolérance à `ffill_limit`
    jours puis on laisse les NaN résiduels, qui seront traités comme
    "non éligible" plutôt que silencieusement valorisés à un prix obsolète.
    """
    prices = df_daily["adj close"].unstack()
    prices = prices.ffill(limit=ffill_limit)
    stale_cols = prices.columns[prices.iloc[-1].isna()]
    if len(stale_cols) > 0:
        logger.debug(f"{len(stale_cols)} tickers exclus (prix obsolètes/manquants) : {list(stale_cols)[:10]}...")
    return prices


# =============================================================================
# 2. MOTEUR DE SIMULATION (VECTORISÉ)
# =============================================================================

def _simulate_period(
    allocation: Dict[str, float],
    drifted_allocation: Dict[str, float],
    trading_days: pd.DatetimeIndex,
    daily_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    portfolio_value: float,
    benchmark_value: float,
    transaction_cost: float = TRANSACTION_COST,
) -> Tuple[pd.DataFrame, float, float, Dict[str, float]]:
    """
    Simule l'évolution quotidienne du portefeuille sur une période de rebalancement,
    de façon entièrement vectorisée (pas de boucle Python jour par jour).

    Le cash non investi (1 - somme des poids) est conservé à rendement nul et
    explicitement réintégré dans la valeur totale du portefeuille — condition
    nécessaire pour que le sizing (vol targeting, conviction tilt, etc.) soit
    comptabilisé correctement.

    Returns:
        period_df: colonnes Strategy / Benchmark / N_Stocks, indexées par date.
        final_portfolio_value, final_benchmark_value: valeurs en fin de période.
        new_drifted_allocation: poids réels (post-dérive de prix) en fin de période,
            utilisés comme référence de turnover au prochain rebalancement.
    """
    if len(trading_days) == 0:
        empty = pd.DataFrame(columns=["Strategy", "Benchmark", "N_Stocks"])
        return empty, portfolio_value, benchmark_value, drifted_allocation

    turnover = _compute_turnover(allocation, drifted_allocation)
    transaction_fees = portfolio_value * turnover * transaction_cost
    portfolio_value -= transaction_fees

    tickers = list(allocation.keys())
    gross_exposure = sum(allocation.values()) if tickers else 0.0
    cash_amount = portfolio_value * (1.0 - gross_exposure)

    if tickers:
        weights = np.array([allocation[t] for t in tickers])
        rets = daily_returns.reindex(index=trading_days, columns=tickers).fillna(0.0).to_numpy()
        growth = np.cumprod(1.0 + rets, axis=0)                      # (n_days, n_tickers)
        stock_values = portfolio_value * weights[np.newaxis, :] * growth
        strategy_values = stock_values.sum(axis=1) + cash_amount
        n_stocks_series = np.full(len(trading_days), len(tickers))

        final_total = float(strategy_values[-1])
        if final_total > 0:
            new_drifted_allocation = dict(zip(tickers, (stock_values[-1, :] / final_total).tolist()))
        else:
            new_drifted_allocation = {}
    else:
        strategy_values = np.full(len(trading_days), portfolio_value)
        n_stocks_series = np.zeros(len(trading_days), dtype=int)
        final_total = portfolio_value
        new_drifted_allocation = {}

    bench_rets = benchmark_returns.reindex(trading_days).fillna(0.0).to_numpy()
    benchmark_values = benchmark_value * np.cumprod(1.0 + bench_rets)

    period_df = pd.DataFrame(
        {"Strategy": strategy_values, "Benchmark": benchmark_values, "N_Stocks": n_stocks_series},
        index=trading_days,
    )
    period_df.index.name = "Date"

    return period_df, final_total, float(benchmark_values[-1]), new_drifted_allocation


# =============================================================================
# 3. ANALYTICS DE PERFORMANCE
# =============================================================================

def compute_performance_metrics(
    results_df: pd.DataFrame,
    rebalance_log: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
) -> Dict[str, float]:
    """
    Calcule le jeu de métriques standard d'un desk quant :
    CAGR, volatilité, Sharpe, Sortino, Calmar, Max Drawdown, Alpha/Beta,
    Tracking Error, Information Ratio, Win Rate, et coût de turnover moyen.
    """
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
    align_df.columns = ["strat", "bench"]

    if len(align_df) > 1 and align_df["bench"].var() > 0:
        beta = np.cov(align_df["strat"], align_df["bench"])[0, 1] / np.var(align_df["bench"])
        bench_cagr = (bench.iloc[-1] / bench.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else 0.0
        alpha = (cagr - risk_free_rate) - beta * (bench_cagr - risk_free_rate)

        active_ret = align_df["strat"] - align_df["bench"]
        tracking_error = active_ret.std() * np.sqrt(TRADING_DAYS_YEAR)
        information_ratio = (
            (active_ret.mean() * TRADING_DAYS_YEAR) / tracking_error if tracking_error > 0 else np.nan
        )
    else:
        beta = alpha = tracking_error = information_ratio = np.nan

    win_rate = float((strat_ret > 0).mean()) if len(strat_ret) > 0 else np.nan
    avg_turnover = _average_turnover(rebalance_log)

    return {
        "CAGR": round(float(cagr), 4),
        "Volatility": round(float(vol), 4),
        "Sharpe": round(float(sharpe), 4),
        "Sortino": round(float(sortino), 4) if pd.notna(sortino) else np.nan,
        "Calmar": round(float(calmar), 4) if pd.notna(calmar) else np.nan,
        "Max_Drawdown": round(float(max_dd), 4),
        "Alpha": round(float(alpha), 4) if pd.notna(alpha) else np.nan,
        "Beta": round(float(beta), 4) if pd.notna(beta) else np.nan,
        "Tracking_Error": round(float(tracking_error), 4) if pd.notna(tracking_error) else np.nan,
        "Information_Ratio": round(float(information_ratio), 4) if pd.notna(information_ratio) else np.nan,
        "Win_Rate": round(win_rate, 4) if pd.notna(win_rate) else np.nan,
        "Avg_Monthly_Turnover": round(avg_turnover, 4),
        "Final_Value": round(float(strat.iloc[-1]), 2),
    }


def _average_turnover(rebalance_log: pd.DataFrame) -> float:
    """Turnover moyen d'un mois sur l'autre — proxy du coût de friction réel de la stratégie."""
    if rebalance_log.empty or "Allocation" not in rebalance_log.columns:
        return 0.0
    turnovers, prev_alloc = [], {}
    for alloc in rebalance_log["Allocation"]:
        alloc = alloc or {}
        turnovers.append(_compute_turnover(alloc, prev_alloc))
        prev_alloc = alloc
    return float(np.mean(turnovers)) if turnovers else 0.0


# =============================================================================
# 4. API PUBLIQUE
# =============================================================================

def backtest_strategy_with_rebalancing(
    df_daily: pd.DataFrame,
    df_monthly: pd.DataFrame,
    model: Any,
    benchmark_ticker: str = "^FCHI",
    proba_min: float = PROBA_MIN,
    max_stocks: int = MAX_STOCKS_SELECT,
    conviction_tilt: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Backtest complet avec rebalancement mensuel.

    Args:
        conviction_tilt: 0.0 = poids Markowitz purs (comportement historique).
            >0 incline les poids vers les tickers à plus forte conviction du modèle.

    Returns:
        (historique_valorisation, journal_de_rebalancement, métriques_de_performance)
        Les métriques sont désormais réellement calculées (CAGR, Sharpe, Alpha,
        Information Ratio, turnover moyen, etc.) — elles n'étaient pas renseignées
        dans la version précédente du pipeline.
    """
    df_monthly_feat = add_all_features(df_monthly.copy())
    daily_prices = _build_price_matrix(df_daily)
    daily_returns = daily_prices.pct_change().fillna(0)
    benchmark_returns = get_benchmark_returns(
        benchmark_ticker,
        df_daily.index.get_level_values("date").min(),
        df_daily.index.get_level_values("date").max(),
        daily_prices.index,
    )

    portfolio_value, benchmark_value = 100.0, 100.0
    drifted_allocation: Dict[str, float] = {}
    period_frames: List[pd.DataFrame] = []
    rebalance_log: List[Dict[str, Any]] = []

    monthly_dates = df_monthly_feat.index.get_level_values("date").unique().sort_values()

    for i, month_date in enumerate(monthly_dates[:-1]):
        month_data = _generate_monthly_signals(
            df_monthly_feat.xs(month_date, level="date").copy(), model
        )

        allocation: Dict[str, float] = {}
        if not month_data.empty:
            tickers = _select_tickers(month_data, proba_min, max_stocks)
            if tickers:
                eligible = [t for t in tickers if t in daily_prices.columns]
                prices_subset = (
                    daily_prices[eligible]
                    .loc[:month_date]
                    .iloc[-TRADING_DAYS_YEAR:]
                    .dropna(axis=1, thresh=int(TRADING_DAYS_YEAR * 0.8))
                )
                if not prices_subset.empty:
                    weights, method = get_optimal_weights(prices_subset)
                    allocation = {t: w for t, w in weights.items() if w > 1e-4}
                    if conviction_tilt > 0:
                        allocation = _blend_with_conviction(
                            allocation, month_data["proba_upside"], conviction_tilt
                        )
                    logger.debug(f"[{month_date.date()}] Optimisation via '{method}', {len(allocation)} positions.")

        trading_days = daily_prices.index[
            (daily_prices.index >= month_date) & (daily_prices.index < monthly_dates[i + 1])
        ]
        period_df, portfolio_value, benchmark_value, drifted_allocation = _simulate_period(
            allocation, drifted_allocation, trading_days, daily_returns, benchmark_returns,
            portfolio_value, benchmark_value,
        )
        period_frames.append(period_df)
        rebalance_log.append({"Date": month_date, "N_Stocks": len(allocation), "Allocation": allocation})

    hist_df = pd.concat(period_frames) if period_frames else pd.DataFrame(columns=["Strategy", "Benchmark", "N_Stocks"])
    rebal_df = pd.DataFrame(rebalance_log).set_index("Date") if rebalance_log else pd.DataFrame()

    metrics = compute_performance_metrics(hist_df, rebal_df) if not hist_df.empty else {}

    return hist_df, rebal_df, metrics


def generate_live_signals(
    df_daily: pd.DataFrame,
    daily_prices: pd.DataFrame,
    model: Any,
    rebalance_history: pd.DataFrame,
    proba_min: float = PROBA_MIN,
    max_stocks: int = MAX_STOCKS_SELECT,
    conviction_tilt: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Génère les signaux du jour (séance N) pour alimenter le dashboard.

    Fix critique : `_build_daily_snapshot` renvoie désormais un index simple
    de tickers (via `.xs(level="date")`) au lieu d'un MultiIndex (ticker, date).
    Avant ce fix, `_select_tickers` retournait des tuples `(ticker, date)` qui ne
    matchaient jamais `daily_prices.columns`, laissant l'allocation vide en
    permanence — d'où les signaux NEUTRAL à 0.00 malgré des probabilités élevées.
    """
    snapshot, last_date = _build_daily_snapshot(df_daily)
    snapshot_scored = _generate_monthly_signals(snapshot, model)

    if snapshot_scored.empty:
        logger.error("Snapshot scoré vide — aucun signal ne peut être généré aujourd'hui.")
        empty_signals = pd.DataFrame(columns=["Ticker", "Signal", "Allocation", "Proba_Hausse"])
        return empty_signals, rebalance_history

    tickers = _select_tickers(snapshot_scored, proba_min, max_stocks)
    allocation: Dict[str, float] = {}
    if tickers:
        eligible = [t for t in tickers if t in daily_prices.columns]
        if len(eligible) < len(tickers):
            missing = set(tickers) - set(eligible)
            logger.warning(f"{len(missing)} ticker(s) sélectionné(s) absents de la matrice de prix : {missing}")

        prices_subset = (
            daily_prices[eligible]
            .loc[:last_date]
            .iloc[-TRADING_DAYS_YEAR:]
            .dropna(axis=1, thresh=int(TRADING_DAYS_YEAR * 0.8))
        )
        if not prices_subset.empty:
            weights, method = get_optimal_weights(prices_subset)
            allocation = {t: w for t, w in weights.items() if w > 1e-4}
            if conviction_tilt > 0:
                allocation = _blend_with_conviction(
                    allocation, snapshot_scored["proba_upside"], conviction_tilt
                )
            logger.info(f"Allocation du jour ({method}) : {len(allocation)} positions sur {len(eligible)} candidats.")
        else:
            logger.warning("Historique de prix insuffisant pour les tickers sélectionnés — allocation vide.")
    else:
        logger.info(f"Aucun ticker au-dessus du seuil de probabilité ({proba_min:.0%}) aujourd'hui.")

    # Log mensuel pour le dashboard (un seul rebalancement enregistré par mois)
    last_rebalance_date = rebalance_history.index.max() if not rebalance_history.empty else None
    if last_rebalance_date is None or (last_date.year, last_date.month) != (last_rebalance_date.year, last_rebalance_date.month):
        new_row = pd.DataFrame([{"N_Stocks": len(allocation), "Allocation": allocation}], index=[last_date])
        new_row.index.name = "Date"
        rebalance_history = pd.concat([rebalance_history, new_row]).sort_index()

    out = snapshot_scored.reset_index().rename(columns={"ticker": "Ticker"})
    out["ticker_root"] = out["Ticker"].apply(lambda t: t.split(".", 1)[0])
    out["Allocation"] = out["ticker_root"].map(allocation).fillna(0.0)
    out["Signal"] = np.where(out["Allocation"] > 0, "BUY", "NEUTRAL")
    out["Proba_Hausse"] = (out["proba_upside"] * 100).round(1)

    return out[["Ticker", "Signal", "Allocation", "Proba_Hausse"]], rebalance_history


def _build_daily_snapshot(df_daily: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """
    Construit le snapshot de la dernière séance, avec un index simple de tickers.

    Avant : filtrage booléen sur un MultiIndex (ticker, date), qui conserve
    le MultiIndex complet dans le résultat.
    Après : `.xs(level="date")`, identique à la logique déjà utilisée (et
    fonctionnelle) dans le backtest — aligne le comportement live sur le
    comportement backtesté.
    """
    last_date = df_daily.index.get_level_values("date").max()
    df_feat = add_all_features(df_daily.copy())
    snapshot = df_feat.xs(last_date, level="date").copy()
    return snapshot, last_date