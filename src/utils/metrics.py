"""
Metrics Utils
=============
Métriques financières — évaluation modèle (walk-forward) et suivi
de performance du portefeuille en production (dashboard).

Fonctions :
  - calculate_financial_metrics() : métriques d'évaluation ML (sharpe/dd/return sur probas)
  - calculate_metrics()           : métriques portefeuille prod (Strategy vs Benchmark)
  - calculate_period_return()     : rendement sur une fenêtre donnée (YTD, 1M, etc.)
  - trim_flat_start()             : nettoie le plateau initial d'un historique
"""

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════
# ÉVALUATION MODÈLE (walk-forward / backtest)
# ══════════════════════════════════════════════════════════════════

def calculate_financial_metrics(df_test: pd.DataFrame, probas: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Calcule sharpe / max_drawdown / total_return à partir des probabilités
    prédites par le modèle sur un jeu de test (index multi = ticker/date).
    """
    signals = (probas > threshold).astype(int)
    strategy_returns = signals * df_test["future_return"]
    portfolio_returns = strategy_returns.groupby(level="date").mean()

    if portfolio_returns.std() == 0:
        return {"sharpe": 0.0, "sortino": 0.0,  "max_drawdown": 0.0, "total_return": 0.0}

    annualization_factor = np.sqrt(12)
    mean_ret = portfolio_returns.mean()
    std_ret = portfolio_returns.std()

    sharpe_ratio = (mean_ret / std_ret) * annualization_factor

    negative_returns = portfolio_returns[portfolio_returns < 0]
    downside_std = negative_returns.std() if len(negative_returns) > 0 else 0.0
    # sortino = semi-ecart-type sur les rendements négatifs only
    sortino_ratio = (mean_ret / downside_std) * annualization_factor if downside_std != 0 else 0.0

    cumulative_returns = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    total_return = cumulative_returns.iloc[-1] - 1 if not cumulative_returns.empty else 0.0

    return {
        "sharpe": round(sharpe_ratio, 4),
        "sortino": round(sortino_ratio, 4),
        "max_drawdown": round(max_drawdown, 4),
        "total_return": round(total_return, 4),
    }


# ══════════════════════════════════════════════════════════════════
# SUIVI PORTEFEUILLE PROD (dashboard)
# ══════════════════════════════════════════════════════════════════

def calculate_metrics(df: pd.DataFrame) -> tuple:
    """
    Calcule les KPI principaux du dashboard à partir de l'historique
    Strategy/Benchmark : total_return, alpha, sharpe, sortino, max_dd, calmar, recovery_time.

    Returns
    -------
    tuple — (total_return, alpha, sharpe, sortino, max_dd, calmar, recovery_days)
    """
    if df.empty or len(df) < 2:
        return 0, 0, 0, 0, 0, 0, 0
    try:
        total_ret = (df["Strategy"].iloc[-1] / df["Strategy"].iloc[0]) - 1
        bench_ret = (df["Benchmark"].iloc[-1] / df["Benchmark"].iloc[0]) - 1
        alpha = total_ret - bench_ret

        strategy_returns = df["Strategy"].pct_change().dropna()
        if strategy_returns.empty:
            return total_ret, alpha, 0, 0, 0, 0, 0

        mean_ret = strategy_returns.mean()
        std_ret = strategy_returns.std()

        sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret != 0 else 0

        # Calcul du Sortino de production (downside risk)
        neg_rets = strategy_returns[strategy_returns < 0]
        downside_std = neg_rets.std() if len(neg_rets) > 0 else 0
        sortino = (mean_ret / downside_std) * np.sqrt(252) if downside_std != 0 else 0

        cum_ret = (1 + strategy_returns).cumprod()
        running_max = cum_ret.cummax()
        dd_series = (cum_ret - running_max) / running_max
        max_dd = dd_series.min()

        # Calcul du Ratio de Calmar (Rendement annualisé / Max Drawdown)
        # Approximation du rendement annualisé basée sur la durée totale
        n_years = len(strategy_returns) / 252
        ann_return = ((1 + total_ret) ** (1 / n_years) - 1) if n_years > 0 else 0
        calmar = (ann_return / abs(max_dd)) if max_dd != 0 else 0

        recovery_time = _compute_recovery_time(dd_series)

        return total_ret, alpha, sharpe, sortino, max_dd, calmar, recovery_time
    except Exception:
        return 0, 0, 0, 0, 0, 0, 0


def _compute_recovery_time(dd_series: pd.Series) -> int:
    """
    Nombre de jours écoulés entre le point bas du dernier drawdown
    significatif et le retour au plus haut (0). Si pas encore récupéré,
    retourne le nombre de jours depuis le point bas jusqu'à aujourd'hui.
    """
    if dd_series.empty:
        return 0
    trough_idx = dd_series.idxmin()
    post_trough = dd_series.loc[trough_idx:]
    recovered = post_trough[post_trough >= -0.0001]
    if len(recovered) > 1:
        recovery_date = recovered.index[1]
        return (recovery_date - trough_idx).days
    return (dd_series.index[-1] - trough_idx).days


def calculate_period_return(df: pd.DataFrame, days: int = None, ytd: bool = False, daily: bool = False) -> float:
    """
    Rendement de la stratégie sur une fenêtre donnée (YTD, N derniers jours,
    variation journalière, ou depuis le début si aucun paramètre n'est fourni).
    """
    if df.empty or "Strategy" not in df.columns or len(df) < 2:
        return 0.0
    try:
        if daily:
            return (df["Strategy"].iloc[-1] / df["Strategy"].iloc[-2]) - 1

        last_price, last_date = df["Strategy"].iloc[-1], df.index[-1]

        if ytd:
            target_date = pd.Timestamp(last_date.year, 1, 1)
        elif days:
            target_date = last_date - pd.Timedelta(days=days)
        else:
            target_date = df.index[0]

        if target_date < df.index[0]:
            start_price = df["Strategy"].iloc[0]
        else:
            start_price = df["Strategy"].iloc[df.index.get_indexer([target_date], method="nearest")[0]]

        return ((last_price / start_price) - 1) if start_price != 0 else 0.0
    except Exception:
        return 0.0
