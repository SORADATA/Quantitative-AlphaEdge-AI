import numpy as np
import pandas as pd


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a.div(b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _rolling_sortino(returns: pd.Series, window: int = 6) -> pd.Series:
    def _sortino_scalar(r: np.ndarray) -> float:
        neg = r[r < 0]
        if len(neg) == 0 or np.std(neg) == 0: return np.nan
        return (np.mean(r) / np.std(neg)) * np.sqrt(12)
    return returns.rolling(window, min_periods=window // 2).apply(_sortino_scalar, raw=True)


def _rolling_maxdrawdown(returns: pd.Series, window: int = 12) -> pd.Series:
    def _mdd(r: np.ndarray) -> float:
        cumulative = np.cumprod(1 + r)
        peak = np.maximum.accumulate(cumulative)
        return ((cumulative - peak) / peak).min()
    return returns.rolling(window, min_periods=window // 2).apply(_mdd, raw=True)


def trim_flat_start(df: pd.DataFrame, tol: float = 1e-6) -> pd.DataFrame:
    """
    Supprime la période initiale "plate" (placeholder constant) présente
    au début de certains historiques, avant le premier rebalancement réel.
    Renvoie l'historique complet si aucun plateau n'est détecté.
    """
    if df.empty or "Strategy" not in df.columns or len(df) < 3:
        return df
    changes = df["Strategy"].diff().abs() > tol
    if "Benchmark" in df.columns:
        changes = changes | (df["Benchmark"].diff().abs() > tol)
    first_move = changes[changes].index
    if len(first_move) == 0:
        return df
    start_idx = df.index.get_loc(first_move[0])
    start_idx = max(0, start_idx - 1)
    return df.iloc[start_idx:]