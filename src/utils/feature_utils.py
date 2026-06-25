"""
Feature Utils
==============
Fonctions utilitaires pour le calcul des indicateurs techniques.
Appelées depuis features.py via groupby().apply().

Chaque fonction reçoit un slice single-ticker et retourne une Series
alignée sur l'index du slice.

Notes
-----
- ATR et MACD sont z-scorés pour homogénéiser l'échelle entre tickers.
  XGBoost/LGB sont invariants à cette transformation, mais elle aide
  Ridge et LR (utilisés dans l'ensemble).
- Garman-Klass est centralisé ici pour faciliter les tests unitaires.
"""

import numpy as np
import pandas as pd
from ta.volatility import AverageTrueRange
from ta.trend import MACD as MACDIndicator

from const import ATR_WINDOW, MACD_SLOW, MACD_FAST, MACD_SIGN, MIN_HISTORY_TA


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def _safe_normalize(series: pd.Series) -> pd.Series:
    """
    Z-score normalization.
    Retourne une série de NaN si std == 0 (évite la division par zéro).
    """
    std = series.std()
    if std == 0 or np.isnan(std):
        return pd.Series(np.nan, index=series.index)
    return series.sub(series.mean()).div(std)


def _get_close(stock_data: pd.DataFrame) -> pd.Series:
    """
    Retourne adj close si disponible, sinon close.
    Garantit la cohérence entre les indicateurs du pipeline.
    """
    if "adj close" in stock_data.columns:
        return stock_data["adj close"]
    if "close" in stock_data.columns:
        return stock_data["close"]
    raise KeyError("Ni 'adj close' ni 'close' trouvés dans le DataFrame.")


# ══════════════════════════════════════════════════════════════════
# INDICATEURS TECHNIQUES
# ══════════════════════════════════════════════════════════════════

def compute_atr(stock_data: pd.DataFrame) -> pd.Series:
    """
    Average True Range normalisé (z-score) pour un single ticker.

    Utilise adj close pour la cohérence avec le reste du pipeline.
    Nécessite les colonnes : high, low, adj close (ou close).

    Parameters
    ----------
    stock_data : pd.DataFrame — slice single ticker

    Returns
    -------
    pd.Series — ATR z-scoré, aligné sur stock_data.index
    """
    required = {"high", "low"}
    if not required.issubset(stock_data.columns):
        return pd.Series(np.nan, index=stock_data.index)

    if len(stock_data) < ATR_WINDOW + 1:
        return pd.Series(np.nan, index=stock_data.index)

    close = _get_close(stock_data)

    atr = AverageTrueRange(
        high=stock_data["high"],
        low=stock_data["low"],
        close=close,
        window=ATR_WINDOW,
    ).average_true_range()

    return _safe_normalize(atr)


def compute_macd(stock_data: pd.DataFrame) -> pd.Series:
    """
    Ligne MACD normalisée (z-score) pour un single ticker.

    Utilise adj close.
    Nécessite la colonne : adj close (ou close).

    Parameters
    ----------
    stock_data : pd.DataFrame — slice single ticker

    Returns
    -------
    pd.Series — MACD z-scoré, aligné sur stock_data.index
    """
    if len(stock_data) < MACD_SLOW + MIN_HISTORY_TA:
        return pd.Series(np.nan, index=stock_data.index)

    close = _get_close(stock_data)

    macd_val = MACDIndicator(
        close=close,
        window_slow=MACD_SLOW,
        window_fast=MACD_FAST,
        window_sign=MACD_SIGN,
    ).macd()

    return _safe_normalize(macd_val)


def compute_garman_klass_vol(stock_data: pd.DataFrame) -> pd.Series:
    """
    Volatilité Garman-Klass pour un single ticker.

    Estimateur de volatilité plus efficace que la volatilité close-to-close.
    Formule : GK = 0.5 * ln(H/L)² - (2*ln(2)-1) * ln(C/O)²

    Nécessite les colonnes : high, low, adj close (ou close), open.

    Parameters
    ----------
    stock_data : pd.DataFrame — slice single ticker

    Returns
    -------
    pd.Series — volatilité GK, alignée sur stock_data.index
    """
    required = {"high", "low", "open"}
    if not required.issubset(stock_data.columns):
        return pd.Series(np.nan, index=stock_data.index)

    close = _get_close(stock_data)

    log_hl = np.log(stock_data["high"]) - np.log(stock_data["low"])
    log_co = np.log(close) - np.log(stock_data["open"])

    gk = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2

    return gk


def compute_macd_histogram(stock_data: pd.DataFrame) -> pd.Series:
    """
    Histogramme MACD normalisé (MACD line - Signal line).

    Plus réactif que la ligne MACD seule pour détecter les retournements.

    Parameters
    ----------
    stock_data : pd.DataFrame — slice single ticker

    Returns
    -------
    pd.Series — histogramme MACD z-scoré
    """
    if len(stock_data) < MACD_SLOW + MIN_HISTORY_TA:
        return pd.Series(np.nan, index=stock_data.index)

    close = _get_close(stock_data)

    indicator = MACDIndicator(
        close=close,
        window_slow=MACD_SLOW,
        window_fast=MACD_FAST,
        window_sign=MACD_SIGN,
    )

    histogram = indicator.macd_diff()

    return _safe_normalize(histogram)
