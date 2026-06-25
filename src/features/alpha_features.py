"""
Alpha Features
=======================================
Features inspired :
  - Barra USE4 / MSCI Factor Model
  - AQR Capital : Momentum, Quality, Value
  - Fama-French
  - George & Hwang (2004) : 52-week high
  - Jegadeesh & Titman (1993) : momentum skip-1m
  - Amihud (2002) : illiquidity ratio
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

from const import (
    RSI_WINDOW, BB_WINDOW, BB_STD,
    MIN_HISTORY_TA, MIN_HISTORY_FF,
    MOMENTUM_LAGS, WINSOR_CUTOFF,
    FAMA_FRENCH_FACTORS,
)
from src.utils.feature_utils import compute_atr, compute_macd
from src.utils.logger import setup_logger

logger = setup_logger("alpha_features")


# ══════════════════════════════════════════════════════════════════
# DAILY FEATURES (ex-features.py)
# ══════════════════════════════════════════════════════════════════

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Garman-Klass volatility, RSI, Bollinger Bands, ATR, MACD,
    and euro volume to a multi-index (date, ticker) DataFrame.
    """
    logger.info("Computing technical indicators...")

    df["garman_klass_vol"] = (
        (np.log(df["high"]) - np.log(df["low"])) ** 2 / 2
        - (2 * np.log(2) - 1)
        * (np.log(df["adj close"]) - np.log(df["open"])) ** 2
    )

    for ticker in df.index.get_level_values(1).unique():
        idx   = (slice(None), ticker)
        close = df.loc[idx, "adj close"]

        if len(close) > MIN_HISTORY_TA:
            df.loc[idx, "rsi"] = RSIIndicator(
                close=close, window=RSI_WINDOW
            ).rsi().values

            bb = BollingerBands(
                close=np.log1p(close), window=BB_WINDOW, window_dev=BB_STD
            )
            df.loc[idx, "bb_low"]  = bb.bollinger_lband().values
            df.loc[idx, "bb_mid"]  = bb.bollinger_mavg().values
            df.loc[idx, "bb_high"] = bb.bollinger_hband().values

    df["atr"]        = df.groupby(level=1, group_keys=False).apply(compute_atr)
    df["macd"]       = df.groupby(level=1, group_keys=False).apply(compute_macd)
    df["euro_volume"] = (df["adj close"] * df["volume"]) / 1e6

    return df


def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds winsorized momentum return columns for each lag in MOMENTUM_LAGS.
    Operates on a single-ticker slice (called via groupby).
    """
    for lag in MOMENTUM_LAGS:
        raw   = df["adj close"].pct_change(lag)
        lower = raw.expanding(min_periods=12).quantile(WINSOR_CUTOFF)
        upper = raw.expanding(min_periods=12).quantile(1 - WINSOR_CUTOFF)
        df[f"return_{lag}m"] = raw.clip(lower=lower, upper=upper)
    return df


def get_fama_french_betas(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fetches Europe 5-Factor data from Kenneth French's library and computes
    rolling 24-month OLS betas for each ticker. Fills with zeros on failure.
    """
    logger.info("Retrieving Fama-French factors (Europe 5)...")

    try:
        import pandas_datareader.data as web
    except ImportError as exc:
        logger.error(f"pandas_datareader unavailable ({exc}). Filling betas with zeros.")
        return data.assign(**{f: 0.0 for f in FAMA_FRENCH_FACTORS})

    try:
        factor_data = (
            web.DataReader("Europe_5_Factors", "famafrench", start="2010")[0]
            .drop("RF", axis=1)
        )
        factor_data.index = pd.to_datetime(
            factor_data.index.to_timestamp()
        ).tz_localize(None)
        factor_data = factor_data.resample("BME").last().div(100)
        factor_data.index.name = "date"

        if "return_1m" not in data.columns:
            return data

        betas_list = []
        for ticker in data.index.get_level_values(1).unique():
            ticker_data = data.xs(ticker, level=1)
            if "return_1m" not in ticker_data.columns or ticker_data["return_1m"].dropna().empty:
                continue

            y = ticker_data["return_1m"].dropna()
            X = factor_data.loc[factor_data.index.intersection(y.index)]
            y = y.loc[X.index]

            if len(y) <= MIN_HISTORY_FF:
                continue

            params = (
                RollingOLS(y, sm.add_constant(X[FAMA_FRENCH_FACTORS]), window=MIN_HISTORY_FF)
                .fit()
                .params.drop("const", axis=1)
            )
            params["ticker"] = ticker
            betas_list.append(params)

        if not betas_list:
            return data

        betas_df = pd.concat(betas_list).set_index("ticker", append=True)
        data = data.join(betas_df.groupby("ticker").shift())
        data[FAMA_FRENCH_FACTORS] = (
            data.groupby(level="ticker", group_keys=False)[FAMA_FRENCH_FACTORS]
            .transform(lambda x: x.fillna(x.mean()))
        )
        return data

    except Exception as exc:
        logger.error(f"Fama-French retrieval failed ({exc}). Filling with zeros.")
        return data.assign(**{f: 0.0 for f in FAMA_FRENCH_FACTORS})


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a.div(b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _rolling_sortino(returns: pd.Series, window: int = 6) -> pd.Series:
    def _sortino_scalar(r: np.ndarray) -> float:
        neg = r[r < 0]
        if len(neg) == 0 or np.std(neg) == 0:
            return np.nan
        return (np.mean(r) / np.std(neg)) * np.sqrt(12)
    return returns.rolling(window, min_periods=window // 2).apply(_sortino_scalar, raw=True)


def _rolling_maxdrawdown(returns: pd.Series, window: int = 12) -> pd.Series:
    def _mdd(r: np.ndarray) -> float:
        cumulative = np.cumprod(1 + r)
        peak = np.maximum.accumulate(cumulative)
        return ((cumulative - peak) / peak).min()
    return returns.rolling(window, min_periods=window // 2).apply(_mdd, raw=True)


# ══════════════════════════════════════════════════════════════════
# MONTHLY ALPHA FEATURES
# ══════════════════════════════════════════════════════════════════

def _add_momentum_factors(df: pd.DataFrame, g) -> pd.DataFrame:
    logger.debug("Computing momentum factors...")
    df["return_1m"] = g["adj close"].transform(lambda x: x.pct_change(1))
    pct_12 = g["adj close"].transform(lambda x: x.pct_change(12))
    pct_1  = g["adj close"].transform(lambda x: x.pct_change(1))
    pct_6  = g["adj close"].transform(lambda x: x.pct_change(6))
    df["mom_12_1"] = _safe_div(pct_12, pct_1.replace(0, np.nan))
    df["mom_6_1"]  = _safe_div(pct_6,  pct_1.replace(0, np.nan))
    df["mom_3_1"]  = g["adj close"].transform(lambda x: x.pct_change(3))
    return df


def _add_mean_reversion_factors(df: pd.DataFrame, g) -> pd.DataFrame:
    logger.debug("Computing mean reversion factors...")
    ma12  = g["adj close"].transform(lambda x: x.rolling(12, min_periods=6).mean())
    std12 = g["adj close"].transform(lambda x: x.rolling(12, min_periods=6).std())
    df["price_zscore_12"]   = _safe_div(df["adj close"] - ma12, std12)
    high52 = g["adj close"].transform(lambda x: x.rolling(12, min_periods=6).max())
    df["nearness_52w_high"] = _safe_div(df["adj close"], high52)
    return df


def _add_volatility_factors(df: pd.DataFrame, g) -> pd.DataFrame:
    logger.debug("Computing volatility factors...")
    df["realized_vol_3m"]  = g["return_1m"].transform(lambda x: x.rolling(3,  min_periods=2).std() * np.sqrt(12))
    df["realized_vol_12m"] = g["return_1m"].transform(lambda x: x.rolling(12, min_periods=6).std() * np.sqrt(12))
    df["vol_ratio"]        = _safe_div(df["realized_vol_3m"], df["realized_vol_12m"])
    if "Mkt-RF" in df.columns:
        excess = df["return_1m"] - df["Mkt-RF"].fillna(0)
        df["idio_vol"] = excess.groupby(level=1).transform(
            lambda x: x.rolling(6, min_periods=3).std() * np.sqrt(12)
        )
    else:
        df["idio_vol"] = np.nan
    return df


def _add_risk_adjusted_factors(df: pd.DataFrame, g) -> pd.DataFrame:
    logger.debug("Computing risk-adjusted factors...")
    df["sharpe_3m"] = _safe_div(
        g["return_1m"].transform(lambda x: x.rolling(3, min_periods=2).mean()),
        g["return_1m"].transform(lambda x: x.rolling(3, min_periods=2).std()),
    ) * np.sqrt(12)
    df["sharpe_6m"] = _safe_div(
        g["return_1m"].transform(lambda x: x.rolling(6, min_periods=3).mean()),
        g["return_1m"].transform(lambda x: x.rolling(6, min_periods=3).std()),
    ) * np.sqrt(12)
    df["sortino_6m"]   = g["return_1m"].transform(lambda x: _rolling_sortino(x, window=6))
    df["calmar_proxy"] = _safe_div(
        g["return_1m"].transform(lambda x: x.rolling(12, min_periods=6).mean() * 12),
        g["return_1m"].transform(lambda x: _rolling_maxdrawdown(x, window=12)).abs(),
    )
    return df


def _add_tail_risk_factors(df: pd.DataFrame, g) -> pd.DataFrame:
    logger.debug("Computing tail risk factors...")
    df["return_skew_6m"] = g["return_1m"].transform(lambda x: x.rolling(6,  min_periods=3).skew())
    df["return_kurt_6m"] = g["return_1m"].transform(lambda x: x.rolling(6,  min_periods=3).kurt())
    df["hist_var_5pct"]  = g["return_1m"].transform(lambda x: x.rolling(12, min_periods=6).quantile(0.05))

    def _cvar(r: np.ndarray) -> float:
        t = np.quantile(r, 0.05)
        tail = r[r <= t]
        return tail.mean() if len(tail) > 0 else np.nan

    df["cvar_5pct"] = g["return_1m"].transform(
        lambda x: x.rolling(12, min_periods=6).apply(_cvar, raw=True)
    )
    return df


def _add_liquidity_factors(df: pd.DataFrame, g) -> pd.DataFrame:
    logger.debug("Computing liquidity factors...")
    if "euro_volume" not in df.columns:
        logger.warning("euro_volume absent — liquidity factors skipped.")
        return df
    df["amihud_illiquidity"] = _safe_div(df["return_1m"].abs(), df["euro_volume"])
    df["volume_trend_3m"]    = g["euro_volume"].transform(lambda x: x.pct_change(3))
    df["volume_zscore"]      = g["euro_volume"].transform(
        lambda x: _safe_div(
            x - x.rolling(12, min_periods=6).mean(),
            x.rolling(12, min_periods=6).std(),
        )
    )
    return df


def _add_technical_enrichment(df: pd.DataFrame, g) -> pd.DataFrame:
    logger.debug("Computing technical enrichment...")
    if "rsi" in df.columns:
        df["rsi_divergence"] = (
            g["adj close"].transform(lambda x: x.pct_change(3)) -
            g["rsi"].transform(lambda x: x.pct_change(3))
        )
    if "bb_low" in df.columns and "bb_high" in df.columns:
        df["bb_position"] = _safe_div(
            df["adj close"] - df["bb_low"],
            (df["bb_high"] - df["bb_low"]).replace(0, np.nan),
        ).clip(0, 1)
    if "macd" in df.columns:
        df["macd_sign"] = np.sign(df["macd"].fillna(0)).astype(int)
    return df


def _add_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Computing cross-sectional rank features...")
    features_to_rank = [
        "mom_12_1", "mom_6_1", "sharpe_6m", "sortino_6m",
        "realized_vol_3m", "realized_vol_12m",
        "amihud_illiquidity", "return_skew_6m", "hist_var_5pct",
    ]
    for date, group in df.groupby(level="date"):
        if len(group) < 5:
            continue
        for feat in features_to_rank:
            if feat in df.columns:
                df.loc[group.index, f"{feat}_rank"] = group[feat].rank(pct=True, na_option="keep")
    return df


def _add_seasonality_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Computing seasonality features...")
    dates = df.index.get_level_values("date")
    df["month_sin"] = np.sin(2 * np.pi * dates.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * dates.month / 12)
    df["is_q_end"]  = dates.month.isin([3, 6, 9, 12]).astype(int)
    df["is_jan"]    = (dates.month == 1).astype(int)
    return df


def _lag_ta_indicators(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Lagging TA indicators (anti-leakage)...")
    ta_cols = ["rsi", "macd", "bb_low", "bb_mid", "bb_high", "atr",
               "cluster", "bb_position", "rsi_divergence", "macd_sign"]
    for col in ta_cols:
        if col in df.columns:
            df[f"{col}_lag1"] = df.groupby(level="ticker")[col].shift(1)
    return df


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("df doit avoir un MultiIndex (date, ticker).")
    if "adj close" not in df.columns:
        raise ValueError("Colonne 'adj close' manquante.")

    df = df.copy()
    g  = df.groupby(level="ticker")

    logger.info("Computing institutional alpha features...")

    df = _add_momentum_factors(df, g)
    df = _add_mean_reversion_factors(df, g)
    df = _add_volatility_factors(df, g)
    df = _add_risk_adjusted_factors(df, g)
    df = _add_tail_risk_factors(df, g)
    df = _add_liquidity_factors(df, g)
    df = _add_technical_enrichment(df, g)
    df = _add_cross_sectional_features(df)
    df = _add_seasonality_features(df)
    df = _lag_ta_indicators(df)

    logger.info(f"✅ Alpha features computed. Total columns : {df.shape[1]}")
    return df