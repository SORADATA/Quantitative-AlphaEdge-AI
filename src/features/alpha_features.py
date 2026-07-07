import warnings
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

from const import (
    RSI_WINDOW, BB_WINDOW, BB_STD,
    MIN_HISTORY_TA, MIN_HISTORY_FF,
    FAMA_FRENCH_FACTORS,
)
from src.utils.feature_utils import compute_atr, compute_macd
from src.utils.logger import setup_logger

logger = setup_logger("alpha_features")

# ══════════════════════════════════════════════════════════════════
# UTILITAIRES
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

def add_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    features_to_rank = [
        "mom_12_1", "mom_6_1", "sharpe_6m", "sortino_6m",
        "realized_vol_3m", "realized_vol_12m",
        "amihud_illiquidity", "return_skew_6m", "hist_var_5pct",
    ]
    for feat in features_to_rank:
        rank_col = f"{feat}_rank"
        if feat in df.columns:
            df[rank_col] = df.groupby(level="date")[feat].transform(lambda x: x.rank(pct=True))
        else:
            df[rank_col] = 0.5
    return df.fillna({f"{f}_rank": 0.5 for f in features_to_rank})

# ══════════════════════════════════════════════════════════════════
# 1. CALCULS TECHNIQUES & FACTEURS
# ══════════════════════════════════════════════════════════════════

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing daily technical indicators...")
    if all(col in df.columns for col in ["high", "low", "open", "adj close"]):
        df["garman_klass_vol"] = (
            (np.log(df["high"]) - np.log(df["low"])) ** 2 / 2
            - (2 * np.log(2) - 1) * (np.log(df["adj close"]) - np.log(df["open"])) ** 2
        )
    else:
        df["garman_klass_vol"] = 0.0

    for ticker in df.index.get_level_values(1).unique():
        idx = (slice(None), ticker)
        close = df.loc[idx, "adj close"]
        if len(close) > MIN_HISTORY_TA:
            df.loc[idx, "rsi"] = RSIIndicator(close=close, window=RSI_WINDOW).rsi().values
            bb = BollingerBands(close=np.log1p(close), window=BB_WINDOW, window_dev=BB_STD)
            df.loc[idx, "bb_low"] = bb.bollinger_lband().values
            df.loc[idx, "bb_mid"] = bb.bollinger_mavg().values
            df.loc[idx, "bb_high"] = bb.bollinger_hband().values
            df.loc[idx, "bb_position"] = ((np.log1p(close) - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-9))

    df["atr"] = df.groupby(level=1, group_keys=False).apply(compute_atr)
    df["macd"] = df.groupby(level=1, group_keys=False).apply(compute_macd)
    df["macd_sign"] = np.sign(df["macd"].fillna(0))

    if "volume" in df.columns:
        df["euro_volume"] = (df["adj close"] * df["volume"]) / 1e6
    else:
        df["euro_volume"] = 0.0
    return df

def get_fama_french_betas(data: pd.DataFrame) -> pd.DataFrame:
    logger.info("Retrieving Fama-French factors...")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*date_parser.*")
            factor_data = web.DataReader("Europe_5_Factors", "famafrench", start="2010")[0].drop("RF", axis=1)

        factor_data.index = pd.to_datetime(factor_data.index.to_timestamp()).tz_localize(None)
        factor_data = factor_data.resample("BME").last().div(100)
        factor_data.index.name = "date"

        if "return_1m" not in data.columns:
            return data

        betas_list = []
        for ticker in data.index.get_level_values(1).unique():
            ticker_data = data.xs(ticker, level=1)
            y = ticker_data["return_1m"].dropna()
            if y.empty: continue
            X = factor_data.loc[factor_data.index.intersection(y.index)]
            y = y.loc[X.index]
            if len(y) <= MIN_HISTORY_FF: continue

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*divide by zero.*")
                params = RollingOLS(y, sm.add_constant(X[FAMA_FRENCH_FACTORS]), window=MIN_HISTORY_FF).fit().params.drop("const", axis=1)
            params["ticker"] = ticker
            betas_list.append(params)

        if betas_list:
            betas_df = pd.concat(betas_list).set_index("ticker", append=True)
            data = data.join(betas_df.groupby("ticker").shift())
            data[FAMA_FRENCH_FACTORS] = data.groupby(level="ticker", group_keys=False)[FAMA_FRENCH_FACTORS].transform(lambda x: x.fillna(x.mean()))
            return data
    except Exception as exc:
        logger.warning(f"Fama-French retrieval failed ({exc}).")
    
    return data.assign(**{f: 0.0 for f in FAMA_FRENCH_FACTORS})

# ══════════════════════════════════════════════════════════════════
# 2. ORCHESTRATION DES FEATURES
# ══════════════════════════════════════════════════════════════════

def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("MultiIndex requis (date, ticker).")
    
    df = df.copy()
    
    # 1. Calcul Fama-French avant tout (Solution au KeyError)
    df = get_fama_french_betas(df)
    
    g = df.groupby(level="ticker")
    logger.info("Computing alpha features...")
    
    # 2. Autres calculs
    df = _add_momentum_factors(df, g)
    df = _add_mean_reversion_factors(df, g)
    df = _add_volatility_factors(df, g)
    df = _add_risk_adjusted_factors(df, g)
    df = _add_tail_risk_factors(df, g)
    df = _add_technical_enrichment(df, g)
    df = _add_seasonality_features(df)
    df = add_rank_features(df)

    # 3. Lags
    cols_to_lag = ["rsi", "macd", "bb_low", "bb_mid", "bb_high", "atr", "garman_klass_vol", "bb_position", "macd_sign"] + FAMA_FRENCH_FACTORS
    for col in cols_to_lag:
        if col not in df.columns:
            df[col] = 0.0
        df[f"{col}_lag1"] = df.groupby(level="ticker")[col].shift(1)

    df = df.fillna(0).replace([np.inf, -np.inf], 0)
    logger.info(f"Features prêtes. Shape : {df.shape}")
    return df

