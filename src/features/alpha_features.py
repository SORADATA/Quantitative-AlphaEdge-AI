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
from src.utils.math_utils import _safe_div, _rolling_sortino, _rolling_maxdrawdown
from src.utils.logger import setup_logger


logger = setup_logger("alpha_features")


# ══════════════════════════════════════════════════════════════════
# CALCULS DES FEATURES
# ══════════════════════════════════════════════════════════════════


def _add_momentum_factors(df: pd.DataFrame, g) -> pd.DataFrame:
    df["return_1m"] = g["adj close"].transform(lambda x: x.pct_change(1))
    for lag in [2, 3, 6, 9, 12]:
        df[f"return_{lag}m"] = g["adj close"].transform(lambda x: x.pct_change(lag))
    pct_12 = g["adj close"].transform(lambda x: x.pct_change(12))
    pct_1 = g["adj close"].transform(lambda x: x.pct_change(1))
    pct_6 = g["adj close"].transform(lambda x: x.pct_change(6))
    df["mom_12_1"] = pct_12.div(pct_1.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    df["mom_6_1"] = pct_6.div(pct_1.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    df["mom_3_1"] = g["adj close"].transform(lambda x: x.pct_change(3))
    return df


def _add_mean_reversion_factors(df: pd.DataFrame, g) -> pd.DataFrame:
    ma12 = g["adj close"].transform(lambda x: x.rolling(12, min_periods=6).mean())
    std12 = g["adj close"].transform(lambda x: x.rolling(12, min_periods=6).std())
    df["price_zscore_12"] = _safe_div(df["adj close"] - ma12, std12)
    high52 = g["adj close"].transform(lambda x: x.rolling(12, min_periods=6).max())
    df["nearness_52w_high"] = _safe_div(df["adj close"], high52)
    return df


def _add_volatility_factors(df: pd.DataFrame, g) -> pd.DataFrame:
    df["realized_vol_3m"] = g["return_1m"].transform(lambda x: x.rolling(3, min_periods=2).std() * np.sqrt(12))
    df["realized_vol_12m"] = g["return_1m"].transform(lambda x: x.rolling(12, min_periods=6).std() * np.sqrt(12))
    df["vol_ratio"] = df["realized_vol_3m"].div(df["realized_vol_12m"]).replace([np.inf, -np.inf], np.nan)
    excess = df["return_1m"] - df["Mkt-RF"].fillna(0) if "Mkt-RF" in df.columns else df["return_1m"]
    df["idio_vol"] = excess.groupby(level=1).transform(lambda x: x.rolling(6, min_periods=3).std() * np.sqrt(12))
    return df


def _add_risk_adjusted_factors(df: pd.DataFrame, g) -> pd.DataFrame:
    df["sharpe_3m"] = _safe_div(g["return_1m"].transform(lambda x: x.rolling(3, min_periods=2).mean()), g["return_1m"].transform(lambda x: x.rolling(3, min_periods=2).std())) * np.sqrt(12)
    df["sharpe_6m"] = _safe_div(g["return_1m"].transform(lambda x: x.rolling(6, min_periods=3).mean()), g["return_1m"].transform(lambda x: x.rolling(6, min_periods=3).std())) * np.sqrt(12)
    df["sortino_6m"] = g["return_1m"].transform(lambda x: _rolling_sortino(x, window=6))
    df["calmar_proxy"] = _safe_div(g["return_1m"].transform(lambda x: x.rolling(12, min_periods=6).mean() * 12), g["return_1m"].transform(lambda x: _rolling_maxdrawdown(x, window=12)).abs())
    return df


def _add_tail_risk_factors(df: pd.DataFrame, g) -> pd.DataFrame:
    df["return_skew_6m"] = g["return_1m"].transform(lambda x: x.rolling(6, min_periods=3).skew())
    df["return_kurt_6m"] = g["return_1m"].transform(lambda x: x.rolling(6, min_periods=3).kurt())
    df["hist_var_5pct"] = g["return_1m"].transform(lambda x: x.rolling(12, min_periods=6).quantile(0.05))

    def _cvar(r: np.ndarray) -> float:
        t = np.quantile(r, 0.05); tail = r[r <= t]
        return tail.mean() if len(tail) > 0 else np.nan
    df["cvar_5pct"] = g["return_1m"].transform(lambda x: x.rolling(12, min_periods=6).apply(_cvar, raw=True))
    return df


def _add_technical_enrichment(df: pd.DataFrame, g) -> pd.DataFrame:
    if "rsi" in df.columns:
        df["rsi_divergence"] = g["adj close"].transform(lambda x: x.pct_change(3)) - g["rsi"].transform(lambda x: x.pct_change(3))

    if "euro_volume" in df.columns:
        df["amihud_illiquidity"] = _safe_div(df["return_1m"].abs(), df["euro_volume"])
        df["volume_trend_3m"] = g["euro_volume"].transform(lambda x: x.pct_change(3))
        df["volume_zscore"] = g["euro_volume"].transform(lambda x: _safe_div(x - x.rolling(12, min_periods=6).mean(), x.rolling(12, min_periods=6).std()))
    return df


def _add_seasonality_features(df: pd.DataFrame) -> pd.DataFrame:
    dates = df.index.get_level_values("date")
    df["month_sin"] = np.sin(2 * np.pi * dates.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * dates.month / 12)
    df["is_q_end"] = dates.month.isin([3, 6, 9, 12]).astype(int)
    df["is_jan"] = (dates.month == 1).astype(int)
    return df


def add_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    features_to_rank = ["mom_12_1", "mom_6_1", "sharpe_6m", "sortino_6m", "realized_vol_3m", "realized_vol_12m", "amihud_illiquidity", "return_skew_6m", "hist_var_5pct"]
    for feat in features_to_rank:
        if feat in df.columns: df[f"{feat}_rank"] = df.groupby(level="date")[feat].transform(lambda x: x.rank(pct=True))
        else: df[f"{feat}_rank"] = 0.5
    return df


# ══════════════════════════════════════════════════════════════════
#  FONCTIONS PRINCIPALES
# ══════════════════════════════════════════════════════════════════


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing daily technical indicators...")
    if all(col in df.columns for col in ["high", "low", "open", "adj close"]):
        df["garman_klass_vol"] = ((np.log(df["high"]) - np.log(df["low"])) ** 2 / 2 - (2 * np.log(2) - 1) * (np.log(df["adj close"]) - np.log(df["open"])) ** 2)
    else:
        df["garman_klass_vol"] = 0.0
    for ticker in df.index.get_level_values(1).unique():
        idx = (slice(None), ticker)
        close = df.loc[idx, "adj close"]
        if len(close) > MIN_HISTORY_TA:
            df.loc[idx, "rsi"] = RSIIndicator(close=close, window=RSI_WINDOW).rsi().values
            bb = BollingerBands(close=np.log1p(close), window=BB_WINDOW, window_dev=BB_STD)
            df.loc[idx, "bb_low"], df.loc[idx, "bb_mid"], df.loc[idx, "bb_high"] = bb.bollinger_lband().values, bb.bollinger_mavg().values, bb.bollinger_hband().values
            df.loc[idx, "bb_position"] = ((np.log1p(close) - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-9))
    df["atr"] = df.groupby(level=1, group_keys=False).apply(compute_atr)
    df["macd"] = df.groupby(level=1, group_keys=False).apply(compute_macd)
    df["macd_sign"] = np.sign(df["macd"].fillna(0))
    df["euro_volume"] = (df["adj close"] * df["volume"]) / 1e6 if "volume" in df.columns else 0.0
    return df


# INJECTION DYNAMIQUE DE LA RÉGION FAMA-FRENCH
def get_fama_french_betas(data: pd.DataFrame, ff_region: str) -> pd.DataFrame:
    logger.info(f"Retrieving Fama-French factors for region: {ff_region}...")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*date_parser.*")
            factor_data = web.DataReader(ff_region, "famafrench", start="2010")[0].drop("RF", axis=1)
        factor_data.index = pd.to_datetime(factor_data.index.to_timestamp()).tz_localize(None)
        factor_data = factor_data.resample("BME").last().div(100)
        factor_data.index.name = "date"
        if "return_1m" not in data.columns: return data
        betas_list = []
        for ticker in data.index.get_level_values(1).unique():
            ticker_data = data.xs(ticker, level=1)
            y = ticker_data["return_1m"].dropna()
            if y.empty: continue
            X = factor_data.loc[factor_data.index.intersection(y.index)]
            y = y.loc[X.index]
            if len(y) <= MIN_HISTORY_FF: continue
            params = RollingOLS(y, sm.add_constant(X[FAMA_FRENCH_FACTORS]), window=MIN_HISTORY_FF).fit().params.drop("const", axis=1)
            params["ticker"] = ticker
            betas_list.append(params)
        if betas_list:
            betas_df = pd.concat(betas_list).set_index("ticker", append=True)
            data = data.join(betas_df.groupby("ticker").shift())
            data[FAMA_FRENCH_FACTORS] = data.groupby(
                level="ticker", group_keys=False)[FAMA_FRENCH_FACTORS].transform(lambda x: x.fillna(x.mean()))
            return data
    except Exception as exc:
        logger.warning(f"Fama-French retrieval failed ({exc}).")
    return data.assign(**{f: 0.0 for f in FAMA_FRENCH_FACTORS})


def add_all_features(df: pd.DataFrame, market_config: dict) -> pd.DataFrame:
    if not isinstance(df.index, pd.MultiIndex): raise ValueError("MultiIndex requis.")
    if "ff_region" not in market_config:
        raise ValueError("ff_region manquant dans market_config — vérifiez le fichier de config du marché.")
    ff_region = market_config["ff_region"]
    df = get_fama_french_betas(df.copy(), ff_region)
    g = df.groupby(level="ticker")
    logger.info("Computing alpha features...")
    df = _add_momentum_factors(df, g)
    df = _add_mean_reversion_factors(df, g)
    df = _add_volatility_factors(df, g)
    df = _add_risk_adjusted_factors(df, g)
    df = _add_tail_risk_factors(df, g)
    df = _add_technical_enrichment(df, g)
    df = _add_seasonality_features(df)
    df = add_rank_features(df)
    cols_to_lag = ["rsi", "macd", "bb_low", "bb_mid", "bb_high", "atr", "garman_klass_vol", "bb_position", "macd_sign"] + FAMA_FRENCH_FACTORS
    for col in cols_to_lag:
        if col not in df.columns: df[col] = 0.0
        df[f"{col}_lag1"] = df.groupby(level="ticker")[col].shift(1)
    return df.fillna(0).replace([np.inf, -np.inf], 0)