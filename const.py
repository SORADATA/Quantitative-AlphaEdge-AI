# const.py

from pathlib import Path
from dotenv import load_dotenv
import os

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "src" / "models"
LOG_DIR = BASE_DIR / "logs"
CONFIG_DIR = BASE_DIR / "config"

load_dotenv(BASE_DIR / ".env")
HF_TOKEN = os.getenv("HF_TOKEN")

TRADING_DAYS_YEAR: int = 252
RISK_FREE_RATE:    float = 0.03

TARGET_CLUSTER:  int = 3
PROBA_THRESHOLD: float = 0.51
PROBA_MIN:       float = 0.50

FEATURE_COLS: list[str] = [
    "rsi_lag1", "macd_lag1", "bb_low_lag1", "bb_mid_lag1",
    "bb_high_lag1", "atr_lag1", "cluster_lag1",
    "return_1m", "return_2m", "return_3m", "return_6m", "return_9m", "return_12m",
    "mom_12_1", "mom_6_1",
    "realized_vol_3m", "realized_vol_12m", "vol_ratio",
    "sharpe_3m", "sharpe_6m", "sortino_6m",
    "return_skew_6m", "hist_var_5pct", "cvar_5pct",
    "amihud_illiquidity", "volume_zscore",
    "price_zscore_12", "nearness_52w_high",
    "mom_12_1_rank", "sharpe_6m_rank", "realized_vol_3m_rank", "amihud_illiquidity_rank",
    "Mkt-RF_lag1", "SMB_lag1", "HML_lag1", "RMW_lag1", "CMA_lag1",
    "euro_volume_lag1", "garman_klass_vol_lag1",
    "month_sin", "month_cos", "is_q_end",
]

RSI_WINDOW: int = 20
BB_WINDOW:  int = 20
BB_STD:     int = 2
ATR_WINDOW: int = 14
MACD_SLOW:  int = 26
MACD_FAST:  int = 12
MACD_SIGN:  int = 9

MIN_HISTORY_TA: int = 20
MIN_HISTORY_FF: int = 24
WINSOR_CUTOFF:  float = 0.005

MOMENTUM_LAGS: list[int] = [1, 2, 3, 6, 9, 12]

VARS_TO_LAG: list[str] = [
    "Mkt-RF", "SMB", "HML", "RMW", "CMA",
    "euro_volume",
    "garman_klass_vol",
]

FAMA_FRENCH_FACTORS: list[str] = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

RESAMPLE_MEAN_COLS: list[str] = ["euro_volume"]

RESAMPLE_LAST_EXCLUDE: list[str] = [
    "euro_volume", "volume", "open", "high", "low", "close",
]

TRANSACTION_COST:  float = 0.0010
MIN_STOCKS_OPTIM:  int = 3
MAX_STOCKS_SELECT: int = 10
WEIGHT_BOUNDS:     tuple = (0.03, 0.20)

SHARPE_THRESHOLD: float = 0.30
MAX_DD_THRESHOLD: float = -0.40

BACKTEST_YEARS: int = 2

FEATURE_GROUPS = {
    "momentum": [
        "return_1m", "return_2m", "return_3m", "return_6m", "return_9m", "return_12m",
        "mom_12_1", "mom_6_1", "mom_3_1", "mom_12_1_rank",
    ],
    "volatility": [
        "realized_vol_3m", "realized_vol_12m", "vol_ratio",
        "realized_vol_3m_rank", "garman_klass_vol_lag1", "idio_vol",
    ],
    "risk_adjusted": [
        "sharpe_3m", "sharpe_6m", "sortino_6m", "calmar_proxy", "sharpe_6m_rank",
    ],
    "tail_risk": [
        "return_skew_6m", "return_kurt_6m", "hist_var_5pct", "cvar_5pct",
    ],
    "technical": [
        "rsi_lag1", "macd_lag1", "bb_low_lag1", "bb_mid_lag1",
        "bb_high_lag1", "atr_lag1", "cluster_lag1",
        "bb_position", "rsi_divergence", "macd_sign",
    ],
    "liquidity": [
        "amihud_illiquidity", "volume_trend_3m", "volume_zscore",
        "amihud_illiquidity_rank", "euro_volume_lag1",
    ],
    "mean_reversion": [
        "price_zscore_12", "nearness_52w_high",
    ],
    "macro": [
        "Mkt-RF_lag1", "SMB_lag1", "HML_lag1", "RMW_lag1", "CMA_lag1",
    ],
    "seasonality": [
        "month_sin", "month_cos", "is_q_end", "is_jan",
    ],
}
