from pathlib import Path
import os
from datetime import date
from dotenv import load_dotenv
import yaml

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config" / "backtest_config.yaml"

load_dotenv(BASE_DIR / ".env")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

backtest_cfg = cfg.get("backtest", {})
strategy_cfg = cfg.get("strategy", {})
transaction_cfg = cfg.get("transaction", {})
huggingface_cfg = cfg.get("huggingface", {})

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "src" / "models"
LOG_DIR = BASE_DIR / "logs"
CONFIG_DIR = BASE_DIR / "config"

DEFAULT_MARKET = "cac40.json"
CONFIG_FILE = CONFIG_DIR / DEFAULT_MARKET

RISK_FREE_RATE = backtest_cfg.get("risk_free_rate", 0.02)
TRADING_DAYS_YEAR = backtest_cfg.get("trading_days_year", 252)
END_TIME = backtest_cfg.get("end_time") or date.today().strftime("%Y-%m-%d")

FEE_BPS = transaction_cfg.get("fee_bps", 0.003)

HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = huggingface_cfg.get("repo_id", "soradata/alphaedge-data")

TARGET_CLUSTER = strategy_cfg.get("target_cluster", 3)
PROBA_THRESHOLD = strategy_cfg.get("proba_threshold", 0.6)
FEATURE_COLS = strategy_cfg.get(
    "feature_cols",
    [
        "rsi",
        "macd",
        "bb_low",
        "bb_high",
        "atr",
        "return_2m",
        "return_3m",
        "return_6m",
        "euro_volume_lag1",
        "garman_klass_vol_lag1",
        "Mkt-RF_lag1",
        "SMB_lag1",
        "HML_lag1",
        "RMW_lag1",
        "CMA_lag1",
        "cluster",
    ],
)

# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

RSI_WINDOW = 20
BB_WINDOW = 20
BB_STD = 2
ATR_WINDOW = 14
MACD_SLOW = 26
MACD_FAST = 12
MACD_SIGN = 9

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

MIN_HISTORY_TA = 20
MIN_HISTORY_FF = 24

MOMENTUM_LAGS = [1, 2, 3, 6, 9, 12]
WINSOR_CUTOFF = 0.005

VARS_TO_LAG = [
    "Mkt-RF",
    "SMB",
    "HML",
    "RMW",
    "CMA",
    "euro_volume",
    "garman_klass_vol",
]

FAMA_FRENCH_FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

# =============================================================================
# RESAMPLING
# =============================================================================

RESAMPLE_MEAN_COLS = ["euro_volume"]

RESAMPLE_LAST_EXCLUDE = [
    "euro_volume",
    "volume",
    "open",
    "high",
    "low",
    "close",
]