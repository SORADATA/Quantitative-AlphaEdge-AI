from pathlib import Path
import os
from dotenv import load_dotenv
import yaml
from datetime import date

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "config" / "backtest_config.yaml"

load_dotenv(BASE_DIR / ".env")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

SETTINGS = {
    "BASE_DIR": BASE_DIR,
    "CONFIG_DIR": BASE_DIR / "config",
    "DATA_DIR": BASE_DIR / "data",
    "MODEL_DIR": BASE_DIR / "src" / "models",
    "LOG_DIR": BASE_DIR / "logs",

    "MARKET_NAME": _cfg.get("market", {}).get("name", "CAC40"),
    "MARKET_SUFFIX": _cfg.get("market", {}).get("suffix", "cac40"),

    "TARGET_CLUSTER": _cfg["strategy"]["target_cluster"],
    "PROBA_THRESHOLD": _cfg["strategy"]["proba_threshold"],
    "FEATURE_COLS": _cfg["strategy"]["feature_cols"],

    "TRADING_DAYS_YEAR": _cfg["backtest"].get("trading_days_year", 252),
    "RISK_FREE_RATE": _cfg["backtest"].get("risk_free_rate", 0.02),
    "END_TIME": _cfg["backtest"].get("end_time") or date.today().strftime("%Y-%m-%d"),

    "HF_TOKEN": os.getenv("HF_TOKEN"),
    "REPO_ID": _cfg.get("huggingface", {}).get("repo_id", "soradata/alphaedge-data"),
}