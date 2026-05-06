
import os
import yaml
from pathlib import Path
from datetime import date

CONFIG_PATH = Path("config/backtest_config.yaml")

with open(CONFIG_PATH, "r") as f:
    _cfg = yaml.safe_load(f)

SETTINGS = {
    "BASE_DIR": Path(_cfg.get("paths", {}).get("base_dir", ".")),
    "CONFIG_DIR": Path(_cfg.get("paths", {}).get("config_dir", "config/markets")),
    "TARGET_CLUSTER": _cfg["strategy"]["target_cluster"],
    "PROBA_THRESHOLD": _cfg["strategy"]["proba_threshold"],
    "FEATURE_COLS": _cfg["strategy"]["feature_cols"],
    "REPO_ID": _cfg.get("huggingface", {}).get("repo_id"),
    "HF_TOKEN": os.getenv("HF_TOKEN"),
    "END_TIME": _cfg["backtest"].get("end_time") or date.today().strftime("%Y-%m-%d"),
}