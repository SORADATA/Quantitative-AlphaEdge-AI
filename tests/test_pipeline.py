# test_pipeline.py
from daily_run import run_pipeline
import json

# Charge un seul marché pour tester
with open("config/markets/CAC40.json", "r") as f:
    config = json.load(f)

# Lance le pipeline sans uploader (tu peux commenter la ligne upload_to_hf dans daily_run)
run_pipeline(config)