import json
from pathlib import Path
# Importe run_pipeline depuis ton fichier daily_run.py
from daily_run import run_pipeline 

def main():
    config_dir = Path("config/markets")
    for config_file in config_dir.glob("*.json"):
        with open(config_file, 'r') as f:
            market_config = json.load(f)
        
        # CORRECTION : On passe le dictionnaire entier, pas des arguments nommés
        run_pipeline(market_config)

if __name__ == "__main__":
    main()