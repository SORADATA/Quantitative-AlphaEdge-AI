import joblib
import sys
from pathlib import Path
import json

# Injection de la racine pour la résolution des imports
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from const import MODEL_DIR
from src.utils.logger import setup_logger

logger = setup_logger("validate")


def get_score_from_card(path):
    with open(path, 'r') as f:
        card = json.load(f)
    return card["metrics_ml"]["auc_test"]
current_score = get_score_from_card(MODEL_DIR / "model_card.json")


def validate_model():
    current_path = MODEL_DIR / "ensemble_model.pkl"
    candidate_path = MODEL_DIR / "candidate_model.pkl"
    
    if not candidate_path.exists():
        logger.error("Erreur : candidate_model.pkl absent.")
        return False

    if not current_path.exists():
        logger.info("Premier déploiement : promotion automatique du candidat.")
        candidate_path.rename(current_path)
        return True

    try:
        current_model = joblib.load(current_path)
        new_model = joblib.load(candidate_path)
        
        # Comparaison sur best_score_ (standard sklearn/optuna)
        current_score = getattr(current_model, 'best_score_', 0.0)
        new_score = getattr(new_model, 'best_score_', 0.0)
        
        if new_score >= current_score:
            candidate_path.replace(current_path)
            logger.info(f"Modèle promu (Nouveau: {new_score:.4f} > Actuel: {current_score:.4f})")
            return True
        else:
            candidate_path.unlink()
            logger.warning(f"Validation échouée (Nouveau: {new_score:.4f} < Actuel: {current_score:.4f})")
            return False
            
    except Exception as e:
        logger.error(f"Erreur système durant la validation : {e}")
        return False

if __name__ == "__main__":
    sys.exit(0 if validate_model() else 1)