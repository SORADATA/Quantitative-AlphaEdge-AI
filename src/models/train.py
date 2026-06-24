import os
import warnings
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score
import mlflow

# 1. INITIALISATION MLFLOW
warnings.filterwarnings('ignore')

# Authentification vers ton Space Hugging Face
os.environ["MLFLOW_TRACKING_USERNAME"] = "SORADATA"
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("HF_TOKEN")

mlflow.set_tracking_uri("https://soradata-alphaedge-registry.hf.space")
mlflow.set_experiment("AlphaEdge_XGBoost_Production")


def main():
    print("Download data clean...")
    feature_cols = ['rsi', 'macd', 'bb_low', 'bb_high', 'atr', 'euro_volume_lag1', 'cluster']
    dates = df.index.get_level_values('date').unique().sort_values()
    split_date = dates[int(len(dates) * 0.8)]
    train_mask = df.index.get_level_values('date') <= split_date
    X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, 'target']
    X_test, y_test = df.loc[~train_mask, feature_cols], df.loc[~train_mask, 'target']

    # RUNNING MLFLOW SESSION
    with mlflow.start_run(run_name="GridSearch_XGBoost_T1"):
        print("🤖 Entraînement et optimisation XGBoost en cours...")
        param_grid = {
            'max_depth': [3, 4],
            'learning_rate': [0.01, 0.05],
            'n_estimators': [100, 200]
        }
        xgb_base = xgb.XGBClassifier(eval_metric='auc', random_state=42, n_jobs=-1)
        tscv = TimeSeriesSplit(n_splits=3)
        grid_search = GridSearchCV(estimator=xgb_base, param_grid=param_grid, scoring='roc_auc', cv=tscv, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        # Récupération du meilleur modèle
        best_model = grid_search.best_estimator_
        
        # Évaluation sur le Test Set
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        y_pred_class = best_model.predict(X_test)
        
        final_auc = roc_auc_score(y_test, y_pred_proba)
        final_acc = accuracy_score(y_test, y_pred_class)
        
        # 3. SAUVEGARDE SUR MLFLOW (Fini les exports manuels !)
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("AUC_Test", final_auc)
        mlflow.log_metric("Accuracy_Test", final_acc)
        
        # On sauvegarde le modèle .pkl directement sur Hugging Face
        mlflow.xgboost.log_model(
            xgb_model=best_model,
            artifact_path="model",
            registered_model_name="AlphaEdge_XGB_CAC40"
        )
        
        print(f"✅ Modèle en production ! AUC: {final_auc:.4f} | Accuracy: {final_acc:.4f}")

if __name__ == "__main__":
    main()