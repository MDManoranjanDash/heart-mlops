# HEART-MLOPS/src/train.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, roc_auc_score

from src.data import load_raw, clean_and_preprocess
from src.features import build_feature_pipeline
from src.models import make_logreg, make_rf

def get_scoring():
    return {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score),
        "recall": make_scorer(recall_score),
        "roc_auc": make_scorer(roc_auc_score),
    }

def train_and_evaluate(df: pd.DataFrame, model_name="logreg"):
    preprocessor, _ = build_feature_pipeline(df)
    X = df.drop(columns=["target"])
    y = df["target"]

    model = make_logreg() if model_name == "logreg" else make_rf()
    
    # Enclose preprocessor and model in one Pipeline for reproducibility 
    pipe = Pipeline(steps=[("pre", preprocessor), ("clf", model)])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_res = cross_validate(pipe, X, y, cv=cv, scoring=get_scoring())

    # Fit on full data for deployment [cite: 41]
    pipe.fit(X, y)

    metrics = {m: np.mean(cv_res[f"test_{m}"]) for m in get_scoring().keys()}
    return pipe, metrics

if __name__ == "__main__":
    raw_df = load_raw()
    df = clean_and_preprocess(raw_df)
    
    log_pipe, log_metrics = train_and_evaluate(df, "logreg")
    rf_pipe, rf_metrics = train_and_evaluate(df, "rf")
    
    # Selection logic based on ROC-AUC [cite: 20]
    best_pipe = rf_pipe if rf_metrics["roc_auc"] >= log_metrics["roc_auc"] else log_pipe
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_pipe, "models/final_model.joblib")
    print(f"Model saved. RF ROC-AUC: {rf_metrics['roc_auc']:.2f}, LogReg ROC-AUC: {log_metrics['roc_auc']:.2f}")