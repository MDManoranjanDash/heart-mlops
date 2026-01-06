# HEART-MLOPS/src/tracking.py

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline
from src.train import train_and_evaluate
from src.data import load_raw, clean_and_preprocess

def run_experiments():
    mlflow.set_experiment("heart-disease-mlops")
    df = clean_and_preprocess(load_raw("data/raw/heart.csv"))

    for model_name in ["logreg", "rf"]:
        with mlflow.start_run(run_name=model_name):
            pipe, metrics = train_and_evaluate(df, model_name)
            # Log params
            mlflow.log_param("model", model_name)
            # Log metrics
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))
            # Log artifact model
            mlflow.sklearn.log_model(pipe, artifact_path=f"model_{model_name}")
            # Log metrics CSV
            pd.DataFrame([metrics]).to_csv("models/tmp_metrics.csv", index=False)
            mlflow.log_artifact("models/tmp_metrics.csv")

if __name__ == "__main__":
    run_experiments()
