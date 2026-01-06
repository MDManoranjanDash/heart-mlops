# HEART-MLOPS/src/data.py

import os
import pandas as pd

# Direct link to the Cleveland processed dataset
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# Column names based on the UCI documentation (14 features)
COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

# Method to download data.
def download_data(save_path="data/raw/heart.csv"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # The dataset uses '?' for missing values
    df = pd.read_csv(UCI_URL, names=COLUMN_NAMES, na_values="?")
    df.to_csv(save_path, index=False)
    return save_path

# Method to load data (raw) from the downloaded csv file.
def load_raw(path="data/raw/heart.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        download_data(path)
    return pd.read_csv(path)

# Method to clean the raw data.
# In src/data.py
def clean_and_preprocess(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df = df.dropna()
    
    # This line is crucial for passing the test
    if 'target' in df.columns:
        df['target'] = (df['target'] > 0).astype(int)
        
    df = df.drop_duplicates()
    return df

# Meethod to save the cleaned data.
def save_processed(df: pd.DataFrame, path="data/processed/heart_clean.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path

if __name__ == "__main__":
    p = download_data()
    df = load_raw(p)
    dfp = clean_and_preprocess(df)
    save_processed(dfp)





