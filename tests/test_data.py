# HEART-MLOPS/tests/test_data.py

import pandas as pd
import numpy as np
from src.data import clean_and_preprocess

def test_clean_and_preprocess_binarization():
    # Create mock data with UCI-style multiclass targets (0-4)
    data = {
        "age": [50, 60, 45, 55],
        "target": [0, 1, 2, 0]  # UCI original: 0 is healthy, 1-4 is disease
    }
    df = pd.DataFrame(data)
    
    processed_df = clean_and_preprocess(df)
    
    # Check if target is binarized (0 or 1)
    assert set(processed_df['target'].unique()).issubset({0, 1})
    # Check that original '2' became '1'
    assert processed_df.iloc[2]['target'] == 1
    
    

def test_data_cleaning_removes_na():
    data = {
        "age": [50, np.nan, 45],
        "target": [0, 1, 0]
    }
    df = pd.DataFrame(data)
    processed_df = clean_and_preprocess(df)
    
    # Check that the row with NaN was dropped
    assert len(processed_df) == 2
    assert processed_df.isnull().sum().sum() == 0
