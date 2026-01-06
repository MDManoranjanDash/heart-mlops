# HEART-MLOPS/tests/test_features.py

import pandas as pd
from src.features import build_feature_pipeline

def test_build_feature_pipeline():
    df = pd.DataFrame({"age":[29,40], "sex":[1,0], "target":[0,1]})
    pre, cols = build_feature_pipeline(df)
    assert len(cols) == 13
    assert pre is not None
