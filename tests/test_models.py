# HEART-MLOPS/tests/test_models.py

from src.models import make_logreg, make_rf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def test_model_instantiation():
    logreg = make_logreg()
    rf = make_rf()
    
    assert isinstance(logreg, LogisticRegression)
    assert isinstance(rf, RandomForestClassifier)
    # Check specific hyperparams from your source
    assert rf.n_estimators == 300