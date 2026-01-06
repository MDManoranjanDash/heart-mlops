# HEART-MLOPS/src/models.py

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def make_logreg():
    return LogisticRegression(max_iter=1000, n_jobs=None, C=1.0, solver="lbfgs")

def make_rf():
    return RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_split=2, min_samples_leaf=1,
        random_state=42, n_jobs=-1
    )
