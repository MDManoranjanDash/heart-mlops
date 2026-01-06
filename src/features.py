
# HEART-MLOPS/src/features.py

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

def test_build_feature_pipeline():
    # Create a mock dataframe with all 13 required features
    data = {col: [0.0, 1.0] for col in [
        "age", "trestbps", "chol", "thalach", "oldpeak", 
        "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
    ]}
    data["target"] = [0, 1]
    df = pd.DataFrame(data)
    
    pre, cols = build_feature_pipeline(df)
    
    # Now it should correctly find all 13 features
    assert len(cols) == 13

def build_feature_pipeline(df):
    # Identify numeric and categorical columns for Task 2 [cite: 19]
    numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor, (numeric_features + categorical_features)