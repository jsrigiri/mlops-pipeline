MODEL_TYPE = "xgboost"      # xgboost or lightgbm
TASK_TYPE = "regression"    # regression or classification
USE_GPU = True

TRAIN_RATIO = 0.8
RANDOM_STATE = 42

DATA_PATH = "data/sample.csv"
MODEL_PATH = "artifacts/model.joblib"
FEATURES_PATH = "artifacts/feature_columns.joblib"
METRICS_PATH = "artifacts/metrics.json"

MLFLOW_TRACKING_URI = "file:./mlruns"