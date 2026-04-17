import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LinearRegression, LogisticRegression
from config import MODEL_TYPE, TASK_TYPE, USE_GPU, RANDOM_STATE


def _build_model(use_gpu: bool):
    used_device = "cpu"

    if MODEL_TYPE == "xgboost":
        params = {
            "random_state": RANDOM_STATE,
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        }

        if USE_GPU and use_gpu:
            params["device"] = "cuda"
            used_device = "gpu"

        if TASK_TYPE == "regression":
            model = xgb.XGBRegressor(**params)
        else:
            params["eval_metric"] = "logloss"
            model = xgb.XGBClassifier(**params)

        return model, used_device

    if MODEL_TYPE == "lightgbm":
        params = {
            "random_state": RANDOM_STATE,
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        }

        if USE_GPU and use_gpu:
            params["device_type"] = "gpu"
            used_device = "gpu"

        if TASK_TYPE == "regression":
            model = lgb.LGBMRegressor(**params)
        else:
            model = lgb.LGBMClassifier(**params)

        return model, used_device

    if MODEL_TYPE == "linear":
        if TASK_TYPE == "regression":
            return LinearRegression(), used_device
        return LogisticRegression(max_iter=1000), used_device

    raise ValueError(f"Unsupported MODEL_TYPE: {MODEL_TYPE}")


def train_model(X, y):
    model, used_device = _build_model(use_gpu=True)

    try:
        model.fit(X, y)
        return model, used_device
    except Exception as e:
        fallback_model, _ = _build_model(use_gpu=False)
        fallback_model.fit(X, y)
        return fallback_model, f"{used_device} -> fallback_cpu ({type(e).__name__})"