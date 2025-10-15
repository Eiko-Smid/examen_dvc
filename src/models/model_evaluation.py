import numpy as np
import pandas as pd

from pathlib import Path

from typing import Dict, List, Optional, Sequence, Tuple, Mapping, Iterable, Any, TypedDict, Callable

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

import json
import joblib


MODEL_PATH = "models/best_model.pkl"
X_TEST_SCALED_PATH = "data/processed_data/X_test_scaled.csv"
Y_TEST_PATH = "data/processed_data/y_test.csv"

PRED_PATH = "data/predictions.csv"
METRIC_PATH = "metrics/scores.json"


def read_data(
        model_path: str,
        X_test_scaled_path: str,
        y_test_path: str
) -> Tuple[Ridge, pd.DataFrame, pd.Series]:
    # Read model
    model = joblib.load(model_path)

    # Read test data
    X_test_scaled = pd.read_csv(X_test_scaled_path)
    y_test = pd.read_csv(y_test_path)
    
    print(type(model))
    return model, X_test_scaled, y_test


def evaluate_model(
        model: Ridge,
        X_test_scaled: pd.DataFrame,
        y_test: pd.Series
) -> Tuple[np.ndarray, float, float]:
    pred = model.predict(X_test_scaled)

    r2 = r2_score(y_true=y_test, y_pred=pred)
    rmse = root_mean_squared_error(y_true=y_test, y_pred=pred)

    return pred, r2, rmse


def store_results(
        pred: np.ndarray,
        r2: float,
        rmse: float,
        pred_path: str,
        metric_path: str
):
    # Store predictions
    pred_path_ = Path(pred_path)
    pred_path_.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pred).to_csv(pred_path)

    # Store metrics
    metric_path_ = Path(metric_path)
    metric_path_.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "r2": float(round(r2, 5)),
        "rmse": float(round(rmse, 5))
    }
    with metric_path_.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def main():
    # Read data
    model, X_test_scaled, y_test = read_data(
        model_path=MODEL_PATH,
        X_test_scaled_path=X_TEST_SCALED_PATH,
        y_test_path=Y_TEST_PATH
    )

    # Evaluate model
    pred, r2, rmse = evaluate_model(
        model=model,
        X_test_scaled=X_test_scaled,
        y_test=y_test
    )

    # Store results (pred, r2, rmse)
    store_results(
        pred=pred,
        r2=r2,
        rmse=rmse,
        pred_path=PRED_PATH,
        metric_path=METRIC_PATH
    )


if __name__ == "__main__":
    main()