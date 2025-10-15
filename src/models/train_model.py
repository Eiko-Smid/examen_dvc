import numpy as np
import pandas as pd

from pathlib import Path

from typing import Dict, List, Optional, Sequence, Tuple, Mapping, Iterable, Any, TypedDict, Callable

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

import joblib

X_TRAIN_SCALED_PATH = "data/processed_data/X_train_scaled.csv"
Y_TRAIN_PATH = "data/processed_data/y_train.csv"
MODEL_PARAM_PATH = "models/best_params.pkl"

TRAIN_RESULT_PATH = "models/best_model.pkl"


def read_data(
        param_path: str,
        X_train_scaled_path: str,
        y_train_path: str
    ) -> Tuple[Dict[str, Any], pd.DataFrame, pd.Series]:
    # Read model data
    params = joblib.load(param_path)
    best_params = params["best_params"]
    
    # Read train data
    X_train_scaled = pd.read_csv(X_train_scaled_path)
    y_train = pd.read_csv(y_train_path)

    return best_params, X_train_scaled, y_train


def train_model(
        best_params: Dict[str, Any],
        X_train_scaled: pd.DataFrame,
        y_train: pd.Series
) -> Ridge:
    # train model 
    ridge_regression = Ridge(**best_params)
    ridge_regression.fit(X=X_train_scaled, y=y_train)
    return ridge_regression


def store_model(model: Ridge, path: str):
    # Create path if needed
    path_ = Path(path)
    path_.parent.mkdir(parents=True, exist_ok=True)

    # Store model
    joblib.dump(model, path_)



def main():
    best_params, X_train_scaled, y_train = read_data(
        param_path=MODEL_PARAM_PATH,
        X_train_scaled_path=X_TRAIN_SCALED_PATH,
        y_train_path=Y_TRAIN_PATH
    )
    best_model = train_model(
        best_params=best_params,
        X_train_scaled=X_train_scaled,
        y_train=y_train
    )

    store_model(model=best_model, path=TRAIN_RESULT_PATH)


if __name__ == "__main__":
    main()