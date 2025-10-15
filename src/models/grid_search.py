import numpy as np
import pandas as pd

from pathlib import Path

from typing import Dict, List, Optional, Sequence, Tuple, Mapping, Iterable, Any, TypedDict, Callable

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score

import joblib
import time

RANDOM_STATE = 42

X_TRAIN_SCALED_PATH = "data/processed_data/X_train_scaled.csv"
X_Test_SCALED_PATH = "data/processed_data/X_test_scaled.csv"
Y_TRAIN_PATH = "data/processed_data/y_train.csv"
Y_TEST_PATH = "data/processed_data/y_test.csv"

TRAIN_RESULT_PATH = "models/best_params.pkl"


def read_data(
        x_train_sclaed_path: str,
        x_test_sclaed_path: str,
        y_train_path: str,
        y_test_path: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # Read data
    X_train_scaled = pd.read_csv(x_train_sclaed_path)
    X_test_scaled = pd.read_csv(x_test_sclaed_path)
    y_train = pd.read_csv(y_train_path)
    y_test = pd.read_csv(y_test_path)
    return X_train_scaled, X_test_scaled, y_train, y_test


def grid_search(
        X_train_scaled: pd.DataFrame | np.ndarray,
        X_test_scaled: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        y_test: pd.Series | np.ndarray,
        cv: int = 5,
        scoring: str = "neg_root_mean_squared_error",
) -> Dict[str, Any]:
    # Define model
    model = Ridge(random_state=RANDOM_STATE)

    # Define parameter grid
    param_grid = {
        "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        "fit_intercept": [True, False]
    }

    # Init grid search 
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1, 
        verbose=0
    )

    # Do grid search
    gs.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = gs.predict(X_test_scaled)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Store results in dict
    results: Dict[str, Any] = {
        "model": "Ridge",
        "best_params": gs.best_params_,
        "cv_best_score": float(gs.best_score_),  # still in 'neg' units due to scoring
        "cv_scoring": scoring,
        "cv_folds": cv,
        "test_metrics": {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return results


def store_data(
        results: Dict[str, Any],
        dst_path: str = "models",
):
    # Save params
    dst_path_ = Path(dst_path)
    dst_path_.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(results, dst_path_)



def main():
    # Read data
    X_train_scaled, X_test_scaled, y_train, y_test = read_data(
        x_train_sclaed_path=X_TRAIN_SCALED_PATH,
        x_test_sclaed_path=X_Test_SCALED_PATH,
        y_train_path=Y_TRAIN_PATH,
        y_test_path=Y_TEST_PATH
    )

    # Do grid search
    results = grid_search(
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
    )

    # Store data
    store_data(
        results=results,
        dst_path=TRAIN_RESULT_PATH
    )


if __name__ == "__main__":
    main()