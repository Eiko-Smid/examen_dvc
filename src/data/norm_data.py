import numpy as np
import pandas as pd

from pathlib import Path

from typing import Dict, List, Optional, Sequence, Tuple, Mapping, Iterable, Any, TypedDict, Callable

from sklearn.preprocessing import StandardScaler


TRAIN_SCR_DATA_PATH = "data/processed_data/X_train.csv"
TEST_SCR_DATA_PATH = "data/processed_data/X_test.csv"
DST_PATH = "data/processed_data"


def read_data(
        train_data_path: str,
        test_data_path: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Read data
    X_train = pd.read_csv(train_data_path)
    X_test = pd.read_csv(test_data_path)

    return X_train, X_test


def scale_data(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
):
    X_train_scaled = None
    X_test_scaled = None
    if X_train is not None and X_test is not None:
        # Scale data
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        print("Failed to load data.")
    
    return X_train_scaled, X_test_scaled


def store_data(
        X_train_scaled: np.ndarray,
        X_test_scaled: np.ndarray,
        dst_path: str = "data/processed_data"
):
    if X_train_scaled is not None and X_test_scaled is not None :
        # Store data
        dst_path_ = Path(dst_path)
        dst_path_.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(X_train_scaled).to_csv(dst_path_ / "X_train_scaled.csv")
        pd.DataFrame(X_test_scaled).to_csv(dst_path_ / "X_test_scaled.csv")
        print("Stored scaled data.")
        return True
    else:
        print("Failed to store scaled data.")
        return False
    


def main():
    # Read data
    X_train, X_test = read_data(
        train_data_path=TRAIN_SCR_DATA_PATH,
        test_data_path=TEST_SCR_DATA_PATH
    )

    # Scale data
    X_train_scaled, X_test_scaled = scale_data(
        X_train=X_train,
        X_test=X_test
    )

    # Store data
    success = store_data(
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        dst_path=DST_PATH
    )
     

    


if __name__ == "__main__":
    main()