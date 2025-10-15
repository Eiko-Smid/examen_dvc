import numpy as np
import pandas as pd

from pathlib import Path

from typing import Dict, List, Optional, Sequence, Tuple, Mapping, Iterable, Any, TypedDict, Callable

from sklearn.model_selection import train_test_split

SRC_PATH = "data/raw_data/raw.csv"
DST_PATH = "data/processed_data"
RANDOM_STATE = 42


def split_and_store_data(
        src_path: str,
        dst_path: str,
        test_size: float = 0.8,
        target_var: str = "silica_concentrate",
        drop_columns: Iterable[str] = ["date", "silica_concentrate"]
    ) -> bool:
    try:
        df = pd.read_csv(src_path)
        print("Loaded raw data.")
    except Exception as e:
        print("Failed to load data ")
        return False

    # Extract target var and train vars
    y = df[target_var]
    X = df.drop(labels=drop_columns, axis=1)

    # Do train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # Store data
    try:
        dst_path_ = Path(dst_path)
        dst_path_.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(X_train).to_csv(dst_path_ / "X_train.csv", index=False)
        pd.DataFrame(X_test).to_csv(dst_path_ / "X_test.csv", index=False)
        pd.DataFrame(y_train).to_csv(dst_path_ / "y_train.csv", index=False)
        pd.DataFrame(y_test).to_csv(dst_path_ / "y_test.csv", index=False)
        print(f"Splitted data into train and test data and stored under:\n{dst_path_}")
        return True
    except Exception as e:
        print(f"Failed to stored data:\n{dst_path_}")
        return False



def main():
    success = split_and_store_data(src_path=SRC_PATH, dst_path=DST_PATH)


if __name__ == "__main__":
    main()