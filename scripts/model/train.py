import json

import polars as pl
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import VotingRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
import joblib

from scripts import logging
from scripts import PROCESSED_DATA_DIR, FULL_DATASET, MODEL_DATA_DIR, PARAMS_FILE, MODEL_FILE, SUBMISSION_FILE


def load_data() -> tuple:
    """Load train and test data."""
    logging.info(f"Loading {PROCESSED_DATA_DIR / FULL_DATASET}.")
    full_data = pl.read_csv(PROCESSED_DATA_DIR / FULL_DATASET)
    train = full_data.filter(pl.col("train")).drop("hex_id")
    X_test = full_data.filter(~pl.col("train")).drop(["cost_of_living", "train"]).to_pandas()
    X_train = train.drop(["cost_of_living", "train"]).to_pandas()
    y_train = train.select("cost_of_living").to_numpy().ravel()
    return X_train, y_train, X_test


def load_params() -> tuple:
    """Load tunned params for each model."""
    with open(MODEL_DATA_DIR / PARAMS_FILE, "r") as f:
        data = json.load(f)
    return data["xgb"]["params"], data["lgbm"]["params"], data["cat"]["params"]


def pipeline() -> Pipeline:
    """Processing pipeline to prepare the data for the model. This applies the following steps:
    - Standardize numerical features.
    - One-hot encode categorical features.
    """
    return Pipeline(
        [
            (
                "preprocessing",
                ColumnTransformer(
                    [
                        ("num", StandardScaler(), make_column_selector(dtype_include=np.number)),
                        (
                            "cat",
                            OneHotEncoder(
                                sparse_output=False,
                                drop="if_binary",
                                handle_unknown="ignore",
                            ),
                            make_column_selector(dtype_exclude=np.number),
                        ),
                    ],
                    remainder="passthrough",
                    verbose_feature_names_out=False,
                ),
            )
        ],
        memory=str(MODEL_DATA_DIR),  # Enable caching to save preprocessing steps
    )


def train(X_train: pl.DataFrame, y_train: pl.DataFrame, seed: int) -> Pipeline:
    """Define the ensemble model, validate it in cross-validation and train it in the whole data."""
    # Preprocessing pipeline
    pipe = pipeline()

    # Load model params
    logging.info(f"Loading models params from {MODEL_DATA_DIR / PARAMS_FILE}.")
    xgb_params, lgbm_params, cat_params = load_params()
    models = [
        ("xgb", xgb.XGBRegressor(**xgb_params, verbosity=0, seed=seed)),
        ("lgbm", lgb.LGBMRegressor(**lgbm_params, verbosity=-1, seed=seed)),
        (
            "cat",
            cat.CatBoostRegressor(
                **cat_params, verbose=0, random_seed=seed, allow_writing_files=False
            ),
        ),
    ]
    regressor = VotingRegressor(models, verbose=False)
    pipe.steps.append(("regression_model", regressor))

    scores = cross_validate(pipe, X_train, y_train, scoring="neg_root_mean_squared_error")
    for key, value in scores.items():
        logging.info(f"{key.upper()}: {value} (Avg {value.mean()})")   

    logging.info("Fitting model...")
    pipe.fit(X_train, y_train)
    return pipe


def predict_test(pipe, X_test: pl.DataFrame) -> np.ndarray:
    """Predict the target for the test data."""
    logging.info("Predicting in test data...")
    return pipe.predict(X_test)


if __name__ == "__main__":
    X_train, y_train, X_test = load_data()
    pipe = train(X_train, y_train, seed=42)

    joblib.dump(pipe, MODEL_DATA_DIR / MODEL_FILE)
    logging.info(f"Model saved to {MODEL_DATA_DIR / MODEL_FILE}")
    
    logging.info("Predicting in test data...")
    y_test = predict_test(pipe, X_test)
    pl.DataFrame(
        [X_test["hex_id"], pl.Series(name="cost_of_living", values=y_test)]
    ).write_csv(MODEL_DATA_DIR / SUBMISSION_FILE)
    logging.info(f"Submission saved to {MODEL_DATA_DIR / SUBMISSION_FILE}.")
