import json
import warnings
# Filter warnings caused by sklearn
# warnings.filterwarnings("ignore", category=UserWarning)

import polars as pl
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cat

from scripts import logging
from scripts import (
    PROCESSED_DATA_DIR, FULL_DATASET, MODEL_DATA_DIR, 
    PARAMS_FILE, SUBMISSION_FILE
)

def load_data():
    logging.info(f"Loading {PROCESSED_DATA_DIR / FULL_DATASET}.")
    full_data = pl.read_csv(PROCESSED_DATA_DIR / FULL_DATASET)
    train = full_data.filter(pl.col('train')).drop('hex_id')
    X_test = full_data.filter(~pl.col('train')).drop(['cost_of_living', 'train'])
    X_train = train.drop(['cost_of_living', 'train'])
    y_train = train.select('cost_of_living').to_numpy().ravel()
    return X_train, y_train, X_test

def load_params():
    with open(MODEL_DATA_DIR / PARAMS_FILE, 'r') as f:
        data = json.load(f)
        return data['xgb']['params'], data['lgbm']['params'], data['cat']['params']

def preprocessing_pipeline(num_features: list[str], cat_features: list[str]):
    return Pipeline([
        ("preprocessing", ColumnTransformer([
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(sparse_output= False, drop="if_binary", handle_unknown= "ignore"), cat_features)
        ],
        remainder="passthrough",
        verbose_feature_names_out=False))],
        memory= str(MODEL_DATA_DIR) # Enable caching to save preprocessing steps
    )

def train(X_train: pl.DataFrame, y_train: pl.DataFrame, scorer: str, seed: int):
    num_features = X_train.select(pl.selectors.integer(), pl.selectors.float()).columns
    cat_features = X_train.select(pl.selectors.string(), pl.selectors.boolean()).columns
    # Preprocessing pipeline
    pipe = preprocessing_pipeline(num_features, cat_features)

    # Load model params
    logging.info(f"Loading models params from {MODEL_DATA_DIR / PARAMS_FILE}.")
    xgb_params, lgbm_params, cat_params = load_params()
    models = [
        ('xgb', xgb.XGBRegressor(**xgb_params, verbosity=0, seed=seed)),
        ('lgbm', lgb.LGBMRegressor(**lgbm_params, verbosity=-1, seed=seed)),
        ('cat', cat.CatBoostRegressor(**cat_params, verbose=0, random_seed=seed, allow_writing_files= False))
    ]
    regressor = VotingRegressor(models, n_jobs=-1, verbose=False)
    pipe.steps.append(('regression_model', regressor))

    logging.info("Cross-validating model.")
    scores = cross_validate(pipe, X_train, y_train, scoring=scorer)
    logging.info("Model performance:")
    for key, value in scores.items():
        logging.info(f"{key.upper()}: {value} (Avg {value.mean():.4f})")

    logging.info("Fit model in the whole data.")
    pipe.fit(X_train, y_train)
    return pipe

def predict_test(pipe, X_test: pl.DataFrame):
    return pipe.predict(X_test)


if __name__ == '__main__':
    X_train, y_train, X_test = load_data()
    pipe = train(X_train, y_train, 'neg_root_mean_squared_error', seed = 42)
    y_test = predict_test(pipe, X_test)
    pl.DataFrame([X_test['hex_id'], pl.Series(name="cost_of_living", values = y_test)]).write_csv(MODEL_DATA_DIR / SUBMISSION_FILE)
    logging.info(f"Submission saved to {MODEL_DATA_DIR / SUBMISSION_FILE}.")
    