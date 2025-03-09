import argparse
import json

import polars as pl
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
import optuna

from scripts import logging
from scripts.model.train import load_data, preprocessing_pipeline
from scripts import MODEL_DATA_DIR, STUDY_FILE, PARAMS_FILE


class OptunaOptimization:
    def __init__(self, X_train: pl.DataFrame, y_train: pl.DataFrame, scorer: str):
        self.X_train = X_train
        self.y_train = y_train
        self.scorer = scorer

        self.num_features = X_train.select(pl.selectors.integer(), pl.selectors.float()).columns
        self.cat_features = X_train.select(pl.selectors.string(), pl.selectors.boolean()).columns

    def pipeline(self, model):
        pipe = preprocessing_pipeline(num_features= self.num_features, cat_features= self.cat_features)
        pipe.steps.append(('regression_model', model))
        return cross_val_score(pipe, self.X_train, self.y_train, scoring=self.scorer).mean()

    def objective_xgb(self, trial):
        # Set model params range and model instance
        XGB_params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "gamma": trial.suggest_float("gamma", 0, 1),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 0.5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2),
            # "device": "cuda"
        }
        model = xgb.XGBRegressor(**XGB_params, verbosity=0)
        return self.pipeline(model)
        
    def objective_lgbm(self, trial):
        LGBM_params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2),
            "num_leaves": trial.suggest_int("num_leaves", 7, 63),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 0.5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.1),
        }
        model = lgb.LGBMRegressor(**LGBM_params, verbosity=-1)
        return self.pipeline(model)
        
    def objective_cat(self, trial):
        CatBoost_params = {
            "iterations": trial.suggest_int("iterations", 50, 300),
            "depth": trial.suggest_int("depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 2),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 0.9),
            "random_strength": trial.suggest_float("random_strength", 0.1, 1),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 20),
            "grow_policy": trial.suggest_categorical("grow_policy", ["Depthwise", "SymmetricTree"]),
            "allow_writing_files": False,
            # "task_type": "GPU" if trial.params.get("gpu", False) else "CPU"
        }
        model = cat.CatBoostRegressor(**CatBoost_params, verbose=0)
        return self.pipeline(model)

    def optimize_params(self, model_name: str):
        logging.info("Hyperparameter Tuning.")
        objectives = {"xgb": self.objective_xgb, "lgbm": self.objective_lgbm, "cat": self.objective_cat}
        objective = objectives[model_name]
        logging.info(f"Initializing study to model {model_name.upper()}.")
        
        # Load storage
        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(str(MODEL_DATA_DIR / STUDY_FILE)),
        )

        # Create a study
        study = optuna.create_study(
            storage=storage,
            study_name=f"{model_name}_study",
            direction='maximize',
            load_if_exists=True
        )

        # Optimize the objective function (model performance)
        logging.info(f"Study will be saved into {MODEL_DATA_DIR / STUDY_FILE}.")
        try:
            study.optimize(objective)
        except KeyboardInterrupt:
            logging.info("Optimization interrupted by user.")
        except Exception as e:
            logging.warning(f"Error during optimization:\n{e}")
        finally:
            logging.info(f"Best score: {study.best_value}.")
            logging.info(f"Best params: {study.best_params}.")
            # Save best params
            params_path = MODEL_DATA_DIR / PARAMS_FILE

            # Load existing parameters or initialize if file is missing/corrupt
            try:
                params = json.loads(params_path.read_text()) if params_path.exists() else {}
            except json.JSONDecodeError:
                params = {}

            # Update parameters for the current model
            params.setdefault(model_name, {})["score"] = study.best_value
            params[model_name]["params"] = study.best_params

            # Save back to file
            params_path.write_text(json.dumps(params, indent=4))
            logging.info(f"Best params saved to {MODEL_DATA_DIR / PARAMS_FILE}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required = True,
        choices = ['xgb', 'lgbm', 'cat'],
        help = "Hyperparameter tuning with Optuna. Choose between xgb, lgbm and cat models. Shutdown with Ctrl-c.",
        type = str
    )
    args = parser.parse_args()

    X_train, y_train, _ = load_data()
    opt = OptunaOptimization(X_train, y_train, 'neg_root_mean_squared_error')
    opt.optimize_params(args.model)
