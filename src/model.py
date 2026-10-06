import argparse
import json
from pathlib import Path

import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "preprocessed.csv"
MODELS_DIR = ROOT / "models"

MODEL_V = 1
TARGET = "Churn"
SEED = 42
N_TRIALS = 50
CV_FOLDS = 5


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)

    # split into train and target data
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return X, y


def split(X, y):
    # evaluate.py uses the same split, so the test set stays untouched until evaluation
    return train_test_split(X, y, random_state=SEED, stratify=y)


def objective(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
    model = xgb.XGBClassifier(**params, objective="binary:logistic", random_state=SEED)

    # score on folds of the training data only, tuning on the test set would leak it into the model
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

    return scores.mean()


def best_params(X, y, n_trials=N_TRIALS):
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)

    print(f"best cv roc_auc: {study.best_value:.4f}")

    return study.best_params


def train(X, y, params):
    m = xgb.XGBClassifier(**params, objective="binary:logistic", random_state=SEED)
    m.fit(X, y)

    return m


def save(m, params, version=MODEL_V, models_dir=MODELS_DIR):
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"model_v{version}.ubj"
    m.save_model(model_path)
    # keep the params next to the model so the run can be reproduced
    (models_dir / f"model_v{version}_params.json").write_text(json.dumps(params, indent=2))

    return model_path


def main(n_trials=N_TRIALS, version=MODEL_V, models_dir=MODELS_DIR):
    X, y = load_data()
    X_train, X_test, y_train, y_test = split(X, y)

    params = best_params(X_train, y_train, n_trials)
    m = train(X_train, y_train, params)
    path = save(m, params, version, models_dir)

    test_auc = roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])
    print(f"test roc_auc: {test_auc:.4f}, saved to {path}")

    return m


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune and train the churn model")
    parser.add_argument("--trials", type=int, default=N_TRIALS, help="number of optuna trials")
    parser.add_argument("--version", type=int, default=MODEL_V, help="saved as models/model_v<version>.ubj")
    args = parser.parse_args()

    main(args.trials, args.version)
