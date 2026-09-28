import optuna
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score 
import numpy as np

df = pd.read_csv("../data/preprocessed.csv")

MODEL_V = 1

# split into train and target data
X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

def objective(trial):
    params = {
        "objective": "binary:logistic",
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }

    model = xgb.XGBClassifier(**params)
    try:
        model.load_model(f"../models/model_v{MODEL_V}.ubj")
    except:
        pass
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    preds_proba = model.predict_proba(X_test)
    ras = np.sqrt(roc_auc_score(y_test, preds_proba[:, 1]))

    return ras

    
def best_params():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    # print("Best ras: ", study.best_value)
    # print("Best params: ", study.best_params)

    return study.best_params

def model():
    m = xgb.XGBClassifier(**best_params())
    try:
        m.load_model(f"../models/model_v{MODEL_V}.ubj")
    except:
        pass
    m.fit(X_train, y_train)
    m.save_model(f"../models/model_v{MODEL_V}.ubj")

    return m

if __name__ == "__main__":
    trained_model = model()
