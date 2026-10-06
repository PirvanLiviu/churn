import os
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocessor import encode  # noqa: E402

MODEL_V = 1
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / f"model_v{MODEL_V}.ubj"))
THRESHOLD = float(os.getenv("THRESHOLD", 0.5))  # same default as src/evaluate.py


@lru_cache
def load_model() -> xgb.XGBClassifier:
    m = xgb.XGBClassifier()
    m.load_model(MODEL_PATH)

    return m


def predict(customers: list[dict]) -> list[dict]:
    """Takes raw customer records (same fields as data/raw.csv, minus customerID and Churn)."""
    m = load_model()
    X = encode(pd.DataFrame(customers))
    # the model expects the exact column order it was trained on
    X = X[m.get_booster().feature_names]

    proba = m.predict_proba(X)[:, 1]

    return [
        {"churn_probability": float(p), "churn": bool(p >= THRESHOLD), "threshold": THRESHOLD}
        for p in proba
    ]
