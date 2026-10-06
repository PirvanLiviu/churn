import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

MODEL_V = 1
THRESHOLD = 0.5  # probability above which a customer is predicted to churn

df = pd.read_csv("../data/preprocessed.csv")

# split into train and target data
X = df.drop(columns=["Churn"])
y = df["Churn"]

# same split as model.py, so we only evaluate on rows the model never trained on
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)


def load_model():
    m = xgb.XGBClassifier()
    m.load_model(f"../models/model_v{MODEL_V}.ubj")

    return m


def evaluate(m, X, y, threshold=THRESHOLD):
    proba = m.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds),
        "recall": recall_score(y, preds),
        "f1": f1_score(y, preds),
        "roc_auc": roc_auc_score(y, proba),
        "confusion_matrix": confusion_matrix(y, preds),
        "report": classification_report(y, preds, target_names=["No churn", "Churn"]),
    }


if __name__ == "__main__":
    m = load_model()
    train_metrics = evaluate(m, X_train, y_train)
    test_metrics = evaluate(m, X_test, y_test)

    print(f"Model v{MODEL_V} (threshold {THRESHOLD})\n")
    print(f"{'metric':<10}{'train':>8}{'test':>8}")
    for name in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"{name:<10}{train_metrics[name]:>8.3f}{test_metrics[name]:>8.3f}")

    # rows = actual, columns = predicted
    tn, fp, fn, tp = test_metrics["confusion_matrix"].ravel()
    print("\nTest confusion matrix")
    print(f"               pred no  pred yes")
    print(f"actual no   {tn:>10}{fp:>10}")
    print(f"actual yes  {fn:>10}{tp:>10}")

    print("\n" + test_metrics["report"])
