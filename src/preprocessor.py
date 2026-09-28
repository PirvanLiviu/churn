import pandas as pd

def preprocessor():
    df: pd.DataFrame = pd.read_csv("../data/raw.csv")
    # dropping customer id since it makes no sense
    df.drop(columns=["customerID"], inplace=True)

    # encoding yes/no columns
    yn = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
    for col in yn:
        df[col] = df[col].map({"Yes": 1, "No": 0}) # type: ignore

    # encoding columns with 3 unique values
    other = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    df["MultipleLines"] = df["MultipleLines"].map({"Yes": 2, "No": 1, "No phone service": 0}) # type: ignore
    df["InternetService"] = df["InternetService"].map({"DSL": 2, "Fiber optic": 1, "No": 0}) # type: ignore
    for col in other:
        df[col] = df[col].map({"Yes": 2, "No": 1, "No internet service": 0}) # type: ignore
    df["Contract"] = df["Contract"].map({"Month-to-month": 0, "One year": 1, "Two year": 2}) # type: ignore
    df["PaymentMethod"] = df["PaymentMethod"].map({"Electronic check": 0, "Mailed check": 1, "Bank transfer (automatic)": 2, "Credit card (automatic)": 3}) # type: ignore
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0}) # type: ignore

    # remove white space (found in total charges column)
    for col in df.columns:
        df = df[df[col].astype(str).str.strip() != ""] # type: ignore

    df.to_csv("../data/preprocessed.csv", index=False)

preprocessor()
