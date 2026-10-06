import pandas as pd

YES_NO = {"Yes": 1, "No": 0}
INTERNET_ADDON = {"Yes": 2, "No": 1, "No internet service": 0}

# raw value -> encoded value for every categorical column
ENCODINGS = {
    # encoding yes/no columns
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "PaperlessBilling": YES_NO,
    "Churn": YES_NO,
    # encoding columns with 3 unique values
    "MultipleLines": {"Yes": 2, "No": 1, "No phone service": 0},
    "InternetService": {"DSL": 2, "Fiber optic": 1, "No": 0},
    "OnlineSecurity": INTERNET_ADDON,
    "OnlineBackup": INTERNET_ADDON,
    "DeviceProtection": INTERNET_ADDON,
    "TechSupport": INTERNET_ADDON,
    "StreamingTV": INTERNET_ADDON,
    "StreamingMovies": INTERNET_ADDON,
    "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2},
    "PaymentMethod": {"Electronic check": 0, "Mailed check": 1, "Bank transfer (automatic)": 2, "Credit card (automatic)": 3},
    "gender": {"Male": 1, "Female": 0},
}


def encode(df: pd.DataFrame) -> pd.DataFrame:
    # also used by the api, where there is no Churn column, so only encode columns that exist
    df = df.copy()
    for col, mapping in ENCODINGS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)  # type: ignore

    return df


def preprocessor():
    df: pd.DataFrame = pd.read_csv("../data/raw.csv")
    # dropping customer id since it makes no sense
    df.drop(columns=["customerID"], inplace=True)

    df = encode(df)

    # remove white space (found in total charges column)
    for col in df.columns:
        df = df[df[col].astype(str).str.strip() != ""] # type: ignore

    df.to_csv("../data/preprocessed.csv", index=False)


if __name__ == "__main__":
    preprocessor()
