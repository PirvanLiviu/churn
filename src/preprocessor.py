from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw.csv"
PREPROCESSED_PATH = ROOT / "data" / "preprocessed.csv"

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
        if col not in df.columns:
            continue

        unknown = set(df[col].dropna().unique()) - set(mapping)
        if unknown:
            # .map() would silently turn these into NaN
            raise ValueError(f"unknown values in {col}: {sorted(unknown)}")
        df[col] = df[col].map(mapping)

    return df


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    # dropping customer id since it makes no sense
    df = raw.drop(columns=["customerID"])

    # TotalCharges is read as text because 11 new customers (tenure 0) have a blank value, drop those rows
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)

    df = encode(df)

    if df.isna().any().any():
        raise ValueError(f"missing values in: {df.columns[df.isna().any()].tolist()}")

    return df


def preprocessor(raw_path=RAW_PATH, out_path=PREPROCESSED_PATH) -> pd.DataFrame:
    raw = pd.read_csv(raw_path)
    df = clean(raw)
    df.to_csv(out_path, index=False)

    print(f"{len(raw)} raw rows -> {len(df)} rows ({len(raw) - len(df)} dropped), saved to {out_path}")

    return df


if __name__ == "__main__":
    preprocessor()
