import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from helpers import PREPROCESSED_CSV, RAW_CSV, cwd, import_fresh

EXPECTED_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]

# allowed encoded values per column, matching the mappings in preprocessor.py
ENCODED_VALUES = {
    "gender": {0, 1},
    "SeniorCitizen": {0, 1},
    "Partner": {0, 1},
    "Dependents": {0, 1},
    "PhoneService": {0, 1},
    "PaperlessBilling": {0, 1},
    "Churn": {0, 1},
    "MultipleLines": {0, 1, 2},
    "InternetService": {0, 1, 2},
    "OnlineSecurity": {0, 1, 2},
    "OnlineBackup": {0, 1, 2},
    "DeviceProtection": {0, 1, 2},
    "TechSupport": {0, 1, 2},
    "StreamingTV": {0, 1, 2},
    "StreamingMovies": {0, 1, 2},
    "Contract": {0, 1, 2},
    "PaymentMethod": {0, 1, 2, 3},
}


class TestPreprocessor(unittest.TestCase):
    """Runs preprocessor.py on a sample of raw.csv inside a temp folder, so the real data is never overwritten."""

    @classmethod
    def setUpClass(cls):
        raw = pd.read_csv(RAW_CSV)
        blank = raw[raw["TotalCharges"].str.strip() == ""]
        cls.raw = pd.concat([raw.head(100), blank]).reset_index(drop=True)
        cls.n_blank = len(blank)

        cls.tmp = Path(tempfile.mkdtemp())
        (cls.tmp / "data").mkdir()
        (cls.tmp / "src").mkdir()
        cls.raw.to_csv(cls.tmp / "data" / "raw.csv", index=False)

        preprocessor = import_fresh("preprocessor")
        with cwd(cls.tmp / "src"):
            preprocessor.preprocessor()
        cls.out = pd.read_csv(cls.tmp / "data" / "preprocessed.csv")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_sample_contains_blank_total_charges(self):
        self.assertGreater(self.n_blank, 0)

    def test_drops_customer_id(self):
        self.assertNotIn("customerID", self.out.columns)

    def test_output_columns(self):
        self.assertEqual(list(self.out.columns), EXPECTED_COLUMNS)

    def test_removes_blank_total_charges_rows(self):
        self.assertEqual(len(self.out), len(self.raw) - self.n_blank)

    def test_no_missing_values(self):
        self.assertFalse(self.out.isna().any().any())

    def test_all_columns_numeric(self):
        for col in self.out.columns:
            with self.subTest(col=col):
                self.assertTrue(pd.api.types.is_numeric_dtype(self.out[col]))

    def test_encoded_values_in_range(self):
        for col, allowed in ENCODED_VALUES.items():
            with self.subTest(col=col):
                self.assertTrue(set(self.out[col].unique()) <= allowed)

    def test_first_row_encoding(self):
        # 7590-VHVEG: Female, Partner Yes, no phone service, DSL, Month-to-month, Electronic check, Churn No
        row = self.out.iloc[0]
        self.assertEqual(row["gender"], 0)
        self.assertEqual(row["Partner"], 1)
        self.assertEqual(row["PhoneService"], 0)
        self.assertEqual(row["MultipleLines"], 0)
        self.assertEqual(row["InternetService"], 2)
        self.assertEqual(row["OnlineBackup"], 2)
        self.assertEqual(row["Contract"], 0)
        self.assertEqual(row["PaymentMethod"], 0)
        self.assertEqual(row["Churn"], 0)
        self.assertAlmostEqual(row["TotalCharges"], 29.85)


class TestPreprocessedData(unittest.TestCase):
    """Sanity checks on the real data/preprocessed.csv that the model is trained on."""

    @classmethod
    def setUpClass(cls):
        if not PREPROCESSED_CSV.exists():
            raise unittest.SkipTest(f"{PREPROCESSED_CSV} not found, run preprocessor.py first")
        cls.df = pd.read_csv(PREPROCESSED_CSV)
        cls.raw = pd.read_csv(RAW_CSV)

    def test_columns(self):
        self.assertEqual(list(self.df.columns), EXPECTED_COLUMNS)

    def test_row_count(self):
        blank = (self.raw["TotalCharges"].str.strip() == "").sum()
        self.assertEqual(len(self.df), len(self.raw) - blank)

    def test_no_missing_values(self):
        self.assertFalse(self.df.isna().any().any())

    def test_encoded_values_in_range(self):
        for col, allowed in ENCODED_VALUES.items():
            with self.subTest(col=col):
                self.assertTrue(set(self.df[col].unique()) <= allowed)

    def test_numeric_ranges(self):
        self.assertTrue((self.df["tenure"] >= 0).all())
        self.assertTrue((self.df["MonthlyCharges"] > 0).all())
        self.assertTrue((self.df["TotalCharges"] > 0).all())

    def test_both_classes_present(self):
        rate = self.df["Churn"].mean()
        self.assertGreater(rate, 0.1)
        self.assertLess(rate, 0.5)


if __name__ == "__main__":
    unittest.main()
