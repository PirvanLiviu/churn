import unittest

from fastapi.testclient import TestClient

from helpers import MODEL_FILE

from api.main import app
from api.schemas import Customer
from api.ui import _predict

EXAMPLE = Customer.model_config["json_schema_extra"]["example"]


@unittest.skipUnless(MODEL_FILE.exists(), "models/model_v1.ubj not found")
class TestApi(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

    def test_root_redirects_to_ui(self):
        r = self.client.get("/", follow_redirects=False)
        self.assertIn(r.status_code, (302, 307))
        self.assertEqual(r.headers["location"], "/ui")

    def test_ui_is_served(self):
        r = self.client.get("/ui/")
        self.assertEqual(r.status_code, 200)

    def test_predict(self):
        r = self.client.post("/predict", json=EXAMPLE)
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(set(body), {"churn_probability", "churn", "threshold"})
        self.assertTrue(0 <= body["churn_probability"] <= 1)
        self.assertEqual(body["churn"], body["churn_probability"] >= body["threshold"])

    def test_risky_customer_scores_higher(self):
        # long two-year contract vs. brand new month-to-month fiber customer
        loyal = {**EXAMPLE, "tenure": 70, "Contract": "Two year", "PaymentMethod": "Credit card (automatic)", "TotalCharges": 2000}
        risky = {**EXAMPLE, "tenure": 1, "Contract": "Month-to-month", "InternetService": "Fiber optic", "MonthlyCharges": 95}
        p_loyal = self.client.post("/predict", json=loyal).json()["churn_probability"]
        p_risky = self.client.post("/predict", json=risky).json()["churn_probability"]
        self.assertGreater(p_risky, p_loyal)

    def test_batch_matches_single(self):
        other = {**EXAMPLE, "tenure": 40, "Contract": "One year"}
        batch = self.client.post("/predict/batch", json=[EXAMPLE, other]).json()
        self.assertEqual(len(batch), 2)
        single = self.client.post("/predict", json=other).json()
        self.assertAlmostEqual(batch[1]["churn_probability"], single["churn_probability"], places=6)

    def test_empty_batch(self):
        r = self.client.post("/predict/batch", json=[])
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), [])

    def test_invalid_category_rejected(self):
        r = self.client.post("/predict", json={**EXAMPLE, "Contract": "Weekly"})
        self.assertEqual(r.status_code, 422)

    def test_negative_tenure_rejected(self):
        r = self.client.post("/predict", json={**EXAMPLE, "tenure": -1})
        self.assertEqual(r.status_code, 422)

    def test_missing_field_rejected(self):
        body = {k: v for k, v in EXAMPLE.items() if k != "gender"}
        r = self.client.post("/predict", json=body)
        self.assertEqual(r.status_code, 422)


@unittest.skipUnless(MODEL_FILE.exists(), "models/model_v1.ubj not found")
class TestUi(unittest.TestCase):

    def test_ui_predict_returns_label(self):
        out = _predict(*EXAMPLE.values())
        self.assertEqual(set(out), {"Churn", "No churn"})
        self.assertAlmostEqual(out["Churn"] + out["No churn"], 1.0)


if __name__ == "__main__":
    unittest.main()
