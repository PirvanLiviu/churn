import unittest

import numpy as np

from helpers import MODEL_FILE, PREPROCESSED_CSV, SRC, cwd, import_fresh

# minimum scores on the held-out test set (model_v1 currently gets roc_auc 0.845, accuracy 0.796)
MIN_ROC_AUC = 0.80
MIN_ACCURACY = 0.75
MIN_F1 = 0.45
MAX_OVERFIT_GAP = 0.05  # max allowed train - test roc_auc


@unittest.skipUnless(PREPROCESSED_CSV.exists(), "data/preprocessed.csv not found")
@unittest.skipUnless(MODEL_FILE.exists(), "models/model_v1.ubj not found")
class TestModelQuality(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ev = import_fresh("evaluate")
        with cwd(SRC):
            cls.model = cls.ev.load_model()
        cls.train_metrics = cls.ev.evaluate(cls.model, cls.ev.X_train, cls.ev.y_train)
        cls.test_metrics = cls.ev.evaluate(cls.model, cls.ev.X_test, cls.ev.y_test)

    def test_metrics_keys(self):
        expected = {"accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix", "report"}
        self.assertEqual(set(self.test_metrics), expected)

    def test_metrics_between_0_and_1(self):
        for name in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            with self.subTest(metric=name):
                self.assertGreaterEqual(self.test_metrics[name], 0.0)
                self.assertLessEqual(self.test_metrics[name], 1.0)

    def test_roc_auc(self):
        self.assertGreaterEqual(self.test_metrics["roc_auc"], MIN_ROC_AUC)

    def test_accuracy(self):
        self.assertGreaterEqual(self.test_metrics["accuracy"], MIN_ACCURACY)

    def test_f1(self):
        self.assertGreaterEqual(self.test_metrics["f1"], MIN_F1)

    def test_beats_majority_baseline(self):
        baseline = 1 - self.ev.y_test.mean()  # always predicting "no churn"
        self.assertGreater(self.test_metrics["accuracy"], baseline)

    def test_not_overfitting(self):
        gap = self.train_metrics["roc_auc"] - self.test_metrics["roc_auc"]
        self.assertLess(gap, MAX_OVERFIT_GAP)

    def test_confusion_matrix_matches_test_set(self):
        cm = self.test_metrics["confusion_matrix"]
        self.assertEqual(cm.shape, (2, 2))
        self.assertEqual(cm.sum(), len(self.ev.y_test))
        # first row = actual no churn, second row = actual churn
        self.assertEqual(cm[1].sum(), self.ev.y_test.sum())

    def test_predict_proba_valid(self):
        proba = self.model.predict_proba(self.ev.X_test)
        self.assertEqual(proba.shape, (len(self.ev.X_test), 2))
        self.assertTrue(np.allclose(proba.sum(axis=1), 1.0))

    def test_lower_threshold_increases_recall(self):
        low = self.ev.evaluate(self.model, self.ev.X_test, self.ev.y_test, threshold=0.3)
        self.assertGreaterEqual(low["recall"], self.test_metrics["recall"])

    def test_threshold_does_not_change_roc_auc(self):
        low = self.ev.evaluate(self.model, self.ev.X_test, self.ev.y_test, threshold=0.3)
        self.assertAlmostEqual(low["roc_auc"], self.test_metrics["roc_auc"])

    def test_predictions_are_deterministic(self):
        first = self.model.predict_proba(self.ev.X_test.head(50))
        second = self.model.predict_proba(self.ev.X_test.head(50))
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
