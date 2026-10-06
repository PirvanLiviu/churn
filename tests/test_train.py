import contextlib
import importlib.util
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import xgboost as xgb

from helpers import PREPROCESSED_CSV, import_fresh

HAS_OPTUNA = importlib.util.find_spec("optuna") is not None

# small params so the tests run fast
FAST_PARAMS = {
    "n_estimators": 50,
    "max_depth": 3,
    "learning_rate": 0.1,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "min_child_weight": 1,
    "reg_lambda": 1.0,
}


@unittest.skipUnless(HAS_OPTUNA, "optuna is not installed")
@unittest.skipUnless(PREPROCESSED_CSV.exists(), "data/preprocessed.csv not found")
class TestTrain(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        cls.train = import_fresh("model")
        cls.X, cls.y = cls.train.load_data()
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = cls.train.split(cls.X, cls.y)

    def test_target_not_in_features(self):
        self.assertNotIn("Churn", self.X.columns)
        self.assertEqual(len(self.X), len(self.y))

    def test_split_sizes(self):
        self.assertEqual(len(self.X_train) + len(self.X_test), len(self.X))
        self.assertAlmostEqual(len(self.X_test) / len(self.X), 0.25, places=2)

    def test_split_is_stratified(self):
        self.assertAlmostEqual(self.y_train.mean(), self.y_test.mean(), places=2)

    def test_split_has_no_overlap(self):
        self.assertEqual(len(self.X_train.index.intersection(self.X_test.index)), 0)

    def test_split_is_reproducible(self):
        X_train, *_ = self.train.split(self.X, self.y)
        self.assertTrue(X_train.index.equals(self.X_train.index))

    def test_split_matches_evaluate(self):
        # evaluate.py must score the model on the same rows that training held out
        ev = import_fresh("evaluate")
        self.assertTrue(ev.X_test.index.equals(self.X_test.index))

    def test_objective_returns_cv_roc_auc(self):
        import optuna

        score = self.train.objective(optuna.trial.FixedTrial(FAST_PARAMS), self.X_train, self.y_train)
        self.assertGreater(score, 0.75)
        self.assertLessEqual(score, 1.0)

    def test_best_params_within_search_space(self):
        with contextlib.redirect_stdout(io.StringIO()):
            params = self.train.best_params(self.X_train.head(500), self.y_train.head(500), n_trials=2)

        self.assertEqual(set(params), set(FAST_PARAMS))
        self.assertTrue(100 <= params["n_estimators"] <= 1000)
        self.assertTrue(3 <= params["max_depth"] <= 10)
        self.assertTrue(0.01 <= params["learning_rate"] <= 0.3)
        self.assertTrue(0.5 <= params["colsample_bytree"] <= 1.0)
        self.assertTrue(0.5 <= params["subsample"] <= 1.0)
        self.assertTrue(1 <= params["min_child_weight"] <= 10)
        self.assertTrue(1e-3 <= params["reg_lambda"] <= 10.0)

    def test_train_is_reproducible(self):
        a = self.train.train(self.X_train, self.y_train, FAST_PARAMS).predict_proba(self.X_test)
        b = self.train.train(self.X_train, self.y_train, FAST_PARAMS).predict_proba(self.X_test)
        self.assertTrue((a == b).all())

    def test_save_writes_model_and_params(self):
        m = self.train.train(self.X_train, self.y_train, FAST_PARAMS)
        tmp = Path(tempfile.mkdtemp())
        try:
            path = self.train.save(m, FAST_PARAMS, version=99, models_dir=tmp)
            self.assertEqual(path, tmp / "model_v99.ubj")
            self.assertEqual(json.loads((tmp / "model_v99_params.json").read_text()), FAST_PARAMS)

            loaded = xgb.XGBClassifier()
            loaded.load_model(path)
            self.assertTrue((loaded.predict_proba(self.X_test) == m.predict_proba(self.X_test)).all())
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
