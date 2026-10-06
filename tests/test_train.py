import importlib.util
import unittest
from unittest import mock

import xgboost as xgb

from helpers import MODELS, PREPROCESSED_CSV, SRC, cwd, import_fresh

HAS_OPTUNA = importlib.util.find_spec("optuna") is not None

# small params so the tests run fast
FAST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.1,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
}


@unittest.skipUnless(HAS_OPTUNA, "optuna is not installed")
@unittest.skipUnless(PREPROCESSED_CSV.exists(), "data/preprocessed.csv not found")
class TestTrain(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        cls.train = import_fresh("model")

    def test_split_sizes(self):
        t = self.train
        self.assertEqual(len(t.X_train) + len(t.X_test), len(t.df))
        self.assertAlmostEqual(len(t.X_test) / len(t.df), 0.25, places=2)

    def test_split_is_stratified(self):
        t = self.train
        self.assertAlmostEqual(t.y_train.mean(), t.y_test.mean(), places=2)

    def test_target_not_in_features(self):
        self.assertNotIn("Churn", self.train.X.columns)

    def test_split_has_no_overlap(self):
        t = self.train
        self.assertEqual(len(t.X_train.index.intersection(t.X_test.index)), 0)

    def test_objective_returns_valid_score(self):
        import optuna

        trial = optuna.trial.FixedTrial(FAST_PARAMS)
        with cwd(SRC):
            score = self.train.objective(trial)

        # objective returns sqrt(roc_auc), so a useful model lands between sqrt(0.5) and 1
        self.assertGreater(score, 0.5 ** 0.5)
        self.assertLessEqual(score, 1.0)

    def test_best_params_returns_search_space(self):
        # swap the real objective for a cheap one so the 50 trials run instantly
        def cheap_objective(trial):
            for name, low, high in [("n_estimators", 100, 1000), ("max_depth", 3, 10), ("min_child_weight", 1, 10)]:
                trial.suggest_int(name, low, high)
            trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
            return trial.suggest_float("colsample_bytree", 0.5, 1.0)

        with mock.patch.object(self.train, "objective", cheap_objective):
            params = self.train.best_params()

        self.assertEqual(set(params), set(FAST_PARAMS))
        self.assertTrue(100 <= params["n_estimators"] <= 1000)
        self.assertTrue(3 <= params["max_depth"] <= 10)
        self.assertTrue(0.01 <= params["learning_rate"] <= 0.3)
        self.assertTrue(0.5 <= params["colsample_bytree"] <= 1.0)
        self.assertTrue(1 <= params["min_child_weight"] <= 10)

    def test_model_trains_and_saves(self):
        # patch save_model so the real models/model_v1.ubj is not overwritten
        with mock.patch.object(self.train, "best_params", return_value=FAST_PARAMS), \
                mock.patch.object(xgb.XGBClassifier, "save_model") as save, \
                cwd(SRC):
            m = self.train.model()

        save.assert_called_once_with(f"../models/model_v{self.train.MODEL_V}.ubj")
        preds = m.predict(self.train.X_test)
        self.assertEqual(len(preds), len(self.train.X_test))
        self.assertTrue(set(preds) <= {0, 1})

    def test_saved_model_path_exists(self):
        self.assertTrue((MODELS / f"model_v{self.train.MODEL_V}.ubj").exists())


if __name__ == "__main__":
    unittest.main()
