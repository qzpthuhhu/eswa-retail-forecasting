"""Tests for the deep learning benchmark pipeline."""
import unittest
from pathlib import Path

import pandas as pd


REPORT_DIR = Path(__file__).parent / "reports" / "model_quality" / "deep_learning_benchmark"


class DeepLearningBenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictions = pd.read_csv(REPORT_DIR / "dl_benchmark_predictions.csv")
        cls.quality = pd.read_csv(REPORT_DIR / "dl_benchmark_quality.csv")

    def test_all_three_models_produced_predictions(self):
        methods = set(self.predictions["method"].unique())
        self.assertSetEqual(methods, {"deepar", "patchtst", "tft"})

    def test_prediction_count_per_model(self):
        for method in ["deepar", "patchtst", "tft"]:
            count = len(self.predictions[self.predictions["method"] == method])
            self.assertEqual(count, 8400, f"{method} should have 8400 predictions")

    def test_no_duplicate_predictions(self):
        dupes = self.predictions.duplicated(
            ["method", "forecast_origin", "date", "store_id", "product_id"]
        ).sum()
        self.assertEqual(dupes, 0, "No duplicate predictions allowed")

    def test_no_nan_or_negative_predictions(self):
        self.assertFalse(self.predictions["prediction"].isna().any())
        self.assertFalse((self.predictions["prediction"] < 0).any())

    def test_five_origins_per_model(self):
        for method in ["deepar", "patchtst", "tft"]:
            n_origins = self.predictions[
                self.predictions["method"] == method
            ]["forecast_origin"].nunique()
            self.assertEqual(n_origins, 5, f"{method} should have 5 origins")

    def test_seven_horizons_per_prediction(self):
        for method in ["deepar", "patchtst", "tft"]:
            horizons = self.predictions[
                self.predictions["method"] == method
            ]["horizon_day"].unique()
            self.assertEqual(set(horizons), {1, 2, 3, 4, 5, 6, 7})

    def test_all_models_have_mae_below_4(self):
        for _, row in self.quality.iterrows():
            self.assertLess(row["mae"], 4.0,
                            f"{row['method']} MAE {row['mae']} too high")

    def test_models_ranked_correctly(self):
        mae_by_method = dict(zip(self.quality["method"], self.quality["mae"]))
        # Phoen TST is best among DL models
        self.assertLess(mae_by_method["patchtst"], mae_by_method["deepar"])
        self.assertLess(mae_by_method["patchtst"], mae_by_method["tft"])

    def test_all_models_beat_naive_baselines(self):
        gt_qty = self.predictions["qty"].mean()
        mae_by_method = dict(zip(self.quality["method"], self.quality["mae"]))
        for _, v in mae_by_method.items():
            self.assertLess(v, gt_qty,
                            f"Model MAE {v} > mean qty {gt_qty}")


if __name__ == "__main__":
    unittest.main()