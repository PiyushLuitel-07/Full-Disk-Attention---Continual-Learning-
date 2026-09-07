import unittest

from modeling.continual_metrics import (
    calculate_continual_metrics,
    calculate_two_stage_metrics,
)


class ContinualMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stage1 = {
            "evaluations": {
                "stage1": {"tss": 0.60, "hss": 0.40},
                "stage2": {"tss": 0.10, "hss": 0.05},
            }
        }
        self.stage2 = {
            "evaluations": {
                "stage1": {"tss": 0.50, "hss": 0.35},
                "stage2": {"tss": 0.55, "hss": 0.45},
            }
        }

    def test_two_stage_tss_metrics(self) -> None:
        result = calculate_two_stage_metrics(self.stage1, self.stage2)
        tss = result["continual_metrics"]["tss"]
        self.assertAlmostEqual(tss["final_average"], 0.525)
        self.assertAlmostEqual(tss["average_incremental_performance"], 0.5625)
        self.assertAlmostEqual(tss["forgetting"], 0.10)
        self.assertAlmostEqual(tss["backward_transfer"], -0.10)
        self.assertAlmostEqual(tss["stage2_gain"], 0.45)

    def test_undefined_base_metric_propagates(self) -> None:
        self.stage2["evaluations"]["stage1"]["tss"] = None
        result = calculate_two_stage_metrics(self.stage1, self.stage2)
        tss = result["continual_metrics"]["tss"]
        self.assertIsNone(tss["final_average"])
        self.assertIsNone(tss["forgetting"])
        self.assertIsNone(tss["backward_transfer"])

    def test_missing_matrix_cell_is_rejected(self) -> None:
        del self.stage1["evaluations"]["stage2"]
        with self.assertRaisesRegex(ValueError, "Missing R_1,2"):
            calculate_two_stage_metrics(self.stage1, self.stage2)

    def test_four_stage_metrics(self) -> None:
        rows = (
            (0.60, 0.10, 0.05, 0.00),
            (0.50, 0.55, 0.15, 0.10),
            (0.48, 0.52, 0.58, 0.20),
            (0.45, 0.50, 0.54, 0.60),
        )
        summaries = {
            trained: {
                "evaluations": {
                    f"stage{evaluated}": {"tss": score, "hss": score}
                    for evaluated, score in enumerate(row, start=1)
                }
            }
            for trained, row in enumerate(rows, start=1)
        }

        result = calculate_continual_metrics(summaries)
        tss = result["continual_metrics"]["tss"]

        self.assertEqual(result["number_of_stages"], 4)
        self.assertAlmostEqual(tss["final_average"], 0.5225)
        self.assertAlmostEqual(
            tss["average_incremental_performance"],
            (0.60 + 0.525 + (1.58 / 3.0) + 0.5225) / 4.0,
        )
        self.assertAlmostEqual(tss["forgetting"], 0.08)
        self.assertAlmostEqual(tss["backward_transfer"], -0.08)
        self.assertAlmostEqual(tss["average_learning_gain"], 1.28 / 3.0)
        self.assertAlmostEqual(tss["learning_gain_by_stage"]["stage4"], 0.40)
        self.assertAlmostEqual(tss["final_stage_after_training"], 0.60)

    def test_forgetting_uses_best_pre_final_score(self) -> None:
        summaries = {
            1: {"evaluations": {"stage1": {"tss": 0.50, "hss": 0.50}, "stage2": {"tss": 0.0, "hss": 0.0}, "stage3": {"tss": 0.0, "hss": 0.0}}},
            2: {"evaluations": {"stage1": {"tss": 0.60, "hss": 0.60}, "stage2": {"tss": 0.50, "hss": 0.50}, "stage3": {"tss": 0.0, "hss": 0.0}}},
            3: {"evaluations": {"stage1": {"tss": 0.55, "hss": 0.55}, "stage2": {"tss": 0.50, "hss": 0.50}, "stage3": {"tss": 0.50, "hss": 0.50}}},
        }
        tss = calculate_continual_metrics(summaries)["continual_metrics"]["tss"]
        self.assertAlmostEqual(tss["forgetting_by_stage"]["stage1"], 0.05)
        self.assertAlmostEqual(tss["backward_transfer_by_stage"]["stage1"], 0.05)


if __name__ == "__main__":
    unittest.main()
