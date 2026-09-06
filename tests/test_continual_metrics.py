import unittest

from modeling.continual_metrics import calculate_two_stage_metrics


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


if __name__ == "__main__":
    unittest.main()
