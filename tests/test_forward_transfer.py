import unittest

from modeling.forward_transfer import calculate_forward_transfer


class ForwardTransferTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stage1 = {
            "evaluations": {
                "stage2": {"tss": 0.20, "hss": 0.10},
            }
        }
        self.random = {"tss": -0.05, "hss": 0.02}
        self.trained = {"tss": 0.55, "hss": 0.45}

    def test_forward_transfer_uses_random_baseline(self) -> None:
        result = calculate_forward_transfer(
            self.stage1, self.random, self.trained
        )
        self.assertAlmostEqual(result["metrics"]["tss"]["forward_transfer"], 0.25)
        self.assertAlmostEqual(result["metrics"]["hss"]["forward_transfer"], 0.08)
        self.assertEqual(
            result["metrics"]["tss"]["stage2_only_after_training"], 0.55
        )

    def test_undefined_score_propagates(self) -> None:
        self.random["tss"] = None
        result = calculate_forward_transfer(
            self.stage1, self.random, self.trained
        )
        self.assertIsNone(result["metrics"]["tss"]["forward_transfer"])

    def test_missing_zero_shot_evaluation_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "R_1,2"):
            calculate_forward_transfer(
                {"evaluations": {}}, self.random, self.trained
            )


if __name__ == "__main__":
    unittest.main()
