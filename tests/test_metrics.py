import math
import unittest

from modeling.metrics import binary_metrics, predictions_from_probabilities


class MetricsTests(unittest.TestCase):
    def test_known_confusion_matrix(self) -> None:
        result = binary_metrics([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual((result["tp"], result["fp"], result["tn"], result["fn"]), (1, 1, 1, 1))
        self.assertAlmostEqual(result["tss"], 0.0)
        self.assertAlmostEqual(result["hss"], 0.0)

    def test_perfect_predictions(self) -> None:
        result = binary_metrics([0, 0, 1, 1], [0, 0, 1, 1])
        self.assertAlmostEqual(result["tss"], 1.0)
        self.assertAlmostEqual(result["hss"], 1.0)

    def test_one_class_returns_nan_instead_of_crashing(self) -> None:
        result = binary_metrics([0, 0], [0, 1])
        self.assertTrue(math.isnan(result["tss"]))

    def test_threshold_is_explicit(self) -> None:
        self.assertEqual(predictions_from_probabilities([0.49, 0.5], 0.5), [0, 1])


if __name__ == "__main__":
    unittest.main()

