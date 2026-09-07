import unittest

from tools.summarize_run import format_summary


class SummarizeRunTests(unittest.TestCase):
    def test_complete_two_stage_matrix_and_forgetting(self) -> None:
        summaries = {
            1: {
                "evaluations": {
                    "stage1": {"tss": 0.50, "hss": 0.40},
                    "stage2": {"tss": 0.10, "hss": 0.05},
                }
            },
            2: {
                "evaluations": {
                    "stage1": {"tss": 0.42, "hss": 0.30},
                    "stage2": {"tss": 0.55, "hss": 0.45},
                }
            },
        }

        lines = format_summary(summaries)

        self.assertIn("tss,0.5,0.1,0.42,0.55", lines)
        self.assertIn("hss,0.4,0.05,0.3,0.45", lines)
        self.assertIn("tss_forgetting,0.08000000000000002", lines)
        self.assertIn("hss_forgetting,0.10000000000000003", lines)

    def test_complete_four_stage_matrix_and_aggregates(self) -> None:
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

        lines = format_summary(summaries)

        self.assertEqual(lines[0].count("after_stage"), 16)
        self.assertIn(
            "tss,0.6,0.1,0.05,0.0,0.5,0.55,0.15,0.1,"
            "0.48,0.52,0.58,0.2,0.45,0.5,0.54,0.6",
            lines,
        )
        forgetting_line = next(line for line in lines if line.startswith("forgetting,"))
        _, tss_forgetting, hss_forgetting = forgetting_line.split(",")
        self.assertAlmostEqual(float(tss_forgetting), 0.08)
        self.assertAlmostEqual(float(hss_forgetting), 0.08)
        self.assertIn("learning_gain_stage4,0.39999999999999997,0.39999999999999997", lines)

    def test_legacy_incomplete_matrix_prints_without_formal_metrics(self) -> None:
        summaries = {
            1: {"evaluations": {"stage1": {"tss": 0.4, "hss": 0.3}}},
            2: {
                "evaluations": {
                    "stage1": {"tss": 0.2, "hss": 0.1},
                    "stage2": {"tss": 0.5, "hss": 0.4},
                }
            },
        }

        lines = format_summary(summaries)

        self.assertIn("tss,0.4,,0.2,0.5", lines)
        self.assertIn("tss_forgetting,0.2", lines)
        self.assertNotIn("continual_metric,tss,hss", lines)


if __name__ == "__main__":
    unittest.main()
