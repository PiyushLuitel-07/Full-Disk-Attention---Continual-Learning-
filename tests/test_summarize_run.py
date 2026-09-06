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


if __name__ == "__main__":
    unittest.main()
