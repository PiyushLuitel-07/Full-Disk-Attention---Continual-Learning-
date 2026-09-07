import unittest

from modeling.continual_metrics import calculate_continual_metrics
from modeling.tracking import WandbTracker, _artifact_name


class FakeTable:
    def __init__(self, columns: list[str]) -> None:
        self.columns = columns
        self.rows: list[tuple] = []

    def add_data(self, *values: object) -> None:
        self.rows.append(values)


class FakeWandb:
    Table = FakeTable


class FakeRun:
    def __init__(self) -> None:
        self.summary: dict[str, object] = {}
        self.logged: list[dict[str, object]] = []

    def log(self, payload: dict[str, object]) -> None:
        self.logged.append(payload)


class TrackingTests(unittest.TestCase):
    def test_disabled_tracking_does_not_require_wandb_import(self) -> None:
        tracker = WandbTracker.start({"tracking": {"enabled": False}}, 1, "cpu")
        self.assertFalse(tracker.enabled)

    def test_artifact_name_is_safe(self) -> None:
        value = "Full disk attention / fold 3: stage sequence"
        self.assertEqual(
            _artifact_name(value),
            "Full-disk-attention-fold-3-stage-sequence",
        )

    def test_four_stage_continual_metrics_are_logged(self) -> None:
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
        run = FakeRun()
        tracker = WandbTracker(run, FakeWandb(), "test", False)

        tracker.log_continual_metrics(result)

        self.assertEqual(len(run.logged), 1)
        table = run.logged[0]["cl/performance_matrix_and_summary"]
        self.assertEqual(len(table.columns), 22)
        self.assertEqual(len(table.rows), 2)
        self.assertIn("cl/tss_forgetting_by_stage/stage1", run.summary)
        self.assertIn("cl/hss_learning_gain_by_stage/stage4", run.summary)


if __name__ == "__main__":
    unittest.main()
