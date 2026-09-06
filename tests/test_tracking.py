import unittest

from modeling.tracking import WandbTracker, _artifact_name


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


if __name__ == "__main__":
    unittest.main()
