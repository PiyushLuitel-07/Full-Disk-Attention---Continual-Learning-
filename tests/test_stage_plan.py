import copy
import json
import unittest
from pathlib import Path

from modeling.train_continual import load_config, validate_stage_plan


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class StagePlanTests(unittest.TestCase):
    def test_two_stage_config_remains_supported(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "pilot.json")
        self.assertEqual(validate_stage_plan(config), [1, 2])

    def test_four_stage_config_is_supported(self) -> None:
        config = load_config(
            PROJECT_ROOT / "configs" / "presentation_4stage_ewc.json"
        )
        self.assertEqual(validate_stage_plan(config), [1, 2, 3, 4])

    def test_nonconsecutive_stage_ids_are_rejected(self) -> None:
        config = json.loads(
            (PROJECT_ROOT / "configs" / "presentation_4stage_ewc.json").read_text(
                encoding="utf-8"
            )
        )
        invalid = copy.deepcopy(config)
        invalid["stages"][2]["id"] = 4
        with self.assertRaisesRegex(ValueError, "Stage IDs"):
            validate_stage_plan(invalid)

    def test_overlapping_year_ranges_are_rejected(self) -> None:
        config = json.loads(
            (PROJECT_ROOT / "configs" / "presentation_4stage_ewc.json").read_text(
                encoding="utf-8"
            )
        )
        invalid = copy.deepcopy(config)
        invalid["stages"][2]["start_year"] = 2014
        with self.assertRaisesRegex(ValueError, "non-overlapping"):
            validate_stage_plan(invalid)


if __name__ == "__main__":
    unittest.main()
