import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> dict:
    return json.loads((PROJECT_ROOT / "configs" / name).read_text(encoding="utf-8"))


def controls(config: dict) -> dict:
    return {
        "seed": config["experiment"]["seed"],
        "outer_fold": config["experiment"]["outer_fold"],
        "stages": config["stages"],
        "model": config["model"],
        "training": config["training"],
        "manifest_dir": config["paths"]["manifest_dir"],
        "image_root": config["paths"]["image_root"],
    }


class MethodConfigTests(unittest.TestCase):
    def test_pilot_configs_are_matched(self) -> None:
        expected = controls(load("pilot.json"))
        self.assertEqual(expected, controls(load("pilot_finetune.json")))
        self.assertEqual(expected, controls(load("pilot_stage2_reference.json")))

    def test_server_templates_are_matched(self) -> None:
        expected = controls(load("server_template.json"))
        self.assertEqual(expected, controls(load("server_finetune_template.json")))
        self.assertEqual(
            expected, controls(load("server_stage2_reference_template.json"))
        )


if __name__ == "__main__":
    unittest.main()
