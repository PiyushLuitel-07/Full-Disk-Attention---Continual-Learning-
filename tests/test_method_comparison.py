import copy
import unittest

from modeling.method_comparison import compare_ewc_with_finetune


def summary(method: str, forgetting: float, final: float, stage2: float) -> dict:
    return {
        "method": method,
        "comparison_controls": {
            "seed": 4,
            "outer_fold": 3,
            "stages": [1, 2],
            "model": {"attention": True},
            "training": {"epochs_per_stage": 10},
            "manifest_dir": "data/manifests/fold3",
            "image_root": "/data/hmi_jpgs_512",
        },
        "continual_metrics": {
            metric: {
                "forgetting": forgetting,
                "final_average": final,
                "stage2_after_training": stage2,
                "backward_transfer": -forgetting,
            }
            for metric in ("tss", "hss")
        },
    }


class MethodComparisonTests(unittest.TestCase):
    def test_positive_forgetting_reduction_favors_ewc(self) -> None:
        result = compare_ewc_with_finetune(
            summary("ewc", forgetting=0.08, final=0.50, stage2=0.52),
            summary("finetune", forgetting=0.20, final=0.45, stage2=0.54),
        )
        tss = result["metrics"]["tss"]
        self.assertAlmostEqual(tss["forgetting_reduction"], 0.12)
        self.assertAlmostEqual(tss["final_average_difference"], 0.05)
        self.assertAlmostEqual(tss["stage2_performance_difference"], -0.02)
        self.assertAlmostEqual(tss["final_stage_performance_difference"], -0.02)

    def test_mismatched_controls_are_rejected(self) -> None:
        ewc = summary("ewc", 0.08, 0.50, 0.52)
        finetune = summary("finetune", 0.20, 0.45, 0.54)
        finetune = copy.deepcopy(finetune)
        finetune["comparison_controls"]["seed"] = 5
        with self.assertRaisesRegex(ValueError, "controls differ"):
            compare_ewc_with_finetune(ewc, finetune)

    def test_four_stage_comparison_uses_generic_final_stage_metric(self) -> None:
        ewc = summary("ewc", 0.08, 0.50, 0.52)
        finetune = summary("finetune", 0.20, 0.45, 0.54)
        for payload, final, incremental, gain in (
            (ewc, 0.48, 0.51, 0.30),
            (finetune, 0.50, 0.47, 0.34),
        ):
            payload["comparison_controls"]["stages"] = [1, 2, 3, 4]
            for metric in ("tss", "hss"):
                values = payload["continual_metrics"][metric]
                values["final_stage_after_training"] = final
                values["average_incremental_performance"] = incremental
                values["average_learning_gain"] = gain

        result = compare_ewc_with_finetune(ewc, finetune)
        tss = result["metrics"]["tss"]
        self.assertEqual(result["number_of_stages"], 4)
        self.assertAlmostEqual(tss["final_stage_performance_difference"], -0.02)
        self.assertAlmostEqual(
            tss["average_incremental_performance_difference"], 0.04
        )
        self.assertAlmostEqual(tss["average_learning_gain_difference"], -0.04)
        self.assertNotIn("stage2_performance_difference", tss)


if __name__ == "__main__":
    unittest.main()
