import csv
import unittest
from pathlib import Path

from data_labeling.make_stage_manifests import STAGES, Row, deterministic_cap


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = PROJECT_ROOT / "data" / "manifests" / "fold3_pilot"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class ManifestTests(unittest.TestCase):
    def test_four_stage_plan_covers_2010_through_2018(self) -> None:
        self.assertEqual(
            STAGES,
            {
                1: (2010, 2012),
                2: (2013, 2014),
                3: (2015, 2016),
                4: (2017, 2018),
            },
        )

    def test_cap_keeps_all_rows_when_a_class_has_fewer_than_limit(self) -> None:
        rows = [
            Row("2017/01/01/HMI.m2017.01.01_00.00.00.jpg", 0),
            Row("2017/01/01/HMI.m2017.01.01_01.00.00.jpg", 0),
            Row("2017/01/01/HMI.m2017.01.01_02.00.00.jpg", 1),
        ]
        selected = deterministic_cap(rows, per_class=2, seed=4)
        self.assertEqual(sum(row.target == 0 for row in selected), 2)
        self.assertEqual(sum(row.target == 1 for row in selected), 1)

    def test_stage_years_and_disjointness(self) -> None:
        paths_by_stage = []
        for stage, expected_years in ((1, {2010, 2011, 2012}), (2, {2013, 2014})):
            train = read(MANIFEST_ROOT / f"stage{stage}_train.csv")
            evaluation = read(MANIFEST_ROOT / f"stage{stage}_eval.csv")
            fisher = read(MANIFEST_ROOT / f"stage{stage}_fisher.csv")
            train_paths = {row["label"] for row in train}
            eval_paths = {row["label"] for row in evaluation}
            self.assertFalse(train_paths & eval_paths)
            self.assertTrue({row["label"] for row in fisher}.issubset(train_paths))
            observed_years = {
                int(row["label"].split("/", 1)[0]) for row in train + evaluation
            }
            self.assertTrue(observed_years.issubset(expected_years))
            paths_by_stage.append(train_paths | eval_paths)
        self.assertFalse(paths_by_stage[0] & paths_by_stage[1])


if __name__ == "__main__":
    unittest.main()
