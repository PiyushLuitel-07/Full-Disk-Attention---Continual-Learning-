"""Print the two-stage TSS/HSS retention table from one run directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    summaries = {}
    for stage in (1, 2):
        path = args.run_dir / "metrics" / f"stage{stage}_summary.json"
        if path.exists():
            summaries[stage] = json.loads(path.read_text(encoding="utf-8"))

    if not summaries:
        raise FileNotFoundError(f"No stage summaries found below {args.run_dir}")
    print("metric,after_stage1_eval_stage1,after_stage2_eval_stage1,after_stage2_eval_stage2")
    for metric in ("tss", "hss"):
        first = summaries.get(1, {}).get("evaluations", {}).get("stage1", {}).get(metric, "")
        old_after_new = summaries.get(2, {}).get("evaluations", {}).get("stage1", {}).get(metric, "")
        current = summaries.get(2, {}).get("evaluations", {}).get("stage2", {}).get(metric, "")
        print(f"{metric},{first},{old_after_new},{current}")
        if isinstance(first, (int, float)) and isinstance(old_after_new, (int, float)):
            print(f"{metric}_forgetting,{first - old_after_new}")


if __name__ == "__main__":
    main()

