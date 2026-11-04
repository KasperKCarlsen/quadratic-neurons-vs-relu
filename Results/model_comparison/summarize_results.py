"""Build comparison tables from finalized repeated-seed evaluation summaries."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "repeated_seed_evaluation" / "Done_models"
OUTPUT = Path(__file__).resolve().parent


def load_rows() -> list[dict[str, object]]:
    rows = []
    for path in sorted(ROOT.glob("*/seed_results.json")):
        data = json.loads(path.read_text())
        config = path.parent.name
        rows.append(
            {
                "configuration": config,
                "mean_accuracy": data.get("mean"),
                "std_accuracy": data.get("std"),
                "hpo_best_value": data.get("hpo_best_value"),
                "seeds": len(data.get("seeds", [])),
                "mean_time_to_best": data.get("mean_time_to_best"),
                "std_time_to_best": data.get("std_time_to_best"),
            }
        )
    return rows


def main() -> None:
    rows = load_rows()
    if not rows:
        raise SystemExit(f"No seed_results.json files found below {ROOT}")

    fields = list(rows[0])
    with (OUTPUT / "model_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    ranked = sorted(
        rows,
        key=lambda row: float(row["mean_accuracy"] or float("-inf")),
        reverse=True,
    )
    (OUTPUT / "model_comparison.json").write_text(json.dumps(ranked, indent=2) + "\n")


if __name__ == "__main__":
    main()
