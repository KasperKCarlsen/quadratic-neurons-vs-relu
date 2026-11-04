"""Inspect the Optuna HPO results without retraining anything.

For every study in optuna_databases/, prints the best trial (highest test
accuracy) and its hyperparameters, plus the top-N trials so you can see the
ranking. Optionally dumps each study's full trial table to a CSV.

Usage:
    python inspect_studies.py                 # best + top 5 per study
    python inspect_studies.py --top 10        # show more trials
    python inspect_studies.py --csv reports   # also write reports/<study>.csv
"""

import os
import glob
import argparse
import sys
from pathlib import Path

import optuna

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def study_name_from_db(db_path):
    """Recover the in-DB study_name from the filename. Train_Model.py names the
    study '{model}_with_ReLU{relu}_optimization' and the file '<name_base>_study.db'
    where name_base = '{model}_with_ReLU{relu}_arc_{num_blocks}'."""
    name_base = os.path.basename(db_path)[: -len("_study.db")]
    model, rest = name_base.split("_with_ReLU", 1)
    relu = rest.split("_arc_", 1)[0]
    return name_base, f"{model}_with_ReLU{relu}_optimization"


def main():
    parser = argparse.ArgumentParser(description="Inspect Optuna HPO studies.")
    parser.add_argument("--top", type=int, default=5,
                        help="How many top trials to list per study.")
    parser.add_argument("--csv", type=str, default=None,
                        help="Directory to write per-study CSVs of all trials.")
    args = parser.parse_args()

    if args.csv:
        os.makedirs(args.csv, exist_ok=True)

    db_paths = sorted(glob.glob("HPO/optuna_databases/*_study.db"))
    print(f"Found {len(db_paths)} study DB(s).\n")

    for db_path in db_paths:
        name_base, study_name = study_name_from_db(db_path)
        study = optuna.load_study(study_name=study_name,
                                  storage=f"sqlite:///{db_path}")

        completed = [t for t in study.trials
                     if t.value is not None]  # skip pruned/failed
        completed.sort(key=lambda t: t.value, reverse=True)

        print(f"=== {name_base} ===")
        print(f"  trials: {len(study.trials)} total, {len(completed)} completed")
        print(f"  BEST  acc={study.best_value:.4f}  params={study.best_params}")
        print(f"  top {min(args.top, len(completed))}:")
        for rank, t in enumerate(completed[:args.top], 1):
            print(f"    {rank:2d}. acc={t.value:6.4f}  trial#{t.number:<3d} {t.params}")

        if args.csv:
            csv_path = os.path.join(args.csv, f"{name_base}.csv")
            study.trials_dataframe().to_csv(csv_path, index=False)
            print(f"  wrote {csv_path}")
        print()


if __name__ == "__main__":
    main()
