"""Launch optuna-dashboard for the journal studies in optuna_databases/.

Each *.log in optuna_databases/ is a SEPARATE Optuna journal storage holding one
study, but optuna-dashboard serves a single storage at a time. So by default this
script copies every study into one combined sqlite storage and serves that, giving
you all three LoRA architectures in one dashboard.

    python dashboard.py                 # all journals, combined (snapshot)
    python dashboard.py --port 8081     # pick a port

The combined view is a SNAPSHOT taken when the script starts (copy_study). For the
imported HPO journals that's fine - they don't change locally. If a journal file is
being written ON THIS machine and you want live auto-refresh, serve that one file
directly instead (a journal storage updates live; the sqlite copy does not):

    python dashboard.py --file "optuna_databases/LoRA_with_ReLUNone_arc_(3_3_3).log"

Requires: pip install optuna-dashboard
"""

import os
import glob
import argparse

import optuna


def open_journal(log_path):
    """Open a journal .log as a storage, using an open()-based lock so reading
    works on Windows without admin/Developer Mode (the default symlink lock
    raises WinError 1314 there). Same approach as Finetune.load_study."""
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
    return JournalStorage(JournalFileBackend(log_path, lock_obj=JournalFileOpenLock(log_path)))


def build_combined(log_paths, combined_path):
    """Copy each journal's study into one fresh sqlite storage and return its URL.
    A snapshot: re-run the script to pick up newly imported journals."""
    if os.path.exists(combined_path):
        os.remove(combined_path)            # start clean so copy_study can't clash
    combined_url = f"sqlite:///{combined_path}"
    for log_path in log_paths:
        storage = open_journal(log_path)
        names = optuna.get_all_study_names(storage)
        if not names:
            print(f"  [skip] {os.path.basename(log_path)}: no study inside")
            continue
        for name in names:
            optuna.copy_study(
                from_study_name=name, from_storage=storage,
                to_storage=combined_url, to_study_name=name,
            )
            n = len(optuna.load_study(study_name=name, storage=combined_url).trials)
            print(f"  + {name}  ({n} trials)")
    return combined_url


def main():
    parser = argparse.ArgumentParser(description="Serve optuna-dashboard for optuna_databases/ journals.")
    parser.add_argument("--dir", default="optuna_databases", help="Folder of *.log journals.")
    parser.add_argument("--file", default=None,
                        help="Serve ONE journal file directly (live auto-refresh) "
                             "instead of the combined snapshot of all of them.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    try:
        from optuna_dashboard import run_server
    except ImportError:
        raise SystemExit("optuna-dashboard is not installed. Run:  pip install optuna-dashboard")

    if args.file:
        # Single journal, served live (the dashboard polls the file for new trials).
        if not os.path.exists(args.file):
            raise SystemExit(f"No such journal: {args.file}")
        storage = open_journal(args.file)
        print(f"Serving {args.file} (live) at http://{args.host}:{args.port}")
    else:
        log_paths = sorted(glob.glob(os.path.join(args.dir, "*.log")))
        if not log_paths:
            raise SystemExit(f"No *.log journals found in {args.dir}/")
        print(f"Combining {len(log_paths)} journal(s) into a snapshot:")
        combined_path = os.path.join(args.dir, "_dashboard_combined.db")
        storage = build_combined(log_paths, combined_path)
        print(f"Serving combined snapshot at http://{args.host}:{args.port}")

    print("Press Ctrl+C to stop.")
    run_server(storage, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
