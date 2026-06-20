"""Final training run using the best hyperparameters found by the Optuna HPO.

Train_Model.py runs the *search*: every trial re-suggests optimizer/lr/batch_size
and trains a short, possibly-pruned model. This script does the *final* training:
for each study in optuna_databases/, it reads `study.best_params`, rebuilds the
model from a fresh random init with exactly those hyperparameters, trains it
across --seeds (default 5) and saves each seed's best-on-test checkpoint plus a
mean +/- std summary to Done_models/<config>/.

Studies are read from optuna_databases/ in either backend: sqlite (*_study.db)
or Optuna journal (*_study.log). Alternatively, --params trains a single config
from hyperparameters supplied directly, with no study at all.

Usage (Windows GPU):
    python Finetune.py --epochs 100
    python Finetune.py --epochs 100 --seeds 0 1 2 3 4
    python Finetune.py --epochs 100 --only "Quad_neuron_with_ReLU2_arc_[5, 5, 5]"

    # Raw hyperparameters (no study DB needed); --params takes inline JSON,
    # a 'key: value' block, or a path to a file containing either:
    python Finetune.py --epochs 100 --model Quad_neuron --relu 0 --num-blocks 9 9 9 \
        --params '{"optimizer": "Adam", "lr": 0.0022126, "batch_size": 147}'
    python Finetune.py --epochs 100 --model Quad_neuron --relu 0 --num-blocks 9 9 9 \
        --params best_params.txt
"""

import os
import re
import ast
import glob
import json
import time
import random
import pickle
import argparse

import torch    
import numpy as np
import optuna

from collections import namedtuple

from funcs import load_batch, get_classes_dataset
from py_files.Quad_Neuron import Quad_neuron
from py_files.Baseline import Baseline
from py_files.Quad_adapters import Quad_adapters
from py_files.Baseline_with_LoRA import LoRA

# Same free Tensor-Core wins Train_Model.py uses; they don't change the model.
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

MODEL_MAP = {
    "Quad_neuron": Quad_neuron,
    "Quad_adapters": Quad_adapters,
    "Baseline": Baseline,
    "LoRA": LoRA,
}
FROM_SCRATCH = (Quad_neuron, Baseline)  # the rest need a --load_path pretrained net

# One config to finalize: where its params came from, plus enough to rebuild it.
# hpo_value is the study's best_value, or None when params are fed in raw.
Spec = namedtuple("Spec", "model relu num_blocks name_base params hpo_value")


def parse_name_base(name_base):
    """Recover (model, relu, num_blocks) from a study name_base of the form
        {model}_with_ReLU{relu}_arc_{arch}
    where {arch} encodes the blocks-per-stage. The architecture is read with a
    digit regex so every spelling the train scripts have used parses the same:
        _arc_(3_3_3)   _arc_[3, 3, 3]   _arc_3-3-3   -> [3, 3, 3]
    (a plain ast.literal_eval is unsafe here - "(3_3_3)" evaluates to the int
    333, since underscores are Python digit-group separators).
    Raises ValueError if the name isn't in this form (e.g. a ViT study), which
    discover_studies relies on to skip non-ResNet files.
    """
    model, rest = name_base.split("_with_ReLU", 1)
    relu_str, arc_str = rest.split("_arc_", 1)
    num_blocks = [int(n) for n in re.findall(r"\d+", arc_str)]
    if not num_blocks:
        raise ValueError(f"no architecture digits in {name_base!r}")
    return model, int(relu_str), num_blocks


def discover_studies():
    """Every ResNet study under optuna_databases/, from either backend: Optuna
    journal (.log) or sqlite (.db). The file stem IS the study name (matching how
    the train scripts and Finetune_ViT.py name them). Files whose stem doesn't
    parse as a known '{model}_with_ReLU..._arc_...' config - e.g. the ViT
    studies - are skipped, so this stays specific to the ResNet models here."""
    paths = glob.glob("optuna_databases/*.log") + glob.glob("optuna_databases/*.db")
    name_bases = set()
    for p in paths:
        nb = os.path.splitext(os.path.basename(p))[0]
        if nb.endswith("_study"):          # tolerate the legacy '..._study.db' form
            nb = nb[: -len("_study")]
        try:
            model, _, _ = parse_name_base(nb)
        except ValueError:
            continue                       # not a ResNet study (e.g. ViT) -> skip
        if model in MODEL_MAP:
            name_bases.add(nb)
    return sorted(name_bases)


def load_study(name_base):
    """Reload a study from whichever backend produced it, journal (.log)
    preferred over sqlite (.db) when both exist. The study name equals the file
    stem. The journal is opened with an open()-based lock so reading works on
    Windows without admin/Developer Mode (the default symlink lock raises
    WinError 1314 there)."""
    log_path = f"optuna_databases/{name_base}.log"
    db_path = f"optuna_databases/{name_base}.db"
    if os.path.exists(log_path):
        from optuna.storages import JournalStorage
        from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
        storage = JournalStorage(
            JournalFileBackend(log_path, lock_obj=JournalFileOpenLock(log_path)))
    elif os.path.exists(db_path):
        storage = f"sqlite:///{db_path}"
    else:
        raise FileNotFoundError(f"No .log or .db storage for study {name_base!r}")
    return optuna.load_study(study_name=name_base, storage=storage)


def parse_params(src):
    """Turn user-supplied hyperparameters into a plain dict, accepting (in order):
    a path to a file holding any of the below; inline JSON; a Python-literal dict;
    or 'key: value' pairs separated by newlines and/or commas. Values are coerced
    via ast.literal_eval where possible, so '147' -> int and 'Adam' -> str. Pairs
    without a colon (e.g. a stray 'for relu 0 9 9 9' note) are ignored.

    The last form is deliberately forgiving so that an inline dict mangled by a
    shell still parses: Windows PowerShell, for instance, strips the inner
    double-quotes when passing JSON to a native exe, turning
        {"optimizer": "Adam", "batch_size": 147}
    into
        {optimizer: Adam, batch_size: 147}
    which is neither valid JSON nor a Python literal. We strip surrounding braces
    and stray quotes and split on commas to recover it anyway. (Passing a file via
    --params sidesteps shell quoting entirely and is the robust option.)
    """
    text = src
    if os.path.exists(src):
        with open(src) as f:
            text = f.read()
    text = text.strip()

    try:                                   # 1) JSON
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    try:                                   # 2) Python-literal dict
        obj = ast.literal_eval(text)
        if isinstance(obj, dict):
            return obj
    except (ValueError, SyntaxError):
        pass

    params = {}                            # 3) 'key: value' pairs (newline/comma)
    body = text.strip().lstrip("{").rstrip("}")
    for pair in re.split(r"[\n,]", body):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        key, _, val = pair.partition(":")
        key = key.strip().strip("\"'")
        val = val.strip().strip("\"'")
        try:
            params[key] = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            params[key] = val
    if not params:
        raise SystemExit(f"Could not parse any hyperparameters from: {src!r}")
    return params


def load_run_args(save_dir):
    """Read the argparse Namespace Train_Model.py pickled for this run, so we
    reuse its exact classes / val / test_split. Falls back to defaults."""
    path = f"{save_dir}/arguments.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def prepare_data(classes, test_split, device):
    """Build three disjoint sets:
      - train : 80% of the official training batches (seed-42 split)
      - val   : the remaining 20% (used ONLY as the HPO target in Train_Model.py)
      - test  : the official test_batch, held out entirely from selection.
    The train/val split mirrors Train_Model.py exactly (same seed 42 and
    test_split), so final models train on the IDENTICAL data HPO used, while
    being scored on a test set that played no part in hyperparameter selection.
    """
    X, y = [], []
    for i in range(1, 6):
        images, labels = load_batch(f"data/cifar-10-batches-py/data_batch_{i}")
        X.append(images); y.append(labels)
    X = np.concatenate(X); y = np.concatenate(y)
    X, y = get_classes_dataset(classes, X, y)
    label_lookup = np.empty(10, dtype=np.int64)
    for new, orig in enumerate(classes):
        label_lookup[orig] = new
    y = label_lookup[y]

    X = torch.from_numpy(X).float() / 255.0
    y = torch.from_numpy(y).long()
    dataset = torch.utils.data.TensorDataset(X, y)
    train_size = int(len(dataset) * (1 - test_split))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    X_train, y_train = X[train_ds.indices], y[train_ds.indices]
    X_val,   y_val   = X[val_ds.indices],   y[val_ds.indices]

    # Official test set, same class filtering / relabeling.
    images_test, labels_test = load_batch("data/cifar-10-batches-py/test_batch")
    Xt, yt = get_classes_dataset(classes, images_test, labels_test)
    yt = label_lookup[yt]
    X_test = torch.from_numpy(Xt).float() / 255.0
    y_test = torch.from_numpy(yt).long()

    return (X_train.to(device), y_train.to(device),
            X_val.to(device),   y_val.to(device),
            X_test.to(device),  y_test.to(device))


def build_model(model_class, relu, num_blocks, classes, load_path, best):
    """Rebuild a model with exactly `best` as its hyperparameters.

    Instead of re-deriving each model's optimizer here (which drifted out of sync
    once Quad_neuron grew per-branch relinear_ratio_* learning rates and dropped
    AdamW), we hand the constructor an optuna FixedTrial wrapping `best`. Every
    `trial.suggest_*` call then returns the fixed value, so the model builds the
    *same* optimizer it built during HPO - relinear param groups and all - with
    no logic duplicated in this script. Extra keys in `best` (e.g. batch_size,
    which is consumed separately) are simply never requested, hence ignored.
    """
    trial = optuna.trial.FixedTrial(best)
    try:
        if model_class in FROM_SCRATCH:
            return model_class(relu, num_blocks, classes=classes, trial=trial)
        return model_class(relu, num_blocks, load_path, classes=classes, trial=trial)
    except ValueError as e:
        # FixedTrial raises ValueError when a suggested param is absent from best.
        raise SystemExit(f"params are missing a hyperparameter the model needs: {e}")


def set_seed(seed):
    """Seed every RNG so a run differs from another only by this seed.

    cudnn.deterministic is intentionally NOT forced: weight init and the
    per-epoch batch-shuffle order already give a genuine spread across seeds,
    and forcing full determinism would disable the cudnn.benchmark speedup for
    no statistical benefit.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_seed(model_class, relu, num_blocks, classes, best, batch_size,
               epochs, load_path, X_train, y_train, X_test, y_test, seed, seed_dir):
    """Train one model from scratch for a single seed. Returns its best test acc,
    the wall-clock seconds spent reaching that best, the epoch it was reached at,
    and the per-epoch accuracy history.

    time_to_best is measured from the start of training to the end of the epoch
    that set the best accuracy - i.e. the time you'd actually have to train to
    obtain the saved checkpoint. model.test() syncs the GPU (its final .item()
    call blocks), so the elapsed time already accounts for queued CUDA work."""
    set_seed(seed)

    # Fresh random init (seed-dependent), with the tuned optimizer rebuilt from
    # `best` inside the constructor via FixedTrial - see build_model.
    model = build_model(model_class, relu, num_blocks, classes, load_path, best)

    os.makedirs(seed_dir, exist_ok=True)
    best_acc, best_time, best_epoch, accs = 0.0, 0.0, -1, {}
    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train(X_train, y_train, batch_size, best["lr"], 1)
        acc = model.test(X_test, y_test)
        accs[epoch] = acc
        improved = acc > best_acc
        if improved:
            best_acc = acc
            best_time = time.perf_counter() - t0
            best_epoch = epoch
            model.save(f"{seed_dir}/best")
        print(f"    [seed {seed}] epoch {epoch + 1}/{epochs}  "
              f"test acc {acc:.2f}%  (best {best_acc:.2f}%)"
              f"{'  *new best' if improved else ''}", flush=True)

    with open(f"{seed_dir}/accuracies.pkl", "wb") as f:
        pickle.dump(accs, f)
    return best_acc, best_time, best_epoch, accs


def plot_seed_summary(config_name, accs_per_seed, mean_best, std_best):
    """Summarize one config's per-seed learning curves into a single figure:
    each seed drawn faintly, the across-seed mean bold, with a +/-1 std band.
    Saved to Visualization/<config_name>_graph.png."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless: no display needed on the cluster/CI
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot] matplotlib not installed - skipping graph.")
        return

    seeds = sorted(accs_per_seed)
    # All seeds share the same epoch grid (same --epochs); align on it.
    epochs = sorted(next(iter(accs_per_seed.values())).keys())
    curves = np.array([[accs_per_seed[s][e] for e in epochs] for s in seeds],
                      dtype=np.float64)
    xs = [e + 1 for e in epochs]
    mean_curve = curves.mean(axis=0)
    std_curve = curves.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros_like(mean_curve)

    os.makedirs("Visualization", exist_ok=True)
    plt.figure(figsize=(8, 5))
    for s, curve in zip(seeds, curves):
        plt.plot(xs, curve, alpha=0.3, linewidth=1, label=f"seed {s}")
    plt.plot(xs, mean_curve, color="black", linewidth=2.5, label="mean")
    plt.fill_between(xs, mean_curve - std_curve, mean_curve + std_curve,
                     color="black", alpha=0.15, label="+/-1 std")
    plt.title(f"{config_name}\nbest test acc {mean_best:.2f} +/- {std_best:.2f} "
              f"over {len(seeds)} seed(s)")
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy (%)")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = f"Visualization/{config_name}_graph.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [plot] wrote {out_path}")


def validate_params(name_base, best):
    """batch_size is the one hyperparameter this script consumes directly (it is
    passed to model.train, not suggested in the constructor), so it must be
    present. Every other key is validated by the model itself via FixedTrial in
    build_model, which fails loudly if a needed one is missing."""
    if "batch_size" not in best:
        raise SystemExit(f"{name_base}: params are missing required 'batch_size'.")


def train_one(spec, args, device):
    model_name, relu, num_blocks, name_base, best, hpo_value = spec
    save_dir = f"models/{model_name}_ReLU{relu}_arc_{num_blocks}"

    model_class = MODEL_MAP[model_name]
    if model_class not in FROM_SCRATCH and args.load_path is None:
        print(f"[skip] {name_base}: {model_name} needs --load_path (pretrained net).")
        return

    print(f"\n=== {name_base} ===")
    if hpo_value is not None:
        print(f"  best_value(HPO)={hpo_value:.4f}  params={best}")
    else:
        print(f"  raw params (no study)={best}")
    validate_params(name_base, best)
    batch_size = best["batch_size"]

    # Reuse this run's original classes / split if recorded.
    run_args = load_run_args(save_dir)
    classes = getattr(run_args, "classes", list(range(10)))
    val = getattr(run_args, "val", True)
    test_split = getattr(run_args, "test_split", 0.2)

    # Prepared once and reused for every seed. The split is fixed (seed 42 inside
    # prepare_data), so all seeds are scored on the IDENTICAL test set and the
    # spread reflects training randomness only, not a different test set.
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(classes, test_split, device)
    if args.refit_full:
        X_fit = torch.cat([X_train, X_val]); y_fit = torch.cat([y_train, y_val])
    else:
        X_fit, y_fit = X_train, y_train
    out_dir = f"Done_models/{model_name}_ReLU{relu}_arc_{num_blocks}"
    os.makedirs(out_dir, exist_ok=True)

    best_per_seed, accs_per_seed = {}, {}
    time_per_seed, epoch_per_seed = {}, {}
    for seed in args.seeds:
        acc, t_best, e_best, accs = train_seed(
            model_class, relu, num_blocks, classes, best, batch_size,
            args.epochs, args.load_path, X_fit, y_fit, X_test, y_test,
            seed, f"{out_dir}/seed{seed}")
        best_per_seed[seed] = acc
        accs_per_seed[seed] = accs
        time_per_seed[seed] = t_best
        epoch_per_seed[seed] = e_best
        print(f"  [seed {seed}] best test acc {acc:.4f}"
              f"  (reached in {t_best:.1f}s at epoch {e_best + 1})")

    vals = np.array(list(best_per_seed.values()), dtype=np.float64)
    mean = float(vals.mean())
    std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0  # sample std over seeds
    times = np.array(list(time_per_seed.values()), dtype=np.float64)
    mean_time = float(times.mean())
    std_time = float(times.std(ddof=1)) if len(times) > 1 else 0.0
    summary = {
        "hpo_best_value": hpo_value,
        "params": best,
        "epochs": args.epochs,
        "seeds": list(args.seeds),
        "best_acc_per_seed": {str(s): best_per_seed[s] for s in args.seeds},
        "mean": mean,
        "std": std,
        # Benchmarking: wall-clock seconds to reach the best test acc, and the
        # epoch (1-indexed) at which it was reached, per seed plus mean +/- std.
        "time_to_best_per_seed": {str(s): time_per_seed[s] for s in args.seeds},
        "best_epoch_per_seed": {str(s): epoch_per_seed[s] + 1 for s in args.seeds},
        "mean_time_to_best": mean_time,
        "std_time_to_best": std_time,
    }
    with open(f"{out_dir}/seed_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  -> {name_base}: {mean:.4f} +/- {std:.4f} over {len(vals)} seed(s)"
          f"  | time-to-best {mean_time:.1f} +/- {std_time:.1f}s"
          f"  ({out_dir}/seed_results.json)")

    config_name = f"{model_name}_ReLU{relu}_arc_{num_blocks}"
    plot_seed_summary(config_name, accs_per_seed, mean, std)


def specs_from_studies(args):
    """Build a Spec per discovered study (sqlite .db or journal .log), honoring
    --only. This is the original behavior, now backend-agnostic."""
    name_bases = discover_studies()
    if args.only:
        name_bases = [n for n in name_bases if n == args.only]
        if not name_bases:
            raise SystemExit(f"No study (.db/.log) matching --only '{args.only}'")
    specs = []
    for nb in name_bases:
        model, relu, num_blocks = parse_name_base(nb)
        study = load_study(nb)
        specs.append(Spec(model, relu, num_blocks, nb, study.best_params,
                          study.best_value))
    return specs


def spec_from_raw(args):
    """Build a single Spec from hyperparameters fed on the command line, with no
    study involved. The config (model/relu/num_blocks) must be given explicitly
    since there is no study filename to recover it from."""
    missing = [f"--{n}" for n in ("model", "relu", "num_blocks")
               if getattr(args, n) is None]
    if missing:
        raise SystemExit(f"--params also requires: {', '.join(missing)}")
    if args.model not in MODEL_MAP:
        raise SystemExit(f"--model must be one of {list(MODEL_MAP)}")
    best = parse_params(args.params)
    name_base = f"{args.model}_with_ReLU{args.relu}_arc_{args.num_blocks}"
    return [Spec(args.model, args.relu, args.num_blocks, name_base, best, None)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Final training from the best HPO hyperparameters (sqlite .db "
                    "or journal .log studies), or from raw params via --params.")
    parser.add_argument("--epochs", type=int, required=True,
                        help="Epochs for the final training run.")
    parser.add_argument("--only", type=str, default=None,
                        help="Only train this study name_base, e.g. "
                             "'Quad_neuron_with_ReLU2_arc_[5, 5, 5]'.")
    parser.add_argument("--load_path", type=str, default=None,
                        help="Pretrained checkpoint for adapter models (LoRA / Quad_adapters).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="Seeds to train across; results reported as mean +/- std.")
    parser.add_argument("--refit-full", dest="refit_full", action="store_true",
                        help="Refit final models on train+val (hyperparameters fixed) "
                             "before evaluating on the official test set.")
    # Raw-params mode: skip studies entirely and train exactly these numbers.
    parser.add_argument("--params", type=str, default=None,
                        help="Hyperparameters to train directly, bypassing any study. "
                             "Accepts inline JSON, a Python dict literal, the 'key: value' "
                             "block Optuna prints, or a path to a file holding any of those. "
                             "Requires --model, --relu and --num-blocks.")
    parser.add_argument("--model", type=str, default=None,
                        help=f"Raw mode: model name, one of {list(MODEL_MAP)}.")
    parser.add_argument("--relu", type=int, default=None,
                        help="Raw mode: ReLU variant (0/1/2).")
    parser.add_argument("--num-blocks", dest="num_blocks", type=int, nargs="+",
                        default=None, help="Raw mode: blocks per stage, e.g. --num-blocks 9 9 9.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.params is not None:
        specs = spec_from_raw(args)
        print(f"Raw-params mode: training {len(specs)} config on {device}.")
    else:
        specs = specs_from_studies(args)
        print(f"Found {len(specs)} study(ies); training on {device}.")
    for spec in specs:
        train_one(spec, args, device)
