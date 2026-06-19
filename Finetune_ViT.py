"""Final multi-seed training for the ViT models, using the best hyperparameters
found by Train_ViT.py.

Parallel to Finetune.py but for the ViT backbone. For each ViT study in
optuna_databases/, it reads study.best_params, rebuilds the model from a fresh
init with those hyperparameters, trains across --seeds, and reports mean +/- std.

Correct split discipline (built in, not retrofitted): the 20% split that
Train_ViT.py optimized against is used ONLY during HPO. Final models train on the
training portion (optionally train+val with --refit-full) and are evaluated on the
official test set, which played no part in selection.

Usage:
    python Finetune_ViT.py --epochs 10 --dataset cifar100
    python Finetune_ViT.py --epochs 10 --dataset cifar100 --seeds 0 1 2 3 4
    python Finetune_ViT.py --epochs 10 --dataset cifar100 --only cifar100_ViT_quad_full1
    python Finetune_ViT.py --epochs 10 --dataset cifar100 --refit-full
"""

import os
import glob
import json
import time
import random
import pickle
import argparse

import torch
import numpy as np
import optuna

from Train_ViT import DATASET_CONFIG
from funcs import load_batch, get_classes_dataset
from py_files.Baseline_ViT import ViT_baseline
from py_files.Quad_ViT import Quad_adapters as ViT_quad

MODEL_MAP = {
    "ViT_baseline": ViT_baseline,
    "ViT_quad": ViT_quad,
}


def parse_name_base(name_base):
    """Recover (model_key, kwargs) from a ViT study name_base (dataset prefix
    stripped first).
        ViT_quad_full1        -> ("ViT_quad", {"full": True})
        ViT_quad_full0        -> ("ViT_quad", {"full": False})
        ViT_baseline          -> ("ViT_baseline", {"freeze_qv_linear": False})
        ViT_baseline_frozenqv -> ("ViT_baseline", {"freeze_qv_linear": True})
    """
    for pfx in ("cifar10_", "cifar100_"):
        if name_base.startswith(pfx):
            name_base = name_base[len(pfx):]
            break
    if name_base.startswith("ViT_quad"):
        full = name_base.endswith("full1")
        return "ViT_quad", {"full": full}
    if name_base.startswith("ViT_baseline"):
        frozen = name_base.endswith("_frozenqv")
        return "ViT_baseline", {"freeze_qv_linear": frozen}
    raise ValueError(f"Unrecognized ViT name_base: {name_base}")


def load_run_args(save_dir):
    """Read the argparse Namespace Train_ViT.py pickled for this run, to reuse its
    exact classes / test_split / grad_clip / full / freeze settings."""
    path = f"{save_dir}/arguments.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def prepare_data(classes, test_split, device, dataset):
    """Three disjoint sets for the chosen dataset: train/val from the training
    files via the seed-42 split (val = HPO target only), test from the official
    test file (untouched during selection)."""
    cfg = DATASET_CONFIG[dataset]
    X, y = [], []
    for fname in cfg["train_files"]:
        images, labels = load_batch(f"{cfg['dir']}/{fname}", label_key=cfg["label_key"])
        X.append(images); y.append(labels)
    X = np.concatenate(X); y = np.concatenate(y)
    X, y = get_classes_dataset(classes, X, y)
    label_lookup = np.empty(cfg["num_classes"], dtype=np.int64)
    for new, orig in enumerate(classes):
        label_lookup[orig] = new
    y = label_lookup[y]
    X = torch.from_numpy(X).float() / 255.0
    y = torch.from_numpy(y).long()

    dataset_t = torch.utils.data.TensorDataset(X, y)
    train_size = int(len(dataset_t) * (1 - test_split))
    val_size = len(dataset_t) - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset_t, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    X_train, y_train = X[train_ds.indices], y[train_ds.indices]
    X_val,   y_val   = X[val_ds.indices],   y[val_ds.indices]

    images_test, labels_test = load_batch(f"{cfg['dir']}/{cfg['test_file']}",
                                          label_key=cfg["label_key"])
    Xt, yt = get_classes_dataset(classes, images_test, labels_test)
    yt = label_lookup[yt]
    X_test = torch.from_numpy(Xt).float() / 255.0
    y_test = torch.from_numpy(yt).long()

    return (X_train.to(device), y_train.to(device),
            X_val.to(device),   y_val.to(device),
            X_test.to(device),  y_test.to(device))


def build_model(model_key, kwargs, classes, grad_clip):
    """Fresh ViT (no trial -> uses default optimizer set in train())."""
    gc = grad_clip if (grad_clip is not None and grad_clip > 0) else None
    if model_key == "ViT_quad":
        return ViT_quad(classes=classes, full=kwargs["full"], grad_clip=gc)
    return ViT_baseline(classes=classes,
                        freeze_qv_linear=kwargs["freeze_qv_linear"], grad_clip=gc)


def build_optimizer(model, params):
    """Rebuild the tuned optimizer from best_params, on the trainable params."""
    trainable = filter(lambda p: p.requires_grad, model.model.parameters())
    name = params["optimizer"]
    lr = params["lr"]
    if name == "AdamW":
        return torch.optim.AdamW(trainable, lr=lr, weight_decay=params["weight_decay"])
    if name == "SGD":
        return torch.optim.SGD(trainable, lr=lr, momentum=params["momentum"])
    return torch.optim.Adam(trainable, lr=lr)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_seed(model_key, kwargs, classes, grad_clip, best, batch_size, epochs,
               X_fit, y_fit, X_test, y_test, seed, seed_dir):
    """Train one ViT from a fresh seed-dependent init. Returns its best test acc,
    the wall-clock seconds to reach that best, the epoch (0-indexed) it was
    reached at, and the per-epoch test-accuracy history. model.test() syncs the
    GPU, so the elapsed time accounts for queued CUDA work."""
    set_seed(seed)
    model = build_model(model_key, kwargs, classes, grad_clip)
    model.optimizer = build_optimizer(model, best)

    os.makedirs(seed_dir, exist_ok=True)
    best_acc, best_time, best_epoch, accs = 0.0, 0.0, -1, {}
    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train(X_fit, y_fit, batch_size, best["lr"], 1)
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
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot] matplotlib not installed - skipping graph.")
        return
    seeds = sorted(accs_per_seed)
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
    plt.xlabel("Epoch"); plt.ylabel("Test accuracy (%)")
    plt.legend(fontsize=8); plt.grid(True, alpha=0.3); plt.tight_layout()
    out_path = f"Visualization/{config_name}_graph.png"
    plt.savefig(out_path, dpi=150); plt.close()
    print(f"  [plot] wrote {out_path}")


def discover_studies(dataset):
    paths = (glob.glob(f"optuna_databases/{dataset}_ViT_*.log")
             + glob.glob(f"optuna_databases/{dataset}_ViT_*.db"))
    return sorted({os.path.splitext(os.path.basename(p))[0] for p in paths})


def load_study(name_base):
    """Reload a study from whichever backend produced it (journal preferred)."""
    log_path = f"optuna_databases/{name_base}.log"
    db_path = f"optuna_databases/{name_base}.db"
    if os.path.exists(log_path):
        from optuna.storages import JournalStorage
        from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
        # open()-based lock: the default symlink lock needs admin/Developer Mode
        # on Windows (WinError 1314). Must match how Train_ViT.py writes the log.
        storage = JournalStorage(
            JournalFileBackend(log_path, lock_obj=JournalFileOpenLock(log_path)))
    elif os.path.exists(db_path):
        storage = f"sqlite:///{db_path}"
    else:
        raise FileNotFoundError(f"No storage for study {name_base}")
    return optuna.load_study(study_name=name_base, storage=storage)


def train_one(name_base, args, device):
    model_key, name_kwargs = parse_name_base(name_base)
    model_class = MODEL_MAP[model_key]
    save_dir = f"models/{name_base}"

    study = load_study(name_base)
    best = study.best_params
    batch_size = best["batch_size"]

    # Reuse this run's recorded settings where present; fall back to name-derived.
    run_args = load_run_args(save_dir)
    classes = getattr(run_args, "classes", None)
    if classes is None:
        classes = list(range(DATASET_CONFIG[args.dataset]["num_classes"]))
    test_split = getattr(run_args, "test_split", 0.2)
    grad_clip = getattr(run_args, "grad_clip", 1.0)
    kwargs = dict(name_kwargs)
    if run_args is not None:
        if model_key == "ViT_quad":
            kwargs["full"] = getattr(run_args, "full", kwargs["full"])
        else:
            kwargs["freeze_qv_linear"] = getattr(run_args, "freeze_qv_linear",
                                                 kwargs["freeze_qv_linear"])

    print(f"\n=== {name_base} ===")
    print(f"  best_value(HPO)={study.best_value:.4f}  params={best}")

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(
        classes, test_split, device, args.dataset)
    if args.refit_full:
        X_fit = torch.cat([X_train, X_val]); y_fit = torch.cat([y_train, y_val])
    else:
        X_fit, y_fit = X_train, y_train

    out_dir = f"Done_models/{name_base}"
    os.makedirs(out_dir, exist_ok=True)

    best_per_seed, accs_per_seed = {}, {}
    time_per_seed, epoch_per_seed = {}, {}
    for seed in args.seeds:
        acc, t_best, e_best, accs = train_seed(
            model_key, kwargs, classes, grad_clip, best,
            batch_size, args.epochs, X_fit, y_fit, X_test, y_test,
            seed, f"{out_dir}/seed{seed}")
        best_per_seed[seed] = acc
        accs_per_seed[seed] = accs
        time_per_seed[seed] = t_best
        epoch_per_seed[seed] = e_best
        print(f"  [seed {seed}] best test acc {acc:.4f}"
              f"  (reached in {t_best:.1f}s at epoch {e_best + 1})")

    vals = np.array(list(best_per_seed.values()), dtype=np.float64)
    mean = float(vals.mean())
    std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    times = np.array(list(time_per_seed.values()), dtype=np.float64)
    mean_time = float(times.mean())
    std_time = float(times.std(ddof=1)) if len(times) > 1 else 0.0
    summary = {
        "hpo_best_value": study.best_value,
        "params": best,
        "epochs": args.epochs,
        "seeds": list(args.seeds),
        "refit_full": args.refit_full,
        "dataset": args.dataset,
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
    plot_seed_summary(name_base, accs_per_seed, mean, std)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Final multi-seed ViT training.")
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--only", type=str, default=None,
                   help="Only train this study name_base, e.g. 'cifar100_ViT_quad_full1' "
                        "(the dataset prefix may be omitted).")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--refit-full", dest="refit_full", action="store_true",
                   help="Refit final models on train+val before the test-set eval.")
    p.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10",
                   help="Must match the dataset the studies were trained with.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    name_bases = discover_studies(args.dataset)
    if args.only:
        name_bases = [n for n in name_bases
                      if n == args.only or n == f"{args.dataset}_{args.only}"]
        if not name_bases:
            raise SystemExit(f"No ViT study matching --only '{args.only}'")
    print(f"Found {len(name_bases)} ViT study(ies); training on {device}.")
    for nb in name_bases:
        train_one(nb, args, device)