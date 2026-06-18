"""Optuna HPO driver for the Vision Transformer models (RQ3).

Parallel to Train_Model.py but for the ViT backbone, which has a different
interface: no --ReLU / --num_blocks / --amp_dtype, and instead --full/--no-full
(quadratic form) and --freeze_qv_linear (matched-ablation baseline). Each
configuration is its own Optuna study; with --storage journal, N MPS workers
sharing one study JOIN it via load_if_exists, exactly as in Train_Model.py.

ViT-specific defaults: far fewer epochs (pretrained finetuning converges fast)
and coarser pruning than the from-scratch CNN runs.

Usage (single process):
    python Train_ViT.py --model ViT_quad --full --epoch 10 --n_trials 20
    python Train_ViT.py --model ViT_baseline --epoch 10 --n_trials 20
    python Train_ViT.py --model ViT_quad --full --dataset cifar100 --epoch 10 --n_trials 20
"""

import os
import pickle
import argparse

import torch
import numpy as np
import optuna
from optuna.trial import TrialState

from funcs import load_batch, get_classes_dataset
from py_files.Baseline_ViT import ViT_baseline
from py_files.Quad_ViT import Quad_adapters as ViT_quad   # aliased to avoid the CNN name clash

MODEL_MAP = {
    "ViT_baseline": ViT_baseline,
    "ViT_quad": ViT_quad,
}

# Dataset configuration: the ONLY place the two datasets differ.
DATASET_CONFIG = {
    "cifar10":  {"dir": "data/cifar-10-batches-py",
                 "train_files": [f"data_batch_{i}" for i in range(1, 6)],
                 "test_file": "test_batch",
                 "label_key": b"labels",
                 "num_classes": 10},
    "cifar100": {"dir": "data/cifar-100-python",
                 "train_files": ["train"],
                 "test_file": "test",
                 "label_key": b"fine_labels",
                 "num_classes": 100},
}

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=list(MODEL_MAP.keys()), required=True)
parser.add_argument("--epoch", type=int, default=10,
                    help="ViT finetuning converges fast; default is small.")
parser.add_argument("--prune_resolution", type=int, default=2,
                    help="Epochs between prune checks. Coarser than the CNN.")
parser.add_argument(
    "--full", dest="full", action=argparse.BooleanOptionalAction, default=True,
    help="Quadratic form for ViT_quad: --full = r*g + b(x^2) (cross-term), "
         "--no-full = r + b(x^2) (reduced, square-term only). Ignored by ViT_baseline.")
parser.add_argument(
    "--freeze_qv_linear", dest="freeze_qv_linear",
    action=argparse.BooleanOptionalAction, default=False,
    help="ViT_baseline only: freeze the linear Q/V projections for the matched "
         "ablation (so the only difference from ViT_quad is the quadratic terms).")
parser.add_argument(
    "--grad_clip", type=float, default=1.0,
    help="Max gradient norm, applied to both ViT models (set <=0 to disable).")
parser.add_argument("--classes", type=int, nargs="+", default=None,
                    help="Class subset. Defaults to all classes of the chosen dataset.")
parser.add_argument("--n_trials", type=int, default=20)
parser.add_argument(
    "--test_split", type=float, default=0.2,
    help="Fraction of the training data reserved as the HPO validation split.")
parser.add_argument("--storage", choices=["sqlite", "journal"], default="journal")
parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10",
                    help="cifar10 (default) or cifar100. Only changes data loading.")


def build_storage(backend, name_base):
    """Always journal storage: writes optuna_databases/<name_base>.log."""
    os.makedirs("optuna_databases", exist_ok=True)
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    return JournalStorage(JournalFileBackend(f"optuna_databases/{name_base}.log"))


def build_name_base(args):
    prefix = f"{args.dataset}_"
    if args.model == "ViT_quad":
        return f"{prefix}ViT_quad_full{int(args.full)}"
    suffix = "_frozenqv" if args.freeze_qv_linear else ""
    return f"{prefix}ViT_baseline{suffix}"


def build_model(model_class, args, trial):
    """Construct the right ViT with the right kwargs for its class."""
    gc = args.grad_clip if args.grad_clip > 0 else None
    if model_class is ViT_quad:
        return model_class(classes=args.classes, full=args.full,
                           grad_clip=gc, trial=trial)
    # ViT_baseline
    return model_class(classes=args.classes,
                       freeze_qv_linear=args.freeze_qv_linear,
                       grad_clip=gc, trial=trial)


def load_cifar(classes, dataset):
    """Load the training split of the chosen dataset, filter to `classes`, and
    relabel to a contiguous range. Returns float tensors in [0,1] (CPU)."""
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
    return X, y


def objective(trial, args, save_dir, X_train, y_train, X_val, y_val):
    model_class = MODEL_MAP[args.model]
    model = build_model(model_class, args, trial)

    trial_dir = f"{save_dir}/trial{trial.number}"
    os.makedirs(trial_dir, exist_ok=True)

    # ViT batch sizes are smaller than the CNN's (224x224 inputs, big backbone).
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    best_acc = 0.0
    accs = {}
    for i in range(args.epoch // args.prune_resolution):
        model.train(X_train, y_train, batch_size, model.lr, args.prune_resolution)
        acc = model.test(X_val, y_val)            # selection on the HPO val split
        accs[i * args.prune_resolution] = acc
        if acc > best_acc:
            best_acc = acc
            model.save(f"{trial_dir}/epoch{i * args.prune_resolution}")
        trial.report(acc, i)
        if trial.should_prune():
            raise optuna.TrialPruned()

    with open(f"{trial_dir}/accuracies.pkl", "wb") as f:
        pickle.dump(accs, f)
    trial.set_user_attr("best_acc", best_acc)
    return best_acc


if __name__ == "__main__":
    args = parser.parse_args()
    if args.classes is None:
        args.classes = list(range(DATASET_CONFIG[args.dataset]["num_classes"]))

    # Load CIFAR and make the HPO train/val split. The official test set is NOT
    # touched here; final test-set evaluation happens in the ViT Finetune, so the
    # number HPO selects on (val) is never the number finally reported (test).
    X, y = load_cifar(args.classes, args.dataset)
    dataset = torch.utils.data.TensorDataset(X, y)
    train_size = int(len(dataset) * (1 - args.test_split))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = X[train_ds.indices].to(device)
    y_train = y[train_ds.indices].to(device)
    X_val = X[val_ds.indices].to(device)
    y_val = y[val_ds.indices].to(device)

    name_base = build_name_base(args)
    save_dir = f"models/{name_base}"
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/arguments.pkl", "wb") as f:
        pickle.dump(args, f)

    storage = build_storage(args.storage, name_base)
    study = optuna.create_study(
        study_name=name_base,
        storage=storage,
        load_if_exists=True,            # workers 2..N join worker 1's study
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2),
    )
    study.optimize(
        lambda trial: objective(trial, args, save_dir, X_train, y_train, X_val, y_val),
        n_trials=args.n_trials,
    )

    pruned = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("Study statistics:")
    print("  finished trials:", len(study.trials))
    print("  pruned:", len(pruned))
    print("  complete:", len(complete))
    print("Best trial:")
    print("  value:", study.best_trial.value)
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")