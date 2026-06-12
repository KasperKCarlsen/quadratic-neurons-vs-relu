"""Final training run using the best hyperparameters found by the Optuna HPO.

Train_Model.py runs the *search*: every trial re-suggests optimizer/lr/batch_size
and trains a short, possibly-pruned model. This script does the *final* training:
for each study in optuna_databases/, it reads `study.best_params`, rebuilds the
model from a fresh random init with exactly those hyperparameters, trains it
across --seeds (default 5) and saves each seed's best-on-test checkpoint plus a
mean +/- std summary to Done_models/<config>/.

Usage (Windows GPU):
    python Finetune.py --epochs 100
    python Finetune.py --epochs 100 --seeds 0 1 2 3 4
    python Finetune.py --epochs 100 --only "Quad_neuron_with_ReLU2_arc_[5, 5, 5]"
"""

import os
import ast
import glob
import json
import random
import pickle
import argparse

import torch    
import numpy as np
import optuna

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


def parse_study_db(db_path):
    """Recover (model, relu, num_blocks, name_base) from an optuna_databases/*.db
    filename. Mirrors the name_base built in Train_Model.build_storage:
        {model}_with_ReLU{relu}_arc_{num_blocks}_study.db
    """
    name_base = os.path.basename(db_path)[: -len("_study.db")]
    model, rest = name_base.split("_with_ReLU", 1)
    relu_str, arc_str = rest.split("_arc_", 1)
    return model, int(relu_str), ast.literal_eval(arc_str), name_base


def load_run_args(save_dir):
    """Read the argparse Namespace Train_Model.py pickled for this run, so we
    reuse its exact classes / val / test_split. Falls back to defaults."""
    path = f"{save_dir}/arguments.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def prepare_data(classes, val, test_split, device):
    """Replicates Train_Model.py's data pipeline for a given class set / split."""
    X, y = [], []
    for i in range(1, 6):
        images, labels = load_batch(f"data/cifar-10-batches-py/data_batch_{i}")
        X.append(images)
        y.append(labels)
    X = np.concatenate(X)
    y = np.concatenate(y)

    X, y = get_classes_dataset(classes, X, y)
    label_lookup = np.empty(10, dtype=np.int64)
    for new, orig in enumerate(classes):
        label_lookup[orig] = new
    y = label_lookup[y]

    if val:
        X = torch.from_numpy(X).float() / 255.0
        y = torch.from_numpy(y).long()
        dataset = torch.utils.data.TensorDataset(X, y)
        train_size = int(len(dataset) * (1 - test_split))
        test_size = len(dataset) - train_size
        train_ds, test_ds = torch.utils.data.random_split(
            dataset, [train_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )
        X_train, y_train = X[train_ds.indices], y[train_ds.indices]
        X_test, y_test = X[test_ds.indices], y[test_ds.indices]
    else:
        X_train = torch.from_numpy(X).float() / 255.0
        y_train = torch.from_numpy(y).long()
        images_test, labels_test = load_batch("data/cifar-10-batches-py/test_batch")
        X_test = torch.from_numpy(images_test).float() / 255.0
        y_test = torch.from_numpy(labels_test).long()

    return (X_train.to(device), y_train.to(device),
            X_test.to(device), y_test.to(device))


def build_optimizer(model, params):
    """Rebuild the tuned optimizer from study.best_params. Mirrors the optimizer
    setup in each model's constructor, but with fixed values instead of trial
    suggestions. Filters to trainable params so it also works for adapter models."""
    trainable = filter(lambda p: p.requires_grad, model.model.parameters())
    name = params["optimizer"]
    lr = params["lr"]
    if name == "AdamW":
        return torch.optim.AdamW(trainable, lr=lr,
                                 weight_decay=params["weight_decay"], fused=model.use_amp)
    if name == "SGD":
        return torch.optim.SGD(trainable, lr=lr, momentum=params["momentum"])
    return torch.optim.Adam(trainable, lr=lr, fused=model.use_amp)


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
    """Train one model from scratch for a single seed; return its best test acc."""
    set_seed(seed)

    # Fresh random init (now seed-dependent); no trial -> we set the tuned optimizer.
    if model_class in FROM_SCRATCH:
        model = model_class(relu, num_blocks, classes=classes)
    else:
        model = model_class(relu, num_blocks, load_path, classes=classes)
    model.optimizer = build_optimizer(model, best)

    os.makedirs(seed_dir, exist_ok=True)
    best_acc, accs = 0.0, {}
    for epoch in range(epochs):
        model.train(X_train, y_train, batch_size, best["lr"], 1)
        acc = model.test(X_test, y_test)
        accs[epoch] = acc
        if acc > best_acc:
            best_acc = acc
            model.save(f"{seed_dir}/best")

    with open(f"{seed_dir}/accuracies.pkl", "wb") as f:
        pickle.dump(accs, f)
    return best_acc, accs


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


def train_one(db_path, args, device):
    model_name, relu, num_blocks, name_base = parse_study_db(db_path)
    save_dir = f"models/{model_name}_ReLU{relu}_arc_{num_blocks}"

    model_class = MODEL_MAP[model_name]
    if model_class not in FROM_SCRATCH and args.load_path is None:
        print(f"[skip] {name_base}: {model_name} needs --load_path (pretrained net).")
        return

    # Recover the best hyperparameters from the study.
    study = optuna.load_study(
        study_name=f"{model_name}_with_ReLU{relu}_optimization",
        storage=f"sqlite:///optuna_databases/{name_base}_study.db",
    )
    best = study.best_params
    batch_size = best["batch_size"]

    # Reuse this run's original classes / split if recorded.
    run_args = load_run_args(save_dir)
    classes = getattr(run_args, "classes", list(range(10)))
    val = getattr(run_args, "val", True)
    test_split = getattr(run_args, "test_split", 0.2)

    print(f"\n=== {name_base} ===")
    print(f"  best_value(HPO)={study.best_value:.4f}  params={best}")
    # Prepared once and reused for every seed. The split is fixed (seed 42 inside
    # prepare_data), so all seeds are scored on the IDENTICAL test set and the
    # spread reflects training randomness only, not a different test set.
    X_train, y_train, X_test, y_test = prepare_data(classes, val, test_split, device)

    out_dir = f"Done_models/{model_name}_ReLU{relu}_arc_{num_blocks}"
    os.makedirs(out_dir, exist_ok=True)

    best_per_seed, accs_per_seed = {}, {}
    for seed in args.seeds:
        acc, accs = train_seed(model_class, relu, num_blocks, classes, best, batch_size,
                               args.epochs, args.load_path, X_train, y_train, X_test, y_test,
                               seed, f"{out_dir}/seed{seed}")
        best_per_seed[seed] = acc
        accs_per_seed[seed] = accs
        print(f"  [seed {seed}] best test acc {acc:.4f}")

    vals = np.array(list(best_per_seed.values()), dtype=np.float64)
    mean = float(vals.mean())
    std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0  # sample std over seeds
    summary = {
        "hpo_best_value": study.best_value,
        "params": best,
        "epochs": args.epochs,
        "seeds": list(args.seeds),
        "best_acc_per_seed": {str(s): best_per_seed[s] for s in args.seeds},
        "mean": mean,
        "std": std,
    }
    with open(f"{out_dir}/seed_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  -> {name_base}: {mean:.4f} +/- {std:.4f} over {len(vals)} seed(s)"
          f"  ({out_dir}/seed_results.json)")

    config_name = f"{model_name}_ReLU{relu}_arc_{num_blocks}"
    plot_seed_summary(config_name, accs_per_seed, mean, std)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Final training with the best HPO hyperparameters.")
    parser.add_argument("--epochs", type=int, required=True,
                        help="Epochs for the final training run.")
    parser.add_argument("--only", type=str, default=None,
                        help="Only train this study name_base, e.g. "
                             "'Quad_neuron_with_ReLU2_arc_[5, 5, 5]'.")
    parser.add_argument("--load_path", type=str, default=None,
                        help="Pretrained checkpoint for adapter models (LoRA / Quad_adapters).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="Seeds to train across; results reported as mean +/- std.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    db_paths = sorted(glob.glob("optuna_databases/*_study.db"))
    if args.only:
        db_paths = [p for p in db_paths
                    if os.path.basename(p) == f"{args.only}_study.db"]
        if not db_paths:
            raise SystemExit(f"No study DB matching --only '{args.only}'")

    print(f"Found {len(db_paths)} study DB(s); training on {device}.")
    for db_path in db_paths:
        train_one(db_path, args, device)
