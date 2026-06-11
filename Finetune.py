"""Final training run using the best hyperparameters found by the Optuna HPO.

Train_Model.py runs the *search*: every trial re-suggests optimizer/lr/batch_size
and trains a short, possibly-pruned model. This script does the *final* training:
for each study in optuna_databases/, it reads `study.best_params`, rebuilds the
model from a fresh random init with exactly those hyperparameters, trains it for
--epochs, and saves the best-on-test checkpoint to models/<config>/final.

Usage (Windows GPU):
    python Finetune.py --epochs 100
    python Finetune.py --epochs 100 --only "Quad_neuron_with_ReLU2_arc_[5, 5, 5]"
"""

import os
import ast
import glob
import json
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
    X_train, y_train, X_test, y_test = prepare_data(classes, val, test_split, device)

    # Fresh random init, no trial -> default optimizer is None; we set the tuned one.
    if model_class in FROM_SCRATCH:
        model = model_class(relu, num_blocks, classes=classes)
    else:
        model = model_class(relu, num_blocks, args.load_path, classes=classes)
    model.optimizer = build_optimizer(model, best)

    out_dir = f"{save_dir}/final"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/best_params.json", "w") as f:
        json.dump({"hpo_best_value": study.best_value, "params": best,
                   "epochs": args.epochs}, f, indent=2)

    best_acc, accs = 0.0, {}
    for epoch in range(args.epochs):
        model.train(X_train, y_train, batch_size, best["lr"], 1)
        acc = model.test(X_test, y_test)
        accs[epoch] = acc
        print(f"  [final {name_base}] epoch {epoch+1}/{args.epochs}  test_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            model.save(f"{out_dir}/best")

    with open(f"{out_dir}/accuracies.pkl", "wb") as f:
        pickle.dump(accs, f)
    print(f"  -> best test acc {best_acc:.4f}  saved to {out_dir}/best")


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
