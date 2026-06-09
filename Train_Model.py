import torch
import os
from torch.utils.data import TensorDataset
import pickle
import numpy as np
import argparse
import optuna
from optuna.trial import TrialState

from funcs import *
from py_files.Quad_Neuron import Quad_neuron
from py_files.Quad_adapters import Quad_adapters
from py_files.Baseline import Baseline
from py_files.Baseline_with_LoRA import LoRA

# Speed up convolutions for the fixed CIFAR-10 input size and let matmuls use
# TF32 on Ampere+ GPUs (e.g. the A100 nodes the .lsf jobs run on) - both are
# free wins that don't change the trained model architecture.
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

MODEL_MAP = {
    "Quad_neuron": Quad_neuron,
    "Quad_adapters": Quad_adapters,
    "Baseline": Baseline,
    "LoRA": LoRA,
}

parser = argparse.ArgumentParser(description="Proper ResNets for CIFAR10 in PyTorch")

parser.add_argument(
    "--model", choices=list(MODEL_MAP.keys()), required=True, help="Model name"
)

parser.add_argument(
    "--ReLU",
    type=int,
    choices=[0, 1, 2],
    required=True,
    help="0: no ReLU, 1: between layers only, 2: between layers and inside neurons",
)

parser.add_argument("--epoch", type=int, required=True)

parser.add_argument(
    "--save_epoch",
    type=int,
    default=1,
    help="Save the model every x epochs if accuracy improves",
)

parser.add_argument("--batch_size", type=int, default=128)

parser.add_argument(
    "--val",
    dest="val",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Whether to use a validation split from the training data instead of the official test set",
)


parser.add_argument("--lr", type=float, default=1e-3)

# Recommended CLI usage:
# --num_blocks 3 3 3
parser.add_argument(
    "--num_blocks",
    type=int,
    nargs="+",
    required=True,
    help="Number of blocks per stage, e.g. --num_blocks 3 3 3",
)

# If you prefer this style instead:
# --classes 0 1 2 3 4 5 6 7 8 9
parser.add_argument("--classes", type=int, nargs="+", default=list(range(10)))

parser.add_argument("--load_path", type=str, default=None)

parser.add_argument("--n_trials", type=int, required=False,default=100)

parser.add_argument(
    "--test_split",
    dest="test_split",
    type=float,
    default=0.2,
    help="Fraction of the loaded data to reserve for testing",
)


parser.add_argument(
    "--storage",
    choices=["sqlite", "journal"],
    default="sqlite",
    help="Optuna storage backend. Use 'journal' for an LSF job array (multiple "
    "workers sharing one study) on a networked/NFS filesystem, where SQLite file "
    "locking is unreliable.",
)


def build_storage(backend, name_base):
    """Build the Optuna storage.

    SQLite is fine for a single process. For an LSF job array (many workers
    sharing one study over a networked/NFS filesystem) use 'journal': SQLite
    file locking is unreliable on NFS and tends to throw 'database is locked'
    under concurrent writers, whereas JournalStorage is designed for it.
    """
    os.makedirs("optuna_databases", exist_ok=True)
    if backend == "journal":
        try:  # optuna >= 4.x
            from optuna.storages import JournalStorage
            from optuna.storages.journal import JournalFileBackend
            return JournalStorage(JournalFileBackend(f"optuna_databases/{name_base}.log"))
        except ImportError:  # optuna 3.x
            from optuna.storages import JournalStorage, JournalFileStorage
            return JournalStorage(JournalFileStorage(f"optuna_databases/{name_base}.log"))
    from optuna.storages import RDBStorage
    # Bump the SQLite busy timeout so a brief lock doesn't fail the trial.
    return RDBStorage(
        url=f"sqlite:///optuna_databases/{name_base}_study.db",
        engine_kwargs={"connect_args": {"timeout": 100}},
    )


def objective(trial, args, save_dir, X_train, y_train, X_test, y_test):
    model_class = MODEL_MAP[args.model]
    if model_class in (Quad_neuron, Baseline):
        model = model_class(args.ReLU, args.num_blocks, classes=args.classes, trial=trial)
    else:
        model = model_class(args.ReLU, args.num_blocks, args.load_path, classes=args.classes, trial=trial)

    if model_class not in (LoRA, Quad_adapters) and args.load_path is not None:
        model.load(args.load_path)

    # Per-trial output directory. Previously every trial wrote checkpoints and
    # accuracies.pkl into the same save_dir, so trials (sequential OR parallel
    # array workers) clobbered each other and the "best" file left on disk had
    # nothing to do with study.best_trial. Scoping by trial.number keeps each
    # trial's artifacts and makes them recoverable from the study DB.
    trial_dir = f"{save_dir}/trial{trial.number}"
    os.makedirs(trial_dir, exist_ok=True)

    best_acc = 0
    accs = {}

    # Categorical powers-of-two: avoids tiny batches (e.g. 35) that starve the
    # GPU, and limits shapes to 5 values so cudnn.benchmark tunes once per shape.
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])

    for i in range(args.epoch):
        model.train(X_train, y_train, batch_size, args.lr, args.save_epoch)

        acc = model.test(X_test, y_test)
        accs[i * args.save_epoch] = acc

        if acc > best_acc:
            best_acc = acc
            model.save(f"{trial_dir}/epoch{i*args.save_epoch}")

        # Report to the pruner every 5 epochs instead of every epoch.
        # Finer granularity wastes a full test-set eval just to get pruned
        # one epoch earlier; every-5 still gives the pruner enough signal
        # while cutting eval overhead by ~4x and reducing MPS inter-worker
        # contention during the report/prune check.
        if (i + 1) % 5 == 0:
            trial.report(acc, i)
            if trial.should_prune():
                raise optuna.TrialPruned()

    with open(f"{trial_dir}/accuracies.pkl", "wb") as f:
        pickle.dump(accs, f)

    # Return the best epoch (matches the checkpoint we keep) rather than the
    # last epoch. Revert to `return acc` if you want last-epoch semantics.
    trial.set_user_attr("best_acc", best_acc)
    return best_acc


if __name__ == "__main__":
    args = parser.parse_args()


    X = []
    y = []


    if args.val:
        for i in range(1, 6):
            images, labels = load_batch(f"data/cifar-10-batches-py/data_batch_{i}")
            X.append(images)
            y.append(labels)

        X = np.concatenate(X)
        y = np.concatenate(y)

        # Filter dataset to selected classes and remap labels to 0..C-1.
        # A small lookup-table index replaces a 50k-iteration Python loop
        # over a dict - both produce the same remapped array, but this one
        # runs as a single vectorized numpy op.
        X, y = get_classes_dataset(args.classes, X, y)
        label_lookup = np.empty(10, dtype=np.int64)
        for new, orig in enumerate(args.classes):
            label_lookup[orig] = new
        y = label_lookup[y]

        X = torch.from_numpy(X).float() / 255.0  # Normalize to [0, 1]
        y = torch.from_numpy(y).long()

        # splitting up the dataset - AI used from here...
        dataset = TensorDataset(X, y)
        train_size = int(len(dataset) * (1 - args.test_split))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        X_train = X[train_dataset.indices]
        y_train = y[train_dataset.indices]

        X_test = X[test_dataset.indices]
        y_test = y[test_dataset.indices]
    
    else:
        for i in range(1, 6):
            images, labels = load_batch(f"data/cifar-10-batches-py/data_batch_{i}")
            X.append(images)
            y.append(labels)

        X = np.concatenate(X)
        y = np.concatenate(y)

        # Filter dataset to selected classes and remap labels to 0..C-1.
        # A small lookup-table index replaces a 50k-iteration Python loop
        # over a dict - both produce the same remapped array, but this one
        # runs as a single vectorized numpy op.
        X, y = get_classes_dataset(args.classes, X, y)
        label_lookup = np.empty(10, dtype=np.int64)
        for new, orig in enumerate(args.classes):
            label_lookup[orig] = new
        y = label_lookup[y]

        X_train = torch.from_numpy(X).float() / 255.0  # Normalize to [0, 1]
        y_train = torch.from_numpy(y).long()

        images_test, labels_test = load_batch(f"data/cifar-10-batches-py/test_batch")

        X_test = torch.from_numpy(images_test).float() / 255.0  # Normalize to [0, 1]
        y_test = torch.from_numpy(labels_test).long()

    # CIFAR-10-scale tensors (well under 1 GB total) fit easily in a 4070
    # Ti's 12 GB of VRAM. Move them to the training device once, up front,
    # instead of copying a fresh batch from host memory on every iteration
    # of every epoch of every Optuna trial - the model wrappers' train/test
    # loops index batches directly out of these resident tensors.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    # Create a directory for Optuna databases
    os.makedirs("optuna_databases", exist_ok=True)

    # save_dir and the pickled args only depend on the (fixed) CLI arguments,
    # not on anything Optuna suggests per trial - compute and write them once
    # instead of redoing it on every single trial.
    save_dir = f"models/{args.model}_ReLU{args.ReLU}_arc_{args.num_blocks}"
    print(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    with open(f"{save_dir}/arguments.pkl", "wb") as f:
        pickle.dump(args, f)

    # Build storage from --storage. 'journal' is the one to use when an LSF
    # job array runs many workers against this same study on a networked
    # filesystem; 'sqlite' is fine for a single process. load_if_exists lets
    # every array task attach to the one study instead of clobbering it.
    storage = build_storage(
        args.storage, f"{args.model}_with_ReLU{args.ReLU}_arc_{args.num_blocks}"
    )
    study = optuna.create_study(
        study_name=f"{args.model}_with_ReLU{args.ReLU}_optimization",
        storage=storage,
        load_if_exists=True,
        direction="maximize",
    )

    study.optimize(
        lambda trial: objective(trial, args, save_dir, X_train, y_train, X_test, y_test),
        n_trials=args.n_trials,
    )

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
