import os
import pickle
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.utils.data import TensorDataset

from funcs import *
from py_files.Quad_Neuron import Quad_neuron
from py_files.Quad_adapters import Quad_adapters
from py_files.Baseline import Baseline
from py_files.Baseline_with_LoRA import LoRA

MODEL_MAP = {
    "Quad_neuron": Quad_neuron,
    "Quad_adapters": Quad_adapters,
    "Baseline": Baseline,
    "LoRA": LoRA,
}


@dataclass(frozen=True)
class Config:
    """Edit these values before submitting the job to the HPC cluster."""

    # Model settings
    model: str = "Baseline"  # Options: "Quad_neuron", "Quad_adapters", "Baseline", "LoRA"
    ReLU: int = 1  # 0: no ReLU, 1: between layers only, 2: between layers and inside neurons
    num_blocks: tuple[int, ...] = (3, 3, 3)
    classes: tuple[int, ...] = tuple(range(10))
    load_path: str | None = None

    # Training settings
    val: bool = True
    epoch: int = 10
    save_epoch: int = 1
    batch_size: int = 128
    lr: float = 1e-3
    test_split: float = 0.2
    seed: int = 42

    # Paths
    data_dir: str = "data/cifar-10-batches-py"
    model_root: str = "models"

    # Keep the original debugging subset behavior. Set to None to use full train/test sets.
    train_limit: int | None = 100
    test_limit: int | None = 100



def main():
    args = Config()

    model_class = MODEL_MAP[args.model]
    if model_class is not Quad_adapters:
        model = model_class(args.ReLU, args.num_blocks, classes=args.classes)
    else:
        model = model_class(
            args.ReLU,
            args.num_blocks,
            args.load_path,
            classes=args.classes
        )

    if model_class is not Quad_adapters and args.load_path is not None:
        model.load(args.load_path)


    X = []
    y = []

    if args.val:
        for i in range(1, 6):
            images, labels = load_batch(f"data/cifar-10-batches-py/data_batch_{i}")
            X.append(images)
            y.append(labels)
        images, labels = load_batch()

        X = np.concatenate(X)
        y = np.concatenate(y)

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

        X_train = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
        y_train = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])

        X_test = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
        y_test = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])
    
    else:
        for i in range(1, 6):
            images, labels = load_batch(f"data/cifar-10-batches-py/data_batch_{i}")
            X.append(images)
            y.append(labels)

        X = np.concatenate(X)
        y = np.concatenate(y)

        X = torch.from_numpy(X).float() / 255.0  # Normalize to [0, 1]
        y = torch.from_numpy(y).long()

        X_train = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
        y_train = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])

        images_test, labels_test = load_batch(f"data/cifar-10-batches-py/test_batch")

        X_test = torch.from_numpy(images_test).float() / 255.0  # Normalize to [0, 1]
        y_test = torch.from_numpy(labels_test).long()


    best_acc = 0
    accs = {}
    save_dir = f"models/{args.model}_ReLU{args.ReLU}"
    print(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    for i in range(args.epoch // args.save_epoch):
        model.train(X_train[0:100], y_train[0:100], args.batch_size, args.lr, args.save_epoch)

        acc = model.test(X_test[0:100], y_test[0:100])
        accs[i*args.save_epoch] = acc

        if acc > best_acc:
            best_acc = acc
            model.save(f"{save_dir}/epoch{i*args.save_epoch}")

    with open(f"{save_dir}/accuracies.pkl", "wb") as f:
        pickle.dump(accs, f)
    
    with open(f"{save_dir}/arguments.pkl", "wb") as f:
        pickle.dump(args, f)

if __name__ == '__main__':
    main()