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

    def validate(self) -> None:
        if self.model not in MODEL_MAP:
            raise ValueError(f"Unknown model '{self.model}'. Choose from: {list(MODEL_MAP)}")
        if self.ReLU not in (0, 1, 2):
            raise ValueError("ReLU must be 0, 1, or 2")
        if self.epoch <= 0:
            raise ValueError("epoch must be positive")
        if self.save_epoch <= 0:
            raise ValueError("save_epoch must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0 < self.test_split < 1:
            raise ValueError("test_split must be between 0 and 1")


def build_model(config: Config):
    model_class = MODEL_MAP[config.model]

    if model_class is Quad_adapters:
        model = model_class(
            config.ReLU,
            list(config.num_blocks),
            config.load_path,
            classes=list(config.classes),
        )
    else:
        model = model_class(
            config.ReLU,
            list(config.num_blocks),
            classes=list(config.classes),
        )
        if config.load_path is not None:
            model.load(config.load_path)

    return model


def load_cifar10_training_batches(data_dir: str):
    X = []
    y = []

    for i in range(1, 6):
        images, labels = load_batch(os.path.join(data_dir, f"data_batch_{i}"))
        X.append(images)
        y.append(labels)

    X = torch.from_numpy(np.concatenate(X)).float() / 255.0
    y = torch.from_numpy(np.concatenate(y)).long()
    return X, y


def split_dataset(X: torch.Tensor, y: torch.Tensor, test_split: float, seed: int):
    dataset = TensorDataset(X, y)
    train_size = int(len(dataset) * (1 - test_split))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    X_train = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
    y_train = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])

    X_test = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
    y_test = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])

    return X_train, y_train, X_test, y_test


def maybe_limit(X: torch.Tensor, y: torch.Tensor, limit: int | None):
    if limit is None:
        return X, y
    return X[:limit], y[:limit]


def main() -> None:
    config = Config()
    config.validate()

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    model = build_model(config)

    X, y = load_cifar10_training_batches(config.data_dir)
    X_train, y_train, X_test, y_test = split_dataset(X, y, config.test_split, config.seed)

    X_train, y_train = maybe_limit(X_train, y_train, config.train_limit)
    X_test, y_test = maybe_limit(X_test, y_test, config.test_limit)

    best_acc = 0
    accs = {}
    save_dir = os.path.join(config.model_root, f"{config.model}_ReLU{config.ReLU}")
    print(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    for i in range(config.epoch // config.save_epoch):
        epoch_number = i * config.save_epoch
        model.train(X_train, y_train, config.batch_size, config.lr, config.save_epoch)

        acc = model.test(X_test, y_test)
        accs[epoch_number] = acc

        if acc > best_acc:
            best_acc = acc
            model.save(os.path.join(save_dir, f"epoch{epoch_number}"))

    with open(os.path.join(save_dir, "accuracies.pkl"), "wb") as f:
        pickle.dump(accs, f)

    with open(os.path.join(save_dir, "config.pkl"), "wb") as f:
        pickle.dump(asdict(config), f)


if __name__ == "__main__":
    main()
