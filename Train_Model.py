import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torchvision
import torchvision.transforms as transforms
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

MODEL_MAP = {
    "Quad_neuron": Quad_neuron,
    "Quad_adapters": Quad_adapters,
    "Baseline": Baseline,
    "LoRA": LoRA,
}

parser = argparse.ArgumentParser(description="Proper ResNets for CIFAR10 in PyTorch")

parser.add_argument("--model", choices=list(MODEL_MAP.keys()),required=True,help="Model name")

parser.add_argument("--ReLU", type=int, choices=[0, 1, 2], required=True, help="0: no ReLU, 1: between layers only, 2: between layers and inside neurons")

parser.add_argument("--epoch", type=int, required=True)

parser.add_argument("--save_epoch", type=int, default=1, help="Save the model every x epochs if accuracy improves")

parser.add_argument("--batch_size", type=int, default=128)

parser.add_argument("--val", dest="val",type = bool, default=False,  help="Whether to use a validation split from the training data instead of the official test set")


parser.add_argument("--lr", type=float, default=1e-3)

# Recommended CLI usage:
# --num_blocks 3 3 3
parser.add_argument("--num_blocks", type=int, nargs="+", required=True, help="Number of blocks per stage, e.g. --num_blocks 3 3 3")

# If you prefer this style instead:
# --classes 0 1 2 3 4 5 6 7 8 9
parser.add_argument("--classes", type=int, nargs="+", default=list(range(10)))

parser.add_argument("--load_path", type=str, default=None)

parser.add_argument("--test_split", dest="test_split", type=float, default=0.2, help="Fraction of the loaded data to reserve for testing")


def objective(trial):
    args = parser.parse_args()
    

    model_class = MODEL_MAP[args.model]
    if model_class is not Quad_adapters:
        model = model_class(args.ReLU, args.num_blocks, classes=args.classes,trial=trial)
    else:
        model = model_class(
            args.ReLU,
            args.num_blocks,
            args.load_path,
            classes=args.classes,
            trial=trial
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

        X_train = torch.from_numpy(X).float() / 255.0  # Normalize to [0, 1]
        y_train = torch.from_numpy(y).long()


        images_test, labels_test = load_batch(f"data/cifar-10-batches-py/test_batch")

        X_test = torch.from_numpy(images_test).float() / 255.0  # Normalize to [0, 1]
        y_test = torch.from_numpy(labels_test).long()


    best_acc = 0
    accs = {}
    save_dir = f"models/{args.model}_ReLU{args.ReLU}"
    print(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    for i in range(args.epoch):
        batch_size = trial.suggest_int("batch_size", 32, 512)
        model.train(X_train, y_train, batch_size, args.lr, args.save_epoch)

        acc = model.test(X_test, y_test)
        accs[i*args.save_epoch] = acc

        if acc > best_acc:
            best_acc = acc
            model.save(f"{save_dir}/epoch{i*args.save_epoch}")

        trial.report(acc, i)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    with open(f"{save_dir}/accuracies.pkl", "wb") as f:
        pickle.dump(accs, f)
    
    with open(f"{save_dir}/arguments.pkl", "wb") as f:
        pickle.dump(args, f)

    return acc


if __name__ == '__main__':
    study = optuna.create_study(direction="maximize")

    study.optimize(objective, n_trials=100, timeout=600)

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