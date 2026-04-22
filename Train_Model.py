import torch
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

from funcs import *
from py_files.Quad_Neuron import Quad_neuron
from py_files.Quad_adapters import Quad_adapters
from py_files.Baseline import Baseline
from py_files.Baseline_with_LoRA import LoRA

model_names = [Quad_neuron, Quad_adapters, Baseline, LoRA]
# model_names = ["Quad_neuron", "Quad_adapters", "Baseline", "LoRA"]

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument("--model", dest="model", choices=model_names)
parser.add_argument("--ReLU", dest="ReLU", choices=[0, 1, 2], help="0 is no ReLU, \n1 is ReLU between layers but not in the neurons, \n2 is ReLU between layers AND in the neurons")
parser.add_argument("--epoch", dest="epoch")
parser.add_argument("--start_epoch", dest="start_epoch")
parser.add_argument("--save_epoch", dest="save_epoch", help="Save the model every x epochs, if it has a higher accuracy", default=1)
parser.add_argument("--batch_size", dest="batch_size", default=128)
parser.add_argument("--lr", dest="lr", default=1E-5)


def main():
    global args, best_prec1
    args = parser.parse_args()

    model = args.model

    X = []
    y = []

    X_test, y_test = load_batch(f"data/cifar-10-batches-py/test_batch")
    
    for i in range(1, 6):
        images, labels = load_batch(f"data/cifar-10-batches-py/data_batch_{i}")
        X.append(images)
        y.append(labels)

    X = np.concatenate(X)
    y = np.concatenate(y)

    X = torch.from_numpy(X).float() / 255.0  # Normalize to [0, 1]
    y = torch.from_numpy(y).long()

    X_test = torch.from_numpy(X_test).float() / 255.0  # Normalize to [0, 1]
    y_test = torch.from_numpy(y_test).long()


    best_acc = 0
    for i in range(args.epoch // args.save_epoch):
        model.train(X, y, args.batch_size, args.lr, args.save_epoch)
        model.test()

if __name__ == '__main__':
    main()