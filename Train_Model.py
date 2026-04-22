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

models = {"Quad_neuron":Quad_neuron, "Quad_adapters":Quad_adapters, "Baseline":Baseline, "LoRA":LoRA}
model_names = ["Quad_neuron", "Quad_adapters", "Baseline", "LoRA"]
model_map = dict(zip(model_names, models))

# python Train_Model.py --model Baseline --ReLU 2 --epoch 5 --num_blocks [3, 3, 3]

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument("--model", dest="model", choices=model_names)
parser.add_argument("--ReLU", dest="ReLU", type=int, choices=[0, 1, 2], help="0 is no ReLU, \n1 is ReLU between layers but not in the neurons, \n2 is ReLU between layers AND in the neurons")
parser.add_argument("--epoch", dest="epoch", type=int)
# parser.add_argument("--start_epoch", dest="start_epoch")
parser.add_argument("--save_epoch", dest="save_epoch", type=int, help="Save the model every x epochs, if it has a higher accuracy", default=1)
parser.add_argument("--batch_size", dest="batch_size", type=int, default=128)
parser.add_argument("--lr", dest="lr", type=float, default=1E-3)
parser.add_argument("--num_blocks", dest="num_blocks")
# parser.add_argument("--block_type", dest="block_type", help="The type of block used in the model", default=BasicBlock)
parser.add_argument("--classes", dest="classes", default=[i for i in range(10)])
parser.add_argument("--load_path", dest="load_path", default=None)


best_acc = 0

def main():
    global args, best_acc
    args = parser.parse_args()

    model_name = models[args.model]

    if not model_name == Quad_adapters:
        model = model_name(args.ReLU, args.num_blocks, classes=args.classes)
    else:
        model = model_name(args.ReLU, args.num_blocks, args.load_path, classes=args.classes)

    if model_name != Quad_adapters and args.load_path != None:
        model.load(args.load_path)


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


    accs = {}
    
    for i in range(args.epoch // args.save_epoch):
        model.train(X, y, args.batch_size, args.lr, args.save_epoch)

        acc = model.test(X_test, y_test)
        accs[i*args.save_epoch] = acc

        if acc > best_acc:
            best_acc = acc
            model.save(f"models/{args.model}_ReLU{args.ReLU}/epoch{i*args.save_epoch}")

    with open(f"models/{args.model}_ReLU{args.ReLU}/accuracies.pkl", "wb") as f:
        pickle.dump(accs, f)
    
    with open(f"models/{args.model}_ReLU{args.ReLU}/arguments.pkl", "wb") as f:
        pickle.dump(args, f)

if __name__ == '__main__':
    main()