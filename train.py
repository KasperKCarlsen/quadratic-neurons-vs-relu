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

from utils import load_batch, save_model



input_channels = 3
path_train = "data/cifar-10-batches-py/data_batch_1"
path_test = "data/cifar-10-batches-py/test_batch"
epochs = 50
relu = True
blocks = [3,3,3]
minst = False
cifar10 = True



def train_model(X,y,model_type_str):
    train_loader = DataLoader(TensorDataset(X, y), batch_size=128, shuffle=True)

    # Build model from existing net_name (e.g., 'qresnet1202')
    model = globals()[model_type_str]()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


    # Train
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {running_loss/total:.4f} | "
            f"Acc: {correct/total:.4f}"
        )
    return model


if True:
    X, y = load_batch("data/cifar-10-batches-py/data_batch_2")
    X = torch.from_numpy(X).float() / 255.0  # Normalize to [0, 1]
    y = torch.from_numpy(y).long()

qresnet20_model = train_model(X=X, y=y, model_type_str="qresnet20")

save_model(qresnet20_model, "qresnet20")