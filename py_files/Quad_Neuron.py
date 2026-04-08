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

#torchvision.datasets.CIFAR10(root='./data', download=True) #run once

class Config:
    def __init__(self):
        # Training
        self.epochs = 25
        self.batch_size = 128
        self.lr = 1e-3

        # Architecture
        self.relu_mode = 1   # 0,1,2
        self.blocks = [3, 3, 3]


config = Config()

def save_model(model, name, relu):
    torch.save(model.state_dict(), f'models/{name}_ReLU_{relu}.pth')

def load_model(path,model_type):
    model = model_type()
    model.load_state_dict(torch.load(f"models/{path}"))
    return model

def load_batch(path):
    with open(path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        
    images = batch[b'data']      # (10000, 3072)
    labels = batch[b'labels']    # length 10000

    images = images.reshape(10000, 3, 32, 32)
    return images, np.array(labels)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, config, stride=1, option='A'):
        super().__init__()
        self.config = config

        self.conv1r = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=True)
        init.kaiming_normal_(self.conv1r.weight)

        self.conv1g = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=True)
        self.conv1g.weight.data.fill_(0)
        self.conv1g.bias.data.fill_(1)

        self.conv1b = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=True)
        self.conv1b.weight.data.fill_(0)
        self.conv1b.bias.data.fill_(0)

        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2r = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        init.kaiming_normal_(self.conv2r.weight)

        self.conv2g = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.conv2g.weight.data.fill_(0)
        self.conv2g.bias.data.fill_(1)

        self.conv2b = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.conv2b.weight.data.fill_(0)
        self.conv2b.bias.data.fill_(0)

        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = LambdaLayer(
                lambda x: F.pad(x[:, :, ::2, ::2],
                                (0, 0, 0, 0, planes//4, planes//4))
            )

    def forward(self, x):
        relu = self.config.relu_mode

        out = self.bn1(self.conv1r(x)*self.conv1g(x) + self.conv1b(x.pow(2)))

        if relu == 2:
            out = F.relu(out)

        out = self.bn2(self.conv2r(out)*self.conv2g(out) + self.conv2b(out.pow(2)))
        out += self.shortcut(x)

        if relu in [1, 2]:
            out = F.relu(out)

        return out


class QResNet(nn.Module):
    def __init__(self, config, block, num_blocks):
        super().__init__()
        self.config = config
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(block, 16, num_blocks[0])
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.linear = nn.Linear(64, len(config.classes))

    def _make_layer(self, block, planes, num_blocks, stride=1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for s in strides:
            layers.append(block(self.in_planes, planes, self.config, s))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.config.relu_mode == 0:
            out = self.bn1(self.conv1(x))
        else:
            out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = F.avg_pool2d(out, out.size(3))
        out = out.view(out.size(0), -1)

        return self.linear(out)


def qresnet_relu_0(config):
    return QResNet(config, BasicBlock, [3, 3, 3])

def train_model(X, y, config, model_fn):
    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=config.batch_size,
        shuffle=True
    )

    model = model_fn(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config.epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * labels.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        print(f"Epoch {epoch+1}/{config.epochs} | "
              f"Loss: {loss_sum/total:.4f} | Acc: {correct/total:.4f}")

    return model

if config.use_mnist:
    df = pd.read_csv("data/mnist_train.csv", header=None)

    y = torch.tensor(df.iloc[:, 0].values, dtype=torch.long)
    X = torch.tensor(df.iloc[:, 1:].values, dtype=torch.float32) / 255.0  # normalize to [0, 1]

    num_pixels = X.shape[1]
    side = int(num_pixels ** 0.5)
    assert side * side == num_pixels, f"Expected square image, got {num_pixels} pixels."
    X = X.view(-1, 1, side, side)  # NCHW for Conv2d: [batch, channel, height, width]

if config.use_cifar10:
    X = []
    y = []
    for i in range(1, 6):
        images, labels = load_batch(f"data/cifar-10-batches-py/data_batch_{i}")
        X.append(images)
        y.append(labels)

    X = np.concatenate(X)
    y = np.concatenate(y)

    X = torch.from_numpy(X).float() / 255.0  # Normalize to [0, 1]
    y = torch.from_numpy(y).long()

def test_model(path, model):
    model.eval()  # disable dropout/batchnorm training behavior
    correct = 0
    total = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    X, y = load_batch(path)
    X = torch.from_numpy(X).float() / 255.0  # Normalize to [0, 1]
    y = torch.from_numpy(y).long()

    train_loader = DataLoader(TensorDataset(X, y), batch_size=128, shuffle=True)

    with torch.no_grad():  # disable gradient computation
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # get class with highest score
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    model.train()

qresnet20_model = train_model(X=X, y=y, model_fn=qresnet_relu_1, config=config)

# qresnet20_model = load_model("qresnet20_noReLU.pth", qresnet20)

# save_model(qresnet20_model, "qresnet20")

test_model("data/cifar-10-batches-py/test_batch", qresnet20_model)

import json
from pathlib import Path

def notebook_to_clean_py(notebook_path, output_path=None):
    notebook_path = Path(notebook_path)
    if output_path is None:
        output_path = notebook_path.with_suffix(".py")
    else:
        output_path = Path(output_path)

    with notebook_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    code_chunks = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            if isinstance(source, list):
                code = "".join(source)
            else:
                code = source
            if code.strip():
                code_chunks.append(code.rstrip())

    output_text = "\n\n".join(code_chunks) + "\n"

    with output_path.open("w", encoding="utf-8") as f:
        f.write(output_text)

    print(f"Saved clean Python file to: {output_path}")

# Example usage:
notebook_to_clean_py("Quad_Neuron.ipynb")
# notebook_to_clean_py("input.ipynb", "output.py")
