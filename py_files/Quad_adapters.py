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

        # Load Pretrained
        self.load_pretrained = True
        self.pretrained_path = "models/baseline_Baseline.pth"

        # Classes
        self.classes=  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


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



def get_classes_dataset(classes, X = X, y = y):

    mask = ""
    for i in classes:
        mask = mask + f"(y == {i}) | "

    mask = mask[:-3]
    mask = eval(mask)
    X_filtered = X[mask]
    y_filtered = y[mask]
    
    print(f"Selected classes: {classes}")
    print(f"Total samples: {len(X_filtered)}")
    
    return X_filtered, y_filtered

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


def qresnet20(config):
    return QResNet(config, BasicBlock, [3, 3, 3])

def load_pretrained(model, pretrained_path):
    pretrained_dict = torch.load(pretrained_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Filter out mismatched layers
    model_dict = model.state_dict()
    
    # Only keep pretrained layers that match in shape
    pretrained_dict_filtered = {
        k: v for k, v in pretrained_dict.items() 
        if k in model_dict and v.shape == model_dict[k].shape
    }
    
    # Update model state dict
    model_dict.update(pretrained_dict_filtered)
    model.load_state_dict(model_dict, strict=False)  # strict=False allows missing/unexpected keys
    
    # Freeze layers as before
    for name, param in model.named_parameters():
        if ('conv1r' in name or 'conv2r' in name):
            param.requires_grad = False
    
    return model

def train_model(X, y, config=config, model_fn=None):
    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=config.batch_size,
        shuffle=True
    )
    if config.load_pretrained:
        model = load_pretrained(model_fn(config), pretrained_path=config.pretrained_path)
    else:
        model = model_fn(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Only optimize parameters that require gradients
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    # Verify what's being trained
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,} / {total_params:,} params ({100*trainable_params/total_params:.1f}%)")

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

def test_model(path, model):
    model.eval()  # disable dropout/batchnorm training behavior
    correct = 0
    total = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    X, y = load_batch(path)
    X = torch.from_numpy(X).float() / 255.0  # Normalize to [0, 1]
    y = torch.from_numpy(y).long()
    X, y = get_classes_dataset(config.classes, X, y)
    train_loader = DataLoader(TensorDataset(X, y), batch_size=128)

    with torch.no_grad():  # disable gradient computation
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # get class with highest score
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')

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

qresnet20_model = train_model(X=X, y=y, model_fn=qresnet20)

# qresnet20_model = load_model("qresnet20_noReLU.pth", qresnet20)

# save_model(qresnet20_model, "qresnet20")

test_model("data/cifar-10-batches-py/test_batch", qresnet20_model)

# import json
# from pathlib import Path

# def notebook_to_clean_py(notebook_path, output_path=None):
#     notebook_path = Path(notebook_path)
#     if output_path is None:
#         output_path = notebook_path.with_suffix(".py")
#     else:
#         output_path = Path(output_path)

#     with notebook_path.open("r", encoding="utf-8") as f:
#         nb = json.load(f)

#     code_chunks = []
#     for cell in nb.get("cells", []):
#         if cell.get("cell_type") == "code":
#             source = cell.get("source", [])
#             if isinstance(source, list):
#                 code = "".join(source)
#             else:
#                 code = source
#             if code.strip():
#                 code_chunks.append(code.rstrip())

#     output_text = "\n\n".join(code_chunks) + "\n"

#     with output_path.open("w", encoding="utf-8") as f:
#         f.write(output_text)

#     print(f"Saved clean Python file to: {output_path}")

# # Example usage:
# notebook_to_clean_py("Quad_Neuron.ipynb")
# # notebook_to_clean_py("input.ipynb", "output.py")
