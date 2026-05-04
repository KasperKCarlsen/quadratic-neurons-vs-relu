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
import loralib as lora
import math


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


# LoRA Layer implementation
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Original weight (frozen)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # LoRA parameters (trainable)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Initialize
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_A)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Original computation (frozen)
        original = F.linear(x, self.weight, self.bias)
        # LoRA computation (trainable)
        lora = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return original + lora


# LoRA-enabled Conv2D (for your quad neuron convolutions)
class LoRAConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        rank=4,
        alpha=1.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        # Original convolution layer (frozen)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        # LoRA convolutions (trainable)
        # Using two small convolutions: A (reduces channels) and B (expands back)
        self.lora_A = nn.Conv2d(
            in_channels, rank, kernel_size, stride=stride, padding=padding, bias=False
        )
        self.lora_B = nn.Conv2d(
            rank, out_channels, 1, bias=False
        )  # 1x1 conv to combine

        # Initialize LoRA layers
        nn.init.zeros_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        # Original path (frozen weights)
        original = self.conv(x)

        # LoRA path (trainable)
        lora = self.lora_B(self.lora_A(x)) * self.scaling

        return original + lora

    def freeze_original(self):
        """Freeze the original convolution weights"""
        for param in self.conv.parameters():
            param.requires_grad = False


# BasicBlock with LoRA
class BasicBlockLoRA(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, config, stride=1, option="A", rank=4):
        super().__init__()
        self.config = config

        # Replace regular Conv2d with LoRAConv2d for each color channel
        self.conv1r = LoRAConv2d(in_planes, planes, 3, stride, 1, bias=True, rank=rank)
        self.conv1g = LoRAConv2d(in_planes, planes, 3, stride, 1, bias=True, rank=rank)
        self.conv1b = LoRAConv2d(in_planes, planes, 3, stride, 1, bias=True, rank=rank)

        # Initialize conv1g and conv1b to near identity
        with torch.no_grad():
            # For conv1g: make it identity-like
            if hasattr(self.conv1g, "conv"):
                self.conv1g.conv.weight.fill_(0)
                if self.conv1g.conv.bias is not None:
                    self.conv1g.conv.bias.fill_(1)
            # For conv1b: make it zero
            if hasattr(self.conv1b, "conv"):
                self.conv1b.conv.weight.fill_(0)
                if self.conv1b.conv.bias is not None:
                    self.conv1b.conv.bias.fill_(0)

        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2r = LoRAConv2d(planes, planes, 3, 1, 1, bias=True, rank=rank)
        self.conv2g = LoRAConv2d(planes, planes, 3, 1, 1, bias=True, rank=rank)
        self.conv2b = LoRAConv2d(planes, planes, 3, 1, 1, bias=True, rank=rank)

        # Initialize conv2g and conv2b to near identity
        with torch.no_grad():
            if hasattr(self.conv2g, "conv"):
                self.conv2g.conv.weight.fill_(0)
                if self.conv2g.conv.bias is not None:
                    self.conv2g.conv.bias.fill_(1)
            if hasattr(self.conv2b, "conv"):
                self.conv2b.conv.weight.fill_(0)
                if self.conv2b.conv.bias is not None:
                    self.conv2b.conv.bias.fill_(0)

        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = LambdaLayer(
                lambda x: F.pad(
                    x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4)
                )
            )

    def forward(self, x):
        # Using only the R channel convolutions as in your original
        out = self.bn1(self.conv1r(x))
        out = F.relu(out)
        out = self.bn2(self.conv2r(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# Modified QResNet with LoRA
class QResNetLoRA(nn.Module):
    def __init__(self, config, block, num_blocks, rank=4):
        super().__init__()
        self.config = config
        self.in_planes = 16
        self.rank = rank

        # First conv layer (can also use LoRA or keep frozen)
        self.conv1 = LoRAConv2d(
            config.input_channels, 16, 3, 1, 1, bias=False, rank=rank
        )
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], rank=rank)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, rank=rank)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, rank=rank)

        self.linear = nn.Linear(64, config.num_classes)

    def _make_layer(self, block, planes, num_blocks, stride=1, rank=4):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for s in strides:
            layers.append(block(self.in_planes, planes, self.config, s, rank=rank))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size(3))
        out = out.view(out.size(0), -1)
        return self.linear(out)


# Function to freeze original weights and only train LoRA parameters
def freeze_original_weights(model):
    """Freeze original weights, only train LoRA parameters"""
    for name, param in model.named_parameters():
        # For the simple LoRAConv2d implementation
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
        elif "conv" in name and ("lora" not in name):
            # Freeze original convolution weights
            param.requires_grad = False
        elif "bn" in name or "linear" in name:
            # Keep batch norm and final linear layer trainable
            param.requires_grad = True
        else:
            param.requires_grad = False


# Modified training function for LoRA


class LoRA:

    def __init__(
        self,
        Relu,
        num_blocks,
        block=BasicBlockLoRA,
        classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        trial=None,
    ):
        self.model = QResNetLoRA(Relu, block, num_blocks, classes)
        self.trial = trial

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        if self.trial is not None:
            self.optimizer_name = self.trial.suggest_categorical(
                "optimizer", ["Adam", "AdamW", "SGD"]
            )
            self.lr = self.trial.suggest_float("lr", 1e-6, 1e-2, log=True)
            if self.optimizer_name == "AdamW":
                self.weight_decay = self.trial.suggest_float(
                    "weight_decay", 1e-5, 1e-1, log=True
                )
                self.optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                )
            elif self.optimizer_name == "SGD":
                self.momentum = self.trial.suggest_float("momentum", 0.5, 0.99)
                self.optimizer = torch.optim.SGD(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=self.lr,
                    momentum=self.momentum,
                )
            else:
                self.optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=self.lr,
                )
        else:
            self.optimizer_name = "Adam"
            self.lr = 1e-3
            self.optimizer = None

    def train(self, X, y, batch_size, lr, epochs, rank=4):
        loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
        self.model.to(self.device)

        # Freeze original weights, only train LoRA parameters
        freeze_original_weights(self.model)

        # Count trainable parameters
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.model.parameters())
        print(
            f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)"
        )

        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr
            )

        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.model.train()
            total, correct, loss_sum = 0, 0, 0

            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                loss_sum += loss.item() * labels.size(0)
                preds = outputs.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Loss: {loss_sum/total:.4f} | Acc: {correct/total:.4f}"
            )

    def test(self, X, y):
        self.model.eval()  # disable dropout/batchnorm training behavior
        correct = 0
        total = 0

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        train_loader = DataLoader(TensorDataset(X, y), batch_size=128, shuffle=True)

        with torch.no_grad():  # disable gradient computation
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)  # get class with highest score
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy

    def save(self, path):
        torch.save(self.model.state_dict(), f"{path}")

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
